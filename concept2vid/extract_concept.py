import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    XCLIPTextModel,
    # AutoProcessor,
    # XCLIPVisionModel,
    XCLIPProcessor,
    XCLIPModel,
)
from torch.autograd import Variable
import torch.nn.functional as F

from accelerate.logging import get_logger


logger = get_logger(__name__, log_level="INFO")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim  # 64
        self._num_embeddings = num_embeddings  # 512

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class QuantizedTransformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
     Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """

    # Constructor
    def __init__(
        self,
        num_tokens,
        d_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        dim_feedforward=2048,
        batch_first=True,
        vq_n_embedding=512,
        vq_embedding_dim=64,
        commitment_cost=0.25,
        decay=0.99,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.d_model = d_model

        # LAYERS
        # self.transformer = nn.Transformer(
        #         d_model=dim_model,
        #         nhead=num_heads,
        #         num_encoder_layers=num_encoder_layers,
        #         num_decoder_layers=num_decoder_layers,
        #         dropout=dropout_p,
        #     )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout_p
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, num_heads, dim_feedforward, dropout_p
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.d_model = d_model
        self.num_heads = num_heads

        self.batch_first = batch_first

        self.out = nn.Linear(d_model, num_tokens)
        self.vec_quantizer = VectorQuantizerEMA(
            vq_n_embedding, vq_embedding_dim, commitment_cost, decay
        )

    def forward(self, src, tgt):
        # we permute to obtain size (sequence length, batch_size, dim_model),
        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError(
                "the feature number of src and tgt must be equal to d_model"
            )

        memory = self.encoder(src)
        # print(" memory ", memory.shape)
        commit_loss, vq_output, _, _ = self.vec_quantizer(memory.unsqueeze(-1))
        # print("vq_output ", vq_output.shape)
        vq_output = vq_output.squeeze(-1)
        # print("vq_output ", vq_output.shape)

        output = self.decoder(tgt, vq_output)
        # print("output ", output.shape)
        return output, vq_output, commit_loss


def get_x_clip_masked_model(
    load_weights, output_dim=512, d_model=512, heads=8, n_layers=4, dropout=0.5
):
    assert d_model % heads == 0
    assert dropout < 1
    model = QuantizedTransformer(
        output_dim, d_model, heads, n_layers, n_layers, dropout
    )

    if load_weights is not None:
        logger.info("Loading Quantized Transformer pretrained weights...")
        checkpoint = torch.load(f"{load_weights}", map_location="cpu")

        state_dict = checkpoint
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith("module."):
                # remove prefix
                state_dict[k[len("module.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        model.load_state_dict(state_dict)
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)

    # model.to(device)

    return model


def get_models_inference(weight_path=None, get_quantized_transformer=True):
    models = {}
    if get_quantized_transformer:
        models["quantized_transformer_model"] = get_x_clip_masked_model(weight_path)

    models["tokenizer"] = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
    models["model"] = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
    # models["model"] = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32").to(
    #     device
    # )

    return models


def get_models_training(weight_path):
    models = {}
    models["quantized_transformer_model"] = get_x_clip_masked_model(weight_path)
    # model_name = "microsoft/xclip-base-patch32"
    # processor = XCLIPProcessor.from_pretrained(model_name)
    # model = XCLIPModel.from_pretrained(model_name)

    models["tokenizer"] = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
    models["model"] = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")
    # models["model"] = XCLIPModel.from_pretrained(model_name).to(
    #     device
    # )
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     models["model"] = nn.DataParallel(models["model"])

    return models


def get_quantized_feature(models, text, video=None, device="cpu", dtype=torch.float32):
    logger.info(f"get_quantized_feature {dtype=}")
    video_features = None
    if video is not None:
        inputs = process_input(models["tokenizer"], video, text, dtype).to(device)
        # print(inputs.keys())
        outputs = models["model"](**inputs, output_hidden_states=True, return_dict=True)
        video_features = outputs["video_embeds"].unsqueeze(-2)
        text_features = outputs["text_model_output"].last_hidden_state
        # print("video_feature ", video_features.shape) # torch.Size([2, 512])
        seperator = torch.zeros(
            video_features.shape[0], 1, video_features.shape[-1]
        ).to(device, dtype=dtype)
        # eos = torch.ones(1, feature_subsets.shape[1])

        # transformer_input = torch.cat((sos, feature_subsets, text_features, eos), dim=0)
        transformer_input = torch.cat(
            (video_features, seperator, text_features), dim=-2
        )

    else:
        text_inputs = models["tokenizer"](
            list(text), padding=True, return_tensors="pt"
        ).to(device)
        text_features = models["model"](**text_inputs)
        text_features = text_features.last_hidden_state
        seperator = torch.zeros(text_features.shape[0], 1, text_features.shape[-1]).to(
            device, dtype=dtype
        )
        transformer_input = torch.cat((seperator, text_features), dim=-2)

    # print("text_feature ", text_features.shape) # torch.Size([2, 4, 512])
    # print("transformer input ", transformer_input.shape) # torch.Size([2, 6, 512])

    _, output, _ = models["quantized_transformer_model"](
        transformer_input, transformer_input
    )
    return output
    # target = torch.zeros(output.shape[0], output.shape[1], 768)
    # # print("output transformer ", output.shape) # torch.Size([2, 6, 512])
    # target[:, :, : output.shape[-1]] = output
    # print(target.shape)
    # return target


def process_input(processor, videos, texts, dtype=torch.float32):
    # videos = tf.constant(videos)
    batch, t, h, w, c = videos.shape
    # print(batch, t, h, w, c) # 256 32 3 256 256
    videos = videos.reshape(batch * t, h, w, c)
    inputs = processor(
        text=texts, videos=list(videos), return_tensors="pt", padding=True
    )
    # print(inputs["pixel_values"].shape)  # torch.Size([1, 8192, 3, 224, 224])
    _, b_t, c, h, w = inputs["pixel_values"].shape
    inputs["pixel_values"] = inputs["pixel_values"].reshape(batch, t, c, h, w)
    inputs["input_ids"] = torch.Tensor(inputs["input_ids"])
    # print(inputs['pixel_values'].shape) # torch.Size([256, 32, 3, 224, 224])
    # print(inputs['input_ids'].shape) # torch.Size([256, 10])
    return inputs


if __name__ == "__main__":

    # while training diffusion model
    # getting models
    weight_path = "/projects/adversarialprototypicalcontrastivelearning/zeroshot/MVS-Transformer/Transformer/output_2023-03-05-13-57-08/checkpoint_best.pth.tar"
    models = get_models_training(weight_path)
    video = torch.rand(2, 8, 3, 224, 224)
    labels = ["playing sports" for j in range(2)]  # torch.randint(10, (256, 10))

    quantized_feature = get_quantized_feature(models, labels, video)
    print(quantized_feature.shape)

    # while inference of diffusion model
    # getting models
    weight_path = "/projects/adversarialprototypicalcontrastivelearning/zeroshot/MVS-Transformer/Transformer/output_2023-03-05-13-57-08/checkpoint_best.pth.tar"
    models = get_models_inference(weight_path)
    labels = ["playing sports" for j in range(2)]  # torch.randint(10, (256, 10))

    quantized_feature = get_quantized_feature(models, labels)
    print(quantized_feature.shape)
