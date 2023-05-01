import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    XCLIPTextModel,
    AutoProcessor,
    XCLIPVisionModel,
    XCLIPProcessor,
    XCLIPModel,
)
from torch.autograd import Variable
import torch.nn.functional as F
import copy
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = (
            self.alpha
            * (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True) + self.eps)
            + self.bias
        )
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        # print("scores ", scores.shape)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm_1(x)
        if mask is not None:
            x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        else:
            x = x + self.dropout_1(self.attn(x2, x2, x2))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask=None, trg_mask=None):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


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


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask=None):
        # x = self.embed(src)
        # x = self.pe(x)
        x = src
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask=None, trg_mask=None):
        # x = self.embed(trg)
        # x = self.pe(x)
        x = trg
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, output_dim, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(d_model, N, heads, dropout)
        self.decoder = Decoder(d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, output_dim)

    def forward(self, src, trg, src_mask=None, trg_mask=None):

        e_outputs = self.encoder(src, src_mask)
        # print("DECODER")
        # print("encoder output ", e_outputs.shape) #  torch.Size([498, 12, 768])
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        # print("decoder output ", d_output.shape) # torch.Size([498, 5, 768])
        output = self.out(d_output)
        # return output, e_outputs[:, -1, :]
        return output, e_outputs


class QuantizedOldTransformer(nn.Module):
    def __init__(
        self,
        output_dim,
        d_model,
        N,
        heads,
        dropout,
        vq_n_embedding=768,
        vq_embedding_dim=64,
        commitment_cost=0.25,
        decay=0.99,
    ):
        super().__init__()
        self.encoder = Encoder(d_model, N, heads, dropout)
        self.decoder = Decoder(d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, output_dim)
        self.vec_quantizer = VectorQuantizerEMA(
            vq_n_embedding, vq_embedding_dim, commitment_cost, decay
        )

    def forward(self, src, trg, src_mask=None, trg_mask=None):

        e_outputs = self.encoder(src, src_mask)

        commit_loss, vq_output, _, _ = self.vec_quantizer(e_outputs.unsqueeze(-1))
        # print("vq_output ", vq_output.shape)
        vq_output = vq_output.squeeze(-1)
        # print("vq output ", vq_output.shape) # torch.Size([750, 11, 768])
        # print("DECODER")
        # print("encoder output ", e_outputs.shape) #  torch.Size([498, 12, 768])
        d_output = self.decoder(trg, vq_output, src_mask, trg_mask)
        # print("decoder output ", d_output.shape) # torch.Size([498, 5, 768])
        output = self.out(d_output)
        # return output, e_outputs[:, -1, :]
        return output, e_outputs, vq_output, commit_loss


def get_model(load_weights, output_dim=768, device="cpu"):
    model = Transformer(output_dim, output_dim, 4, 8, 0.5)

    if load_weights is not None:
        checkpoint = torch.load(f"{load_weights}", map_location=device)

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


def get_quantized_old_model(load_weights, output_dim=768):
    model = QuantizedOldTransformer(output_dim, output_dim, 4, 8, 0.5)

    if load_weights is not None:
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

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    return model


def get_models_inference(weight_path=None, device="cpu"):
    models = {}
    models["transformer_model"] = (
        None if weight_path is None else get_model(weight_path, device=device)
    )

    models["tokenizer"] = AutoTokenizer.from_pretrained(
        "microsoft/xclip-large-patch14-kinetics-600"
    )
    models["model"] = XCLIPTextModel.from_pretrained(
        "microsoft/xclip-large-patch14-kinetics-600"
    )
    # models["model"] = XCLIPTextModel.from_pretrained(
    #     "microsoft/xclip-large-patch14-kinetics-600"
    # ).to(device)

    return models


def get_models_training(weight_path, device="cpu"):
    models = {}
    models["transformer_model"] = get_model(weight_path, device=device)
    model_name = "microsoft/xclip-large-patch14-kinetics-600"
    # processor = XCLIPProcessor.from_pretrained(model_name)
    # model = XCLIPModel.from_pretrained(model_name)

    models["tokenizer"] = XCLIPProcessor.from_pretrained(
        "microsoft/xclip-large-patch14-kinetics-600"
    )
    models["model"] = XCLIPModel.from_pretrained(
        "microsoft/xclip-large-patch14-kinetics-600"
    )
    # models["model"] = XCLIPModel.from_pretrained(
    #     "microsoft/xclip-large-patch14-kinetics-600"
    # ).to(device)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     models["model"] = nn.DataParallel(models["model"])

    return models


def get_quantized_feature(models, text, video=None, device="cpu"):
    video_features = None
    if video is not None:
        inputs = process_input(models["tokenizer"], video, text).to(device)
        # print(inputs.keys())
        outputs = models["model"](**inputs, output_hidden_states=True, return_dict=True)
        video_features = outputs["video_embeds"].unsqueeze(-2).to(device)
        text_features = outputs["text_model_output"].last_hidden_state.to(device)
        # print("video_feature ", video_features.shape) # torch.Size([2, 512])
        seperator = torch.zeros(
            video_features.shape[0], 1, video_features.shape[-1]
        ).to(device)
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
            device
        )
        transformer_input = torch.cat((seperator, text_features), dim=-2)

    # print("text_feature ", text_features.shape) # torch.Size([2, 4, 512])
    # print("transformer input ", transformer_input.shape) # torch.Size([2, 6, 512])

    _, output = models["transformer_model"](transformer_input, transformer_input)
    return output
    # target = torch.zeros(output.shape[0], output.shape[1], 768)
    # # print("output transformer ", output.shape) # torch.Size([2, 6, 512])
    # target[:, :, : output.shape[-1]] = output
    # # print(target.shape)
    # return target


def process_input(processor, videos, texts):
    # videos = tf.constant(videos)
    batch, t, h, w, c = videos.shape
    # print(batch, t, h, w, c) # 256 32 3 256 256
    videos = videos.reshape(batch * t, h, w, c)
    inputs = processor(
        text=texts, videos=list(videos), return_tensors="pt", padding=True
    )
    # print(inputs["pixel_values"].shape)  # torch.Size([1, 8192, 3, 224, 224])
    _, b_t, c, h, w = inputs["pixel_values"].shape
    inputs["pixel_values"] = Variable(inputs["pixel_values"].reshape(batch, t, c, h, w))
    inputs["input_ids"] = torch.tensor(inputs["input_ids"])
    # print(inputs['pixel_values'].shape) # torch.Size([256, 32, 3, 224, 224])
    # print(inputs['input_ids'].shape) # torch.Size([256, 10])
    return inputs


if __name__ == "__main__":

    # while training diffusion model
    # getting models
    weight_path = "/projects/adversarialprototypicalcontrastivelearning/zeroshot/MVS-Transformer/Transformer/output_2023-04-14-18-56-08/checkpoint_best.pth.tar"
    models = get_models_training(weight_path)
    video = torch.rand(1, 8, 3, 224, 224)
    labels = ["playing sports" for j in range(1)]  # torch.randint(10, (256, 10))

    quantized_feature = get_quantized_feature(models, labels, video)
    print(quantized_feature.shape)

    # while inference of diffusion model
    # getting models
    weight_path = "/projects/adversarialprototypicalcontrastivelearning/zeroshot/MVS-Transformer/Transformer/output_2023-04-14-18-56-08/checkpoint_best.pth.tar"
    models = get_models_inference(weight_path)
    labels = ["playing sports" for j in range(2)]  # torch.randint(10, (256, 10))

    quantized_feature = get_quantized_feature(models, labels)
    print(quantized_feature.shape)
