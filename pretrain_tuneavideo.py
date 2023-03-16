import argparse
import datetime
import logging
import inspect
import math
import os
import json
from typing import Dict, Optional, Tuple
from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed  # , DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, XCLIPTextModel, AutoTokenizer

from torchmetrics.functional.multimodal.clip_score import (
    clip_score as clip_score_metric,
)

from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.data.dataset import TuneAVideoDataset, TuneAVideoKineticsPretrainDataset
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.util import save_videos_grid
from einops import rearrange

from concept2vid.extract_concept import (
    get_models_training,
    get_models_inference,
    get_quantized_feature,
)


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    validation_steps: int = 100,
    text_encoder_name: str = "clip",
    trainable_modules: Tuple[str] = (
        "attn1.to_q",
        "attn2.to_q",
        "attn_temp",
    ),
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
    quantized_transformer_weights_path: str = None,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        # kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, "config.yaml"))
        logger.info(f"Output directory: {output_dir}")

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_path, subfolder="scheduler"
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    train_models = None
    tokenizer = None
    text_encoder = None
    if text_encoder_name == "clip":
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_path, subfolder="text_encoder"
        )
    elif text_encoder_name == "xclip-base":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
        text_encoder = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
    elif text_encoder_name == "xclip-large":
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/xclip-large-patch14-kinetics-600"
        )
        text_encoder = XCLIPTextModel.from_pretrained(
            "microsoft/xclip-large-patch14-kinetics-600"
        )
    elif text_encoder_name == "quantized":
        train_models = get_models_training(
            quantized_transformer_weights_path,
            device=accelerator.device,
            dtype=weight_dtype,
        )
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path, subfolder="unet"
    )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    # text_encoder.requires_grad_(False)
    if text_encoder_name != "quantized":
        text_encoder.requires_grad_(False)
    else:
        train_models["model"].requires_grad_(False)
        train_models["quantized_transformer_model"].requires_grad_(False)

    unet.requires_grad_(False)
    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = (
            learning_rate
            * gradient_accumulation_steps
            * train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    pretrain_dataset = TuneAVideoKineticsPretrainDataset(**train_data)
    if pretrain_dataset.tokenizer is None:
        pretrain_dataset.tokenizer = tokenizer
    pretrain_dataloader = torch.utils.data.DataLoader(
        pretrain_dataset, batch_size=train_batch_size
    )

    # # Get the training dataset
    # train_dataset = TuneAVideoDataset(**train_data)

    # # Preprocessing the dataset
    # train_dataset.prompt_ids = tokenizer(
    #     train_dataset.prompt,
    #     max_length=tokenizer.model_max_length,
    #     padding="max_length",
    #     truncation=True,
    #     return_tensors="pt",
    # ).input_ids[0]

    # # DataLoaders creation:
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=train_batch_size
    # )

    # Get the validation pipeline
    if text_encoder_name == "quantized":
        val_models = get_models_inference(get_quantized_transformer=False)
        val_tokenizer = val_models["tokenizer"]
        val_model = val_models["model"]
        val_model.to(accelerator.device)
    validation_pipeline = TuneAVideoPipeline(
        vae=vae,
        text_encoder=text_encoder if text_encoder_name != "quantized" else val_model,
        tokenizer=tokenizer if text_encoder_name != "quantized" else val_tokenizer,
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(
            pretrained_model_path, subfolder="scheduler"
        ),
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    # unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     unet, optimizer, train_dataloader, lr_scheduler
    # )
    unet, optimizer, pretrain_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, pretrain_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    # weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    # if text_encoder is not None:
    #     text_encoder.to(accelerator.device, dtype=weight_dtype)
    if text_encoder_name != "quantized":
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    else:
        train_models["model"].to(accelerator.device)  # , dtype=weight_dtype)
        train_models["quantized_transformer_model"].to(
            accelerator.device, dtype=weight_dtype
        )
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(pretrain_dataloader) / gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # Initialize metric
    # clip_score_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(accelerator.device, dtype=weight_dtype)
    clip_scores = {}

    # Train!
    total_batch_size = (
        train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(pretrain_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(pretrain_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            # videopaths = [Path(videopath) for videopath in batch["videopath"]]
            # for videopath in videopaths:
            #     logger.info(f"File: {str(videopath.relative_to(videopath.parents[1]))}")

            with accelerator.accumulate(unet):
                # Convert videos to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                if text_encoder_name != "quantized":
                    encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]
                else:
                    pixel_values = rearrange(
                        pixel_values, "(b f) c h w -> b f c h w", f=video_length
                    )
                    encoder_hidden_states = get_quantized_feature(
                        train_models,
                        batch["prompt"],
                        video=pixel_values,
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )

                if encoder_hidden_states.shape[-1] < 768:
                    encoder_hidden_states = F.pad(
                        encoder_hidden_states,
                        (0, 768 - encoder_hidden_states.shape[-1]),
                        "constant",
                        0.0,
                    )

                # Get the target for loss depending on the prediction type
                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.prediction_type}"
                    )

                # Predict the noise residual and compute loss
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if global_step % validation_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            output_dir, f"samples/sample-{global_step}.gif"
                        )
                        clip_scores[f"samples/sample-{global_step}.gif"] = {}
                        samples = []
                        generator = torch.Generator(device=latents.device)
                        generator.manual_seed(seed)
                        for idx, prompt in enumerate(validation_data.prompts):
                            sample = validation_pipeline(
                                prompt,
                                generator=generator,
                                **validation_data,
                                quantized_transformer=train_models[
                                    "quantized_transformer_model"
                                ]
                                if text_encoder_name == "quantized"
                                else None,
                            ).videos
                            save_videos_grid(
                                sample,
                                os.path.join(
                                    output_dir,
                                    f"samples/sample-{global_step}/{prompt}.gif",
                                ),
                            )
                            samples.append(sample)
                            video = rearrange(sample.squeeze(0), "c t h w -> t c h w")
                            clip_score = (
                                clip_score_metric(
                                    video,
                                    [prompt] * video.shape[0],
                                    "openai/clip-vit-base-patch32",
                                )
                                .detach()
                                .item()
                            )
                            clip_scores[f"samples/sample-{global_step}.gif"][
                                prompt
                            ] = clip_score
                            with open(
                                os.path.join(
                                    output_dir,
                                    f"samples/sample-{global_step}-scores.json",
                                ),
                                "w",
                            ) as f:
                                json.dump(
                                    clip_scores[f"samples/sample-{global_step}.gif"],
                                    f,
                                    indent=4,
                                )
                        samples = torch.concat(samples)
                        save_videos_grid(samples, save_path)
                        logger.info(f"Saved samples to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    if accelerator.is_main_process:
        with open(os.path.join(output_dir, f"all-scores.json"), "w") as f:
            json.dump(clip_scores, f, indent=4)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = TuneAVideoPipeline.from_pretrained(
            pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/pretrain.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
