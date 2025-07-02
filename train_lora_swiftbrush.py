import argparse
import logging
import math
import os
import re
import random
import shutil
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig, get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from diffusers.optimization import get_scheduler

from dataset import T2I_DATASET_NAME_MAPPING

os.environ["WANDB_DISABLED"] = "true"
logger = get_logger(__name__, log_level="INFO")
PLACEHOLDER_TOKEN_TEMPLATE = "<class_{}>"


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/sd-turbo", required=True)
    parser.add_argument("--swiftbrush_checkpoint_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="swiftbrush-finetuned-lora")
    parser.add_argument("--resume_from_checkpoint", type=str, default="latest")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--ti_learning_rate", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_main_process:
        logger.info("=" * 40)
        for k, v in vars(args).items():
            logger.info(f"  - {k}: {v}")
        logger.info("=" * 40)
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.swiftbrush_checkpoint_path)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
    vae.to(accelerator.device, dtype=weight_dtype)

    with accelerator.main_process_first():
        train_dataset = T2I_DATASET_NAME_MAPPING[args.dataset_name](
            image_train_dir=args.train_data_dir, resolution=args.resolution, use_placeholder=True,
        )
    placeholder_tokens = [PLACEHOLDER_TOKEN_TEMPLATE.format(i) for i in range(train_dataset.num_classes)]
    tokenizer.add_tokens(placeholder_tokens)
    placeholder_tokens = [PLACEHOLDER_TOKEN_TEMPLATE.format(i) for i in range(train_dataset.num_classes)]
    tokenizer.add_tokens(placeholder_tokens)
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    name2placeholder = {name: placeholder for name, placeholder in zip(train_dataset.class_names, placeholder_tokens)}
    train_dataset.name2placeholder = name2placeholder
    text_encoder.resize_token_embeddings(len(tokenizer))
    text_encoder.text_model.embeddings.token_embedding.requires_grad_(True)

    lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"],
        lora_dropout=0.1, bias="none",
    )
    unet.add_adapter(lora_config)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device)

    params_to_optimize = [
        {"params": list(filter(lambda p: p.requires_grad, unet.parameters())), "lr": args.learning_rate},
        {"params": text_encoder.text_model.embeddings.token_embedding.parameters(), "lr": args.ti_learning_rate},
    ]
    optimizer = torch.optim.AdamW(params_to_optimize, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-08)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        input_ids = tokenizer(
            [example["caption"] for example in examples], padding="max_length", truncation=True,
            max_length=tokenizer.model_max_length, return_tensors="pt",
        ).input_ids
        return {"pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float(), "input_ids": input_ids}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=4,
    )

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    )
    unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
    orig_embeds_params = unwrapped_text_encoder.get_input_embeddings().weight.data.clone()

    global_step = 0
    first_epoch = 0  # Khởi tạo first_epoch

    if args.resume_from_checkpoint:
        path = None
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            if len(dirs) > 0:
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1]

        if path is not None:
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, path))
                global_step = int(path.split("-")[1])
                num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
                first_epoch = global_step // num_update_steps_per_epoch
            except (KeyError, RuntimeError) as e:
                accelerator.print(f"Could not load state completely due to {str(e)}. Continuing training.")
        else:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting a new training run.")

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, text_encoder):
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.full((bsz,), noise_scheduler.config.num_train_timesteps - 1, device=latents.device,
                                       dtype=torch.long)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    all_token_indices = torch.arange(len(tokenizer))
                    placeholder_mask = torch.zeros(len(tokenizer), dtype=torch.bool)
                    placeholder_mask[placeholder_token_ids] = True
                    non_placeholder_indices = all_token_indices[~placeholder_mask]
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[non_placeholder_indices] = \
                    orig_embeds_params[non_placeholder_indices]

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": loss.detach().item()}, step=global_step)

                # --- ĐÂY LÀ PHẦN THAY ĐỔI ---
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved training state to {save_path}")

                        # <<< BẮT ĐẦU PHẦN BỔ SUNG >>>
                        # Lưu cả embedding của Textual Inversion tại mỗi checkpoint
                        logger.info(f"Saving learned embeddings to {save_path}...")
                        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
                        learned_embeds = unwrapped_text_encoder.get_input_embeddings().weight[placeholder_token_ids]
                        learned_embeds_dict = {
                            "text_embeds": learned_embeds,
                            "placeholder_tokens": placeholder_tokens
                        }
                        torch.save(learned_embeds_dict, os.path.join(save_path, "learned_embeds.bin"))
                        logger.info(f"Successfully saved learned_embeds.bin in {save_path}")

                        # Lưu cả trọng số LoRA tại mỗi checkpoint
                        logger.info(f"Saving LoRA weights to {save_path}...")
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=lora_state_dict
                        )
                        logger.info(f"Successfully saved pytorch_lora_weights.safetensors in {save_path}")
                        # <<< KẾT THÚC PHẦN BỔ SUNG >>>

            progress_bar.set_postfix(loss=loss.detach().item())
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # (Phần lưu kết quả cuối cùng giữ nguyên, vì giờ nó chỉ là backup)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Saving final model...")
        unet = accelerator.unwrap_model(unet)
        text_encoder = accelerator.unwrap_model(text_encoder)
        unet_lora_layers = get_peft_model_state_dict(unet)
        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
        )
        learned_embeds = text_encoder.get_input_embeddings().weight[placeholder_token_ids]
        learned_embeds_dict = {
            "text_embeds": learned_embeds,
            "placeholder_tokens": placeholder_tokens
        }
        torch.save(learned_embeds_dict, os.path.join(args.output_dir, "learned_embeds.bin"))

    accelerator.end_training()


if __name__ == "__main__":
    main()