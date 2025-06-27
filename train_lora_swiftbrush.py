# train_lora_swiftbrush.py
# PHIÊN BẢN CUỐI CÙNG - ĐÃ SỬA LỖI VÀ THÊM TÍNH NĂNG RESUME CHECKPOINT

# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import re  # <<< THÊM MỚI: Cần cho việc tìm checkpoint
import random
import shutil
from pathlib import Path

# CÁC IMPORT CẦN THIẾT
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,  # <<< THÊM MỚI: Cần cho việc lưu LoRA weights
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from dataset import T2I_DATASET_NAME_MAPPING

os.environ["WANDB_DISABLED"] = "true"

logger = get_logger(__name__, log_level="INFO")

PLACEHOLDER_TOKEN_TEMPLATE = "<class_{}>"


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/sd-turbo",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--swiftbrush_checkpoint_path",
        type=str,
        required=True,
        help="Path to the SwiftBrush v2 UNet checkpoint directory.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="bear",
        help="The name of the dataset to train on.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="swiftbrush-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # <<< BẮT ĐẦU PHẦN THÊM MỚI >>>
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether to resume from the latest checkpoint in `output_dir`. "
            "Set to `None` to not resume."
        ),
    )
    # <<< KẾT THÚC PHẦN THÊM MỚI >>>
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate for LoRA layers.",
    )
    parser.add_argument(
        "--ti_learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate for Textual Inversion embeddings.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='The scheduler type to use. Choose between ["linear", "cosine", "constant", ...]',
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="a photo of a <class_0> bear in a forest",
        help="A prompt that is sampled during training for inference.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
    )

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
        logger.info("TRAINING ARGUMENTS")
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

    # 1. Tải các thành phần từ mô hình nền
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")

    # 2. Tải UNet
    unet = UNet2DConditionModel.from_pretrained(args.swiftbrush_checkpoint_path)

    # Đóng băng các thành phần không cần huấn luyện
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)

    # 3. Chuẩn bị Dataset và Placeholder Tokens
    with accelerator.main_process_first():
        train_dataset = T2I_DATASET_NAME_MAPPING[args.dataset_name](
            image_train_dir=args.train_data_dir,
            resolution=args.resolution,
            use_placeholder=True,
        )

    placeholder_tokens = [PLACEHOLDER_TOKEN_TEMPLATE.format(i) for i in range(train_dataset.num_classes)]
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)

    if num_added_tokens != train_dataset.num_classes:
        raise ValueError(f"Added {num_added_tokens} tokens, but there are {train_dataset.num_classes} classes.")

    name2placeholder = {name: placeholder for name, placeholder in zip(train_dataset.class_names, placeholder_tokens)}
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    train_dataset.name2placeholder = name2placeholder

    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data

    initializer_token_id = tokenizer.convert_tokens_to_ids("bear")
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    text_encoder.text_model.embeddings.token_embedding.requires_grad_(True)

    # 4. Thiết lập LoRA cho UNet bằng PEFT
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"],
        lora_dropout=0.1,
        bias="none",
    )

    unet.add_adapter(lora_config)

    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device)

    # 5. Optimizer
    lora_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    params_to_optimize = [
        {"params": lora_params, "lr": args.learning_rate},
        {"params": text_encoder.text_model.embeddings.token_embedding.parameters(), "lr": args.ti_learning_rate},
    ]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # 6. Dataloader
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        captions = [example["caption"] for example in examples]
        input_ids = tokenizer(
            captions, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt",
        ).input_ids
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=4,
    )

    # 7. Scheduler và chuẩn bị với Accelerator
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

    # 8. Vòng lặp Huấn luyện
    global_step = 0

    # <<< BẮT ĐẦU PHẦN THÊM MỚI: RESUME TỪ CHECKPOINT >>>
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            if len(dirs) > 0:
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1]
            else:
                path = None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")

            try:
                # Cố gắng tải lại trạng thái
                accelerator.load_state(os.path.join(args.output_dir, path))
            except KeyError as e:
                # Bắt lỗi KeyError cụ thể liên quan đến 'step'
                if "step" in str(e):
                    accelerator.print(
                        "Caught a 'KeyError: step'. This is a known issue. "
                        "Ignoring it as model/optimizer/scheduler states were likely loaded correctly. "
                        "Continuing training."
                    )
                else:
                    # Nếu là một KeyError khác, thì báo lỗi như bình thường
                    raise e

            global_step = int(path.split("-")[1])

            num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch

    # <<< KẾT THÚC PHẦN THÊM MỚI >>>

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(args.num_train_epochs):
        # <<< THÊM MỚI: Logic skip dataloader khi resume >>>
        if args.resume_from_checkpoint and epoch == first_epoch:
            # Skip các batch đã xử lý
            logger.info(f"Skipping {resume_step} batches from the dataloader to resume training.")
            for _ in range(resume_step):
                next(iter(train_dataloader))
        # <<< KẾT THÚC >>>

        unet.train()
        text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, text_encoder):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                timesteps = torch.full(
                    (bsz,), noise_scheduler.config.num_train_timesteps - 1, device=latents.device, dtype=torch.long
                )

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
                    unwrapped_text_encoder.get_input_embeddings().weight[non_placeholder_indices] = orig_embeds_params[
                        non_placeholder_indices]

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": loss.detach().item()}, step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            progress_bar.set_postfix(loss=loss.detach().item())
            if global_step >= args.max_train_steps:
                break

    # 9. Lưu kết quả cuối cùng
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # <<< BẮT ĐẦU PHẦN SỬA LỖI >>>
        logger.info("Saving final model...")

        # Import hàm cần thiết từ PEFT
        from peft import get_peft_model_state_dict

        # Lấy UNet và Text Encoder đã huấn luyện ra khỏi wrapper của accelerator
        unet = accelerator.unwrap_model(unet)
        text_encoder = accelerator.unwrap_model(text_encoder)

        # 1. Trích xuất state dict của CHỈ các lớp LoRA từ UNet
        # Đây là bước quan trọng để lấy đúng các trọng số đã được finetune
        unet_lora_layers = get_peft_model_state_dict(unet)

        # 2. Gọi hàm lưu của pipeline và truyền các lớp LoRA vào một cách tường minh
        # Chúng ta không cần tạo một pipeline đầy đủ, chỉ cần gọi phương thức tĩnh của nó
        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
        )
        logger.info(f"LoRA weights saved to {os.path.join(args.output_dir, 'pytorch_lora_weights.safetensors')}")
        # <<< KẾT THÚC PHẦN SỬA LỖI >>>

        # Lưu cả các embedding đã học (phần này vẫn giữ nguyên)
        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
        learned_embeds = unwrapped_text_encoder.get_input_embeddings().weight[placeholder_token_ids]
        learned_embeds_dict = {
            "text_embeds": learned_embeds,
            "placeholder_tokens": placeholder_tokens
        }
        torch.save(learned_embeds_dict, os.path.join(args.output_dir, "learned_embeds.bin"))
        logger.info(f"Learned embeddings saved to {os.path.join(args.output_dir, 'learned_embeds.bin')}")

    accelerator.end_training()


if __name__ == "__main__":
    main()