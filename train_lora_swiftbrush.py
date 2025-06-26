# train_lora_swiftbrush.py
# Dựa trên train_lora.py từ diffmix, đã được chỉnh sửa cho SwiftBrush v2

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
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

# MODIFICATION: Import các lớp Dataset từ các tệp bạn đã cung cấp
from dataset import T2I_DATASET_NAME_MAPPING

os.environ["WANDB_DISABLED"] = "true"

logger = get_logger(__name__, log_level="INFO")

PLACEHOLDER_TOKEN_TEMPLATE = "<class_{}>"


# MODIFICATION: Gỡ bỏ Initializer token không cần thiết, vì chúng ta sẽ khởi tạo từ "bear"
# INITIALIZER_TOKEN = 42170

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/sd-turbo",  # MODIFICATION: Mặc định là sd-turbo, nền tảng của swiftbrushv2
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    # MODIFICATION: Thêm đối số cho đường dẫn checkpoint của SwiftBrush v2 UNet
    parser.add_argument(
        "--swiftbrush_checkpoint_path",
        type=str,
        required=True,
        help="Path to the SwiftBrush v2 UNet checkpoint directory.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="bear",  # MODIFICATION: Mặc định là 'bear' để khớp với bear.py
        help="The name of the dataset to train on.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="A folder containing the training data for the bears (e.g., the parent folder of '10_gobi_bear bear').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="swiftbrush-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,  # MODIFICATION: Giảm batch size mặc định để tiết kiệm VRAM
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
        default=5e-4,  # MODIFICATION: Tăng một chút LR cho TI để học token mới nhanh hơn
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
        default=0,  # MODIFICATION: Bỏ warmup cho scheduler 'constant'
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",  # MODIFICATION: Mặc định là fp16 để huấn luyện nhanh hơn
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,  # MODIFICATION: Tăng rank mặc định để có khả năng biểu diễn tốt hơn
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
    # Thêm các đối số khác nếu cần...
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
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # 1. Tải các thành phần từ mô hình nền (sd-turbo)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")

    # 2. Tải UNet từ checkpoint của SwiftBrush v2
    # MODIFICATION: Tải UNet từ đường dẫn riêng biệt do người dùng cung cấp
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

    # MODIFICATION: Di chuyển text_encoder và unet lên device sau để xử lý embedding

    # 3. Chuẩn bị Dataset và Placeholder Tokens
    with accelerator.main_process_first():
        train_dataset = T2I_DATASET_NAME_MAPPING[args.dataset_name](
            image_train_dir=args.train_data_dir,
            resolution=args.resolution,
            use_placeholder=True,  # Quan trọng: để dataset tạo prompt với placeholder
        )

    placeholder_tokens = [PLACEHOLDER_TOKEN_TEMPLATE.format(i) for i in range(train_dataset.num_classes)]
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)

    if num_added_tokens != train_dataset.num_classes:
        raise ValueError(f"Added {num_added_tokens} tokens, but there are {train_dataset.num_classes} classes.")

    name2placeholder = {name: placeholder for name, placeholder in zip(train_dataset.class_names, placeholder_tokens)}
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    train_dataset.name2placeholder = name2placeholder

    # Thay đổi kích thước token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data

    # Khởi tạo các token mới bằng embedding của một token có sẵn (ví dụ: "bear")
    initializer_token_id = tokenizer.convert_tokens_to_ids("bear")
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    # Chỉ cho phép huấn luyện các embedding của token mới
    text_encoder.text_model.embeddings.token_embedding.requires_grad_(True)

    # 4. Thiết lập LoRA cho UNet
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            # Fallback for unexpected layer names
            hidden_size = unet.config.block_out_channels[0]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.rank,
        ).to(accelerator.device)  # Di chuyển lên device ngay
    unet.set_attn_processor(lora_attn_procs)

    # Di chuyển UNet và Text Encoder lên device
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    lora_layers = AttnProcsLayers(unet.attn_processors)

    # 5. Optimizer
    params_to_optimize = [
        {"params": lora_layers.parameters(), "lr": args.learning_rate},
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

        # Tokenize captions trong collate_fn
        captions = [example["caption"] for example in examples]
        input_ids = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=4,
    )

    # 7. Scheduler và chuẩn bị với Accelerator
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Chuẩn bị tất cả
    lora_layers, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # Cần phải unwrap text_encoder để giữ lại bản sao của embedding gốc
    unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
    orig_embeds_params = unwrapped_text_encoder.get_input_embeddings().weight.data.clone()

    # 8. Vòng lặp Huấn luyện
    global_step = 0
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(args.num_train_epochs):
        unet.train()
        text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, text_encoder):
                # Mã hóa ảnh sang không gian latent
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Thêm nhiễu vào latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # ==================== MODIFICATION CORE ====================
                # Đối với mô hình một bước, luôn sử dụng timestep cuối cùng.
                # Đây là thay đổi quan trọng nhất so với mã gốc của diffmix.
                timesteps = torch.full(
                    (bsz,),
                    noise_scheduler.config.num_train_timesteps - 1,
                    device=latents.device,
                    dtype=torch.long
                )
                # =========================================================

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Lấy text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Dự đoán nhiễu
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Tính loss
                # Vì prediction_type của sd-turbo là 'epsilon'
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)  # Tối ưu hóa bộ nhớ

                # Giữ nguyên các embedding không được huấn luyện
                with torch.no_grad():
                    # Lấy lại index của các token không phải là placeholder
                    all_token_indices = torch.arange(len(tokenizer))
                    placeholder_mask = torch.zeros(len(tokenizer), dtype=torch.bool)
                    placeholder_mask[placeholder_token_ids] = True
                    non_placeholder_indices = all_token_indices[~placeholder_mask]

                    # Khôi phục embedding gốc
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
        unet = accelerator.unwrap_model(unet)
        unet.save_attn_procs(args.output_dir)

        # Lưu cả các embedding đã học
        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
        learned_embeds = unwrapped_text_encoder.get_input_embeddings().weight[placeholder_token_ids]
        learned_embeds_dict = {
            "text_embeds": learned_embeds,
            "placeholder_tokens": placeholder_tokens
        }
        torch.save(learned_embeds_dict, os.path.join(args.output_dir, "learned_embeds.bin"))

    accelerator.end_training()


if __name__ == "__main__":
    main()