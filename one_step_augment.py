# one_step_augment.py

import torch
from PIL import Image
from typing import Tuple

from augmentation.base_augmentation import GenerativeMixup
from diffusers import StableDiffusionImg2ImgPipeline, UNet2DConditionModel
from diffusers.utils import logging

# Tắt các thanh tiến trình không cần thiết
logging.disable_progress_bar()


def load_swiftbrush_embeddings(embed_path: str, pipeline: StableDiffusionImg2ImgPipeline):
    """
    Hàm helper để tải các file embedding .bin được tạo ra từ kịch bản training của bạn.
    """
    # Tải file .bin
    embed_ckpt = torch.load(embed_path, map_location="cpu")
    placeholder_tokens = embed_ckpt["placeholder_tokens"]
    learned_embeds = embed_ckpt["text_embeds"]

    # Thêm các token mới vào tokenizer của pipeline
    pipeline.tokenizer.add_tokens(placeholder_tokens)
    token_ids = pipeline.tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize và gán trọng số embedding mới
    pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer))
    with torch.no_grad():
        pipeline.text_encoder.get_input_embeddings().weight[token_ids] = learned_embeds.to(
            pipeline.text_encoder.get_input_embeddings().weight.dtype
        )

    # Tạo mapping từ tên lớp sang placeholder token (ví dụ: 'Bear' -> '<class_0>')
    # Giả định rằng class_names được sắp xếp đúng thứ tự
    # Đây là một điểm cần chú ý, cần đảm bảo dataset của bạn có thuộc tính class_names
    name2placeholder = {f"class_{i}": token for i, token in enumerate(placeholder_tokens)}

    return name2placeholder


class OneStepLoraAugment(GenerativeMixup):
    pipe = None

    def __init__(
            self,
            base_model_name: str,
            swiftbrush_checkpoint_path: str,
            lora_path: str,
            embed_path: str,
            prompt: str = "a photo of a {name}",
            device="cuda",
            **kwargs,
    ):
        super(OneStepLoraAugment, self).__init__()

        # Chỉ tải pipeline một lần để tiết kiệm bộ nhớ
        if OneStepLoraAugment.pipe is None:
            print("Initializing One-Step Augmentation Pipeline...")

            # Tải pipeline Img2Img từ base model (sd-turbo)
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
            )

            # Tải và thay thế UNet bằng UNet của SwiftBrush
            print(f"Loading SwiftBrush UNet from {swiftbrush_checkpoint_path}...")
            unet = UNet2DConditionModel.from_pretrained(swiftbrush_checkpoint_path, torch_dtype=torch.float16)
            pipe.unet = unet

            # Tải trọng số LoRA đã finetune
            print(f"Loading LoRA weights from {lora_path}...")
            pipe.load_lora_weights(lora_path)

            # Tải các embedding đã học
            print(f"Loading custom embeddings from {embed_path}...")
            self.name2placeholder = load_swiftbrush_embeddings(embed_path, pipe)

            # Không thay đổi scheduler, giữ nguyên scheduler của SD-Turbo
            pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            if pipe.safety_checker is not None:
                pipe.safety_checker = None

            OneStepLoraAugment.pipe = pipe
            print("Pipeline initialized successfully.")

        self.prompt = prompt

    def forward(
            self,
            image: Image.Image,
            label: int,  # Không dùng trực tiếp, sẽ lấy từ metadata
            metadata: dict,
            strength: float = 0.5,
            resolution=512,
    ) -> Tuple[Image.Image, int]:

        canvas = [img.resize((resolution, resolution), Image.BILINEAR) for img in image]

        # Lấy tên lớp mục tiêu từ metadata
        target_class_name = metadata.get("name", "")

        # Tìm placeholder token tương ứng (ví dụ: '<class_0>')
        placeholder_token = self.name2placeholder.get(target_class_name, target_class_name)

        # Tạo prompt với placeholder token
        # Ví dụ: "a photo of a <class_0> bear"
        final_prompt = f"a photo of a {placeholder_token} {metadata.get('super_class', '')}".strip()
        print(f"Generating with prompt: '{final_prompt}' and strength: {strength}")

        # Các tham số cho pipeline one-step
        kwargs = dict(
            image=canvas,
            prompt=[final_prompt] * len(canvas),
            strength=strength,
            num_inference_steps=1,  # QUAN TRỌNG: Chỉ 1 bước
            guidance_scale=0.0,  # QUAN TRỌNG: Tham số chuẩn cho SD-Turbo
            num_images_per_prompt=len(canvas),
        )

        with torch.no_grad(), torch.autocast("cuda"):
            outputs = self.pipe(**kwargs).images

        # Resize ảnh về kích thước gốc
        output_images = []
        for orig, out in zip(image, outputs):
            output_images.append(out.resize(orig.size, Image.BILINEAR))

        return output_images, metadata['label']