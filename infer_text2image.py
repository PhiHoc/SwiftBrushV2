# infer_one_step_final.py
# Kịch bản inference chuẩn xác cuối cùng: Sử dụng Pipeline để tải LoRA và thực hiện inference 1 bước.

import torch
import typer
from pathlib import Path
from PIL import Image
from accelerate.utils import set_seed
# Sử dụng StableDiffusionPipeline làm nền tảng
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from torchvision.utils import save_image

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
        prompt: str = typer.Argument(..., help="Prompt to generate, e.g., 'a photo of a <class_0> bear'"),
        swiftbrush_checkpoint_path: Path = typer.Argument(...,
                                                          help="Path to the original SwiftBrush v2 UNet checkpoint directory."),
        lora_path: Path = typer.Argument(...,
                                         help="Path to the finetuned LoRA directory (the one containing the weights file)."),
        text_embeds_path: Path = typer.Argument(...,
                                                help="Path to the 'learned_embeds.bin' file containing the finetuned text embeddings."),
        output_dir: Path = typer.Option("generated_images_final", help="Path to the output directory.", dir_okay=True),
        base_model_name: str = typer.Option("stabilityai/sd-turbo", help="Base Hugging Face model name."),
        seed: int = typer.Option(42, help="A seed for reproducible generation."),
        nsamples: int = typer.Option(4, help="Number of images to generate."),
):
    """
    Generates high-quality ONE-STEP images using the correct pipeline-centric approach.
    """
    set_seed(seed)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")

    device = "cuda"
    weight_dtype = torch.float16

    # --- 1. Tải Pipeline cơ sở và ngay lập tức thay thế UNet ---
    print(f"Loading base pipeline from '{base_model_name}'...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_name,
        torch_dtype=weight_dtype
    )

    print(f"Loading and replacing with SwiftBrush v2 UNet from: {swiftbrush_checkpoint_path}")
    # Tải UNet của SwiftBrush và gán nó vào pipeline
    unet = UNet2DConditionModel.from_pretrained(swiftbrush_checkpoint_path, torch_dtype=weight_dtype)
    pipeline.unet = unet

    # --- 2. Cập nhật Text Encoder với các Embedding đã học (thao tác trên pipeline) ---
    print(f"Loading and applying custom text embeddings from: {text_embeds_path}")
    learned_embeds_dict = torch.load(text_embeds_path, map_location="cpu")
    placeholder_tokens = learned_embeds_dict["placeholder_tokens"]
    learned_embeds = learned_embeds_dict["text_embeds"]
    pipeline.tokenizer.add_tokens(placeholder_tokens)
    token_ids = pipeline.tokenizer.convert_tokens_to_ids(placeholder_tokens)
    pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer))
    with torch.no_grad():
        pipeline.text_encoder.get_input_embeddings().weight[token_ids] = learned_embeds.to(
            pipeline.text_encoder.get_input_embeddings().weight.dtype
        )

    # --- 3. Tải LoRA vào Pipeline (đây là cách làm đúng) ---
    print(f"Loading and applying Diffusers LoRA weights from: {lora_path}")
    pipeline.load_lora_weights(lora_path)

    # Chuyển toàn bộ pipeline đã được tùy chỉnh lên GPU
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)  # Tắt progress bar cho gọn

    # --- 4. Chạy Inference One-Step ---
    print(f"Generating {nsamples} images with prompt: '{prompt}'...")

    generator = torch.Generator(device=device)
    for i in range(nsamples):
        generator.manual_seed(seed + i)

        # Gọi thẳng pipeline với các tham số chuẩn cho model one-step
        image = pipeline(
            prompt,
            num_inference_steps=1,
            guidance_scale=0.0,
            generator=generator
        ).images[0]

        prompt_safe_name = "".join(c if c.isalnum() else "_" for c in prompt)[:50]
        output_path = output_dir / f"seed_{seed + i}_{prompt_safe_name}.png"

        image.save(output_path)
        print(f"Saved image to {output_path}")

    print("Generation complete.")


if __name__ == "__main__":
    app()