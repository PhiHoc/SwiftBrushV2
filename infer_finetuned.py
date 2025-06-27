# infer_finetuned.py
# Phiên bản cuối cùng, sử dụng Pipeline làm trung tâm để đảm bảo tính chính xác và ổn định.

import torch
import typer
from pathlib import Path
from accelerate.utils import set_seed
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
        output_dir: Path = typer.Option("generated_images", help="Path to the output directory.", dir_okay=True),
        base_model_name: str = typer.Option("stabilityai/sd-turbo", help="Base Hugging Face model name."),
        seed: int = typer.Option(42, help="A seed for reproducible generation."),
        nsamples: int = typer.Option(4, help="Number of images to generate."),
):
    """
    Generates images using a SwiftBrush v2 model finetuned with Diffusers LoRA.
    """
    set_seed(seed)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")

    device = "cuda"
    weight_dtype = torch.float16

    # --- 1. Tải Pipeline cơ sở và thay thế UNet ---
    print(f"Loading base pipeline from '{base_model_name}'...")
    # Tải pipeline hoàn chỉnh từ model gốc
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_name,
        torch_dtype=weight_dtype
    )

    print(f"Loading and replacing with SwiftBrush v2 UNet from: {swiftbrush_checkpoint_path}")
    # Tải UNet của SwiftBrush và thay thế UNet mặc định trong pipeline
    unet = UNet2DConditionModel.from_pretrained(swiftbrush_checkpoint_path, torch_dtype=weight_dtype)
    pipeline.unet = unet

    # --- 2. Cập nhật Text Encoder với các Embedding đã học ---
    print(f"Loading and applying custom text embeddings from: {text_embeds_path}")
    if not text_embeds_path.exists():
        raise FileNotFoundError(f"Text embeddings file not found: {text_embeds_path}")

    learned_embeds_dict = torch.load(text_embeds_path, map_location="cpu")
    placeholder_tokens = learned_embeds_dict["placeholder_tokens"]
    learned_embeds = learned_embeds_dict["text_embeds"]

    # Thêm token mới vào tokenizer của pipeline
    pipeline.tokenizer.add_tokens(placeholder_tokens)
    token_ids = pipeline.tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Thay đổi kích thước embedding của text encoder trong pipeline
    pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer))

    # Gán các embedding đã học
    with torch.no_grad():
        pipeline.text_encoder.get_input_embeddings().weight[token_ids] = learned_embeds.to(
            pipeline.text_encoder.get_input_embeddings().weight.dtype
        )

    # --- 3. Tải và áp dụng Trọng số LoRA vào Pipeline ---
    print(f"Loading and applying Diffusers LoRA weights from: {lora_path}")
    # Bây giờ, chúng ta gọi phương thức `load_lora_weights` trên chính pipeline
    # Pipeline sẽ tự động áp dụng các trọng số này vào UNet bên trong nó.
    pipeline.load_lora_weights(lora_path)

    # Chuyển toàn bộ pipeline lên GPU
    pipeline.to(device)

    # --- 4. Chạy Inference (đơn giản hơn rất nhiều) ---
    print(f"Generating {nsamples} images with prompt: '{prompt}'...")

    # SD-Turbo chỉ cần 1 bước inference và không cần guidance
    generator = torch.Generator(device=device)
    for i in range(nsamples):
        generator.manual_seed(seed + i)

        # Gọi thẳng pipeline, nó sẽ tự xử lý các bước denoising
        image = pipeline(
            prompt,
            num_inference_steps=1,
            guidance_scale=0.0,
            generator=generator
        ).images[0]

        # Tạo tên file an toàn
        prompt_safe_name = "".join(c if c.isalnum() else "_" for c in prompt)[:50]
        output_path = output_dir / f"seed_{seed + i}_{prompt_safe_name}.png"

        # Lưu ảnh
        image.save(output_path)
        print(f"Saved image to {output_path}")

    print("Generation complete.")


if __name__ == "__main__":
    app()