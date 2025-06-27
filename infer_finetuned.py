# infer_peft_lora.py
# Kịch bản inference cuối cùng, tương thích với LoRA được huấn luyện bằng PEFT (`add_adapter`)

import torch
import typer
from pathlib import Path
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel
from torchvision.utils import save_image

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
        prompt: str = typer.Argument(..., help="Prompt to generate, e.g., 'a photo of a <class_0> bear'"),
        swiftbrush_checkpoint_path: Path = typer.Argument(...,
                                                          help="Path to the original SwiftBrush v2 UNet checkpoint directory."),
        # MODIFICATION: Đối số bây giờ trỏ đến thư mục chứa adapter_model.safetensors
        lora_path: Path = typer.Argument(...,
                                         help="Path to the finetuned LoRA directory (saved by `save_lora_weights`)."),
        text_embeds_path: Path = typer.Argument(...,
                                                help="Path to the 'learned_embeds.bin' file containing the finetuned text embeddings."),
        output_dir: Path = typer.Option("generated_peft_lora", help="Path to the output directory.", dir_okay=True),
        model_name: str = typer.Option("stabilityai/sd-turbo", help="Base Hugging Face model name."),
        seed: int = typer.Option(42, help="A seed for reproducible generation."),
        nsamples: int = typer.Option(4, help="Number of images to generate."),
):
    """
    Generates images using a SwiftBrush v2 model finetuned with PEFT LoRA adapters.
    """
    set_seed(seed)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")

    # --- 1. Tải Mô hình Nền ---
    print(f"Loading base models from '{model_name}'...")
    weight_dtype = torch.float16
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to("cuda", dtype=weight_dtype)

    print(f"Loading SwiftBrush v2 UNet from: {swiftbrush_checkpoint_path}")
    unet = UNet2DConditionModel.from_pretrained(swiftbrush_checkpoint_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")

    # --- 2. Cập nhật Text Encoder với các Embedding đã học ---
    print(f"Loading and applying custom text embeddings from: {text_embeds_path}")
    if not text_embeds_path.exists():
        raise FileNotFoundError(f"Text embeddings file not found: {text_embeds_path}")

    learned_embeds_dict = torch.load(text_embeds_path, map_location="cpu")
    placeholder_tokens = learned_embeds_dict["placeholder_tokens"]
    learned_embeds = learned_embeds_dict["text_embeds"]

    tokenizer.add_tokens(placeholder_tokens)
    token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        text_encoder.get_input_embeddings().weight[token_ids] = learned_embeds.to(
            text_encoder.get_input_embeddings().weight.dtype)

    # --- 3. Áp dụng Trọng số LoRA vào UNet ---
    # MODIFICATION START: Sử dụng phương thức tải LoRA mới của PEFT
    print(f"Loading and applying PEFT LoRA adapter from: {lora_path}")
    # Đây là phương thức hiện đại để tải các trọng số LoRA được lưu bởi `save_lora_weights`
    unet.load_adapter(lora_path)
    # MODIFICATION END

    # Chuyển tất cả mô hình lên GPU với dtype phù hợp
    unet.to("cuda", dtype=weight_dtype)
    text_encoder.to("cuda", dtype=weight_dtype)

    # --- 4. Chuẩn bị và chạy Inference Một Bước ---
    unet.eval()
    text_encoder.eval()

    timestep = torch.tensor([noise_scheduler.config.num_train_timesteps - 1], device="cuda")
    alphas_cumprod = noise_scheduler.alphas_cumprod.to("cuda")
    alpha_t = (alphas_cumprod[timestep] ** 0.5).view(-1, 1, 1, 1).to(dtype=weight_dtype)
    sigma_t = ((1 - alphas_cumprod[timestep]) ** 0.5).view(-1, 1, 1, 1).to(dtype=weight_dtype)

    print(f"Generating {nsamples} images with prompt: '{prompt}'...")
    for i in range(nsamples):
        noise = torch.randn(1, unet.config.in_channels, 64, 64, device="cuda", dtype=weight_dtype,
                            generator=torch.manual_seed(seed + i))

        with torch.no_grad():
            input_ids = tokenizer([prompt], padding="max_length", truncation=True,
                                  max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to("cuda")
            encoder_hidden_state = text_encoder(input_ids)[0].to(dtype=weight_dtype)

        with torch.no_grad():
            model_pred = unet(noise, timestep, encoder_hidden_state).sample

        pred_original_sample = (noise - sigma_t * model_pred) / alpha_t
        pred_original_sample = pred_original_sample / vae.config.scaling_factor

        with torch.no_grad():
            image = vae.decode(pred_original_sample.to(dtype=vae.dtype)).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        prompt_safe_name = prompt.replace(" ", "_").replace("<", "").replace(">", "")
        output_path = output_dir / f"{seed}_{i:03d}_{prompt_safe_name}.png"
        save_image(image, output_path)
        print(f"Saved image to {output_path}")

    print("Generation complete.")


if __name__ == "__main__":
    app()