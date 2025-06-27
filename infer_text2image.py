# infer_one_step_text2img.py
# Kịch bản inference chuẩn, tuân thủ logic gốc của SwiftBrush/SD-Turbo.

import torch
import typer
from pathlib import Path
from PIL import Image
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, AutoTokenizer
from torchvision.utils import save_image

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
        prompt: str = typer.Argument(..., help="Prompt to generate, e.g., 'a photo of a <class_0> bear'"),
        lora_path: Path = typer.Argument(...,
                                         help="Path to the finetuned LoRA directory (the one containing the weights file)."),
        text_embeds_path: Path = typer.Argument(...,
                                                help="Path to the 'learned_embeds.bin' file containing the finetuned text embeddings."),
        output_dir: Path = typer.Option("generated_images_final", help="Path to the output directory.", dir_okay=True),
        # <<< QUAN TRỌNG: Model nền cho UNet và Pipeline phải khác nhau >>>
        base_model_name: str = typer.Option("stabilityai/sd-turbo", help="Base model for VAE, Scheduler, Tokenizer."),
        swiftbrush_checkpoint_path: Path = typer.Argument(...,
                                                          help="Path to the original SwiftBrush v2 UNet checkpoint directory to apply LoRA onto."),
        seed: int = typer.Option(42, help="A seed for reproducible generation."),
        nsamples: int = typer.Option(4, help="Number of images to generate."),
):
    """
    Generates images using a finetuned One-Step model with the standard Text-to-Image process.
    """
    set_seed(seed)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")

    device = "cuda"
    weight_dtype = torch.float16  # Sử dụng float16 để tối ưu tốc độ và bộ nhớ

    # --- 1. Tải các thành phần riêng lẻ ---
    # Tải các thành phần nền từ sd-turbo
    noise_scheduler = DDPMScheduler.from_pretrained(base_model_name, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(base_model_name, subfolder="vae").to(device, dtype=weight_dtype)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model_name, subfolder="text_encoder").to(
        device)  # Giữ ở fp32 để ổn định

    # Tải UNet gốc của SwiftBrush (đây là UNet bạn đã finetune LoRA lên trên)
    unet = UNet2DConditionModel.from_pretrained(swiftbrush_checkpoint_path).to(device, dtype=weight_dtype)

    # --- 2. Áp dụng LoRA và Textual Inversion ---
    print(f"Loading and applying Diffusers LoRA weights from: {lora_path}")
    unet.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")

    print(f"Loading and applying custom text embeddings from: {text_embeds_path}")
    learned_embeds_dict = torch.load(text_embeds_path, map_location="cpu")
    placeholder_tokens = learned_embeds_dict["placeholder_tokens"]
    learned_embeds = learned_embeds_dict["text_embeds"]
    tokenizer.add_tokens(placeholder_tokens)
    token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        text_encoder.get_input_embeddings().weight[token_ids] = learned_embeds.to(
            text_encoder.get_input_embeddings().weight.dtype
        )

    # --- 3. Chuẩn bị cho Inference One-Step ---
    unet.eval()
    vae.eval()
    text_encoder.eval()

    # Lấy các hằng số sigma và alpha tại timestep cuối cùng
    timestep = torch.tensor([noise_scheduler.config.num_train_timesteps - 1], device=device)
    alpha_t = noise_scheduler.alphas_cumprod[timestep].clone().to(dtype=weight_dtype)
    sigma_t = (1 - alpha_t).sqrt()
    alpha_t = alpha_t.sqrt()

    alpha_t = alpha_t.view(-1, 1, 1, 1)
    sigma_t = sigma_t.view(-1, 1, 1, 1)

    # --- 4. Vòng lặp tạo ảnh ---
    print(f"Generating {nsamples} images with prompt: '{prompt}'...")
    generator = torch.Generator(device=device)

    for i in range(nsamples):
        generator.manual_seed(seed + i)

        # Bắt đầu từ nhiễu ngẫu nhiên
        noise = torch.randn(1, unet.config.in_channels, 64, 64, generator=generator, device=device, dtype=weight_dtype)

        # Mã hóa prompt
        input_ids = tokenizer([prompt], padding="max_length", truncation=True, max_length=tokenizer.model_max_length,
                              return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            encoder_hidden_states = text_encoder(input_ids)[0].to(dtype=weight_dtype)

            # Chạy UNet một bước
            model_pred = unet(noise, timestep, encoder_hidden_states).sample

            # Công thức giải mã của SD-Turbo
            pred_original_sample = (noise - sigma_t * model_pred) / alpha_t

            # Giải mã bằng VAE
            image = vae.decode(pred_original_sample / vae.config.scaling_factor).sample

        # Chuẩn hóa và lưu ảnh
        image = (image / 2 + 0.5).clamp(0, 1)
        prompt_safe_name = "".join(c if c.isalnum() else "_" for c in prompt)[:50]
        output_path = output_dir / f"seed_{seed + i}_{prompt_safe_name}.png"

        save_image(image, output_path)
        print(f"Saved image to {output_path}")

    print("Generation complete.")


if __name__ == "__main__":
    app()