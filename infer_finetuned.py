# infer_from_checkpoint.py
# Kịch bản để chạy inference trực tiếp từ checkpoint được lưu bởi `accelerate.save_state()`

import torch
import typer
from pathlib import Path
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from transformers import AutoTokenizer, CLIPTextModel
from torchvision.utils import save_image
from safetensors.torch import load_file
import os

# MODIFICATION: Định nghĩa các token định danh một cách thủ công.
# Giả sử bạn có 10 loài gấu, tương ứng với <class_0> đến <class_9>.
# Con số này phải khớp với số lớp trong dataset của bạn khi huấn luyện.
NUM_CLASSES = 10
PLACEHOLDER_TOKENS = [f"<class_{i}>" for i in range(NUM_CLASSES)]

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
        prompt: str = typer.Argument(..., help="Prompt to generate, e.g., 'a photo of a <class_0> bear'"),
        swiftbrush_checkpoint_path: Path = typer.Argument(...,
                                                          help="Path to the original SwiftBrush v2 UNet checkpoint directory."),
        # MODIFICATION: Thay đổi các đối số để chỉ cần một đường dẫn đến checkpoint
        finetuned_checkpoint_path: Path = typer.Argument(...,
                                                         help="Path to the finetuning checkpoint directory (containing model.safetensors, etc.)."),
        output_dir: Path = typer.Option("generated_from_checkpoint", help="Path to the output directory.",
                                        dir_okay=True),
        model_name: str = typer.Option("stabilityai/sd-turbo", help="Base Hugging Face model name."),
        seed: int = typer.Option(42, help="A seed for reproducible generation."),
        nsamples: int = typer.Option(4, help="Number of images to generate."),
):
    """
    Generates images using a finetuned SwiftBrush v2 model directly from an accelerate checkpoint.
    """
    set_seed(seed)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")

    # --- 1. Tải Mô hình Nền ---
    print("Loading base models from 'stabilityai/sd-turbo'...")
    weight_dtype = torch.float16
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to("cuda", dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(swiftbrush_checkpoint_path).to("cuda", dtype=weight_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to("cuda")  # Tải ở float32

    # --- 2. Áp dụng các thành phần đã Finetune từ Checkpoint ---

    # MODIFICATION START: Logic tải mới
    print(f"Loading finetuned components from checkpoint: {finetuned_checkpoint_path}")

    # 2a. Tải và áp dụng trọng số LoRA vào UNet
    # Tệp model.safetensors chứa trạng thái của các lớp LoRA.
    lora_state_dict_path = finetuned_checkpoint_path / "model.safetensors"
    if not lora_state_dict_path.exists():
        raise FileNotFoundError(f"LoRA state dict not found at {lora_state_dict_path}")

    # Để tải state dict này, trước tiên ta cần thiết lập các processor rỗng trên UNet
    # rồi dùng AttnProcsLayers để load state dict vào đúng cấu trúc.
    unet.set_attn_processor(
        AttnProcsLayers(unet.attn_processors).to("cuda", dtype=weight_dtype)
    )
    # Tải trọng số LoRA trực tiếp vào các attention processors
    unet.load_attn_procs(lora_state_dict_path)
    print("Successfully loaded LoRA weights into UNet.")

    # 2b. Tải và áp dụng trọng số Text Encoder
    # Tệp model_1.safetensors chứa trạng thái của toàn bộ Text Encoder đã finetune.
    text_encoder_state_dict_path = finetuned_checkpoint_path / "model_1.safetensors"
    if not text_encoder_state_dict_path.exists():
        raise FileNotFoundError(f"Text encoder state dict not found at {text_encoder_state_dict_path}")

    # Trước khi tải state dict, ta cần đảm bảo tokenizer và text_encoder có các token mới.
    tokenizer.add_tokens(PLACEHOLDER_TOKENS)
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Tải toàn bộ trạng thái của text encoder
    text_encoder.load_state_dict(load_file(text_encoder_state_dict_path))
    print("Successfully loaded finetuned Text Encoder state.")

    # MODIFICATION END

    # Chuyển các mô hình sang device và dtype phù hợp
    unet.to("cuda", dtype=weight_dtype)
    text_encoder.to("cuda", dtype=weight_dtype)

    # --- 3. Chuẩn bị và chạy Inference ---
    unet.eval()
    text_encoder.eval()

    timestep = torch.tensor([noise_scheduler.config.num_train_timesteps - 1], device="cuda")
    alphas_cumprod = noise_scheduler.alphas_cumprod.to("cuda")
    alpha_t = (alphas_cumprod[timestep] ** 0.5).view(-1, 1, 1, 1).to(dtype=weight_dtype)
    sigma_t = ((1 - alphas_cumprod[timestep]) ** 0.5).view(-1, 1, 1, 1).to(dtype=weight_dtype)

    print(f"Generating {nsamples} images with prompt: '{prompt}'...")
    for i in range(nsamples):
        noise = torch.randn(1, unet.config.in_channels, 64, 64, device="cuda", dtype=weight_dtype)

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