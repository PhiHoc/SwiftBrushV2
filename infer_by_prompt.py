import random
from pathlib import Path

import numpy as np
import torch
import typer
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from sklearn.decomposition import PCA
from torchvision.utils import save_image
from transformers import AutoTokenizer, CLIPTextModel


def tokenize_captions(examples, tokenizer, is_train=False):
    captions = []
    for caption in examples:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(f"Captions should contain either strings or lists of strings but got {examples}.")
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return inputs.input_ids


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    prompt: str = typer.Argument(..., help="Prompt"),
    dir: Path = typer.Argument(..., help="model path"),
    output_dir: Path = typer.Option("generated", help="Path to the base outdir", dir_okay=True),
    model_name: str = typer.Option("stabilityai/sd-turbo", help="huggingface model name"),
    seed: int = typer.Option(0, help="seed"),
    nsamples: int = typer.Option(1, help="seed"),
    output_vae: bool = typer.Option(True, help="output PCA-ed VAE output"),
    subfolder: str = typer.Option("unet_ema"),
):
    set_seed(seed)
    weight_dtype = torch.float32
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to("cuda", dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(dir, subfolder=subfolder).to("cuda", dtype=weight_dtype)
    unet.eval()

    timestep = torch.ones((1,), dtype=torch.int64, device="cuda")
    timestep = timestep * (noise_scheduler.config.num_train_timesteps - 1)

    alphas_cumprod = noise_scheduler.alphas_cumprod.to("cuda")
    alpha_t = (alphas_cumprod[timestep] ** 0.5).view(-1, 1, 1, 1)
    sigma_t = ((1 - alphas_cumprod[timestep]) ** 0.5).view(-1, 1, 1, 1)
    del alphas_cumprod

    p = output_dir / f"{dir.parent.name}_{dir.name}"
    p.mkdir(exist_ok=True, parents=True)
    print(f"Outdir={p}")

    def predict_img(prompt):
        noise = torch.randn(1, 4, 64, 64, device="cuda")
        input_id = tokenize_captions([prompt], tokenizer).to("cuda")
        encoder_hidden_state = text_encoder(input_id)[0].to(dtype=weight_dtype)

        model_pred = unet(noise, timestep, encoder_hidden_state).sample
        if model_pred.shape[1] == noise.shape[1] * 2:
            model_pred, _ = torch.split(model_pred, noise.shape[1], dim=1)

        pred_original_sample = (noise - sigma_t * model_pred) / alpha_t
        if noise_scheduler.config.thresholding:
            pred_original_sample = noise_scheduler._threshold_sample(pred_original_sample)
        elif noise_scheduler.config.clip_sample:
            clip_sample_range = noise_scheduler.config.clip_sample_range
            pred_original_sample = pred_original_sample.clamp(-clip_sample_range, clip_sample_range)

        pred_original_sample = pred_original_sample / vae.config.scaling_factor
        image = (vae.decode(pred_original_sample).sample + 1) / 2

        return image, pred_original_sample * vae.config.scaling_factor

    tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to("cuda")
    if Path(prompt).exists():
        with open(prompt) as f:
            prompts = f.readlines()
            prompts = [p.strip() for p in prompts]
    else:
        prompts = [prompt]
    for prompt in prompts:
        op = p / f'{prompt.replace(" ", "_")}.png'
        vop = p / f'{prompt.replace(" ", "_")}_vae.png'
        op.parent.mkdir(exist_ok=True, parents=True)

        for i in range(nsamples):
            op = p / f'{i}_{prompt.replace(" ", "_")}.png'
            image, vae_latent = predict_img(prompt)
            save_image(image, op.as_posix())

            if output_vae:
                latent_reshaped = vae_latent[0].detach().cpu().numpy().reshape(4, -1).T
                vop = p / f'{i}_{prompt.replace(" ", "_")}_vae.png'
                pca = PCA(n_components=3)
                latent_pca = pca.fit_transform(latent_reshaped)
                latent_pca_reshaped = latent_pca.T.reshape(3, 64, 64)
                latent_normalized = (
                    (latent_pca_reshaped - latent_pca_reshaped.min())
                    / (latent_pca_reshaped.max() - latent_pca_reshaped.min())
                    * 255
                ).astype(np.uint8)
                latent_image = np.transpose(latent_normalized, (1, 2, 0))
                vae_latent = Image.fromarray(latent_image, "RGB")
                vae_latent.save(vop)


if __name__ == "__main__":
    app()
