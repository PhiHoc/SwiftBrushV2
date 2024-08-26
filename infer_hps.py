import random
from pathlib import Path

import hpsv2
import numpy as np
import torch
import typer
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel

app = typer.Typer(pretty_exceptions_show_locals=False)


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


@app.command()
def main(
    dir: Path = typer.Argument(..., help="Path to the checkpoint directory", dir_okay=True, exists=True),
    output_dir: Path = typer.Option("hps-data", help="Path to the base outdir", dir_okay=True),
    model_name: str = typer.Option("stabilityai/sd-turbo", help="huggingface model name"),
    nsamples: int = typer.Option(30000, help="number of inference image"),
    seed: int = typer.Option(0, help="seed"),
    subfolder: str = typer.Option("unet_ema"),
    dtype: str = typer.Option("fp32"),
):
    set_seed(seed)
    if dtype == "fp16":
        weight_dtype = torch.float16
    elif dtype == "torch.bfloat16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to("cuda", dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(dir, subfolder=subfolder).to("cuda", dtype=weight_dtype)
    unet.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to("cuda")

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
        noise = torch.randn(1, 4, 64, 64, device="cuda", dtype=weight_dtype)
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
        image = (vae.decode(pred_original_sample.to(dtype=vae.dtype)).sample.float() + 1) / 2
        image = image.clamp(0, 1)

        return image

    all_prompts = hpsv2.benchmark_prompts("all")
    for style, prompts in all_prompts.items():
        for idx, prompt in enumerate(prompts):
            op = p / style / f"{idx:05d}.png"
            if not op.parent.exists():
                op.parent.mkdir(exist_ok=True, parents=True)
            if op.exists():
                continue
            image = predict_img(prompt)
            image.save(op.as_posix())


if __name__ == "__main__":
    app()
