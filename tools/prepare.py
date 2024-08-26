import functools
import json
import math
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import typer
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig


def process_prompt_data(index, batch_start, prompt_embed, output_path):
    np.save(output_path / f"{str(batch_start+index).zfill(8)}.npy", prompt_embed)
    return index


def wrapper_process_prompt_data(args):
    return process_prompt_data(*args)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompts, text_encoder, tokenizer, is_train=True):
    captions = []
    for caption in prompts:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
        )[0]

    return {"prompt_embeds": prompt_embeds.cpu()}


app = typer.Typer(pretty_exceptions_show_locals=False)


# ./journeydb_laion2256k.txt
@app.command()
def main(
    caption_path: Path = typer.Option("dataset.json", help="Path to the caption", file_okay=True, exists=True),
    model_name: str = typer.Option("stabilityai/stable-diffusion-2-1-base", help="huggingface model name"),
    batch_size: int = typer.Option(32, help="Batch size for encoding"),
    num_processes: int = typer.Option(16, help="#process to save"),
):
    # Load the tokenizers
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(model_name, None)

    text_encoder = text_encoder_cls.from_pretrained(model_name, subfolder="text_encoder")

    text_encoder.requires_grad_(False)
    text_encoder.to("cuda", dtype=torch.float32)

    # Let's first compute all the embeddings so that we can free up the text encoders
    compute_embeddings_fn = functools.partial(
        encode_prompt,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    if caption_path.suffix == ".txt":
        with open(caption_path) as f:
            prompts = f.read().splitlines()
    elif caption_path.suffix == ".json":
        with open(caption_path) as f:
            p = json.load(f)
            prompts = p["labels"]
            prompts = [x[1] for x in prompts]
    else:
        raise Exception("Unknown file format")

    op = Path(f"textembeds/{model_name}/{caption_path.name}")
    op.mkdir(exist_ok=True, parents=True)

    if num_processes <= 1:
        for i in tqdm(range(math.ceil(len(prompts) / batch_size))):
            start = batch_size * i
            end = min(len(prompts), batch_size * i + batch_size)

            if start >= len(prompts):
                break

            p = op / Path(f"{str(end).zfill(8)}.npy")
            if p.exists():
                print("here")
                continue
            else:
                prompt_dicts = compute_embeddings_fn(prompts=prompts[start:end])
                prompt_dicts = {key: value.detach().cpu().numpy() for key, value in prompt_dicts.items()}

                for j, prompt_embed in enumerate(prompt_dicts["prompt_embeds"]):
                    fop = op / f"{str(batch_size*i+j).zfill(8)}.npy"
                    np.save(fop, prompt_embed)

    else:
        with Pool(num_processes) as pool:
            # ceil so that it will handle the left-over batch
            for i in tqdm(range(math.ceil(len(prompts) / batch_size)), desc="Batch Progress"):
                start = batch_size * i
                end = min(len(prompts), batch_size * i + batch_size)

                if start >= len(prompts):
                    break
                p = op / Path(f"{str(end).zfill(8)}.npy")
                if p.exists():
                    continue

                prompt_dicts = compute_embeddings_fn(prompts=prompts[start:end])
                prompt_dicts = {key: value.detach().cpu().numpy() for key, value in prompt_dicts.items()}

                batch_start = batch_size * i
                prompt_embeds = prompt_dicts["prompt_embeds"]

                _ = list(
                    tqdm(
                        pool.imap_unordered(
                            wrapper_process_prompt_data,
                            [
                                (
                                    j,
                                    batch_start,
                                    prompt_embeds[j],
                                    op,
                                )
                                for j in range(len(prompt_embeds))
                            ],
                        ),
                        total=len(prompt_embeds),
                        desc="Processing Prompts",
                        leave=False,
                    )
                )
    # np.save(op / "prompts.npy", np.array(prompts))


if __name__ == "__main__":
    app()
