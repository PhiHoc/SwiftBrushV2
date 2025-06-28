import argparse
import os
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Process, Queue
from queue import Empty

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import safetensors.torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset import DATASET_NAME_MAPPING

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["WANDB_DISABLED"] = "true"

def load_textual_inversion(tokenizer, text_encoder, embeds_path):
    learned_embeds_dict = torch.load(embeds_path, map_location="cpu")
    placeholder_tokens = learned_embeds_dict["placeholder_tokens"]
    learned_embeds = learned_embeds_dict["text_embeds"]
    tokenizer.add_tokens(placeholder_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    for token_id, embed in zip(placeholder_token_ids, learned_embeds):
        token_embeds[token_id] = embed

def apply_lora(unet, lora_path):
    lora_state_dict = safetensors.torch.load_file(lora_path)
    unet.load_state_dict(lora_state_dict, strict=False)

def tokenize_prompt(prompt, tokenizer):
    inputs = tokenizer(prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
    return inputs.input_ids

def generate_image(prompt, tokenizer, text_encoder, unet, vae, scheduler, device):
    input_ids = tokenize_prompt([prompt], tokenizer).to(device)
    encoder_hidden_state = text_encoder(input_ids)[0]

    noise = torch.randn(1, 4, 64, 64, device=device)
    timestep = torch.tensor([scheduler.config.num_train_timesteps - 1], device=device)

    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    alpha_t = (alphas_cumprod[timestep] ** 0.5).view(-1, 1, 1, 1)
    sigma_t = ((1 - alphas_cumprod[timestep]) ** 0.5).view(-1, 1, 1, 1)

    with torch.no_grad():
        model_pred = unet(noise, timestep, encoder_hidden_state).sample
        if model_pred.shape[1] == noise.shape[1] * 2:
            model_pred, _ = torch.split(model_pred, noise.shape[1], dim=1)

        pred_original_sample = (noise - sigma_t * model_pred) / alpha_t
        pred_original_sample = pred_original_sample / vae.config.scaling_factor
        image = (vae.decode(pred_original_sample).sample + 1) / 2
    return image

def sample_func(args, in_queue, gpu_id, process_id):
    device = f"cuda:{gpu_id}"
    torch.manual_seed(args.seed + process_id)
    np.random.seed(args.seed + process_id)
    random.seed(args.seed + process_id)

    tokenizer = CLIPTokenizer.from_pretrained(args.base_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae").to(device)
    scheduler = DDPMScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(args.swiftbrush_unet_path).to(device)

    load_textual_inversion(tokenizer, text_encoder, args.textual_inversion_embeds_path)
    if args.lora_weights_path and os.path.exists(args.lora_weights_path):
        apply_lora(unet, args.lora_weights_path)

    train_dataset = DATASET_NAME_MAPPING[args.dataset](split="train", seed=args.seed, examples_per_class=args.examples_per_class, image_train_dir=args.train_data_dir)
    name2placeholder = {name: f"<class_{i}>" for i, name in enumerate(train_dataset.class_names)}

    while True:
        try:
            tasks = [in_queue.get(timeout=5) for _ in range(args.batch_size)]
        except Empty:
            break

        prompts, save_paths = [], []
        for index, source_context_label, target_label in tasks:
            target_name = train_dataset.label2class[target_label]
            target_placeholder = name2placeholder[target_name]
            if args.sample_strategy == "one-step-aug":
                prompt = f"a photo of a {target_placeholder}"
            elif args.sample_strategy == "one-step-mix":
                source_context_name = train_dataset.label2class[source_context_label]
                prompt = f"a photo of a {target_placeholder}, in the environment of a {source_context_name}"
            else:
                prompt = f"a photo of a {target_placeholder}"

            prompts.append(prompt)
            save_dir = os.path.join(args.output_path, "data", target_name.replace(" ", "_").replace("/", "_"))
            os.makedirs(save_dir, exist_ok=True)
            save_name = f"{target_name.replace(' ', '_').replace('/', '_')}-{index:06d}.png"
            save_paths.append(os.path.join(save_dir, save_name))

        if all(os.path.exists(p) for p in save_paths): continue

        for prompt, save_path in zip(prompts, save_paths):
            image = generate_image(prompt, tokenizer, text_encoder, unet, vae, scheduler, device)
            image = image.mul(255).clamp(0, 255).byte().cpu().squeeze(0).permute(1, 2, 0).numpy()
            Image.fromarray(image).save(save_path)

def main(args):
    torch.multiprocessing.set_start_method("spawn")
    os.makedirs(args.output_path, exist_ok=True)

    train_dataset = DATASET_NAME_MAPPING[args.dataset](split="train", seed=args.seed, examples_per_class=args.examples_per_class, image_train_dir=args.train_data_dir)
    num_classes = len(train_dataset.class_names)
    num_real_images = len(train_dataset) if args.examples_per_class == -1 else args.examples_per_class * num_classes
    num_synthetic_tasks = int(args.syn_dataset_mulitiplier * num_real_images)
    samples_per_class = num_synthetic_tasks // num_classes

    tasks = []
    all_class_indices = list(range(num_classes))
    for target_idx in all_class_indices:
        for _ in range(samples_per_class):
            source_context_idx = target_idx if args.sample_strategy == 'one-step-aug' else random.choice(all_class_indices)
            tasks.append((source_context_idx, target_idx))

    remainder = num_synthetic_tasks % num_classes
    if remainder > 0:
        extra_targets = random.choices(all_class_indices, k=remainder)
        for target_idx in extra_targets:
            source_context_idx = target_idx if args.sample_strategy == 'one-step-aug' else random.choice(all_class_indices)
            tasks.append((source_context_idx, target_idx))
    random.shuffle(tasks)

    in_queue = Queue()
    for i, (source_context_label, target_label) in enumerate(tasks):
        in_queue.put((i, source_context_label, target_label))

    print(f"Total tasks created: {len(tasks)}. Enqueuing...")

    processes = []
    with tqdm(total=len(tasks), desc="Generating Images") as pbar:
        for process_id, gpu_id in enumerate(args.gpu_ids):
            process = Process(target=sample_func, args=(args, in_queue, gpu_id, process_id))
            process.start()
            processes.append(process)

        initial_tasks = len(tasks)
        while any(p.is_alive() for p in processes):
            pbar.n = initial_tasks - in_queue.qsize()
            pbar.refresh()
            time.sleep(1)
        pbar.n = initial_tasks
        pbar.refresh()
        for p in processes: p.join()

    print("Generation complete. Generating meta.csv...")
    rootdir = os.path.join(args.output_path, "data")
    data_dict = defaultdict(list)
    if not os.path.exists(rootdir) or not os.listdir(rootdir):
        print("!!! WARNING: 'data' directory is empty or does not exist. No images were generated.")
        df = pd.DataFrame({'Path': [], 'Target Class': []})
    else:
        for dir_name in os.listdir(rootdir):
            class_dir = os.path.join(rootdir, dir_name)
            if not os.path.isdir(class_dir): continue
            target_dir_name = dir_name.replace("_", " ")
            for file_name in os.listdir(class_dir):
                path = os.path.join(dir_name, file_name)
                data_dict["Path"].append(path)
                data_dict["Target Class"].append(target_dir_name)
        df = pd.DataFrame(data_dict)
        if not df.empty:
            df = df.sort_values(by=["Target Class", "Path"]).reset_index(drop=True)

    csv_path = os.path.join(args.output_path, "meta.csv")
    df.to_csv(csv_path, index=False)
    print(f"meta.csv saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("One-Step Mixup/Augmentation Sampling Script without Pipeline")
    parser.add_argument("--sample_strategy", type=str, default="one-step-mix", choices=["one-step-mix", "one-step-aug"])
    parser.add_argument("--base_model_path", type=str, default="stabilityai/sd-turbo")
    parser.add_argument("--swiftbrush_unet_path", type=str, required=True)
    parser.add_argument("--lora_weights_path", type=str, default=None)
    parser.add_argument("--textual_inversion_embeds_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="bear")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples_per_class", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--syn_dataset_mulitiplier", type=int, default=1)
    args = parser.parse_args()
    main(args)
