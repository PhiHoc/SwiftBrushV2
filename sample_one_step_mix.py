# sample_one_step_mix.py
# MỤC ĐÍCH: Tạo dữ liệu tăng cường bằng mô hình đã fine-tune.
# ĐẦU VÀO: Các file model từ Script 1.
# ĐẦU RA: Thư mục chứa ảnh tổng hợp và file meta.csv.

import argparse
import os
import random
import re
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
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset import DATASET_NAME_MAPPING

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["WANDB_DISABLED"] = "true"


class OneStepMixupPipeline:
    def __init__(self, args, device):
        self.device = device
        self.args = args
        self.pipe = self._load_pipeline()

    def _load_textual_inversion_embeds(self, tokenizer, text_encoder):
        if not os.path.exists(self.args.textual_inversion_embeds_path):
            raise FileNotFoundError(f"TI embeds not found: {self.args.textual_inversion_embeds_path}")
        learned_embeds_dict = torch.load(self.args.textual_inversion_embeds_path, map_location="cpu")
        placeholder_tokens = learned_embeds_dict["placeholder_tokens"]
        learned_embeds = learned_embeds_dict["text_embeds"]
        tokenizer.add_tokens(placeholder_tokens)
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        for token_id, embed in zip(placeholder_token_ids, learned_embeds):
            token_embeds[token_id] = embed
        return tokenizer, text_encoder

    def _load_pipeline(self):
        weight_dtype = torch.float16
        tokenizer = CLIPTokenizer.from_pretrained(self.args.base_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.args.base_model_path, subfolder="text_encoder")
        tokenizer, text_encoder = self._load_textual_inversion_embeds(tokenizer, text_encoder)
        vae = AutoencoderKL.from_pretrained(self.args.base_model_path, subfolder="vae")
        scheduler = DDPMScheduler.from_pretrained(self.args.base_model_path, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(self.args.swiftbrush_unet_path)
        pipe = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                                       scheduler=scheduler, safety_checker=None, feature_extractor=None)
        if self.args.lora_weights_path and os.path.exists(self.args.lora_weights_path):
            pipe.load_lora_weights(self.args.lora_weights_path)
        pipe.to(self.device, dtype=weight_dtype)
        return pipe

    def __call__(self, prompts, num_images_per_prompt=1):
        with torch.no_grad():
            return self.pipe(prompt=prompts, num_inference_steps=1, guidance_scale=0.0,
                             num_images_per_prompt=num_images_per_prompt).images


def sample_func(args, in_queue, gpu_id, process_id):
    device = f"cuda:{gpu_id}"
    random.seed(args.seed + process_id)
    np.random.seed(args.seed + process_id)
    torch.manual_seed(args.seed + process_id)
    train_dataset = DATASET_NAME_MAPPING[args.dataset](split="train", seed=args.seed,
                                                       examples_per_class=args.examples_per_class,
                                                       image_train_dir=args.train_data_dir)
    name2placeholder = {name: f"<class_{i}>" for i, name in enumerate(train_dataset.class_names)}
    pipeline = OneStepMixupPipeline(args, device)
    batch_size = args.batch_size
    while True:
        try:
            tasks = [in_queue.get(timeout=5) for _ in range(batch_size)]
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
        try:
            images = pipeline(prompts, num_images_per_prompt=1)
            for image, save_path in zip(images, save_paths):
                image.save(save_path)
        except Exception as e:
            print(f"Process {process_id}: ERROR generating images. Skipping batch. Error: {e}", file=sys.stderr)


def main(args):
    torch.multiprocessing.set_start_method("spawn")
    os.makedirs(args.output_path, exist_ok=True)
    train_dataset = DATASET_NAME_MAPPING[args.dataset](split="train", seed=args.seed,
                                                       examples_per_class=args.examples_per_class,
                                                       image_train_dir=args.train_data_dir)
    num_classes = len(train_dataset.class_names)
    num_real_images = len(train_dataset) if args.examples_per_class == -1 else args.examples_per_class * num_classes
    num_synthetic_tasks = int(args.syn_dataset_mulitiplier * num_real_images)
    samples_per_class = num_synthetic_tasks // num_classes

    tasks = []
    all_class_indices = list(range(num_classes))
    for target_idx in all_class_indices:
        for _ in range(samples_per_class):
            source_context_idx = target_idx if args.sample_strategy == 'one-step-aug' else random.choice(
                all_class_indices)
            tasks.append((source_context_idx, target_idx))

    remainder = num_synthetic_tasks % num_classes
    if remainder > 0:
        extra_targets = random.choices(all_class_indices, k=remainder)
        for target_idx in extra_targets:
            source_context_idx = target_idx if args.sample_strategy == 'one-step-aug' else random.choice(
                all_class_indices)
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
    # (Phần argparse không thay đổi)
    parser = argparse.ArgumentParser("One-Step Mixup/Augmentation Sampling Script")
    parser.add_argument("--sample_strategy", type=str, default="one-step-mix", choices=["one-step-mix", "one-step-aug"],
                        help="Sampling strategy: inter-class ('mix') or intra-class ('aug').")
    parser.add_argument("--base_model_path", type=str, default="stabilityai/sd-turbo",
                        help="Path to the base one-step model.")
    parser.add_argument("--swiftbrush_unet_path", type=str, required=True,
                        help="Path to the pre-trained SwiftBrush UNet checkpoint directory.")
    parser.add_argument("--lora_weights_path", type=str, default=None,
                        help="Optional path to the fine-tuned LoRA weights (.safetensors).")
    parser.add_argument("--textual_inversion_embeds_path", type=str, required=True,
                        help="Path to the learned TI embeddings (.bin).")
    parser.add_argument("--output_path", type=str, required=True, help="The output directory for synthetic data.")
    parser.add_argument("--dataset", type=str, default="bear", help="Dataset name for metadata.")
    parser.add_argument("--train_data_dir", type=str, required=True, help="A folder containing the training data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--examples_per_class", type=int, default=-1,
                        help="Examples per class in the original dataset.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU.")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0], help="GPU ids.")
    parser.add_argument("--syn_dataset_mulitiplier", type=int, default=1, help="Multiplier for synthetic dataset size.")
    args = parser.parse_args()
    main(args)