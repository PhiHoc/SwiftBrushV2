# scripts/sample_one_step_mix.py

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
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel

# Đảm bảo có thể import các module từ thư mục gốc của dự án
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset import DATASET_NAME_MAPPING

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["WANDB_DISABLED"] = "true"


# --- Lớp Pipeline One-Step Tùy chỉnh ---
class OneStepMixupPipeline:
    """
    Một lớp pipeline tùy chỉnh để tải mô hình one-step, checkpoint UNet,
    trọng số LoRA và các embedding Textual Inversion.
    """

    def __init__(self, args, device):
        self.device = device
        self.args = args
        self.pipe = self._load_pipeline()

    def _load_textual_inversion_embeds(self, tokenizer, text_encoder):
        """Tải và áp dụng các embedding đã học."""
        if not os.path.exists(self.args.textual_inversion_embeds_path):
            raise FileNotFoundError(
                f"Textual Inversion embeds file not found at: {self.args.textual_inversion_embeds_path}")

        learned_embeds_dict = torch.load(self.args.textual_inversion_embeds_path, map_location="cpu")
        placeholder_tokens = learned_embeds_dict["placeholder_tokens"]
        learned_embeds = learned_embeds_dict["text_embeds"]

        # Thêm token vào tokenizer
        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != len(placeholder_tokens):
            print(f"Warning: Some placeholder tokens might already exist in the tokenizer.")

        # Thay đổi kích thước embedding của text encoder
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data

        # Gán các embedding đã học
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        for token_id, embed in zip(placeholder_token_ids, learned_embeds):
            token_embeds[token_id] = embed

        print(f"Successfully loaded {len(placeholder_tokens)} textual inversion embeddings.")
        return tokenizer, text_encoder

    def _load_pipeline(self):
        """Tải tất cả các thành phần của pipeline."""
        weight_dtype = torch.float16

        # 1. Tải tokenizer và text encoder gốc, sau đó áp dụng TI
        tokenizer = CLIPTokenizer.from_pretrained(self.args.base_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.args.base_model_path, subfolder="text_encoder")
        tokenizer, text_encoder = self._load_textual_inversion_embeds(tokenizer, text_encoder)

        # 2. Tải các thành phần khác của pipeline
        vae = AutoencoderKL.from_pretrained(self.args.base_model_path, subfolder="vae")
        scheduler = DDPMScheduler.from_pretrained(self.args.base_model_path, subfolder="scheduler")

        # 3. Tải UNet từ checkpoint SwiftBrush
        print(f"Loading SwiftBrush UNet from: {self.args.swiftbrush_unet_path}")
        unet = UNet2DConditionModel.from_pretrained(self.args.swiftbrush_unet_path)

        # 4. Tạo pipeline đầy đủ
        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
        )

        # 5. Tải trọng số LoRA
        print(f"Loading LoRA weights from: {self.args.lora_weights_path}")
        pipe.load_lora_weights(self.args.lora_weights_path)
        pipe.to(self.device, dtype=weight_dtype)

        return pipe

    def __call__(self, prompts, num_images_per_prompt=1):
        """Thực hiện sinh ảnh one-step."""
        with torch.no_grad():
            images = self.pipe(
                prompt=prompts,
                num_inference_steps=1,  # Quan trọng: chỉ 1 bước suy luận
                guidance_scale=0.0,  # SD-Turbo/SwiftBrush thường dùng guidance_scale=0
                num_images_per_prompt=num_images_per_prompt,
            ).images
        return images


# --- Hàm thực thi cho mỗi tiến trình ---
def sample_func(args, in_queue, out_queue, gpu_id, process_id):
    device = f"cuda:{gpu_id}"
    random.seed(args.seed + process_id)
    np.random.seed(args.seed + process_id)
    torch.manual_seed(args.seed + process_id)

    # Khởi tạo dataset để lấy metadata
    train_dataset = DATASET_NAME_MAPPING[args.dataset](
        split="train",
        seed=args.seed,
        examples_per_class=args.examples_per_class,
        image_train_dir=args.train_data_dir,
    )
    # Tạo map từ tên lớp sang placeholder token
    name2placeholder = {name: f"<class_{i}>" for i, name in enumerate(train_dataset.class_names)}

    # Khởi tạo pipeline one-step
    pipeline = OneStepMixupPipeline(args, device)
    batch_size = args.batch_size

    while True:
        try:
            tasks = [in_queue.get(timeout=5) for _ in range(batch_size)]
        except Empty:
            print(f"Process {process_id} finished.")
            break

        prompts = []
        save_paths = []

        for index, source_label, target_label in tasks:
            # Lấy metadata
            target_name = train_dataset.label2class[target_label]
            source_name = train_dataset.label2class[source_label]

            # Lấy placeholder token cho lớp đích
            target_placeholder = name2placeholder[target_name]

            # --- Logic "Mixup" One-Step ---
            # Tạo prompt để sinh ảnh lớp đích. Sự đa dạng đến từ
            # việc chọn ngẫu nhiên các cặp nguồn-đích và từ nhiễu ngẫu nhiên.
            prompt = f"a photo of a {target_placeholder}"
            prompts.append(prompt)

            # Tạo đường dẫn lưu file theo phong cách DiffMix
            # Tên file sẽ chứa thông tin về lớp nguồn và lớp đích
            save_dir = os.path.join(args.output_path, "data", source_name.replace(" ", "_").replace("/", "_"))
            os.makedirs(save_dir, exist_ok=True)
            save_name = f"{target_name.replace(' ', '_').replace('/', '_')}-{index:06d}.png"
            save_paths.append(os.path.join(save_dir, save_name))

        # Bỏ qua nếu đã tồn tại
        if all(os.path.exists(p) for p in save_paths):
            print(f"Skipping batch, all files exist.")
            continue

        # Sinh ảnh
        images = pipeline(prompts, num_images_per_prompt=1)

        # Lưu ảnh
        for image, save_path in zip(images, save_paths):
            image.save(save_path)
        print(f"Process {process_id}: Saved {len(images)} images, last one to {save_paths[-1]}")


def main(args):
    torch.multiprocessing.set_start_method("spawn")
    os.makedirs(args.output_path, exist_ok=True)

    # ... (phần code tạo task queue giữ nguyên từ sample_mp.py) ...
    # Tạo dataset để xác định số lượng task
    train_dataset = DATASET_NAME_MAPPING[args.dataset](
        split="train", seed=args.seed, examples_per_class=args.examples_per_class, image_train_dir=args.train_data_dir
    )
    num_classes = len(train_dataset.class_names)
    num_tasks = args.syn_dataset_mulitiplier * len(train_dataset)

    # Tạo danh sách các cặp (nguồn, đích)
    target_classes = []
    samples_per_class = num_tasks // num_classes
    target_classes.extend(list(range(num_classes)) * samples_per_class)
    target_classes.extend(random.sample(range(num_classes), num_tasks % num_classes))
    random.shuffle(target_classes)

    # Đối với mixup, lớp nguồn được chọn ngẫu nhiên
    source_classes = random.choices(range(num_classes), k=num_tasks)

    # Đưa task vào queue
    in_queue = Queue()
    for i in range(num_tasks):
        in_queue.put((i, source_classes[i], target_classes[i]))

    print(f"Total tasks to generate: {in_queue.qsize()}")

    # Khởi chạy các tiến trình
    processes = []
    with tqdm(total=num_tasks, desc="Generating Images") as pbar:
        for process_id, gpu_id in enumerate(args.gpu_ids):
            process = Process(target=sample_func, args=(args, in_queue, out_queue, gpu_id, process_id))
            process.start()
            processes.append(process)

        while any(p.is_alive() for p in processes):
            pbar.n = num_tasks - in_queue.qsize()
            pbar.refresh()
            time.sleep(1)

        for process in processes:
            process.join()

    # ... (phần code tạo meta.csv giữ nguyên từ sample_mp.py) ...
    print("Generation complete. Generating meta.csv...")
    rootdir = os.path.join(args.output_path, "data")
    pattern_level_2 = r"(.+)-(\d+).png"  # Pattern đơn giản hơn
    data_dict = defaultdict(list)
    for dir_name in os.listdir(rootdir):
        if not os.path.isdir(os.path.join(rootdir, dir_name)): continue
        source_dir_name = dir_name.replace("_", " ")
        for file_name in os.listdir(os.path.join(rootdir, dir_name)):
            match = re.match(pattern_level_2, file_name)
            if match:
                target_dir_name = match.group(1).replace("_", " ")
                num = int(match.group(2))
                data_dict["First Directory"].append(source_dir_name)  # Lớp nguồn
                data_dict["Second Directory"].append(target_dir_name)  # Lớp đích
                data_dict["Number"].append(num)
                data_dict["Path"].append(os.path.join(dir_name, file_name))

    df = pd.DataFrame(data_dict)
    csv_path = os.path.join(args.output_path, "meta.csv")
    df.to_csv(csv_path, index=False)
    print(f"meta.csv saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("One-Step Mixup Sampling Script")
    # Các arguments mới và được điều chỉnh
    parser.add_argument("--base_model_path", type=str, default="stabilityai/sd-turbo",
                        help="Path to the base one-step model.")
    parser.add_argument("--swiftbrush_unet_path", type=str, required=True,
                        help="Path to the pre-trained SwiftBrush UNet checkpoint directory.")
    parser.add_argument("--lora_weights_path", type=str, required=True,
                        help="Path to the fine-tuned LoRA weights (.safetensors).")
    parser.add_argument("--textual_inversion_embeds_path", type=str, required=True,
                        help="Path to the learned TI embeddings (.bin).")

    # Các arguments giữ lại từ DiffMix
    parser.add_argument("--output_path", type=str, required=True, help="The output directory for synthetic data.")
    parser.add_argument("--dataset", type=str, default="bear", help="Dataset name for metadata.")
    parser.add_argument("--train_data_dir", type=str, required=True, help="A folder containing the training data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--examples_per_class", type=int, default=-1,
                        help="Examples per class in the original dataset.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU.")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0], help="GPU ids.")
    parser.add_argument("--syn_dataset_mulitiplier", type=int, default=5, help="Multiplier for synthetic dataset size.")

    args = parser.parse_args()
    main(args)