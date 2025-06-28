# sample_one_step.py
import argparse
import os
import random
import sys
import time
import yaml
from collections import defaultdict
from multiprocessing import Process, Queue
from queue import Empty

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Trỏ đến các file bạn đã cung cấp
from dataset import T2I_DATASET_NAME_MAPPING  # Giả định bạn có file này định nghĩa dataset "bear"
from one_step_augment import OneStepLoraAugment  # Import "động cơ" mới của chúng ta

# Ánh xạ chiến lược tới lớp xử lý mới
AUGMENT_METHODS = {
    "diff-mix": OneStepLoraAugment,
    "diff-aug": OneStepLoraAugment,
}


# (Hàm sample_func và các hàm khác được giữ lại và sửa đổi)
def sample_func(args, in_queue, gpu_id, process_id):
    # Thiết lập seed cho từng tiến trình
    random.seed(args.seed + process_id)
    np.random.seed(args.seed + process_id)
    torch.manual_seed(args.seed + process_id)

    # Tải dataset
    # Đảm bảo DATASET_NAME_MAPPING trỏ đúng đến lớp BearHugDatasetForT2I của bạn
    train_dataset = T2I_DATASET_NAME_MAPPING[args.dataset](image_train_dir=args.train_data_dir)

    # Khởi tạo "động cơ" one-step với các tham số mới
    model = AUGMENT_METHODS[args.sample_strategy](
        base_model_name=args.base_model_name,
        swiftbrush_checkpoint_path=args.swiftbrush_checkpoint_path,
        lora_path=args.lora_path,
        embed_path=args.embed_path,
        device=f"cuda:{gpu_id}",
    )

    batch_size = args.batch_size

    while True:
        # Lấy nhiệm vụ từ hàng đợi
        try:
            tasks = [in_queue.get(timeout=10) for _ in range(batch_size)]
        except Empty:
            print(f"Process {process_id} finished.")
            break

        source_labels = [task[1] for task in tasks]
        target_labels = [task[2] for task in tasks]
        strengths = [task[3] for task in tasks]
        indices = [task[0] for task in tasks]

        # Lấy ảnh nguồn
        source_images = []
        source_indices = [random.choice(train_dataset.label_to_indices[label]) for label in source_labels]
        for idx in source_indices:
            source_images.append(train_dataset.get_image_by_idx(idx))

        # Lấy metadata của lớp đích (chỉ cần 1 vì trong batch này chúng giống nhau)
        target_label = target_labels[0]
        target_indice = random.choice(train_dataset.label_to_indices[target_label])
        target_metadata = train_dataset.get_metadata_by_idx(target_indice)

        # Gọi "động cơ" để xử lý
        generated_images, _ = model(
            image=source_images,
            label=target_label,
            metadata=target_metadata,
            strength=strengths[0],  # Giả định strength giống nhau trong 1 batch
        )

        # Lưu kết quả
        for i, gen_image in enumerate(generated_images):
            source_class_name = train_dataset.label2class[source_labels[i]].replace(" ", "_")
            target_class_name = target_metadata["name"].replace(" ", "_")
            strength_val = strengths[i]
            index_val = indices[i]

            save_dir = os.path.join(args.output_path, "data", source_class_name)
            os.makedirs(save_dir, exist_ok=True)

            save_name = f"{target_class_name}-{index_val:06d}-{strength_val}.png"
            save_path = os.path.join(save_dir, save_name)

            if not os.path.exists(save_path):
                gen_image.save(save_path)
                print(f"Saved: {save_path}")


# ... (Phần còn lại của sample_mp.py có thể giữ nguyên hoặc rút gọn, dưới đây là phiên bản đầy đủ đã sửa)

def main(args):
    torch.multiprocessing.set_start_method("spawn")

    # Tạo thư mục output
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "data"), exist_ok=True)

    # Thiết lập seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    in_queue = Queue()

    # Tải dataset để lấy thông tin về các lớp
    train_dataset = T2I_DATASET_NAME_MAPPING[args.dataset](image_train_dir=args.train_data_dir)
    num_classes = train_dataset.num_classes
    for name in train_dataset.class_names:
        os.makedirs(os.path.join(args.output_path, "data", name.replace(" ", "_")), exist_ok=True)

    # Lên kế hoạch sản xuất
    num_tasks = args.syn_dataset_multiplier * len(train_dataset)

    # Tạo danh sách lớp đích
    samples_per_class = num_tasks // num_classes
    target_classes = [i for i in range(num_classes) for _ in range(samples_per_class)]
    target_classes.extend(random.sample(range(num_classes), num_tasks % num_classes))
    random.shuffle(target_classes)

    # Tạo danh sách lớp nguồn
    if args.sample_strategy == "diff-aug":
        source_classes = target_classes
    elif args.sample_strategy == "diff-mix":
        source_classes = random.choices(range(num_classes), k=num_tasks)
    else:
        raise ValueError("Strategy not supported for one-step model")

    # Tạo danh sách strength
    strength_list = [args.aug_strength] * num_tasks

    # Đưa nhiệm vụ vào hàng đợi
    for task in zip(range(num_tasks), source_classes, target_classes, strength_list):
        in_queue.put(task)

    total_tasks = in_queue.qsize()
    print(f"Total tasks to generate: {total_tasks}")

    # Khởi động các tiến trình worker
    processes = []
    gpu_ids = args.gpu_ids * (args.num_processes // len(args.gpu_ids) + 1)

    for i in range(args.num_processes):
        process = Process(target=sample_func, args=(args, in_queue, gpu_ids[i], i))
        process.start()
        processes.append(process)

    # Theo dõi tiến trình
    with tqdm(total=total_tasks) as pbar:
        while total_tasks > in_queue.qsize():
            pbar.update(total_tasks - in_queue.qsize() - pbar.n)
            time.sleep(1)
            if in_queue.qsize() == 0 and not any(p.is_alive() for p in processes):
                break

    for p in processes:
        p.join()

    print("Data generation complete.")
    # (Bạn có thể thêm phần tạo meta.csv ở đây nếu cần)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("One-Step Data Augmentation Script")
    # Các tham số cho model của bạn
    parser.add_argument("--base_model_name", type=str, default="stabilityai/sd-turbo")
    parser.add_argument("--swiftbrush_checkpoint_path", type=str, required=True,
                        help="Path to original SwiftBrush UNet")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to your finetuned LoRA directory")
    parser.add_argument("--embed_path", type=str, required=True, help="Path to your learned_embeds.bin file")

    # Các tham số cho dataset và quá trình sinh ảnh
    parser.add_argument("--dataset", type=str, default="bear", help="Dataset name from your T2I_DATASET_NAME_MAPPING")
    parser.add_argument("--train_data_dir", type=str, required=True, help="Directory of your training images")
    parser.add_argument("--output_path", type=str, default="augmented_one_step",
                        help="Output directory for generated data")
    parser.add_argument("--sample_strategy", type=str, default="diff-mix", choices=["diff-mix", "diff-aug"])
    parser.add_argument("--syn_dataset_multiplier", type=int, default=5,
                        help="Num of synthetic images = multiplier * num of real images")
    parser.add_argument("--aug_strength", type=float, default=0.7, help="Augmentation strength for img2img")

    # Các tham số cho hệ thống
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--num_processes", type=int, default=1, help="Total number of parallel processes")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0], help="List of GPU IDs to use")

    args = parser.parse_args()
    main(args)