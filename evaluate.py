import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import (vit_b_16, ViT_B_16_Weights,
                                resnet50, ResNet50_Weights,
                                resnet18, ResNet18_Weights)
from tqdm import tqdm

# Đảm bảo rằng bạn đã import đúng Dataset class từ project của mình
from dataset import DATASET_NAME_MAPPING


def set_seed(seed):
    """Hàm để đảm bảo các lần chạy có thể được tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(args):
    """Hàm đánh giá chính."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # 1. TẢI DATASET TEST
    DatasetClass = DATASET_NAME_MAPPING[args.dataset_name]
    test_set = DatasetClass(
        split='val',  # Hoặc 'test' tùy thuộc vào cách bạn định nghĩa trong class Dataset
        image_test_dir=args.test_data_dir,
        examples_per_class=-1,
        return_onehot=False,
        synthetic_dir=None,
        image_size=args.resize,
        crop_size=args.crop_size,
    )

    nb_class = test_set.num_classes
    print(f"Dataset: {args.dataset_name} với {nb_class} lớp.")
    print(f"Sử dụng mô hình: {args.model}")
    print(f"Đánh giá trên {len(test_set)} mẫu thật.")

    # 2. THIẾT LẬP MODEL
    if args.model == "resnet18" or args.model == "resnet18pretrain":
        net = resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif args.model == "resnet50":
        net = resnet50(weights=None)
        net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif args.model == "vit_b_16":
        net = vit_b_16(weights=None)
        net.heads.head = nn.Linear(net.heads.head.in_features, nb_class)
    else:
        raise ValueError(f"Mô hình {args.model} không được hỗ trợ.")

    # Tải trọng số đã huấn luyện
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Không tìm thấy tệp mô hình tại: {args.model_path}")

    # Tải state_dict vào mô hình. Sử dụng map_location để đảm bảo nó hoạt động trên cả CPU và GPU.
    state_dict = torch.load(args.model_path, map_location=device)

    # Xử lý trường hợp mô hình được lưu dưới dạng DataParallel hoặc torch.compile
    # Loại bỏ tiền tố 'module.' hoặc '_orig_mod.' nếu có
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if k.startswith('module.'):
            name = k[7:]  # Loại bỏ 'module.'
        elif k.startswith('_orig_mod.'):
            name = k[10:]  # Loại bỏ '_orig_mod.'
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    print(f"Đã tải thành công trọng số từ: {args.model_path}")

    net.to(device)
    if torch.__version__ >= "2.0.0":
        net = torch.compile(net)

    # 3. THIẾT LẬP DATALOADER
    def collate_fn(batch):
        batch = [b for b in batch if b is not None and 'pixel_values' in b and 'label' in b]
        if not batch:
            return None, None
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        return pixel_values, labels

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
                             collate_fn=collate_fn)

    # 4. VÒNG LẶP ĐÁNH GIÁ
    net.eval()  # Chuyển mô hình sang chế độ đánh giá
    eval_correct = 0
    eval_samples = 0

    pbar = tqdm(test_loader, desc="Đang đánh giá")

    with torch.no_grad():  # Không cần tính toán gradient khi đánh giá
        for inputs, labels in pbar:
            if inputs is None: continue
            inputs, labels = inputs.to(device), labels.to(device)

            # Sử dụng autocast nếu bạn muốn tận dụng AMP để tăng tốc độ inference
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            eval_correct += (predicted == labels).sum().item()
            eval_samples += inputs.size(0)

            if eval_samples > 0:
                current_acc = eval_correct / eval_samples
                pbar.set_postfix(accuracy=f"{current_acc:.4f}")

    if eval_samples > 0:
        final_accuracy = eval_correct / eval_samples
        print("\n" + "=" * 50)
        print(f"Đánh giá hoàn tất.")
        print(f"Tổng số mẫu: {eval_samples}")
        print(f"Số mẫu dự đoán đúng: {eval_correct}")
        print(f"Độ chính xác cuối cùng trên tập test: {final_accuracy:.4f} ({final_accuracy:.2%})")
        print("=" * 50)
    else:
        print("Không có mẫu nào trong tập test để đánh giá.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mô hình trên tập dữ liệu test.")

    # --- Các tham số chính cho việc đánh giá ---
    parser.add_argument("--model_path", type=str, required=True, help="Đường dẫn đến tệp model.pth đã lưu.")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Thư mục chứa dữ liệu test.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=DATASET_NAME_MAPPING.keys(),
                        help="Tên dataset định nghĩa trong __init__.py.")
    parser.add_argument("--model", type=str, required=True,
                        choices=["resnet18", "resnet18pretrain", "resnet50", "vit_b_16"],
                        help="Kiến trúc mô hình tương ứng với tệp model.pth.")

    # --- Các tham số cấu hình khác ---
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    evaluate(args)