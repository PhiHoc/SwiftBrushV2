import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Thêm resnet18 và ResNet18_Weights vào import
from torchvision.models import (vit_b_16, ViT_B_16_Weights,
                                resnet50, ResNet50_Weights,
                                resnet18, ResNet18_Weights)
from tqdm import tqdm

# Sử dụng lại hệ thống dataset đã có trong codebase của bạn
from dataset import DATASET_NAME_MAPPING


def set_seed(seed):
    """Hàm để đảm bảo các lần chạy có thể được tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    """Hàm huấn luyện chính."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    # 1. TẢI DATASET
    DatasetClass = DATASET_NAME_MAPPING[args.dataset_name]
    train_set = DatasetClass(
        split='train',
        image_train_dir=args.real_data_dir,
        examples_per_class=-1,
        return_onehot=False,
        synthetic_dir=args.synthetic_data_dir,
        synthetic_probability=args.synthetic_probability,
        image_size=args.resize,
        crop_size=args.crop_size,
    )
    test_set = DatasetClass(
        split='val',
        image_test_dir=args.test_data_dir,
        examples_per_class=-1,
        return_onehot=False,
        synthetic_dir=None,
        image_size=args.resize,
        crop_size=args.crop_size,
    )

    nb_class = train_set.num_classes
    print(f"Dataset: {args.dataset_name} with {nb_class} classes.")
    print(f"Using model: {args.model}")
    print(f"Training with `synthetic_probability` = {args.synthetic_probability}")
    print(f"Training with `label_smoothing` = {args.label_smoothing}")
    print(f"Testing with {len(test_set)} real samples.")

    # 2. THIẾT LẬP MODEL, LOSS, VÀ OPTIMIZER
    # --- CẬP NHẬT LOGIC TẢI MODEL ---
    if args.model == "resnet18":
        net = resnet18(weights=None)  # Huấn luyện từ đầu
        net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif args.model == "resnet18pretrain":
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Dùng trọng số pre-trained
        net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif args.model == "resnet50":
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif args.model == "vit_b_16":
        net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        net.heads.head = nn.Linear(net.heads.head.in_features, nb_class)
    else:
        raise ValueError(f"Model {args.model} is not supported.")

    net.to(device)
    if torch.__version__ >= "2.0.0":
        net = torch.compile(net)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepoch)

    # 3. THIẾT LẬP DATALOADER
    def collate_fn(batch):
        batch = [b for b in batch if b is not None and 'pixel_values' in b and 'label' in b]
        if not batch:
            return None, None
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        return pixel_values, labels

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
                             collate_fn=collate_fn)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_accuracy = 0.0
    patience_counter = 0
    # 4. VÒNG LẶP HUẤN LUYỆN
    for epoch in range(args.nepoch):
        net.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.nepoch}")

        for inputs, labels in pbar:
            if inputs is None: continue
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += inputs.size(0)
            if total_samples > 0:
                pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}", acc=f"{total_correct / total_samples:.4f}")

        # Đánh giá trên tập test sau mỗi epoch
        net.eval()
        eval_correct, eval_samples = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                if inputs is None: continue
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                eval_correct += (predicted == labels).sum().item()
                eval_samples += inputs.size(0)

        if eval_samples > 0:
            eval_acc = eval_correct / eval_samples
            print(f"\nEpoch {epoch + 1} - Test Accuracy: {eval_acc:.4f}")

            if eval_acc > best_accuracy:
                patience_counter = 0
                best_accuracy = eval_acc
                torch.save(net.state_dict(), output_dir / "best_model.pth")
                print(f"*** New best model saved with Test Acc: {best_accuracy:.4f} ***\n")
            else:
                patience_counter += 1

        if patience_counter >= 20:
            print(f"Early stopping at epoch {epoch}")
            break

        scheduler.step()

    print(f"Training finished. Best Test Accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thực nghiệm tăng cường ảnh với Label Smoothing.")

    parser.add_argument("--real_data_dir", type=str, required=True, help="Thư mục chứa dữ liệu thật.")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Thư mục chứa dữ liệu test.")
    parser.add_argument("--synthetic_data_dir", type=str, default=None,
                        help="Thư mục dữ liệu giả. Bỏ trống nếu không dùng.")
    parser.add_argument("--output_dir", type=str, required=True, help="Thư mục để lưu model.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=DATASET_NAME_MAPPING.keys(),
                        help="Tên dataset định nghĩa trong __init__.py.")

    parser.add_argument("--synthetic_probability", type=float, default=0.0,
                        help="Xác suất lấy một mẫu tổng hợp. Đặt 0.0 để chỉ train trên dữ liệu thật.")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Hệ số epsilon cho Label Smoothing. Đặt 0.0 để tắt.")

    # --- CẬP NHẬT DANH SÁCH MODEL ---
    parser.add_argument("--model", type=str, default="resnet18pretrain",
                        choices=["resnet18", "resnet18pretrain", "resnet50", "vit_b_16"],
                        help="Lựa chọn kiến trúc model để huấn luyện.")

    parser.add_argument("--nepoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate cho AdamW.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)