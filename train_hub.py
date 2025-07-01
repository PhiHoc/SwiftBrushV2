import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet50, vit_b_16, ViT_B_16_Weights, ResNet18_Weights, ResNet50_Weights
from torchvision import transforms, datasets
from tqdm import tqdm


class CombinedDataset(Dataset):
    def __init__(self, real_data_dir, synthetic_data_dir, class_to_idx, image_size=256, crop_size=224, is_train=True,
                 limit_synthetic_samples_per_class=0):
        self.image_size, self.crop_size, self.class_to_idx, self.items = image_size, crop_size, class_to_idx, []

        # Bước 1: Luôn tải TẤT CẢ dữ liệu thật (real data)
        if real_data_dir and os.path.exists(real_data_dir):
            for class_name in os.listdir(real_data_dir):
                class_dir = os.path.join(real_data_dir, class_name)
                if os.path.isdir(class_dir) and class_name in self.class_to_idx:
                    label_idx = self.class_to_idx[class_name]
                    for img_name in os.listdir(class_dir):
                        self.items.append({"path": os.path.join(class_dir, img_name), "label": label_idx})

        # Bước 2: Chỉ giới hạn số lượng dữ liệu tổng hợp (synthetic data) nếu cần
        if synthetic_data_dir and os.path.exists(synthetic_data_dir):
            meta_path = os.path.join(synthetic_data_dir, 'meta.csv')
            if os.path.exists(meta_path):
                meta_df = pd.read_csv(meta_path)

                # Áp dụng logic lấy mẫu ngẫu nhiên CHỈ cho dữ liệu tổng hợp
                if is_train and limit_synthetic_samples_per_class > 0:
                    print(f"Giới hạn {limit_synthetic_samples_per_class} ảnh tổng hợp cho mỗi lớp.")
                    # Nhóm theo lớp và lấy ngẫu nhiên `n` mẫu cho mỗi nhóm
                    # Sử dụng lambda để xử lý trường hợp một lớp có ít hơn `n` mẫu
                    meta_df = meta_df.groupby('Target Class', group_keys=False).apply(
                        lambda x: x.sample(n=min(len(x), limit_synthetic_samples_per_class))
                    )

                for _, row in meta_df.iterrows():
                    if row['Target Class'] in self.class_to_idx:
                        self.items.append({
                            "path": os.path.join(synthetic_data_dir, 'data', row['Path']),
                            "label": self.class_to_idx[row['Target Class']]
                        })

        # Xáo trộn toàn bộ tập dữ liệu để đảm bảo tính ngẫu nhiên
        random.shuffle(self.items)

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomCrop(self.crop_size, padding=8) if is_train else transforms.CenterCrop(self.crop_size),
            transforms.RandomHorizontalFlip() if is_train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            img = Image.open(item['path']).convert("RGB")
            return self.transform(img), item['label']
        except Exception:
            return torch.zeros((3, self.crop_size, self.crop_size)), -1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = True

    temp_dataset = datasets.ImageFolder(root=args.real_data_dir)
    class_to_idx = temp_dataset.class_to_idx
    nb_class = len(class_to_idx)

    if args.model == "resnet18":
        net = resnet18(weights=None);
        net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif args.model == "resnet18pretrain":
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1);
        net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif args.model == "resnet50":
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2);
        net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif args.model == "vit_b_16":
        net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1);
        net.heads.head = nn.Linear(net.heads.head.in_features, nb_class)
    else:
        raise ValueError("Model not supported")

    net.to(device)
    if torch.__version__ >= "2.0.0":
        net = torch.compile(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepoch)

    train_set = CombinedDataset(
        args.real_data_dir, args.synthetic_data_dir, class_to_idx,
        args.resize, args.crop_size, is_train=True,
        limit_synthetic_samples_per_class=args.limit_synthetic_samples_per_class  # <-- Tham số mới
    )
    test_set = CombinedDataset(args.test_data_dir, None, class_to_idx, args.resize, args.crop_size, is_train=False)

    print(f"Tổng số ảnh trong tập huấn luyện: {len(train_set)}")
    print(f"Tổng số ảnh trong tập kiểm tra: {len(test_set)}")

    def collate_fn(batch):
        batch = [b for b in batch if b[1] != -1]
        if not batch: return None, None
        return torch.stack([item[0] for item in batch]), torch.LongTensor([item[1] for item in batch])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              collate_fn=collate_fn, pin_memory=True, persistent_workers=True)
    test_loader_full = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                  collate_fn=collate_fn, pin_memory=True, persistent_workers=True)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(args.nepoch):
        net.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.nepoch}")

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

        net.eval()
        eval_correct, eval_samples = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader_full:
                if inputs is None: continue
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                eval_correct += (predicted == labels).sum().item()
                eval_samples += inputs.size(0)

        if eval_samples > 0:
            eval_acc = eval_correct / eval_samples
            print(f"Epoch {epoch} - Test Accuracy: {eval_acc:.4f}")

            if eval_acc > best_accuracy:
                best_accuracy = eval_acc
                patience_counter = 0
                torch.save(net.state_dict(), output_dir / "best_model.pth")
                print(f"*** New best model saved with Test Acc: {best_accuracy:.4f} ***")
            else:
                patience_counter += 1

        if patience_counter >= 20:
            print(f"Early stopping at epoch {epoch}")
            break

        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình phân loại ảnh với tùy chọn giới hạn dữ liệu tổng hợp.")
    parser.add_argument("--real_data_dir", type=str, required=True, help="Thư mục chứa dữ liệu thật.")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Thư mục chứa dữ liệu test.")
    parser.add_argument("--synthetic_data_dir", type=str, help="Thư mục chứa dữ liệu tổng hợp (tùy chọn).")
    parser.add_argument("--output_dir", type=str, required=True, help="Thư mục để lưu model.")

    # --- Tham số được cập nhật ---
    parser.add_argument("--limit_synthetic_samples_per_class", type=int, default=0,
                        help="Giới hạn số lượng ảnh TỔNG HỢP cho mỗi lớp. Mặc định là 0 (sử dụng tất cả).")

    parser.add_argument("--model", type=str, default="resnet18",
                        help="Tên model: resnet18, resnet18pretrain, resnet50, vit_b_16")
    parser.add_argument("--nepoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)