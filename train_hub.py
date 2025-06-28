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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50, vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from tqdm import tqdm


# --- Lớp Dataset Tùy chỉnh ---
class CombinedDataset(Dataset):
    """
    Một Dataset tùy chỉnh để tải cả ảnh thật và ảnh tổng hợp.
    Nó đọc file meta.csv để gán nhãn chính xác cho ảnh tổng hợp.
    """

    def __init__(self, real_data_dir, synthetic_data_dir, class_to_idx, image_size=256, crop_size=224, is_train=True):
        self.image_size = image_size
        self.crop_size = crop_size
        self.class_to_idx = class_to_idx
        self.items = []

        # 1. Tải ảnh thật
        if real_data_dir and os.path.exists(real_data_dir):
            for class_name in os.listdir(real_data_dir):
                class_dir = os.path.join(real_data_dir, class_name)
                if os.path.isdir(class_dir):
                    label_idx = self.class_to_idx[class_name]
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        self.items.append({"path": img_path, "label": label_idx})
            print(f"Đã tải {len(self.items)} ảnh thật.")
        else:
            print(f"Cảnh báo: Không tìm thấy thư mục ảnh thật tại: {real_data_dir}")

        # 2. Tải ảnh tổng hợp từ meta.csv
        if synthetic_data_dir and os.path.exists(synthetic_data_dir):
            meta_path = os.path.join(synthetic_data_dir, 'meta.csv')
            if os.path.exists(meta_path):
                meta_df = pd.read_csv(meta_path)
                initial_item_count = len(self.items)
                for _, row in meta_df.iterrows():
                    # Lấy nhãn từ 'Second Directory' (Lớp Đích)
                    target_class_name = row['Second Directory']
                    label_idx = self.class_to_idx[target_class_name.replace('_', ' ')]

                    # Đường dẫn đầy đủ đến ảnh tổng hợp
                    img_path = os.path.join(synthetic_data_dir, 'data', row['Path'])

                    self.items.append({"path": img_path, "label": label_idx})
                print(f"Đã tải {len(self.items) - initial_item_count} ảnh tổng hợp từ meta.csv.")
            else:
                print(f"Cảnh báo: Không tìm thấy file meta.csv tại: {meta_path}")

        # Định nghĩa phép biến đổi ảnh
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomCrop(self.crop_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:  # is_val / is_test
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = item['path']
        label = item['label']

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Lỗi khi tải ảnh {img_path}: {e}. Sẽ trả về một ảnh rỗng.")
            image = torch.zeros((3, self.crop_size, self.crop_size))  # Trả về tensor rỗng nếu lỗi
            label = -1  # Gán nhãn không hợp lệ để có thể bỏ qua sau này

        return image, label


def main(args):
    # --- Thiết lập cơ bản ---
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, output_dir / "train_script.py")
    with open(output_dir / "args.json", 'w') as f:
        json.dump(vars(args), f, indent=4)

    print(f"Thiết lập hoàn tất. Sử dụng device: {device}. Lưu kết quả tại: {output_dir}")

    # --- Tải dữ liệu ---
    # Lấy class_to_idx từ ImageFolder để đảm bảo tính nhất quán
    temp_dataset = transforms.torchvision.datasets.ImageFolder(root=args.real_data_dir)
    class_to_idx = temp_dataset.class_to_idx
    nb_class = len(class_to_idx)
    print(f"Đã tìm thấy {nb_class} lớp từ dữ liệu thật.")

    # Tạo training set bằng CombinedDataset
    train_set = CombinedDataset(
        real_data_dir=args.real_data_dir,
        synthetic_data_dir=args.synthetic_data_dir,
        class_to_idx=class_to_idx,
        image_size=args.resize,
        crop_size=args.crop_size,
        is_train=True
    )

    # Tạo test set (chỉ từ dữ liệu thật)
    test_set = CombinedDataset(
        real_data_dir=args.test_data_dir,
        synthetic_data_dir=None,  # Không dùng ảnh tổng hợp để test
        class_to_idx=class_to_idx,
        image_size=args.resize,
        crop_size=args.crop_size,
        is_train=False
    )

    def collate_fn(batch):
        # Lọc ra các mẫu bị lỗi
        batch = [b for b in batch if b[1] != -1]
        if not batch: return None, None
        images = torch.stack([item[0] for item in batch])
        labels = torch.LongTensor([item[1] for item in batch])
        return images, labels

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # --- Thiết lập mô hình, optimizer, loss ---
    if args.model == "resnet50":
        net = resnet50(weights='IMAGENET1K_V2')
        net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif args.model == "vit_b_16":
        net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        net.heads.head = nn.Linear(net.heads.head.in_features, nb_class)
    else:
        raise ValueError("Model not supported")

    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepoch * len(train_loader))

    # --- Logic Resume Checkpoint ---
    start_epoch = 0
    best_acc = 0.0
    train_log = []

    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"Đang tải lại checkpoint từ: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        train_log = checkpoint.get('log', [])
        print(f"Tiếp tục huấn luyện từ epoch {start_epoch}")

    # --- Vòng lặp huấn luyện ---
    for epoch in range(start_epoch, args.nepoch):
        net.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.nepoch} [Training]")
        for inputs, labels in pbar:
            if inputs is None: continue  # Bỏ qua batch lỗi
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += inputs.size(0)
            pbar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item() / inputs.size(0):.2f}")

        scheduler.step()
        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # --- Vòng lặp đánh giá ---
        net.eval()
        eval_correct = 0
        eval_samples = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Epoch {epoch}/{args.nepoch} [Evaluating]"):
                if inputs is None: continue
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                eval_correct += (predicted == labels).sum().item()
                eval_samples += inputs.size(0)

        eval_acc = eval_correct / eval_samples

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {eval_acc:.4f}")

        # --- Lưu log và checkpoint ---
        log_entry = {'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'test_acc': eval_acc}
        train_log.append(log_entry)

        is_best = eval_acc > best_acc
        if is_best:
            best_acc = eval_acc
            torch.save(net.state_dict(), output_dir / "best_model.pth")
            print(f"*** Đã lưu model tốt nhất mới với Test Acc: {best_acc:.4f} ***")

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'log': train_log
        }
        torch.save(checkpoint_data, output_dir / "latest_checkpoint.pth")

        # Ghi log ra file json để dễ đọc
        with open(output_dir / "training_log.json", 'w') as f:
            json.dump(train_log, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwiftBrush Classifier Training Script")
    # Các tham số quan trọng
    parser.add_argument("--real_data_dir", type=str, required=True, help="Đường dẫn đến thư mục ảnh thật (gốc).")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Đường dẫn đến thư mục ảnh test thật.")
    parser.add_argument("--synthetic_data_dir", type=str, required=True,
                        help="Đường dẫn đến thư mục chứa ảnh tổng hợp (chứa meta.csv).")
    parser.add_argument("--output_dir", type=str, required=True, help="Thư mục để lưu kết quả, log và checkpoint.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Đường dẫn đến file checkpoint để tiếp tục huấn luyện.")

    # Các tham số huấn luyện
    parser.add_argument("-m", "--model", default="resnet50", choices=["resnet50", "vit_b_16"], help="Tên model.")
    parser.add_argument("-ne", "--nepoch", type=int, default=100, help="Số lượng epoch.")
    parser.add_argument("-b", "--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--resize", default=256, type=int)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("-g", "--gpu", default=0, type=int)
    parser.add_argument("-s", "--seed", default=42, type=int)

    args = parser.parse_args()
    main(args)