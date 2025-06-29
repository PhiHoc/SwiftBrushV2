# downstream_tasks/train_swiftbrush_classifier.py
# PHIÊN BẢN HOÀN CHỈNH - ĐÃ BỔ SUNG LOGGING METRICS GIỐNG TRAIN_HUB.PY

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


# (Lớp CombinedDataset không thay đổi)
class CombinedDataset(Dataset):
    def __init__(self, real_data_dir, synthetic_data_dir, class_to_idx, image_size=256, crop_size=224, is_train=True):
        self.image_size, self.crop_size, self.class_to_idx, self.items = image_size, crop_size, class_to_idx, []
        if real_data_dir and os.path.exists(real_data_dir):
            real_images_loaded = 0
            for class_name in os.listdir(real_data_dir):
                class_dir = os.path.join(real_data_dir, class_name)
                if os.path.isdir(class_dir) and class_name in self.class_to_idx:
                    label_idx = self.class_to_idx[class_name]
                    for img_name in os.listdir(class_dir):
                        self.items.append({"path": os.path.join(class_dir, img_name), "label": label_idx});
                        real_images_loaded += 1
            print(f"Loaded {real_images_loaded} real images.")
        if synthetic_data_dir and os.path.exists(synthetic_data_dir):
            meta_path = os.path.join(synthetic_data_dir, 'meta.csv')
            if os.path.exists(meta_path):
                meta_df, initial_count = pd.read_csv(meta_path), len(self.items)
                for _, row in meta_df.iterrows():
                    if row['Target Class'] in self.class_to_idx:
                        self.items.append({"path": os.path.join(synthetic_data_dir, 'data', row['Path']),
                                           "label": self.class_to_idx[row['Target Class']]})
                print(f"Loaded {len(self.items) - initial_count} synthetic images.")
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomCrop(self.crop_size, padding=8) if is_train else transforms.CenterCrop(self.crop_size),
            transforms.RandomHorizontalFlip() if is_train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            return self.transform(Image.open(item['path']).convert("RGB")), item['label']
        except Exception as e:
            print(f"Error loading image {item['path']}: {e}. Skipping.", file=sys.stderr)
            return torch.zeros((3, self.crop_size, self.crop_size)), -1


def set_seed(seed):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# <<< THÊM HÀM LƯU METRICS >>>
def save_metrics_to_json(metrics, filename):
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, path)


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    output_dir = Path(args.output_dir);
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, output_dir / "train_script.py")
    with open(output_dir / "args.json", 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Setup complete. Using device: {device}. Saving results to: {output_dir}")

    try:
        temp_dataset = datasets.ImageFolder(root=args.real_data_dir)
        class_to_idx = temp_dataset.class_to_idx;
        nb_class = len(class_to_idx)
        print(f"Found {nb_class} classes: {list(class_to_idx.keys())}")
    except FileNotFoundError:
        print(f"ERROR: Real data directory not found at '{args.real_data_dir}'.");
        return

    train_set = CombinedDataset(real_data_dir=args.real_data_dir, synthetic_data_dir=args.synthetic_data_dir,
                                class_to_idx=class_to_idx, image_size=args.resize, crop_size=args.crop_size,
                                is_train=True)
    test_set = CombinedDataset(real_data_dir=args.test_data_dir, synthetic_data_dir=None, class_to_idx=class_to_idx,
                               image_size=args.resize, crop_size=args.crop_size, is_train=False)

    def collate_fn(batch):
        batch = [b for b in batch if b[1] != -1]
        if not batch: return None, None
        return torch.stack([item[0] for item in batch]), torch.LongTensor([item[1] for item in batch])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn,
                              pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn,
                             pin_memory=True)

    if args.model == "resnet18":
        net = resnet18(weights=None); net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif args.model == "resnet18pretrain":
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif args.model == "resnet50":
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2); net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif args.model == "vit_b_16":
        net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1); net.heads.head = nn.Linear(
            net.heads.head.in_features, nb_class)
    else:
        raise ValueError("Model not supported")
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepoch)

    start_epoch = 0
    best_accuracy = 0.0
    train_losses, train_accuracies, eval_losses, eval_accuracies = [], [], [], []
    metrics_path = output_dir / "metrics.json"
    checkpoint_path = output_dir / "latest_checkpoint.pth"

    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        # Tải lại metrics từ checkpoint
        metrics = checkpoint.get('metrics', {})
        train_losses = metrics.get('train_losses', [])
        train_accuracies = metrics.get('train_accuracies', [])
        eval_losses = metrics.get('eval_losses', [])
        eval_accuracies = metrics.get('eval_accuracies', [])
        best_accuracy = metrics.get('best_accuracy', 0.0)

        print(f"Resumed successfully. Continuing from epoch {start_epoch}")

    for epoch in range(start_epoch, args.nepoch):
        net.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.nepoch} [Training]")
        for inputs, labels in pbar:
            if inputs is None: continue
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
            pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}", acc=f"{total_correct / total_samples:.4f}")

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # <<< THAY ĐỔI: THÊM TÍNH TOÁN EVAL_LOSS >>>
        net.eval()
        eval_correct, eval_samples, eval_total_loss = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Epoch {epoch}/{args.nepoch} [Evaluating]"):
                if inputs is None: continue
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                eval_total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                eval_correct += (predicted == labels).sum().item()
                eval_samples += inputs.size(0)
        eval_acc = eval_correct / eval_samples
        eval_loss = eval_total_loss / eval_samples

        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}")

        # <<< THAY ĐỔI: LOGIC LƯU METRICS MỚI >>>
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        eval_losses.append(eval_loss)
        eval_accuracies.append(eval_acc)

        if eval_acc > best_accuracy:
            best_accuracy = eval_acc
            torch.save(net.state_dict(), output_dir / "best_model.pth")
            print(f"*** New best model saved with Test Acc: {best_accuracy:.4f} ***")

        metrics = {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "eval_losses": eval_losses,
            "eval_accuracies": eval_accuracies,
            "best_accuracy": best_accuracy
        }

        save_metrics_to_json(metrics, metrics_path)
        save_checkpoint(net, optimizer, scheduler, epoch, metrics, checkpoint_path)

        scheduler.step()


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
    parser.add_argument("-m", "--model", default="resnet18", help="Tên model.")
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