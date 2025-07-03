import abc
import math
import os
import random
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.0
    return vec


class SyntheticDataset(Dataset):
    def __init__(
        self,
        synthetic_dir: Union[str, List[str]] = None,
        gamma: int = 1,
        soft_scaler: float = 1,
        num_syn_seeds: int = 999,
        image_size: int = 512,
        crop_size: int = 448,
        class2label: dict = None,
    ) -> None:
        super().__init__()

        self.synthetic_dir = synthetic_dir
        self.num_syn_seeds = num_syn_seeds  # number of seeds to generate synthetic data
        self.gamma = gamma
        self.soft_scaler = soft_scaler
        self.class_names = None

        self.parse_syn_data_pd(synthetic_dir)

        test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.transform = test_transform
        self.class2label = (
            {name: i for i, name in enumerate(self.class_names)}
            if class2label is None
            else class2label
        )
        self.num_classes = len(self.class2label.keys())

    def set_transform(self, transform) -> None:
        self.transform = transform

    def parse_syn_data_pd(self, synthetic_dir) -> None:
        if isinstance(synthetic_dir, list):
            pass
        elif isinstance(synthetic_dir, str):
            synthetic_dir = [synthetic_dir]
        else:
            raise NotImplementedError("Not supported type")
        meta_df_list = []

        for _dir in synthetic_dir:
            meta_dir = os.path.join(_dir, self.csv_file)
            meta_df = pd.read_csv(meta_dir)
            meta_df.loc[:, "Path"] = meta_df["Path"].apply(
                lambda x: os.path.join(_dir, "data", x)
            )
            meta_df_list.append(meta_df)
        self.meta_df = pd.concat(meta_df_list).reset_index(drop=True)

        self.syn_nums = len(self.meta_df)
        self.class_names = list(set(self.meta_df["Target Class"].values))
        print(f"Syn numbers: {self.syn_nums}\n")

    def get_syn_item_raw(self, idx: int):
        df_data = self.meta_df.iloc[idx]
        src_label = self.class2label[df_data["First Directory"]]
        tar_label = self.class2label[df_data["Second Directory"]]
        path = df_data["Path"]
        return path, src_label, tar_label

    def __len__(self) -> int:
        return self.syn_nums

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, src_label, target_label = self.get_syn_item_raw(idx)
        image = Image.open(path).convert("RGB")
        return {
            "pixel_values": self.transform(image),
            "src_label": src_label,
            "tar_label": target_label,
        }


class HugFewShotDataset(Dataset):

    num_classes: int = None
    class_names: int = None
    class2label: dict = None
    label2class: dict = None

    def __init__(
        self,
        split: str = "train",
        examples_per_class: int = None,
        synthetic_probability: float = 0.5,
        return_onehot: bool = False,
        soft_scaler: float = 1,
        synthetic_dir: Union[str, List[str]] = None,
        image_size: int = 512,
        crop_size: int = 448,
        gamma: int = 1,
        num_syn_seeds: int = 99999,
        clip_filtered_syn: bool = False,
        target_class_num: int = None,
        **kwargs,
    ):

        self.examples_per_class = examples_per_class
        self.num_syn_seeds = num_syn_seeds  # number of seeds to generate synthetic data

        self.synthetic_dir = synthetic_dir
        self.clip_filtered_syn = clip_filtered_syn
        self.return_onehot = return_onehot

        if self.synthetic_dir is not None:
            # assert self.return_onehot == True
            self.synthetic_probability = synthetic_probability
            self.soft_scaler = soft_scaler
            self.gamma = gamma
            self.target_class_num = target_class_num
            self.parse_syn_data_pd(synthetic_dir)

        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop(crop_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.transform = {"train": train_transform, "val": test_transform}[split]

    def set_transform(self, transform) -> None:
        self.transform = transform

    @abc.abstractmethod
    def get_image_by_idx(self, idx: int) -> Image.Image:

        return NotImplemented

    @abc.abstractmethod
    def get_label_by_idx(self, idx: int) -> int:

        return NotImplemented

    @abc.abstractmethod
    def get_metadata_by_idx(self, idx: int) -> dict:

        return NotImplemented

    def parse_syn_data_pd(self, synthetic_dir, filter=True) -> None:
        if isinstance(synthetic_dir, list):
            pass
        elif isinstance(synthetic_dir, str):
            synthetic_dir = [synthetic_dir]
        else:
            raise NotImplementedError("Not supported type")
        meta_df_list = []
        for _dir in synthetic_dir:
            df_basename = (
                "meta.csv" if not self.clip_filtered_syn else "remained_meta.csv"
            )
            meta_dir = os.path.join(_dir, df_basename)
            meta_df = self.filter_df(pd.read_csv(meta_dir))
            meta_df.loc[:, "Path"] = meta_df["Path"].apply(
                lambda x: os.path.join(_dir, "data", x)
            )
            meta_df_list.append(meta_df)
        self.meta_df = pd.concat(meta_df_list).reset_index(drop=True)
        self.syn_nums = len(self.meta_df)

        print(f"Syn numbers: {self.syn_nums}\n")

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.target_class_num is not None:
            selected_indexs = []
            for source_name in self.class_names:
                target_classes = random.sample(self.class_names, self.target_class_num)
                indexs = df[
                    (df["First Directory"] == source_name)
                    & (df["Second Directory"].isin(target_classes))
                ]
                selected_indexs.append(indexs)

            meta2 = pd.concat(selected_indexs, axis=0)
            total_num = min(len(meta2), 18000)
            idxs = random.sample(range(len(meta2)), total_num)
            meta2 = meta2.iloc[idxs]
            meta2.reset_index(drop=True, inplace=True)
            df = meta2
            print("filter_df", self.target_class_num, len(df))
        return df

    def get_syn_item(self, idx: int):

        df_data = self.meta_df.iloc[idx]
        src_label = self.class2label[df_data["First Directory"]]
        tar_label = self.class2label[df_data["Second Directory"]]
        path = df_data["Path"]
        strength = df_data["Strength"]
        onehot_label = torch.zeros(self.num_classes)
        onehot_label[src_label] += self.soft_scaler * (
            1 - math.pow(strength, self.gamma)
        )
        onehot_label[tar_label] += self.soft_scaler * math.pow(strength, self.gamma)

        return path, onehot_label

    def __getitem__(self, idx: int) -> dict:
        # Logic xác suất để chọn giữa ảnh thật và ảnh giả
        use_synthetic = self.synthetic_dir is not None and np.random.uniform() < self.synthetic_probability

        try:
            if use_synthetic:
                # === LOGIC MỚI CHO ẢNH GIẢ ===
                # 1. Chọn ngẫu nhiên một ảnh giả từ metadata
                syn_idx = np.random.choice(self.syn_nums)
                df_data = self.meta_df.iloc[syn_idx]

                # 2. Đọc đường dẫn và tên lớp từ cột "Path" và "Target Class"
                image_path = df_data["Path"]  # Đường dẫn này đã là đường dẫn đầy đủ
                class_name = df_data["Target Class"]  # Đọc đúng tên cột

                # 3. Lấy chỉ số (label) của lớp
                label = self.class2label[class_name]
                image = Image.open(image_path).convert("RGB")
            else:
                image = self.get_image_by_idx(idx)
                label = self.get_label_by_idx(idx)

            # Trả về kết quả sau khi áp dụng transform
            return dict(pixel_values=self.transform(image), label=label)

        except Exception as e:
            # Nếu có lỗi khi đọc ảnh (vd: ảnh bị hỏng), bỏ qua mẫu này
            # print(f"Warning: Skipping problematic image at index {idx}. Error: {e}")
            return None
