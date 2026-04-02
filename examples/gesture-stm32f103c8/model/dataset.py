import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

gesture_class_list = [
    "RightAngle",
    "SharpAngle",
    "Lightning",
    "Triangle",
    "Letter_h",
    "letter_R",
    "letter_W",
    "letter_phi",
    "Circle",
    "UpAndDown",
    "Horn",
    "Wave",
    "NoMotion",
]
gesture_class_encoding = {category: index for index, category in enumerate(gesture_class_list)}


USE_COLS = (3, 4, 5)
FILE_FORMAT = ".txt"


def make_loaders(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    num_samples = features.shape[0]
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_size = int(num_samples * (1 - val_ratio))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_ds = TensorDataset(features[train_idx], labels[train_idx])
    val_ds = TensorDataset(features[val_idx], labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def load_gesture_data(root: str, batch_size: int, val_ratio: float = 0.2) -> tuple[DataLoader, DataLoader]:
    data_list: list[np.ndarray] = []  # (N, channel, step)
    labels: list[int] = []
    path = Path(root).expanduser()

    for path in path.iterdir():
        # 跳过非文件或扩展名不符的条目
        if not path.is_file() or path.suffix != FILE_FORMAT:
            continue

        filename = path.name
        match = re.match(rf"^([\w]+)_([\d]+){FILE_FORMAT}$", filename)
        if not match:
            print(f"Skip invalid filename: {filename}")
            continue

        motion_name = match.group(1)

        # 校验动作名称合法性
        if motion_name not in gesture_class_encoding:
            print(f"Skip unknown motion: {filename}")
            continue

        # 读取指定列的数据，并记录样本与标签
        data = np.loadtxt(path, delimiter=" ", usecols=USE_COLS).astype(np.float32)  # (step, channel)
        data = data.T  # (channel, step)
        data_list.append(data)
        labels.append(gesture_class_encoding[motion_name])

    # 将列表转换为张量
    data_np = np.stack(data_list, axis=0)  # (N, channel, step)
    labels_np = np.array(labels)  # (N,)

    features = torch.from_numpy(data_np)  # (N, 3, max_len)
    targets = torch.from_numpy(labels_np)
    print(f"Loaded {len(features)} samples with shape {features.shape} and labels with shape {targets.shape}")

    return make_loaders(features, targets, batch_size=batch_size, val_ratio=val_ratio)
