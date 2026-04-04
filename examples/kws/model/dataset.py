import torch
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torchaudio import datasets

SAMPLE_RATE = 16000
NUM_SAMPLES = 16000
NUM_MFCC = 12

mfcc_transform = T.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=NUM_MFCC,
    melkwargs={
        "n_fft": 512,
        "hop_length": 256,
        "n_mels": 40,
        "center": False,
    },
)

speech_commands_class = [
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
]

speech_commands_map = {label: idx for idx, label in enumerate(speech_commands_class)}


def _pad_or_trim_waveform(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    """如果 waveform 的长度超过 target_num_samples，则截断；如果不足，则在末尾补零。"""
    num_samples = waveform.size(1)
    if num_samples == target_num_samples:
        return waveform
    if num_samples > target_num_samples:
        return waveform[:, :target_num_samples]
    pad_len = target_num_samples - num_samples
    return torch.nn.functional.pad(waveform, (0, pad_len))


def _extract_mfcc(
    waveform: torch.Tensor,
    sample_rate: int,
    resamplers: dict[int, T.Resample],
) -> torch.Tensor:
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != SAMPLE_RATE:
        if sample_rate not in resamplers:
            resamplers[sample_rate] = T.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resamplers[sample_rate](waveform)

    waveform = _pad_or_trim_waveform(waveform, target_num_samples=NUM_SAMPLES)

    # MFCC 输出是 [channel, n_mfcc, time]，转成 [channel, time, n_mfcc] 以匹配 KWS 输入约定。
    return mfcc_transform(waveform).transpose(1, 2)


def _compute_global_mfcc_stats(train_ds: datasets.SPEECHCOMMANDS) -> tuple[torch.Tensor, torch.Tensor]:
    """基于训练集统计全局 MFCC 均值和标准差。"""
    sum_mfcc = torch.zeros(NUM_MFCC, dtype=torch.float64)
    sumsq_mfcc = torch.zeros(NUM_MFCC, dtype=torch.float64)
    total_count = 0
    resamplers: dict[int, T.Resample] = {}

    with torch.no_grad():
        for waveform, sample_rate, _, _, _ in train_ds:
            mfcc = _extract_mfcc(waveform, sample_rate, resamplers)
            flat = mfcc.reshape(-1, NUM_MFCC).to(torch.float64)
            sum_mfcc += flat.sum(dim=0)
            sumsq_mfcc += (flat * flat).sum(dim=0)
            total_count += flat.size(0)

    if total_count == 0:
        raise RuntimeError("Failed to compute global MFCC stats: training dataset is empty.")

    mean = sum_mfcc / total_count
    var = (sumsq_mfcc / total_count) - (mean * mean)
    std = torch.sqrt(var.clamp_min(1e-12))
    mean = mean.to(torch.float32).view(1, 1, 1, NUM_MFCC)
    std = std.to(torch.float32).view(1, 1, 1, NUM_MFCC)
    return mean, std


def load_speech_commands(root: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_ds = datasets.SPEECHCOMMANDS(root=root, subset="training", download=True)
    test_ds = datasets.SPEECHCOMMANDS(root=root, subset="testing", download=True)

    global_mean = torch.tensor([-164.69, 37.94, -0.2, 5.39, -5.43, -0.63, -4.05, -0.92, -3.31, -0.44, -4.02, -0.75])
    global_std = torch.tensor([106.45, 41.18, 23.14, 17.16, 15.91, 13.01, 11.38, 9.81, 8.6, 7.92, 7.46, 7.01])
    resamplers: dict[int, T.Resample] = {}

    def collate_fn(batch: list[tuple[torch.Tensor, int, str, str, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """将原始音频数据转换为 MFCC 特征，并将标签转换为整数索引"""
        features: list[torch.Tensor] = []
        labels: list[int] = []

        for waveform, sample_rate, label, _, _ in batch:
            mfcc = _extract_mfcc(waveform, sample_rate, resamplers)
            features.append(mfcc)
            labels.append(speech_commands_map[label])

        x = torch.stack(features, dim=0)
        x = (x - global_mean) / global_std
        y = torch.tensor(labels, dtype=torch.long)
        return x, y

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader


if __name__ == "__main__":
    train_ds = datasets.SPEECHCOMMANDS(root="/home/linkwanna/.data", subset="training", download=True)
    global_mean, global_std = _compute_global_mfcc_stats(train_ds)

    # 保留两位小数，打印 global_mean 和 global_std 的值，
    global_mean = global_mean.flatten().tolist()
    global_std = global_std.flatten().tolist()
    global_mean = [round(x, 2) for x in global_mean]
    global_std = [round(x, 2) for x in global_std]
    print("Global MFCC mean:", global_mean)
    print("Global MFCC std:", global_std)
