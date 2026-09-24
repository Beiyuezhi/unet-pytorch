from pathlib import Path
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils.ab1_alignment import labels_from_ab1_pair
from utils.ab1_features import load_ab1_base_features


def discover_pairs(raw_dir: str, trimmed_dir: str) -> List[Tuple[Path, Path]]:
    raw_root = Path(raw_dir)
    trimmed_root = Path(trimmed_dir)

    raw_files = {
        p.name.lower(): p
        for p in raw_root.iterdir()
        if p.is_file() and p.suffix.lower() == ".ab1"
    }
    trimmed_files = {
        p.name.lower(): p
        for p in trimmed_root.iterdir()
        if p.is_file() and p.suffix.lower() == ".ab1"
    }

    common = sorted(set(raw_files) & set(trimmed_files))
    if not common:
        raise ValueError(
            "No same-name .ab1 pairs found. "
            "raw_dir and trimmed_dir must contain matching filenames."
        )

    return [(raw_files[name], trimmed_files[name]) for name in common]


class AB1PairDataset(Dataset):
    def __init__(
        self,
        pairs,
        min_query_coverage: float = 0.80,
    ):
        self.pairs = list(pairs)
        self.min_query_coverage = min_query_coverage

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        raw_path, trimmed_path = self.pairs[index]

        features, raw_sequence = load_ab1_base_features(str(raw_path))
        alignment = labels_from_ab1_pair(
            str(raw_path),
            str(trimmed_path),
            min_query_coverage=self.min_query_coverage,
        )

        length = features.shape[-1]
        target = torch.zeros(length, dtype=torch.float32)
        target[alignment.start:alignment.end] = 1.0

        return {
            "features": torch.from_numpy(features),
            "target": target,
            "length": length,
            "filename": raw_path.name,
            "start": alignment.start,
            "end": alignment.end,
            "coverage": alignment.query_coverage,
        }


def collate_ab1_batch(batch):
    # pad_sequence expects [L, C], so transpose before/after padding.
    feature_list = [item["features"].transpose(0, 1) for item in batch]
    features = pad_sequence(feature_list, batch_first=True).transpose(1, 2)

    targets = pad_sequence(
        [item["target"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_len = targets.shape[1]
    valid_mask = (
        torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    ).float()

    return {
        "features": features,
        "target": targets,
        "valid_mask": valid_mask,
        "lengths": lengths,
        "filenames": [item["filename"] for item in batch],
        "starts": torch.tensor([item["start"] for item in batch]),
        "ends": torch.tensor([item["end"] for item in batch]),
    }
