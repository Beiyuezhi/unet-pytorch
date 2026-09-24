from pathlib import Path
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

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
    """
    Paired AB1 training dataset.

    The expensive raw/trimmed sequence alignment is performed once when the
    dataset is created. __getitem__ only loads raw AB1 model features and reuses
    the prepared start/end labels, so alignment is not repeated every epoch.
    """

    def __init__(
        self,
        pairs,
        min_query_coverage: float = 0.80,
        show_prepare_progress: bool = True,
        prepare_desc: str = "Preparing AB1 labels",
    ):
        self.samples = []

        iterator = tqdm(
            list(pairs),
            desc=prepare_desc,
            unit="pair",
            disable=not show_prepare_progress,
        )
        for raw_path, trimmed_path in iterator:
            alignment = labels_from_ab1_pair(
                str(raw_path),
                str(trimmed_path),
                min_query_coverage=min_query_coverage,
            )
            self.samples.append(
                {
                    "raw_path": raw_path,
                    "trimmed_path": trimmed_path,
                    "start": alignment.start,
                    "end": alignment.end,
                    "coverage": alignment.query_coverage,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        raw_path = sample["raw_path"]

        features, _ = load_ab1_base_features(str(raw_path))
        length = features.shape[-1]

        start = min(int(sample["start"]), length)
        end = min(int(sample["end"]), length)

        target = torch.zeros(length, dtype=torch.float32)
        target[start:end] = 1.0

        return {
            "features": torch.from_numpy(features),
            "target": target,
            "length": length,
            "filename": raw_path.name,
            "start": start,
            "end": end,
            "coverage": sample["coverage"],
        }


def collate_ab1_batch(batch):
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
