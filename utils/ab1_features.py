from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from Bio import SeqIO


BASES = "ACGT"


def _as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", errors="ignore")
    return str(value)


def _trace_channels(abif_raw: Dict) -> Dict[str, np.ndarray]:
    """
    Return A/C/G/T electropherogram channels from ABI DATA9..DATA12.

    ABI stores the channel order in FWO_1. DATA9..DATA12 follow that order.
    """
    order = _as_text(abif_raw.get("FWO_1", "GATC")).strip("\x00")
    traces = {}
    for base, tag in zip(order[:4], ("DATA9", "DATA10", "DATA11", "DATA12")):
        if tag in abif_raw:
            traces[base.upper()] = np.asarray(abif_raw[tag], dtype=np.float32)

    # Keep a deterministic fallback for unusual files.
    for base in BASES:
        traces.setdefault(base, np.zeros(1, dtype=np.float32))
    return traces


def load_ab1_base_features(path: str) -> Tuple[np.ndarray, str]:
    """
    Convert one raw AB1 file to base-level features.

    Output:
        features: float32 array [9, n_bases]
            0..3  : base one-hot A/C/G/T
            4..7  : normalized A/C/G/T trace intensity at each called peak
            8     : normalized Phred quality
        sequence: called base sequence

    The network predicts one keep/discard value for each called base.
    """
    record = SeqIO.read(str(Path(path)), "abi")
    seq = str(record.seq).upper()
    n = len(seq)
    if n == 0:
        raise ValueError(f"No base calls found in AB1 file: {path}")

    abif_raw = record.annotations.get("abif_raw", {})
    peaks = np.asarray(abif_raw.get("PLOC2", np.arange(n)), dtype=np.int64)
    if len(peaks) < n:
        peaks = np.pad(peaks, (0, n - len(peaks)), mode="edge")
    peaks = peaks[:n]

    traces = _trace_channels(abif_raw)

    one_hot = np.zeros((4, n), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in BASES:
            one_hot[BASES.index(base), i] = 1.0

    peak_values = np.zeros((4, n), dtype=np.float32)
    for channel_idx, base in enumerate(BASES):
        trace = traces[base]
        if trace.size == 0:
            continue
        safe_peaks = np.clip(peaks, 0, trace.size - 1)
        peak_values[channel_idx] = trace[safe_peaks]

    # Per-file robust scaling preserves the relative competition among dye channels
    # while avoiding dependence on absolute instrument signal amplitude.
    scale = float(np.percentile(peak_values, 99.5))
    if scale <= 0:
        scale = float(np.max(peak_values))
    if scale > 0:
        peak_values = np.clip(peak_values / scale, 0.0, 2.0)

    qualities = record.letter_annotations.get("phred_quality", [0] * n)
    quality = np.asarray(qualities, dtype=np.float32)[:n]
    if quality.size < n:
        quality = np.pad(quality, (0, n - quality.size))
    quality = np.clip(quality / 60.0, 0.0, 1.0)[None, :]

    features = np.concatenate([one_hot, peak_values, quality], axis=0)
    return features.astype(np.float32), seq


def load_ab1_sequence(path: str) -> str:
    return str(SeqIO.read(str(Path(path)), "abi").seq).upper()
