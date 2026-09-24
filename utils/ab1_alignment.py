from dataclasses import dataclass

from Bio.Align import PairwiseAligner

from utils.ab1_features import load_ab1_sequence


@dataclass
class TrimAlignment:
    start: int
    end: int
    score: float
    query_coverage: float


def align_trimmed_to_raw(
    raw_sequence: str,
    trimmed_sequence: str,
    min_query_coverage: float = 0.80,
) -> TrimAlignment:
    """
    Locate the manually trimmed read inside the untrimmed read.

    start/end are zero-based base indices in the raw read, with end exclusive.
    This alignment is only used to create supervised labels; it is not used
    during model inference.
    """
    if not raw_sequence or not trimmed_sequence:
        raise ValueError("Raw and trimmed sequences must both be non-empty.")

    aligner = PairwiseAligner(mode="local")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -3.0
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(raw_sequence, trimmed_sequence)
    if len(alignments) == 0:
        raise ValueError("No alignment found between raw and trimmed AB1 sequences.")

    aln = alignments[0]
    raw_blocks, trimmed_blocks = aln.aligned
    if len(raw_blocks) == 0:
        raise ValueError("Alignment contained no aligned blocks.")

    start = int(raw_blocks[0][0])
    end = int(raw_blocks[-1][1])
    aligned_query_bases = sum(int(b - a) for a, b in trimmed_blocks)
    coverage = aligned_query_bases / max(len(trimmed_sequence), 1)

    if coverage < min_query_coverage:
        raise ValueError(
            f"Trimmed/raw alignment coverage too low: {coverage:.3f} "
            f"(required >= {min_query_coverage:.3f})"
        )

    return TrimAlignment(
        start=start,
        end=end,
        score=float(aln.score),
        query_coverage=float(coverage),
    )


def labels_from_ab1_pair(
    raw_path: str,
    trimmed_path: str,
    min_query_coverage: float = 0.80,
):
    raw_sequence = load_ab1_sequence(raw_path)
    trimmed_sequence = load_ab1_sequence(trimmed_path)
    alignment = align_trimmed_to_raw(
        raw_sequence,
        trimmed_sequence,
        min_query_coverage=min_query_coverage,
    )
    return alignment
