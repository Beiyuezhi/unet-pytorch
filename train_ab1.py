import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.ab1_dataset import AB1PairDataset, collate_ab1_batch, discover_pairs
from nets.unet_resnet_1d import ResNet50UNet1D, ResNet101UNet1D


def build_model(backbone: str, input_channels: int = 9):
    if backbone == "resnet50":
        return ResNet50UNet1D(input_channels=input_channels)
    if backbone == "resnet101":
        return ResNet101UNet1D(input_channels=input_channels)
    raise ValueError(f"Unsupported backbone: {backbone}")


def boundary_weighted_bce_with_logits(
    logits,
    targets,
    valid_mask,
    starts,
    ends,
    boundary_radius=12,
    boundary_weight=4.0,
):
    per_position = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )

    batch_size, length = logits.shape
    positions = torch.arange(length, device=logits.device).unsqueeze(0)
    starts = starts.to(logits.device).view(batch_size, 1)
    ends = ends.to(logits.device).view(batch_size, 1)

    near_start = (positions - starts).abs() <= boundary_radius
    near_end = (positions - ends).abs() <= boundary_radius
    boundary = (near_start | near_end).float()

    weights = 1.0 + (boundary_weight - 1.0) * boundary
    weights = weights * valid_mask

    return (per_position * weights).sum() / weights.sum().clamp_min(1.0)


def masked_dice_loss(logits, targets, valid_mask, eps=1e-6):
    probs = torch.sigmoid(logits) * valid_mask
    targets = targets * valid_mask
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def masked_boundary_losses(boundary_logits, starts, ends, lengths):
    """
    boundary_logits: [B, 2, L]
      channel 0 -> start position
      channel 1 -> inclusive end position (true end - 1)

    Invalid padded positions are masked before cross entropy.
    Losses are normalized by log(sequence length) so their scale is comparable
    to the segmentation loss.
    """
    device = boundary_logits.device
    _, _, max_len = boundary_logits.shape

    lengths = lengths.to(device)
    starts = starts.to(device).long().clamp_min(0)
    end_targets = (ends.to(device).long() - 1).clamp_min(0)

    starts = torch.minimum(starts, lengths - 1)
    end_targets = torch.minimum(end_targets, lengths - 1)

    positions = torch.arange(max_len, device=device).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)

    start_logits = boundary_logits[:, 0, :].masked_fill(~valid, -1e4)
    end_logits = boundary_logits[:, 1, :].masked_fill(~valid, -1e4)

    start_ce = F.cross_entropy(start_logits, starts)
    end_ce = F.cross_entropy(end_logits, end_targets)

    mean_length = lengths.float().mean().clamp_min(2.0)
    normalizer = math.log(float(mean_length.item()) + 1.0)

    return start_ce / normalizer, end_ce / normalizer


def interval_from_probs(probs, threshold=0.5):
    keep = probs >= threshold
    best = (0, 0, -1.0)
    start = None

    for i, flag in enumerate(keep.tolist() + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            end = i
            score = float(probs[start:end].sum())
            if (end - start) > (best[1] - best[0]) or (
                (end - start) == (best[1] - best[0]) and score > best[2]
            ):
                best = (start, end, score)
            start = None

    return best[0], best[1]


def interval_from_model_outputs(seg_probs, boundary_logits, length):
    """
    Boundary head is the primary predictor.
    If it produces an invalid interval, fall back to the segmentation mask.
    """
    start_logits = boundary_logits[0, :length]
    end_logits = boundary_logits[1, :length]

    pred_start = int(torch.argmax(start_logits).item())
    pred_end = int(torch.argmax(end_logits).item()) + 1

    used_fallback = False
    if pred_end <= pred_start:
        pred_start, pred_end = interval_from_probs(seg_probs[:length])
        used_fallback = True

    return pred_start, pred_end, used_fallback


def compute_loss(
    outputs,
    batch,
    device,
    boundary_radius,
    boundary_weight,
    start_head_weight,
    end_head_weight,
):
    y = batch["target"].to(device)
    valid = batch["valid_mask"].to(device)

    seg_logits = outputs["seg_logits"].squeeze(1)
    boundary_logits = outputs["boundary_logits"]

    bce = boundary_weighted_bce_with_logits(
        seg_logits,
        y,
        valid,
        batch["starts"],
        batch["ends"],
        boundary_radius=boundary_radius,
        boundary_weight=boundary_weight,
    )
    dice = masked_dice_loss(seg_logits, y, valid)
    seg_loss = 0.5 * bce + 0.5 * dice

    start_loss, end_loss = masked_boundary_losses(
        boundary_logits,
        batch["starts"],
        batch["ends"],
        batch["lengths"],
    )

    total = (
        seg_loss
        + start_head_weight * start_loss
        + end_head_weight * end_loss
    )

    return total, seg_loss, start_loss, end_loss


def evaluate(
    model,
    loader,
    device,
    boundary_radius,
    boundary_weight,
    start_head_weight,
    end_head_weight,
    epoch=None,
):
    model.eval()
    total_loss = 0.0
    total_start_abs = 0.0
    total_end_abs = 0.0
    total_start_signed = 0.0
    total_end_signed = 0.0
    total_iou = 0.0
    fallback_count = 0

    start_within = {5: 0, 10: 0, 20: 0}
    end_within = {5: 0, 10: 0, 20: 0}
    count = 0

    desc = f"Epoch {epoch:03d} val" if epoch is not None else "Validation"
    progress = tqdm(loader, desc=desc, unit="batch", leave=False)

    with torch.no_grad():
        for batch in progress:
            x = batch["features"].to(device)
            outputs = model(x)

            loss, _, _, _ = compute_loss(
                outputs,
                batch,
                device,
                boundary_radius,
                boundary_weight,
                start_head_weight,
                end_head_weight,
            )
            total_loss += float(loss.item()) * x.size(0)

            seg_probs = torch.sigmoid(outputs["seg_logits"].squeeze(1)).cpu()
            boundary_logits = outputs["boundary_logits"].cpu()

            for i in range(x.size(0)):
                length = int(batch["lengths"][i])
                pred_start, pred_end, used_fallback = interval_from_model_outputs(
                    seg_probs[i],
                    boundary_logits[i],
                    length,
                )
                fallback_count += int(used_fallback)

                true_start = int(batch["starts"][i])
                true_end = int(batch["ends"][i])

                start_error = pred_start - true_start
                end_error = pred_end - true_end

                total_start_abs += abs(start_error)
                total_end_abs += abs(end_error)
                total_start_signed += start_error
                total_end_signed += end_error

                for threshold in start_within:
                    if abs(start_error) <= threshold:
                        start_within[threshold] += 1
                    if abs(end_error) <= threshold:
                        end_within[threshold] += 1

                inter = max(0, min(pred_end, true_end) - max(pred_start, true_start))
                union = max(pred_end, true_end) - min(pred_start, true_start)
                total_iou += inter / union if union > 0 else 0.0
                count += 1

            progress.set_postfix(val_loss=f"{loss.item():.4f}")

    denom = max(count, 1)
    start_mae = total_start_abs / denom
    end_mae = total_end_abs / denom

    return {
        "loss": total_loss / denom,
        "start_mae": start_mae,
        "end_mae": end_mae,
        "boundary_mae": (start_mae + end_mae) / 2.0,
        "start_bias": total_start_signed / denom,
        "end_bias": total_end_signed / denom,
        "start_within_5bp": start_within[5] / denom,
        "start_within_10bp": start_within[10] / denom,
        "start_within_20bp": start_within[20] / denom,
        "end_within_5bp": end_within[5] / denom,
        "end_within_10bp": end_within[10] / denom,
        "end_within_20bp": end_within[20] / denom,
        "interval_iou": total_iou / denom,
        "boundary_fallback_rate": fallback_count / denom,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--trimmed-dir", required=True)
    parser.add_argument("--output-dir", default="logs_ab1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--min-query-coverage", type=float, default=0.80)
    parser.add_argument(
        "--backbone",
        choices=["resnet50", "resnet101"],
        default="resnet50",
    )
    parser.add_argument("--boundary-radius", type=int, default=12)
    parser.add_argument("--boundary-weight", type=float, default=4.0)
    parser.add_argument(
        "--start-head-weight",
        type=float,
        default=0.3,
        help="Weight for normalized start-boundary CE loss.",
    )
    parser.add_argument(
        "--end-head-weight",
        type=float,
        default=0.7,
        help="Weight for normalized end-boundary CE loss.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="0 disables early stopping and runs all epochs.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    args = parser.parse_args()

    if args.boundary_radius < 0:
        raise ValueError("--boundary-radius must be >= 0")
    if args.boundary_weight < 1.0:
        raise ValueError("--boundary-weight must be >= 1.0")
    if args.start_head_weight < 0 or args.end_head_weight < 0:
        raise ValueError("boundary head weights must be >= 0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = discover_pairs(args.raw_dir, args.trimmed_dir)
    print(f"Found {len(pairs)} paired AB1 files.", flush=True)
    random.shuffle(pairs)

    if len(pairs) < 2:
        raise ValueError("At least 2 paired AB1 files are required.")

    val_size = max(1, int(round(len(pairs) * args.val_ratio)))
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]

    print(
        f"Preparing labels once: train={len(train_pairs)}, val={len(val_pairs)}",
        flush=True,
    )
    train_ds = AB1PairDataset(
        train_pairs,
        args.min_query_coverage,
        prepare_desc="Preparing train labels",
    )
    val_ds = AB1PairDataset(
        val_pairs,
        args.min_query_coverage,
        prepare_desc="Preparing val labels",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_ab1_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_ab1_batch,
    )

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training device: {device}", flush=True)
    print(f"Backbone: {args.backbone}", flush=True)
    print("Architecture: ECA + Dilated(1,2,4,8) + Attention U-Net + Boundary Head", flush=True)
    print(
        f"Loss weights: segmentation=1.0 start={args.start_head_weight} "
        f"end={args.end_head_weight}",
        flush=True,
    )

    model = build_model(args.backbone, input_channels=9).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {parameter_count:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_boundary_mae = float("inf")
    early_stop_reference = float("inf")
    epochs_without_improvement = 0
    best_iou = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        running_seg = 0.0
        running_start = 0.0
        running_end = 0.0
        seen = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{args.epochs:03d} train",
            unit="batch",
        )

        for batch in progress:
            x = batch["features"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)

            loss, seg_loss, start_loss, end_loss = compute_loss(
                outputs,
                batch,
                device,
                args.boundary_radius,
                args.boundary_weight,
                args.start_head_weight,
                args.end_head_weight,
            )

            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            running += float(loss.item()) * batch_size
            running_seg += float(seg_loss.item()) * batch_size
            running_start += float(start_loss.item()) * batch_size
            running_end += float(end_loss.item()) * batch_size
            seen += batch_size

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                seg=f"{seg_loss.item():.4f}",
                start=f"{start_loss.item():.3f}",
                end=f"{end_loss.item():.3f}",
            )

        scheduler.step()

        metrics = evaluate(
            model,
            val_loader,
            device,
            args.boundary_radius,
            args.boundary_weight,
            args.start_head_weight,
            args.end_head_weight,
            epoch=epoch,
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = running / max(seen, 1)
        metrics["train_seg_loss"] = running_seg / max(seen, 1)
        metrics["train_start_head_loss"] = running_start / max(seen, 1)
        metrics["train_end_head_loss"] = running_end / max(seen, 1)
        history.append(metrics)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={metrics['train_loss']:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"iou={metrics['interval_iou']:.4f} "
            f"start_mae={metrics['start_mae']:.2f} "
            f"end_mae={metrics['end_mae']:.2f} "
            f"end_bias={metrics['end_bias']:+.2f} "
            f"end<=10bp={metrics['end_within_10bp']:.1%} "
            f"end<=20bp={metrics['end_within_20bp']:.1%} "
            f"fallback={metrics['boundary_fallback_rate']:.1%}",
            flush=True,
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "input_channels": 9,
            "backbone": args.backbone,
            "epoch": epoch,
            "metrics": metrics,
            "architecture": "enhanced_resnet_unet_1d_v1",
            "training_config": vars(args),
        }
        torch.save(checkpoint, output_dir / "last.pth")

        if metrics["boundary_mae"] < best_boundary_mae:
            best_boundary_mae = metrics["boundary_mae"]
            torch.save(checkpoint, output_dir / "best.pth")
            torch.save(checkpoint, output_dir / "best_boundary.pth")

        if metrics["interval_iou"] > best_iou:
            best_iou = metrics["interval_iou"]
            torch.save(checkpoint, output_dir / "best_iou.pth")

        if (
            metrics["boundary_mae"]
            < early_stop_reference - args.early_stopping_min_delta
        ):
            early_stop_reference = metrics["boundary_mae"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}; "
                f"best boundary_mae={best_boundary_mae:.2f} bp.",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
