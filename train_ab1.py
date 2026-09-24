import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.ab1_dataset import AB1PairDataset, collate_ab1_batch, discover_pairs
from nets.unet_1d import UNet1D
from nets.unet_resnet_1d import ResNet50UNet1D, ResNet101UNet1D


def build_model(backbone: str, input_channels: int = 9):
    if backbone == "plain":
        return UNet1D(input_channels=input_channels, base_channels=32)
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


def evaluate(
    model,
    loader,
    device,
    boundary_radius,
    boundary_weight,
    epoch=None,
):
    model.eval()
    total_loss = 0.0
    total_start_abs = 0.0
    total_end_abs = 0.0
    total_start_signed = 0.0
    total_end_signed = 0.0
    total_iou = 0.0

    start_within = {5: 0, 10: 0, 20: 0}
    end_within = {5: 0, 10: 0, 20: 0}
    count = 0

    desc = f"Epoch {epoch:03d} val" if epoch is not None else "Validation"
    progress = tqdm(loader, desc=desc, unit="batch", leave=False)

    with torch.no_grad():
        for batch in progress:
            x = batch["features"].to(device)
            y = batch["target"].to(device)
            valid = batch["valid_mask"].to(device)

            logits = model(x).squeeze(1)
            bce = boundary_weighted_bce_with_logits(
                logits,
                y,
                valid,
                batch["starts"],
                batch["ends"],
                boundary_radius=boundary_radius,
                boundary_weight=boundary_weight,
            )
            dice = masked_dice_loss(logits, y, valid)
            loss = 0.5 * bce + 0.5 * dice
            total_loss += float(loss.item()) * x.size(0)

            probs = torch.sigmoid(logits).cpu()
            for i in range(x.size(0)):
                length = int(batch["lengths"][i])
                pred_start, pred_end = interval_from_probs(probs[i, :length])
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
        choices=["plain", "resnet50", "resnet101"],
        default="plain",
        help="plain = original 1D U-Net, resnet50/resnet101 = 1D ResNet encoder + U-Net decoder.",
    )
    parser.add_argument(
        "--boundary-radius",
        type=int,
        default=12,
        help="Number of bases on each side of start/end receiving extra BCE weight.",
    )
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=4.0,
        help="BCE weight multiplier around the true start/end boundaries.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=12,
        help="Stop after this many epochs without boundary-MAE improvement. 0 disables.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.05,
        help="Minimum boundary-MAE improvement in bp required to reset patience.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. Use --device cpu to force CPU training.",
    )
    args = parser.parse_args()

    if args.boundary_radius < 0:
        raise ValueError("--boundary-radius must be >= 0")
    if args.boundary_weight < 1.0:
        raise ValueError("--boundary-weight must be >= 1.0")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be >= 0")

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
    if not train_pairs:
        train_pairs = pairs[:-1]
        val_pairs = pairs[-1:]

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
    print(
        f"Boundary weighting: radius={args.boundary_radius} bp, "
        f"weight={args.boundary_weight:.1f}x",
        flush=True,
    )
    if args.early_stopping_patience > 0:
        print(
            f"Early stopping: patience={args.early_stopping_patience}, "
            f"min_delta={args.early_stopping_min_delta:.2f} bp "
            f"(monitors boundary_mae)",
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
    best_iou = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{args.epochs:03d} train",
            unit="batch",
        )

        for batch in progress:
            x = batch["features"].to(device)
            y = batch["target"].to(device)
            valid = batch["valid_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x).squeeze(1)

            bce = boundary_weighted_bce_with_logits(
                logits,
                y,
                valid,
                batch["starts"],
                batch["ends"],
                boundary_radius=args.boundary_radius,
                boundary_weight=args.boundary_weight,
            )
            dice = masked_dice_loss(logits, y, valid)
            loss = 0.5 * bce + 0.5 * dice

            loss.backward()
            optimizer.step()

            running += float(loss.item()) * x.size(0)
            seen += x.size(0)
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()
        metrics = evaluate(
            model,
            val_loader,
            device,
            boundary_radius=args.boundary_radius,
            boundary_weight=args.boundary_weight,
            epoch=epoch,
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = running / max(seen, 1)
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
            f"end<=20bp={metrics['end_within_20bp']:.1%}",
            flush=True,
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "input_channels": 9,
            "base_channels": 32 if args.backbone == "plain" else None,
            "backbone": args.backbone,
            "epoch": epoch,
            "metrics": metrics,
            "training_config": {
                "backbone": args.backbone,
                "boundary_radius": args.boundary_radius,
                "boundary_weight": args.boundary_weight,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
            },
        }
        torch.save(checkpoint, output_dir / "last.pth")

        boundary_improved = (
            metrics["boundary_mae"]
            < best_boundary_mae - args.early_stopping_min_delta
        )
        if boundary_improved:
            best_boundary_mae = metrics["boundary_mae"]
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best.pth")
            torch.save(checkpoint, output_dir / "best_boundary.pth")
        else:
            epochs_without_improvement += 1

        if metrics["interval_iou"] > best_iou:
            best_iou = metrics["interval_iou"]
            torch.save(checkpoint, output_dir / "best_iou.pth")

        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}: boundary_mae has not improved "
                f"by at least {args.early_stopping_min_delta:.2f} bp for "
                f"{args.early_stopping_patience} epochs. "
                f"Best boundary_mae={best_boundary_mae:.2f} bp.",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
