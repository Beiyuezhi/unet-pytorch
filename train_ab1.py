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


def masked_bce_with_logits(logits, targets, valid_mask):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = loss * valid_mask
    return loss.sum() / valid_mask.sum().clamp_min(1.0)


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


def evaluate(model, loader, device, epoch=None):
    model.eval()
    total_loss = 0.0
    total_start_mae = 0.0
    total_end_mae = 0.0
    total_iou = 0.0
    count = 0

    desc = f"Epoch {epoch:03d} val" if epoch is not None else "Validation"
    progress = tqdm(loader, desc=desc, unit="batch", leave=False)

    with torch.no_grad():
        for batch in progress:
            x = batch["features"].to(device)
            y = batch["target"].to(device)
            valid = batch["valid_mask"].to(device)

            logits = model(x).squeeze(1)
            bce = masked_bce_with_logits(logits, y, valid)
            dice = masked_dice_loss(logits, y, valid)
            loss = 0.5 * bce + 0.5 * dice
            total_loss += float(loss.item()) * x.size(0)

            probs = torch.sigmoid(logits).cpu()
            for i in range(x.size(0)):
                length = int(batch["lengths"][i])
                pred_start, pred_end = interval_from_probs(probs[i, :length])
                true_start = int(batch["starts"][i])
                true_end = int(batch["ends"][i])

                total_start_mae += abs(pred_start - true_start)
                total_end_mae += abs(pred_end - true_end)

                inter = max(0, min(pred_end, true_end) - max(pred_start, true_start))
                union = max(pred_end, true_end) - min(pred_start, true_start)
                total_iou += inter / union if union > 0 else 0.0
                count += 1

            progress.set_postfix(val_loss=f"{loss.item():.4f}")

    return {
        "loss": total_loss / max(count, 1),
        "start_mae": total_start_mae / max(count, 1),
        "end_mae": total_end_mae / max(count, 1),
        "interval_iou": total_iou / max(count, 1),
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
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. Use --device cpu to force CPU training.",
    )
    args = parser.parse_args()

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

    model = UNet1D(input_channels=9, base_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_iou = -1.0
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

            bce = masked_bce_with_logits(logits, y, valid)
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
        metrics = evaluate(model, val_loader, device, epoch=epoch)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running / max(seen, 1)
        history.append(metrics)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={metrics['train_loss']:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"iou={metrics['interval_iou']:.4f} "
            f"start_mae={metrics['start_mae']:.2f} "
            f"end_mae={metrics['end_mae']:.2f}",
            flush=True,
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "input_channels": 9,
            "base_channels": 32,
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(checkpoint, output_dir / "last.pth")

        if metrics["interval_iou"] > best_iou:
            best_iou = metrics["interval_iou"]
            torch.save(checkpoint, output_dir / "best.pth")

        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
