import argparse
from pathlib import Path

import torch

from nets.unet_resnet_1d import ResNet50UNet1D, ResNet101UNet1D
from train_ab1 import interval_from_model_outputs
from utils.ab1_features import load_ab1_base_features
from utils.ab1_trim import write_trimmed_ab1


def build_model(backbone, input_channels=9):
    if backbone == "resnet50":
        return ResNet50UNet1D(input_channels=input_channels)
    if backbone == "resnet101":
        return ResNet101UNet1D(input_channels=input_channels)
    raise ValueError(f"Unsupported backbone in checkpoint: {backbone}")


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def discover_ab1_files(input_path: Path, excluded_dir: Path | None = None):
    """
    Accept either one AB1 file or a directory.

    Directory input is scanned recursively and case-insensitively for .ab1.
    An output directory located inside the input tree is excluded so rerunning
    prediction does not process already-generated trimmed files.
    """
    if input_path.is_file():
        if input_path.suffix.lower() != ".ab1":
            raise ValueError(f"Input file is not .ab1: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"AB1 input does not exist: {input_path}")

    files = []
    for path in input_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".ab1":
            continue
        if excluded_dir is not None and is_under(path, excluded_dir):
            continue
        files.append(path)

    return sorted(files, key=lambda p: str(p).lower())


def predict_one(model, device, ab1_path: Path):
    features, sequence = load_ab1_base_features(str(ab1_path))
    x = torch.from_numpy(features).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)

    seg_probs = torch.sigmoid(
        outputs["seg_logits"].squeeze(0).squeeze(0)
    ).cpu()
    boundary_logits = outputs["boundary_logits"].squeeze(0).cpu()

    start, end, used_fallback = interval_from_model_outputs(
        seg_probs,
        boundary_logits,
        len(sequence),
    )

    return {
        "sequence": sequence,
        "bases": len(sequence),
        "start": start,
        "end": end,
        "kept_bases": max(0, end - start),
        "boundary_fallback": used_fallback,
    }


def resolve_output_root(input_path: Path, custom_output_dir: str | None):
    if custom_output_dir:
        return Path(custom_output_dir)

    if input_path.is_dir():
        return input_path / "predicted_trimmed"

    return input_path.parent / "predicted_trimmed"


def output_path_for(
    source_path: Path,
    input_path: Path,
    output_root: Path,
):
    if input_path.is_dir():
        relative = source_path.relative_to(input_path)
        return output_root / relative

    return output_root / source_path.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ab1",
        required=True,
        help=(
            "Input .ab1 file or directory. Directory input is scanned "
            "recursively, including all subdirectories."
        ),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--save-trimmed-ab1",
        action="store_true",
        help="Write predicted intervals as new AB1 files. Disabled by default.",
    )
    parser.add_argument(
        "--trimmed-ab1-dir",
        default=None,
        help=(
            "Root directory for trimmed AB1 output. "
            "For directory input, relative subdirectory structure is preserved. "
            "Default: <input>/predicted_trimmed"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    args = parser.parse_args()

    input_path = Path(args.ab1)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model, map_location=device)
    backbone = checkpoint["backbone"]
    input_channels = checkpoint.get("input_channels", 9)

    model = build_model(backbone, input_channels=input_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    output_root = (
        resolve_output_root(input_path, args.trimmed_ab1_dir)
        if args.save_trimmed_ab1
        else None
    )

    excluded_dir = None
    if (
        input_path.is_dir()
        and output_root is not None
        and is_under(output_root, input_path)
    ):
        excluded_dir = output_root

    ab1_files = discover_ab1_files(
        input_path,
        excluded_dir=excluded_dir,
    )

    if not ab1_files:
        raise FileNotFoundError(
            f"No .ab1 files found in input: {input_path}"
        )

    batch_mode = input_path.is_dir()
    print(f"backbone={backbone}")
    print(
        f"architecture={checkpoint.get('architecture', 'enhanced_resnet_unet_1d_v1')}"
    )
    print(f"device={device}")
    print(f"input_mode={'directory' if batch_mode else 'file'}")
    print(f"ab1_files={len(ab1_files)}")

    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        print(f"trimmed_ab1_root={output_root}")

    success = 0
    failed = 0

    for index, source_path in enumerate(ab1_files, start=1):
        try:
            result = predict_one(model, device, source_path)

            print(
                f"[{index}/{len(ab1_files)}] "
                f"file={source_path} "
                f"bases={result['bases']} "
                f"start={result['start']} "
                f"end={result['end']} "
                f"kept_bases={result['kept_bases']} "
                f"boundary_fallback={result['boundary_fallback']}"
            )

            if not batch_mode:
                print(
                    f"trimmed_sequence="
                    f"{result['sequence'][result['start']:result['end']]}"
                )

            if output_root is not None:
                output_path = output_path_for(
                    source_path,
                    input_path,
                    output_root,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

                written_path = write_trimmed_ab1(
                    source_path=str(source_path),
                    output_path=str(output_path),
                    start=result["start"],
                    end=result["end"],
                )
                print(f"trimmed_ab1={written_path}")

            success += 1

        except Exception as exc:
            failed += 1
            print(
                f"[{index}/{len(ab1_files)}] "
                f"file={source_path} ERROR: {exc}"
            )

    print(
        f"summary total={len(ab1_files)} "
        f"success={success} failed={failed}"
    )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
