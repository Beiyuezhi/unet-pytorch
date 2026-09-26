import argparse

import torch

from nets.unet_resnet_1d import ResNet50UNet1D, ResNet101UNet1D
from train_ab1 import interval_from_model_outputs
from utils.ab1_features import load_ab1_base_features


def build_model(backbone, input_channels=9):
    if backbone == "resnet50":
        return ResNet50UNet1D(input_channels=input_channels)
    if backbone == "resnet101":
        return ResNet101UNet1D(input_channels=input_channels)
    raise ValueError(f"Unsupported backbone in checkpoint: {backbone}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ab1", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    args = parser.parse_args()

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

    features, sequence = load_ab1_base_features(args.ab1)
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

    print(f"backbone={backbone}")
    print(f"architecture={checkpoint.get('architecture', 'enhanced_resnet_unet_1d_v1')}")
    print(f"bases={len(sequence)}")
    print(f"start={start}")
    print(f"end={end}")
    print(f"kept_bases={max(0, end - start)}")
    print(f"boundary_fallback={used_fallback}")
    print(f"trimmed_sequence={sequence[start:end]}")


if __name__ == "__main__":
    main()
