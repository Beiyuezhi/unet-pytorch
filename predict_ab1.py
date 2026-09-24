import argparse

import torch

from nets.unet_1d import UNet1D
from train_ab1 import interval_from_probs
from utils.ab1_features import load_ab1_base_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ab1", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device)

    model = UNet1D(
        input_channels=checkpoint.get("input_channels", 9),
        base_channels=checkpoint.get("base_channels", 32),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    features, sequence = load_ab1_base_features(args.ab1)
    x = torch.from_numpy(features).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(x).squeeze(0).squeeze(0)).cpu()

    start, end = interval_from_probs(probs, args.threshold)

    print(f"bases={len(sequence)}")
    print(f"start={start}")
    print(f"end={end}")
    print(f"kept_bases={max(0, end - start)}")
    print(f"trimmed_sequence={sequence[start:end]}")


if __name__ == "__main__":
    main()
