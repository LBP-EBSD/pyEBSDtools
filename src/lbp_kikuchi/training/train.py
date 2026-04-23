import argparse
import torch
import torch.nn as nn
from lbp_kikuchi.models.pair_model import PairModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    print(f"Running with config: {args.config}")

    model = PairModel()
    x1 = torch.randn(4, 1, 64, 64)
    x2 = torch.randn(4, 1, 64, 64)

    out = model(x1, x2)
    print("Output shape:", out.shape)

if __name__ == "__main__":
    main()