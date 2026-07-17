from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the minimum, maximum, and complete pixel array of a depth PNG."
    )
    parser.add_argument("depth_png", type=Path, help="Path to a depth PNG file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    depth_image = cv2.imread(str(args.depth_png), cv2.IMREAD_UNCHANGED)

    if depth_image is None:
        raise SystemExit(f"Could not read depth PNG: {args.depth_png}")
    if depth_image.ndim != 2:
        raise SystemExit(
            f"Expected a single-channel depth PNG, got shape {depth_image.shape}"
        )
    if depth_image.size == 0:
        raise SystemExit(f"Depth PNG is empty: {args.depth_png}")

    print(f"Min depth: {int(depth_image.min())}")
    print(f"Max depth: {int(depth_image.max())}")
    print("Depth values:")
    with np.printoptions(threshold=np.inf, linewidth=200):
        print(depth_image)


if __name__ == "__main__":
    main()
