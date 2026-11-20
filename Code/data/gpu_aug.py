"""GPU augmentation: random 224 crops, flips, and format equalization.

Format equalization = JPEG at random Q in [65, 95] applied to both classes on the GPU
(torchvision.io.encode_jpeg / decode_jpeg on CUDA), with random crop offsets so the 8x8 grid phase
carries no cue. Also the test-time JPEG sweep for E4 and the format-flip test for E3.

Owner: Modeling. Not implemented yet -- see the interfaces in Code/common/schemas.py.
"""
from __future__ import annotations


def main() -> int:
    raise NotImplementedError("gpu_aug.py is a placeholder for the Modeling role")


if __name__ == "__main__":
    raise SystemExit(main())
