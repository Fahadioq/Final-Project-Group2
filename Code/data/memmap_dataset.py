"""Dataset over the chunked view store.

Reads a split file, joins it to the index for (shard, view_off), and serves uint8 [256, 256, 3] crops from
np.load(views/<view>/<pool>/<shard>.npy, mmap_mode='r') staged on NVMe. No image decoding in the loop.

Owner: Modeling. Not implemented yet -- see the interfaces in Code/common/schemas.py.
"""
from __future__ import annotations


def main() -> int:
    raise NotImplementedError("memmap_dataset.py is a placeholder for the Modeling role")


if __name__ == "__main__":
    raise SystemExit(main())
