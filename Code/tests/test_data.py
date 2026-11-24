"""Self-contained checks for the Modeling data path (no pytest needed, no GPU needed).

    cd Code && python -m tests.test_data

Runs on the synthetic store from data/standin.py, so it works on a laptop before any real split exists.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from data.gpu_aug import GpuAugment, center_crop, jpeg, to_device
from data.memmap_dataset import ViewDataset, make_loader, read_split
from data.standin import decode_id, make_standin_store


def _store(tmp: Path) -> Path:
    return make_standin_store(tmp, n_shards=3, rows_per_shard=120, seed=1)


def test_standin_matches_schemas(tmp: Path):
    split_path = _store(tmp)
    split = read_split(split_path)
    assert len(split) > 0 and set(split["role"]) == {"train", "dev"}


def test_items_point_to_the_right_image(tmp: Path):
    """Each image's top-left pixel holds its position in the split file: the join must preserve it."""
    split_path = _store(tmp)
    expected = {rid: k for k, rid in enumerate(pq.read_table(split_path).column("row_id").to_pylist())}
    for view in ("C", "R", "N"):
        ds = ViewDataset(split_path, ["train", "dev"], view=view, data_root=tmp, views_root=tmp)
        for i in range(len(ds)):
            image, label, attr, mask, idx = ds[i]
            assert image.shape == (256, 256, 3) and image.dtype == torch.uint8
            assert decode_id(image.numpy()) == expected[ds.rows["row_id"].iloc[idx]], (view, i)
            assert label == float(ds.rows["label"].iloc[idx] == "fake")
            assert ds.classes[attr] == ds.rows["attr_class"].iloc[idx] and mask


def test_roles_are_separate(tmp: Path):
    split_path = _store(tmp)
    train = ViewDataset(split_path, ["train"], data_root=tmp, views_root=tmp)
    dev = ViewDataset(split_path, ["dev"], data_root=tmp, views_root=tmp)
    assert not set(train.rows["row_id"]) & set(dev.rows["row_id"])
    assert train.classes == dev.classes                      # same class order for every role


def test_loader_batches(tmp: Path):
    split_path = _store(tmp)
    ds = ViewDataset(split_path, ["train"], data_root=tmp, views_root=tmp)
    for workers in (0, 2):
        images, labels, attr, mask, idx = next(iter(make_loader(ds, batch_size=16, num_workers=workers)))
        assert images.shape == (16, 256, 256, 3) and labels.shape == (16,) and attr.dtype == torch.int64


def test_augment_shapes_and_repeatability():
    images = torch.randint(0, 256, (6, 256, 256, 3), dtype=torch.uint8)
    a = GpuAugment(device="cpu", seed=3).train(images)
    b = GpuAugment(device="cpu", seed=3).train(images)
    c = GpuAugment(device="cpu", seed=4).train(images)
    assert a.shape == (6, 3, 224, 224) and a.dtype == torch.float32
    assert torch.equal(a, b) and not torch.equal(a, c)       # same seed, same batch; new seed, new batch


def test_eval_center_crop_is_exact():
    images = torch.randint(0, 256, (2, 256, 256, 3), dtype=torch.uint8)
    x = GpuAugment(device="cpu", channels_last=False).eval(images)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    pixels = torch.round((x * std + mean) * 255).to(torch.uint8)
    assert torch.equal(pixels, center_crop(to_device(images, "cpu"), 224))


def test_jpeg_mask_only_touches_selected_images():
    x = torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8)
    mask = torch.tensor([True, False, True, False])
    out = jpeg(x, torch.full((4,), 50), mask)
    assert torch.equal(out[1], x[1]) and torch.equal(out[3], x[3])
    assert not torch.equal(out[0], x[0]) and not torch.equal(out[2], x[2])


def test_lower_jpeg_quality_changes_more():
    x = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    err = {q: (jpeg(x, torch.full((2,), q)).float() - x.float()).abs().mean().item() for q in (95, 50)}
    assert err[50] > err[95] > 0


def main() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        if fn.__code__.co_argcount:
            with tempfile.TemporaryDirectory() as tmp:
                fn(Path(tmp))
        else:
            fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
