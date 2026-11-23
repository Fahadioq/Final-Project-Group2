"""Self-contained checks for the ETL building blocks (no pytest needed).

    cd Code && python -m tests.test_etl
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image

from common import paths, schemas
from etl import views as V
from etl.header_feats import header_features
from etl.selection import DEFAULT_KEY, Selector, Targets, bucket_of, row_hash, waterfill
from etl.shard import process_shard

RNG = np.random.default_rng(0)


def _img(w: int, h: int, mode: str = "RGB") -> Image.Image:
    arr = RNG.integers(0, 256, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    im = Image.fromarray(arr).resize((w, h), Image.BILINEAR)
    return im.convert(mode)


def _encode(im: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format=fmt, **kw)
    return buf.getvalue()


def test_jpeg_quality_estimate():
    for q in (50, 75, 90, 95):
        hf = header_features(_encode(_img(320, 240), "JPEG", quality=q))
        assert hf["fmt"] == "JPEG" and abs(hf["jpeg_quality"] - q) <= 2, (q, hf["jpeg_quality"])
    assert header_features(_encode(_img(320, 240), "JPEG", quality=90, subsampling=0))["jpeg_subsampling"] == "444"


def test_header_png_and_garbage():
    hf = header_features(_encode(_img(300, 200), "PNG"))
    assert hf["fmt"] == "PNG" and hf["jpeg_quality"] is None and (hf["width"], hf["height"]) == (300, 200)
    assert header_features(b"not an image")["header_ok"] is False
    assert header_features(None)["header_ok"] is False


def test_views_and_eligibility():
    big = _img(3000, 2000)                       # 6 MP -> capped to ~1 MP for C, N stored
    assert V.needs_native(*big.size)
    assert abs(np.prod(V.capped_size(*big.size)) / 1e6 - 1.0) < 0.01
    for f in (V.view_c, V.view_r, V.view_n):
        out = f(big)
        assert out.shape == (256, 256, 3) and out.dtype == np.uint8
    small = _img(1024, 1024)
    assert not V.needs_native(*small.size)
    assert np.array_equal(V.view_c(small), V.view_n(small))   # <= 2 MP: C is native pixels
    assert V.eligibility(dict(header_ok=True, width=300, height=200))[1] == "too_small"
    assert V.eligibility(dict(header_ok=True, width=20000, height=300))[1] == "too_small_after_cap"
    assert V.eligibility(dict(header_ok=True, width=512, height=512))[0]
    assert V.view_c_scaled(small, 0.5).shape == (256, 256, 3)


def test_selection_math():
    alloc = waterfill({"a": 100, "b": 5000, "c": 9000}, total_cap=3000, item_cap=2500)
    assert abs(sum(alloc.values()) - 3000) < 1 and alloc["a"] == 100 and alloc["b"] <= 2500
    assert waterfill({"a": 10, "b": 20}, 1000, 15) == {"a": 10, "b": 15}
    u = row_hash("row-1")
    assert u == row_hash("row-1") and 0 <= u < 1
    tax = {"sd-1.5": "sd12"}
    assert bucket_of("real", "laion", tax) == ("real", "real:laion")
    assert bucket_of("fake", "sd-1.5", tax) == ("sd12", "fake:sd-1.5")
    assert bucket_of("fake", "brand-new", tax) == ("unmapped", "fake:brand-new")

    t = Targets(lineage_cap=1000, string_cap=600, real_caps={"laion": 500}, real_default_cap=100,
                unmapped_cap=10, exclude_cap=5)
    s = Selector(t)
    assert s.thresholds(10) == {DEFAULT_KEY: 1.0}            # warm-up keeps everything
    s.update(dict(eligible={"fake:a": 1000, "fake:b": 50, "real:laion": 400}, kept={"fake:a": 1000, "fake:b": 50,
                  "real:laion": 400}, lineage={"fake:a": "sd12", "fake:b": "sd12", "real:laion": "real"}))
    th = s.thresholds(9)
    assert th["fake:a"] == 0.0                                # already above its string cap
    assert th["fake:b"] == 1.0                                # rare string: keep everything
    assert th["real:laion"] == 0.0 or th["real:laion"] < 0.05
    assert all(0.0 <= v <= 1.0 for v in th.values())

    # a lineage with a single model string must be able to reach the lineage cap, not the string cap
    s1 = Selector(t)
    s1.update(dict(eligible={"fake:solo": 500}, kept={"fake:solo": 500}, lineage={"fake:solo": "sd35"}))
    tgt, _ = s1.targets(horizon=10)
    assert tgt["fake:solo"] == 1000, tgt          # lineage_cap, not string_cap (600)
    assert abs(s1.thresholds(9)["fake:solo"] - 500 / 4500) < 1e-9


def test_splits_config_consistent():
    """configs/splits.yaml names only lineages and styles the taxonomy defines, and its lists don't overlap."""
    import csv
    cfg = yaml.safe_load(open(paths.CONFIG_DIR / "splits.yaml"))
    rows = list(csv.DictReader(open(paths.TAXONOMY_CSV, newline="")))
    assert all(None not in r and None not in r.values() for r in rows)   # every row has every column
    lineages = {r["lineage"] for r in rows}
    groups = [set(cfg[k]) for k in ("train_lineages", "test_only_lineages", "never_use_lineages")]
    for g in groups:
        assert g <= lineages, g - lineages
    assert sum(map(len, groups)) == len(set().union(*groups)), "a lineage is listed twice"
    styles = {r["style"] for r in rows if r["style"]}
    assert set(cfg["test_only_styles"]) <= styles
    for name, run in cfg.get("extra_runs", {}).items():
        assert set(run["train_lineages"]) <= groups[0], name           # only training families
        assert set(run.get("test_only_styles", [])) <= styles, name


def _synthetic_shard(path: Path) -> None:
    rows = [  # (label, model, bytes)
        ("real", "laion", _encode(_img(3000, 2000), "JPEG", quality=80)),     # > 2 MP: C, R, N
        ("fake", "sd-1.5", _encode(_img(1024, 1024), "PNG")),
        ("real", "laion", _encode(_img(200, 150), "JPEG", quality=75)),       # too small
        ("fake", "sd-1.5", b"\x89PNG broken"),                                # header error
        ("fake", "flux.1-dev", _encode(_img(768, 512), "WEBP", quality=90)),
        ("fake", "flux.1-dev", _encode(_img(512, 512, "RGBA"), "PNG")),
        ("real", "pexels", _encode(_img(600, 400, "L"), "JPEG", quality=95)),
    ]
    image_t = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table({
        "image": pa.array([{"bytes": b, "path": None} for _, _, b in rows], type=image_t),
        "prompt": [f"prompt {i}" for i in range(len(rows))],
        "label": [r[0] for r in rows], "model": [r[1] for r in rows], "type": ["base"] * len(rows),
        "release_date": ["2024-01"] * len(rows), "row_id": [f"r{i}" for i in range(len(rows))],
        "image_sha256": [f"h{i}" for i in range(len(rows))],
    })
    pq.write_table(table, path, row_group_size=3)


def test_process_shard_end_to_end():
    cfg = yaml.safe_load(open(paths.CONFIG_DIR / "etl_core_train.yaml"))
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "train-99999-of-00032-00000.parquet"
        _synthetic_shard(src)
        stats = process_shard(dict(path=str(src), shard="core/" + src.name, pool="test_pool",
                                   thresholds={DEFAULT_KEY: 1.0}, taxonomy={"sd-1.5": "sd12", "flux.1-dev": "flux1"},
                                   cfg=cfg, of_data=tmp))
        assert stats["n_rows"] == 7 and stats["n_kept"] == 5 and stats["n_native"] == 1, stats
        assert stats["skips"] == {"too_small": 1, "header_error": 1}, stats["skips"]
        idx = pq.read_table(paths.index_path("test_pool", src.stem, tmp))
        schemas.validate(idx, schemas.INDEX_SCHEMA)
        d = idx.to_pandas()
        assert sorted(d.loc[d.kept, "view_off"]) == [0, 1, 2, 3, 4]
        assert (d.loc[~d.kept, "view_off"] == -1).all()
        c = np.load(paths.view_path("C", "test_pool", src.stem, tmp), mmap_mode="r")
        r = np.load(paths.view_path("R", "test_pool", src.stem, tmp), mmap_mode="r")
        n = np.load(paths.view_path("N", "test_pool", src.stem, tmp), mmap_mode="r")
        assert c.shape == r.shape == (5, 256, 256, 3) and n.shape == (1, 256, 256, 3)
        assert not list(Path(tmp).rglob("*.tmp*"))
        del c, r, n

        # resume check: complete outputs pass; a truncated array or a missing index fails
        from etl.stream import outputs_ok
        import etl.stream as stream_mod
        old_pool, stream_mod.POOL = stream_mod.POOL, "test_pool"
        try:
            entry = {"stats": stats}
            assert outputs_ok(entry, tmp)
            cpath = paths.view_path("C", "test_pool", src.stem, tmp)
            data = cpath.read_bytes()
            cpath.write_bytes(data[: len(data) // 2])
            assert not outputs_ok(entry, tmp)
            cpath.write_bytes(data)
            assert outputs_ok(entry, tmp)
            paths.index_path("test_pool", src.stem, tmp).unlink()
            assert not outputs_ok(entry, tmp)
        finally:
            stream_mod.POOL = old_pool


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
