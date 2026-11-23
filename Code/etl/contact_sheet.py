"""Contact sheet of stored views for one model string or lineage (Phase 0 checks). Owner: Data.

    cd Code && python -m etl.contact_sheet --model tiny-random-sana --out ~/of_data/logs/sana.png
    cd Code && python -m etl.contact_sheet --lineage unmapped --view R --out ~/of_data/logs/unmapped.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
from PIL import Image

from common import paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="core_train")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model")
    g.add_argument("--lineage")
    ap.add_argument("--view", default="C", choices=["C", "R", "N"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--thumb", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    key, val = ("model", args.model) if args.model else ("lineage", args.lineage)
    off_col = "n_off" if args.view == "N" else "view_off"
    t = ds.dataset(paths.index_dir(args.pool), format="parquet").to_table(
        columns=["row_id", "model", "shard", "hash_u", off_col],
        filter=(pc.field(key) == val) & pc.field("kept") & (pc.field(off_col) >= 0)).to_pandas()
    if t.empty:
        print(f"no kept rows with {key}={val!r} in view {args.view}")
        return 1
    t = t.sort_values("hash_u").head(args.n)
    cols = 6
    rows = (len(t) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * args.thumb, rows * args.thumb), "white")
    for i, r in enumerate(t.itertuples()):
        arr = np.load(paths.view_path(args.view, args.pool, r.shard), mmap_mode="r")[getattr(r, off_col)]
        sheet.paste(Image.fromarray(np.asarray(arr)).resize((args.thumb, args.thumb)),
                    ((i % cols) * args.thumb, (i // cols) * args.thumb))
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"{len(t)} images ({key}={val}, view {args.view}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
