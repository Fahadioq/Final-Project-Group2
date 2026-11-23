"""Census-gate tables from the index. Owner: Data.

    cd Code && python -m etl.census --pool core_train

Writes (small, committed):
  results/census_<pool>.csv          rows / eligible / kept per (label, lineage, model)
  results/header_summary_<pool>.csv  format and resolution statistics per (label, lineage)
"""
from __future__ import annotations

import argparse

import pyarrow.dataset as ds

from common import paths
from etl.selection import bucket_of, load_taxonomy

COLUMNS = ["row_id", "label", "lineage", "model", "eligible", "kept", "header_ok", "fmt", "width", "height",
           "jpeg_quality", "has_exif", "skip_reason"]


def load_index(pool: str, columns: list[str] = COLUMNS, remap: bool = True):
    """Read a pool's index. With `remap`, `lineage` comes from the current taxonomy CSV rather than the value
    stored at ETL time, so strings mapped after a shard was processed (e.g. kling-video in Pass A) are counted
    correctly. Every reader of the index (census, make_splits) should use this."""
    df = ds.dataset(paths.index_dir(pool), format="parquet").to_table(columns=columns).to_pandas()
    if remap and {"label", "model", "lineage"} <= set(df.columns):
        tax = load_taxonomy(paths.TAXONOMY_CSV)
        pairs = df[["label", "model"]].drop_duplicates()
        lut = {(l, m): bucket_of(l, m, tax)[0] for l, m in pairs.itertuples(index=False)}
        new = [lut[k] for k in zip(df["label"], df["model"])]
        changed = int((df["lineage"] != new).sum())
        if changed:
            print(f"lineage remapped from the current taxonomy for {changed:,} rows")
        df["lineage"] = new
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="core_train")
    args = ap.parse_args(argv)

    df = load_index(args.pool)
    census = (df.groupby(["label", "lineage", "model"], dropna=False)
                .agg(rows=("row_id", "size"), eligible=("eligible", "sum"), kept=("kept", "sum"))
                .reset_index().sort_values(["label", "lineage", "rows"], ascending=[True, True, False]))

    h = df[df["header_ok"]].copy()
    h["mp"] = h["width"] * h["height"] / 1e6
    h["div64"] = (h["width"] % 64 == 0) & (h["height"] % 64 == 0)
    summary = (h.groupby(["label", "lineage"])
                .agg(rows=("row_id", "size"),
                     frac_jpeg=("fmt", lambda s: (s == "JPEG").mean()),
                     frac_png=("fmt", lambda s: (s == "PNG").mean()),
                     frac_webp=("fmt", lambda s: (s == "WEBP").mean()),
                     mp_p10=("mp", lambda s: s.quantile(0.1)),
                     mp_median=("mp", "median"),
                     mp_p90=("mp", lambda s: s.quantile(0.9)),
                     frac_div64=("div64", "mean"),
                     jpeg_q_median=("jpeg_quality", "median"),
                     frac_exif=("has_exif", "mean"))
                .reset_index().round(4))

    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    c_path = paths.RESULTS_DIR / f"census_{args.pool}.csv"
    s_path = paths.RESULTS_DIR / f"header_summary_{args.pool}.csv"
    census.to_csv(c_path, index=False)
    summary.to_csv(s_path, index=False)
    n_chunks = len(list(paths.index_dir(args.pool).glob("*.parquet")))
    print(f"{args.pool}: {len(df):,} rows from {n_chunks} shards; kept {int(df['kept'].sum()):,}")
    print(census.groupby(["label", "lineage"])[["rows", "eligible", "kept"]].sum().to_string())
    print("skip reasons:", df["skip_reason"].replace("", None).dropna().str.split(":").str[0].value_counts().to_dict())
    print(f"wrote {c_path} and {s_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
