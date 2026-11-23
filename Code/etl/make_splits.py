"""Build split files for E0-E6 after the census gate.

Writes ~/of_data/splits/<experiment>/<variant>.parquet in common.schemas.SPLIT_SCHEMA: trims buckets
to their caps by hash rank, groups dev by prompt_hash, assigns attribution classes and outlier-exposure
unknowns inside each run's allowed lineages, and sets bce_mask=False for auxiliary reals.

Census-gate decisions live in configs/splits.yaml: training families, test-only lineages (other_dit), and
test-only taxonomy styles (anime). Test-only strings never enter a training pool, dev, or outlier exposure.
`extra_runs` there adds runs that override those defaults (e.g. E1 sdxl_anime, the style ablation).

Read the index with etl.census.load_index(pool, columns): it takes `lineage` from the current taxonomy CSV,
not the value stored at ETL time (56 kling-video rows from Pass A are stored as 'unmapped').

Owner: Data. Not implemented yet -- see the interfaces in Code/common/schemas.py.
"""
from __future__ import annotations


def main() -> int:
    raise NotImplementedError("make_splits.py is a placeholder for the Data role")


if __name__ == "__main__":
    raise SystemExit(main())
