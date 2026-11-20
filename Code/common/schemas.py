"""Data contracts between the Data, Modeling, and Analysis & App roles.

Change a schema only after telling the other two roles. Producers call `validate()` before
writing; consumers may rely on every column listed here existing with exactly this type.
"""
from __future__ import annotations

import pyarrow as pa

VIEW_SIZE = 256
VIEW_NAMES = ("C", "R", "N")          # C: density-capped (main), R: resized, N: native (images > 2 MP only)
ROBUSTNESS_VIEWS = ("C075", "C050")   # evaluation pools only (E4)

# ---------------------------------------------------------------------------------------------
# Index: one row per streamed image, for every pool (core_train, core_validation, ...).
# View store layout (chunk = source shard stem, e.g. "train-00000-of-00032-00000"):
#   views/C/<pool>/<chunk>.npy   row `view_off`   } C and R share the same row order
#   views/R/<pool>/<chunk>.npy   row `view_off`   }
#   views/N/<pool>/<chunk>.npy   row `n_off`      (-1 when the image is <= 2 MP, so N == C)
# `view_off` is -1 for rows that were not kept.
# ---------------------------------------------------------------------------------------------
INDEX_SCHEMA = pa.schema([
    ("row_id", pa.string()),
    ("image_sha256", pa.string()),
    ("pool", pa.string()),
    ("shard", pa.string()),
    ("row_group", pa.int32()),
    ("row_in_group", pa.int32()),
    ("label", pa.string()),            # "real" | "fake"
    ("model", pa.string()),            # raw OpenFake string; canonical name lives in the taxonomy
    ("type", pa.string()),
    ("release_date", pa.string()),
    ("prompt_hash", pa.string()),      # for grouping the dev split; prompts are never model input
    ("lineage", pa.string()),          # taxonomy lineage id at ETL time (draft taxonomy)
    ("bucket", pa.string()),           # selection bucket: "fake:<model>" | "real:<source>"
    # header features (no decoding) -- the metadata-only shortcut baseline uses these
    ("fmt", pa.string()),
    ("width", pa.int32()),
    ("height", pa.int32()),
    ("mode", pa.string()),
    ("file_bytes", pa.int64()),
    ("jpeg_quality", pa.int16()),      # IJG-equivalent estimate from the luminance table; null if not JPEG
    ("jpeg_subsampling", pa.string()), # "444" | "422" | "420" | null
    ("progressive", pa.bool_()),
    ("has_exif", pa.bool_()),
    ("has_icc", pa.bool_()),
    ("n_frames", pa.int32()),
    ("header_ok", pa.bool_()),
    # selection
    ("eligible", pa.bool_()),
    ("skip_reason", pa.string()),
    ("hash_u", pa.float64()),          # deterministic U[0,1) from row_id
    ("p_keep", pa.float32()),          # threshold in force for this row's bucket when the shard ran
    ("kept", pa.bool_()),
    ("decode_ok", pa.bool_()),
    ("view_off", pa.int32()),
    ("n_off", pa.int32()),
])

# Split files: ~/of_data/splits/<experiment>/<variant>.parquet  (written by Data, read by Modeling)
SPLIT_SCHEMA = pa.schema([
    ("row_id", pa.string()),
    ("pool", pa.string()),
    ("role", pa.string()),             # "train" | "dev" | "test:<set>"
    ("attr_class", pa.string()),       # attribution-head class, incl. "unknown-generator" / "unknown-real"
    ("bce_mask", pa.bool_()),          # False for auxiliary reals that train only the attribution head
])

# Run directories: ~/of_data/runs/<run_id>/  (written by Modeling, read by Analysis & App)
RUN_FILES = ("config.yaml", "metrics.json", "preds.parquet", "model.pt", "model_card.json")
MODEL_CARD_KEYS = ("run_id", "backbone", "classes", "tau", "view", "eval_crop")

PREDS_SCHEMA = pa.schema([
    ("row_id", pa.string()),
    ("set", pa.string()),              # evaluation set name, e.g. "core_test", "reddit_test"
    ("label", pa.int8()),              # 1 = fake
    ("p_fake", pa.float32()),
    ("attr_probs", pa.list_(pa.float32())),  # order = model_card.json["classes"]
    ("p_unknown_gen", pa.float32()),
    ("p_unknown_real", pa.float32()),
])


def validate(table: pa.Table, schema: pa.Schema, *, allow_extra: bool = False) -> None:
    """Raise if `table` does not match `schema` (names and types)."""
    names = table.schema.names
    missing = [f.name for f in schema if f.name not in names]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    for f in schema:
        got = table.schema.field(f.name).type
        if got != f.type:
            raise TypeError(f"column {f.name!r}: expected {f.type}, got {got}")
    if not allow_extra:
        extra = [n for n in names if n not in schema.names]
        if extra:
            raise ValueError(f"unexpected columns: {extra}")
