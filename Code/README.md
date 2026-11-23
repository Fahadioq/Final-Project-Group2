# Code

All project code. The data contracts between roles are in
[`common/schemas.py`](common/schemas.py) and the paths in [`common/paths.py`](common/paths.py).

Run every command from the repo root on `Final-Project-Group2` unless noted. Python is `~/dl-venv/bin/python`.
Data lives outside the repo in `~/of_data/` (EBS, durable) and `/mnt/nvme/` (instance store, wiped on
every stop/start). Never commit images, crops, or checkpoints: the data is CC BY-NC 4.0.

## Run order

| Step | Command | Status | What it does |
|---|---|---|---|
| 0 | `bash Code/scripts/mount_nvme.sh` | ready | Mount the instance store at `/mnt/nvme` (every session; formats it only if blank) |
| 1 | `bash Code/scripts/run_etl.sh A` (inside tmux) | ready | Pass A: stream 150 `core/train` shards, index every row, keep views of a stratified subset |
| 2 | `bash Code/scripts/run_etl.sh B` (inside tmux) | ready | Pass B: the remaining 458 shards; tops up rare buckets, completes the census |
| 3 | `cd Code && python -m etl.census --pool core_train` | ready | Census-gate tables in `results/` |
| 4 | `cd Code && python -m etl.contact_sheet --model tiny-random-sana --out ~/of_data/logs/sana.png` | ready | Visual check of a model string (Phase 0) |
| 5 | evaluation pools (validation, test, reddit, supplements) | to do | Same ETL for the evaluation sets |
| 6 | `cd Code && python -m etl.make_splits` | stub | Split files for E0-E6 |
| 7 | `bash Code/scripts/stage_data.sh C R` | ready | Copy views from EBS to NVMe before training |
| 8 | `bash Code/scripts/gpu_run.sh python Code/train.py ...` | stub | Train (queued behind the shared GPU lock) |
| 9 | `python Code/evaluate.py ...` | stub | Predictions and metrics for every evaluation set |
| 10 | `Code/analysis/*` | stub | Metrics, metadata baseline, transfer matrix, figures |
| 11 | `streamlit run Code/app/Home.py` | stub | The app |

ETL controls: stop gracefully with `touch ~/of_data/manifests/STOP` (delete the file before resuming);
resume by re-running the same command; check progress with `tail -f ~/of_data/logs/etl_core_train_pass*.log`;
`--dry-run` prints the current per-bucket thresholds.

Tests: `cd Code && ~/dl-venv/bin/python -m tests.test_etl`.

## Layout and ownership

| Path | Owner | Contents |
|---|---|---|
| `common/` | shared (changes reviewed by another role) | `schemas.py` (index, split, run, prediction contracts), `paths.py` |
| `etl/` | Data | `stream.py` (pass driver), `shard.py` (per-shard worker), `selection.py` (hash-threshold sampler), `views.py` (C/N/R recipes), `header_feats.py`, `census.py`, `contact_sheet.py`, `make_splits.py` |
| `taxonomy/` | Data | `generator_taxonomy.csv`: model string -> canonical name, lineage, denoiser, autoencoder (draft; verify) |
| `configs/` | each role, its own files | `etl_core_train.yaml` (caps, view recipe, runtime) |
| `scripts/` | Data | `mount_nvme.sh`, `run_etl.sh`, `stage_data.sh`, `gpu_run.sh` |
| `data/`, `models/`, `train.py`, `evaluate.py`, `inference.py` | Modeling | loader, GPU augmentation, backbones, two-head model, training, evaluation, inference API |
| `analysis/`, `app/` | Analysis & App | metrics, shortcut baseline, transfer matrix, figures, mock runs, Streamlit app |
| `tests/` | each role, its own files | `test_etl.py` |

## Data on disk

```
~/of_data/
  manifests/core_train_passes.json   fixed Pass A / Pass B shard lists
  manifests/core_train.jsonl         one line per finished shard (thresholds used + per-bucket counts)
  index/<pool>/<shard>.parquet       one row per streamed image (common.schemas.INDEX_SCHEMA)
  views/{C,R,N}/<pool>/<shard>.npy   uint8 [n, 256, 256, 3]; rows addressed by index view_off / n_off
  logs/
/mnt/nvme/landing/                   raw parquet shards, deleted after processing
```
