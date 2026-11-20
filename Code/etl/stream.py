"""Stream core/train through NVMe and keep 256x256 views of a stratified subset. Owner: Data.

    cd Code && python -m etl.stream --pass A            # 150 shards
    cd Code && python -m etl.stream --pass B            # the remaining 458
    (use Code/scripts/run_etl.sh, which also sets HF_HOME, nice, and logging)

Resumable: completed shards are recorded in manifests/core_train.jsonl and skipped on restart, but only
if their outputs are actually on disk with the expected row counts. Entries whose index or view files
are missing or incomplete (e.g. after a partial copy to another machine) are dropped from the manifest,
which is backed up first, and those shards are reprocessed.
Graceful stop: `touch ~/of_data/manifests/STOP` (in-flight shards finish; nothing new starts).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from common import paths
from etl.selection import Selector, Targets, load_taxonomy
from etl.shard import process_shard

REPO_ID = "ComplexDataLab/OpenFake"
POOL = "core_train"
log = logging.getLogger("etl.stream")


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _append_jsonl(path: Path, obj) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(obj) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _shard_key(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def load_passes(cfg: dict) -> dict:
    """Deterministic, persisted split of the 608 train shards into Pass A and Pass B."""
    p = paths.MANIFEST_DIR / f"{POOL}_passes.json"
    if p.exists():
        return json.loads(p.read_text())
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset", revision=cfg["revision"])
    shards = sorted(f for f in files if f.startswith("core/train-") and f.endswith(".parquet"))
    order = sorted(shards, key=_shard_key)
    n = int(cfg["pass_a_shards"])
    passes = {"revision": cfg["revision"], "n_shards": len(shards), "A": sorted(order[:n]), "B": sorted(order[n:])}
    _write_json_atomic(p, passes)
    return passes


def read_manifest() -> list[dict]:
    p = paths.MANIFEST_DIR / f"{POOL}.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def outputs_ok(entry: dict, root: Path | str | None = None) -> bool:
    """True if a manifest entry's index and view arrays exist with the row counts it recorded."""
    st = entry["stats"]
    chunk = st["chunk"]
    try:
        if pq.ParquetFile(paths.index_path(POOL, chunk, root)).metadata.num_rows != st["n_rows"]:
            return False
        for view, n in (("C", st["n_kept"]), ("R", st["n_kept"]), ("N", st["n_native"])):
            if n and np.load(paths.view_path(view, POOL, chunk, root), mmap_mode="r").shape[0] != n:
                return False
    except (OSError, ValueError):  # missing file, or truncated / unreadable array
        return False
    return True


def validated_manifest(rewrite: bool) -> list[dict]:
    """Manifest entries whose outputs are complete; optionally rewrite the file without the others."""
    manifest = read_manifest()
    valid = [m for m in manifest if outputs_ok(m)]
    if len(valid) != len(manifest):
        log.warning("%d of %d manifest entries have missing or incomplete outputs; those shards will be reprocessed",
                    len(manifest) - len(valid), len(manifest))
        if rewrite:
            p = paths.MANIFEST_DIR / f"{POOL}.jsonl"
            backup = p.with_name(f"{p.name}.bak-{time.strftime('%Y%m%dT%H%M%S')}")
            shutil.copy2(p, backup)
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text("".join(json.dumps(m) + "\n" for m in valid))
            os.replace(tmp, p)
            log.warning("manifest rewritten with %d entries; previous version saved as %s", len(valid), backup.name)
    return valid


def download(fname: str, revision: str, landing: Path) -> tuple[str, str, float]:
    from huggingface_hub import hf_hub_download
    last = None
    for attempt in range(4):
        try:
            t0 = time.time()
            local = hf_hub_download(REPO_ID, fname, repo_type="dataset", revision=revision, local_dir=str(landing))
            return fname, local, time.time() - t0
        except Exception as e:  # transient network / throttling
            last = e
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"download failed for {fname}: {last!r}")


def cleanup_tmp() -> None:
    root = paths.OF_DATA
    for sub in ("index", "views"):
        for p in (root / sub).rglob("*.tmp*"):
            p.unlink(missing_ok=True)


def stop_reason(cfg: dict) -> str | None:
    if (paths.MANIFEST_DIR / "STOP").exists():
        return "STOP file present"
    free_gb = shutil.disk_usage(paths.OF_DATA).free / 1e9
    if free_gb < float(cfg["runtime"]["min_free_gb"]):
        return f"only {free_gb:.0f} GB free on {paths.OF_DATA}"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass", dest="pass_", choices=["A", "B"], required=True)
    ap.add_argument("--config", default=str(paths.CONFIG_DIR / "etl_core_train.yaml"))
    ap.add_argument("--workers", type=int, help="decode processes (default from config)")
    ap.add_argument("--downloads", type=int, help="concurrent shard downloads (default from config)")
    ap.add_argument("--limit", type=int, help="process at most N shards this run (smoke tests)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and current thresholds, then exit")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(open(args.config))
    workers = args.workers or int(cfg["runtime"]["workers"])
    downloads = args.downloads or int(cfg["runtime"]["downloads"])
    for d in (paths.LANDING, paths.MANIFEST_DIR, paths.LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    taxonomy = load_taxonomy(paths.TAXONOMY_CSV)
    passes = load_passes(cfg)
    if passes["revision"] != cfg["revision"]:
        raise SystemExit(f"pass file is for revision {passes['revision']}, config says {cfg['revision']}")
    manifest = validated_manifest(rewrite=not args.dry_run)
    done = {m["shard"] for m in manifest}
    selector = Selector(Targets.from_cfg(cfg["targets"]))
    for m in manifest:
        selector.update(m["stats"])

    pass_shards = sorted(passes[args.pass_], key=_shard_key)
    remaining = [s for s in pass_shards if s not in done]
    todo = remaining[: args.limit] if args.limit else list(remaining)
    log.info("pass %s: %d shards, %d already done, %d to process now (workers=%d, downloads=%d, OF_DATA=%s)",
             args.pass_, len(pass_shards), len(pass_shards) - len(remaining), len(todo), workers, downloads,
             paths.OF_DATA)
    log.info("kept so far by lineage: %s", selector.kept_by_lineage())
    if args.dry_run:
        th = selector.thresholds(len(remaining))
        log.info("thresholds for next shard: %s", json.dumps({k: round(v, 3) for k, v in sorted(th.items())}))
        return 0
    if not todo:
        log.info("nothing to do")
        return 0

    cleanup_tmp()
    failures = paths.MANIFEST_DIR / f"{POOL}_failures.jsonl"
    manifest_path = paths.MANIFEST_DIR / f"{POOL}.jsonl"
    queue = collections.deque(todo)
    dl_futs: dict[cf.Future, str] = {}
    proc_futs: dict[cf.Future, tuple[str, str, dict]] = {}
    ready: collections.deque = collections.deque()
    slots = workers + downloads   # max shards on the landing disk at once
    completed = 0
    t_start = time.time()
    dl_bytes = 0
    dl_secs = 0.0

    with cf.ThreadPoolExecutor(downloads) as dpool, \
            cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=25) as ppool:
        while queue or dl_futs or proc_futs or ready:
            reason = stop_reason(cfg)
            if reason and queue:
                log.warning("stopping early: %s (%d shards left unstarted)", reason, len(queue))
                queue.clear()
            while queue and len(dl_futs) < downloads and len(dl_futs) + len(ready) + len(proc_futs) < slots:
                s = queue.popleft()
                dl_futs[dpool.submit(download, s, cfg["revision"], paths.LANDING)] = s
            while ready and len(proc_futs) < workers:
                s, local = ready.popleft()
                th = selector.thresholds(len(remaining) - completed)
                task = dict(path=local, shard=s, pool=POOL, thresholds=th, taxonomy=taxonomy, cfg=cfg,
                            of_data=str(paths.OF_DATA))
                proc_futs[ppool.submit(process_shard, task)] = (s, local, th)
            if not dl_futs and not proc_futs:
                continue
            finished, _ = cf.wait(list(dl_futs) + list(proc_futs), timeout=120, return_when=cf.FIRST_COMPLETED)
            for f in finished:
                if f in dl_futs:
                    s = dl_futs.pop(f)
                    try:
                        _, local, secs = f.result()
                    except Exception as e:
                        log.error("download failed: %s: %r", s, e)
                        _append_jsonl(failures, dict(shard=s, stage="download", error=repr(e), at=time.time()))
                        continue
                    size = os.path.getsize(local)
                    dl_bytes += size
                    dl_secs += secs
                    log.info("downloaded %s (%.1f GB in %.0fs, %.0f MB/s)", Path(s).name, size / 1e9, secs,
                             size / 1e6 / max(secs, 1e-6))
                    ready.append((s, local))
                else:
                    s, local, th = proc_futs.pop(f)
                    try:
                        stats = f.result()
                        _append_jsonl(manifest_path, dict(shard=s, pass_=args.pass_, finished_at=time.time(),
                                                          thresholds={k: round(v, 4) for k, v in th.items()},
                                                          stats=stats))
                        selector.update(stats)
                        completed += 1
                        elapsed = time.time() - t_start
                        eta = elapsed / completed * (len(todo) - completed)
                        log.info("[%s %d/%d] %s rows=%d kept=%d native=%d %.0fs (pass1 %.0fs) skips=%s | "
                                 "elapsed %.0f min, ETA %.0f min", args.pass_, completed, len(todo), stats["chunk"],
                                 stats["n_rows"], stats["n_kept"], stats["n_native"], stats["secs"],
                                 stats["secs_pass1"], stats["skips"], elapsed / 60, eta / 60)
                        if completed % 10 == 0 or completed == len(todo):
                            log.info("kept by lineage: %s", selector.kept_by_lineage())
                    except Exception as e:
                        log.exception("processing failed: %s", s)
                        _append_jsonl(failures, dict(shard=s, stage="process", error=repr(e), at=time.time()))
                    finally:
                        Path(local).unlink(missing_ok=True)

    log.info("pass %s run finished: %d/%d shards in %.0f min; download avg %.0f MB/s per stream",
             args.pass_, completed, len(todo), (time.time() - t_start) / 60, dl_bytes / 1e6 / max(dl_secs, 1e-6))
    log.info("kept by lineage: %s", selector.kept_by_lineage())
    return 0 if completed == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())
