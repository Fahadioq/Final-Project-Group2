"""Process one downloaded parquet shard (runs in a worker process). Owner: Data.

Pass 1 reads every row: metadata, header features, eligibility, keep decision -> index (ALL rows).
Pass 2 re-reads only the row groups that contain kept rows, decodes those images once, and writes
the C, R (and, for images > 2 MP, N) views into uint8 arrays. Everything is written to `*.tmp`
first and renamed at the end, so a crash never leaves a half-written chunk behind.
"""
from __future__ import annotations

import collections
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.lib.format import open_memmap

from common import paths, schemas
from etl import views as V
from etl.header_feats import header_features
from etl.selection import DEFAULT_KEY, bucket_of, prompt_hash, row_hash

META_COLS = ("row_id", "label", "model", "type", "release_date", "prompt", "image_sha256")


def _image_bytes(table: pa.Table) -> pa.Array:
    return table.column("image").combine_chunks().field("bytes")


def _finish_array(arr, tmp: Path, final: Path, n_used: int) -> None:
    """Flush a memmap; if some rows failed to decode, rewrite it compactly; then rename."""
    n_alloc = arr.shape[0]
    arr.flush()
    if n_used == n_alloc:
        del arr
        os.replace(tmp, final)
        return
    data = np.array(arr[:n_used])
    del arr
    compact = tmp.with_name(tmp.name + ".compact")
    with open(compact, "wb") as fh:
        np.save(fh, data)
    os.replace(compact, final)
    tmp.unlink(missing_ok=True)


def process_shard(task: dict) -> dict:
    t0 = time.time()
    src = task["path"]
    pool = task["pool"]
    chunk = Path(task["shard"]).stem
    root = task.get("of_data")
    th = task["thresholds"]
    default_p = float(th.get(DEFAULT_KEY, 1.0))
    taxonomy = task["taxonomy"]
    vcfg, ecfg = task["cfg"]["views"], task["cfg"]["eligibility"]
    size, max_mp, target_mp = int(vcfg["size"]), float(vcfg["max_mp"]), float(vcfg["target_mp"])

    pf = pq.ParquetFile(src)
    names = set(pf.schema_arrow.names)
    cols = [c for c in META_COLS if c in names]

    # ---- pass 1: index every row ------------------------------------------------------------
    recs: list[dict] = []
    for rg in range(pf.num_row_groups):
        tb = pf.read_row_group(rg, columns=cols + ["image"])
        meta = {c: tb.column(c).to_pylist() for c in cols}
        img = _image_bytes(tb)
        for i in range(tb.num_rows):
            b = img[i].as_py()
            hf = header_features(b)
            label = meta["label"][i] if "label" in meta else None
            model = meta["model"][i] if "model" in meta else None
            lineage, bucket = bucket_of(label, model, taxonomy)
            ok, reason = V.eligibility(hf, min_side=int(ecfg["min_side"]), max_mp=max_mp, target_mp=target_mp)
            rid = meta["row_id"][i]
            u = row_hash(rid)
            p = float(th.get(bucket, default_p))
            recs.append(dict(
                row_id=rid, image_sha256=meta.get("image_sha256", [None] * tb.num_rows)[i],
                pool=pool, shard=chunk, row_group=rg, row_in_group=i,
                label=label, model=model, type=meta.get("type", [None] * tb.num_rows)[i],
                release_date=meta.get("release_date", [None] * tb.num_rows)[i],
                prompt_hash=prompt_hash(meta.get("prompt", [None] * tb.num_rows)[i]),
                lineage=lineage, bucket=bucket, **hf,
                eligible=ok, skip_reason=reason, hash_u=u, p_keep=p, kept=bool(ok and u < p),
                decode_ok=False, view_off=-1, n_off=-1,
            ))
        del tb, img
    t1 = time.time()

    # ---- pass 2: decode kept rows, write views ----------------------------------------------
    keep_by_rg: dict[int, list[int]] = collections.defaultdict(list)
    for k, r in enumerate(recs):
        if r["kept"]:
            keep_by_rg[r["row_group"]].append(k)
    n_keep = sum(len(v) for v in keep_by_rg.values())
    n_big = sum(1 for ks in keep_by_rg.values() for k in ks
                if V.needs_native(recs[k]["width"], recs[k]["height"], max_mp))

    final = {v: paths.view_path(v, pool, chunk, root) for v in ("C", "R", "N")}
    tmp = {v: p.with_name(p.name + ".tmp") for v, p in final.items()}
    for p in final.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.unlink(missing_ok=True)  # reprocessing a shard replaces its outputs
    arrs = {}
    if n_keep:
        arrs["C"] = open_memmap(tmp["C"], mode="w+", dtype=np.uint8, shape=(n_keep, size, size, 3))
        arrs["R"] = open_memmap(tmp["R"], mode="w+", dtype=np.uint8, shape=(n_keep, size, size, 3))
    if n_big:
        arrs["N"] = open_memmap(tmp["N"], mode="w+", dtype=np.uint8, shape=(n_big, size, size, 3))

    vi = ni = 0
    for rg in sorted(keep_by_rg):
        img = _image_bytes(pf.read_row_group(rg, columns=["image"]))
        for k in keep_by_rg[rg]:
            r = recs[k]
            big = V.needs_native(r["width"], r["height"], max_mp)
            try:
                im = V.load_rgb(img[r["row_in_group"]].as_py())
                c = V.view_c(im, size, max_mp, target_mp)
                rr = V.view_r(im, size)
                n = V.view_n(im, size) if big else None
            except Exception as e:  # corrupt or unusual file: record and move on
                r.update(kept=False, skip_reason=f"decode_error:{type(e).__name__}")
                continue
            arrs["C"][vi] = c
            arrs["R"][vi] = rr
            r.update(decode_ok=True, view_off=vi)
            vi += 1
            if n is not None:
                arrs["N"][ni] = n
                r["n_off"] = ni
                ni += 1
        del img
    for v, n_used in (("C", vi), ("R", vi), ("N", ni)):
        if v in arrs:
            arr = arrs.pop(v)
            _finish_array(arr, tmp[v], final[v], n_used)
    for v in ("C", "R", "N"):  # an all-failed chunk leaves no empty arrays behind
        if final[v].exists() and np.load(final[v], mmap_mode="r").shape[0] == 0:
            final[v].unlink()

    table = pa.Table.from_pylist(recs, schema=schemas.INDEX_SCHEMA)
    schemas.validate(table, schemas.INDEX_SCHEMA)
    out_idx = paths.index_path(pool, chunk, root)
    out_idx.parent.mkdir(parents=True, exist_ok=True)
    tmp_idx = out_idx.with_name(out_idx.name + ".tmp")
    pq.write_table(table, tmp_idx, compression="zstd")
    os.replace(tmp_idx, out_idx)

    seen, eligible, kept = collections.Counter(), collections.Counter(), collections.Counter()
    lineage = {}
    skips = collections.Counter()
    for r in recs:
        b = r["bucket"]
        lineage[b] = r["lineage"]
        seen[b] += 1
        eligible[b] += r["eligible"]
        kept[b] += r["kept"]
        if r["skip_reason"]:
            skips[r["skip_reason"].split(":")[0]] += 1
    return dict(
        shard=task["shard"], chunk=chunk, n_rows=len(recs), n_kept=vi, n_native=ni,
        seen=dict(seen), eligible=dict(eligible), kept=dict(kept), lineage=lineage, skips=dict(skips),
        secs=round(time.time() - t0, 1), secs_pass1=round(t1 - t0, 1),
    )
