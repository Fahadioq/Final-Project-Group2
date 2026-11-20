"""Hash-threshold selection with adaptive per-bucket thresholds. Owner: Data.

A row is kept iff it is eligible and hash(row_id) < p[bucket]. Buckets are "fake:<model>" and
"real:<source>". Before each shard is dispatched, p[bucket] is recomputed from the running census
of completed shards so that the expected number kept meets the bucket's target:

    target   = water-fill of the lineage cap across its model strings, using the expected eligible
               count over every shard seen by the end of the current pass. The per-string cap is
               max(string_cap, lineage_cap / n_strings): it limits one fine-tune's share in lineages
               with many strings, but never stops a lineage with few strings (e.g. SD3.5, which has
               one) from reaching its lineage cap
    p        = clip((target - kept) / (rate * shards_remaining), 0, 1)

Shards are i.i.d. random samples of core/train, so this is an unbiased stratified sample. The
thresholds used for every shard are recorded in the manifest and every row's `p_keep` and `hash_u`
are stored in the index, so the selection is fully auditable. Any overshoot is trimmed by hash rank
at the census gate.
"""
from __future__ import annotations

import collections
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

REAL = "real"
UNMAPPED = "unmapped"
EXCLUDE = "exclude"
DEFAULT_KEY = "__default__"   # threshold for buckets not seen yet (rare by construction)


def row_hash(row_id: str) -> float:
    d = hashlib.blake2b(row_id.encode("utf-8"), digest_size=8, person=b"openfake-select").digest()
    return int.from_bytes(d, "big") / 2.0**64


def prompt_hash(prompt: str | None) -> str | None:
    if not prompt:
        return None
    norm = " ".join(prompt.lower().split())
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=8).hexdigest()


def load_taxonomy(path: Path | str) -> dict[str, str]:
    """model string -> lineage id."""
    with open(path, newline="") as fh:
        return {row["model"].strip(): row["lineage"].strip() for row in csv.DictReader(fh) if row.get("model")}


def bucket_of(label: str | None, model: str | None, taxonomy: dict[str, str]) -> tuple[str, str]:
    model = model or ""
    if label == "real":
        return REAL, f"real:{model}"
    lineage = taxonomy.get(model, UNMAPPED)
    if lineage == REAL:  # a real-source string labelled fake would be a data error; keep it visible
        lineage = UNMAPPED
    return lineage, f"fake:{model}"


def waterfill(avail: dict[str, float], total_cap: float, item_cap: float) -> dict[str, float]:
    """Allocate up to `total_cap` across items, each at most min(avail, item_cap), as evenly as possible."""
    capped = {k: min(v, item_cap) for k, v in avail.items()}
    if sum(capped.values()) <= total_cap:
        return capped
    lo, hi = 0.0, max(capped.values())
    for _ in range(64):
        mid = (lo + hi) / 2
        if sum(min(v, mid) for v in capped.values()) > total_cap:
            hi = mid
        else:
            lo = mid
    return {k: min(v, lo) for k, v in capped.items()}


@dataclass
class Targets:
    lineage_cap: float
    string_cap: float
    real_caps: dict
    real_default_cap: float
    unmapped_cap: float
    exclude_cap: float

    @classmethod
    def from_cfg(cls, d: dict) -> "Targets":
        return cls(**d)


class Selector:
    def __init__(self, targets: Targets):
        self.t = targets
        self.shards_done = 0
        self.eligible: collections.Counter = collections.Counter()
        self.kept: collections.Counter = collections.Counter()
        self.lineage: dict[str, str] = {}

    def update(self, stats: dict) -> None:
        """Fold in one completed shard's stats (as returned by etl.shard.process_shard)."""
        self.shards_done += 1
        self.eligible.update(stats["eligible"])
        self.kept.update(stats["kept"])
        self.lineage.update(stats["lineage"])

    def targets(self, horizon: float) -> tuple[dict[str, float], dict[str, float]]:
        rate = {b: self.eligible[b] / self.shards_done for b in self.lineage}
        est = {b: r * horizon for b, r in rate.items()}
        by_lineage: dict[str, dict[str, float]] = collections.defaultdict(dict)
        for b, lin in self.lineage.items():
            by_lineage[lin][b] = est[b]
        tgt: dict[str, float] = {}
        for lin, items in by_lineage.items():
            if lin == REAL:
                for b, v in items.items():
                    cap = self.t.real_caps.get(b.split(":", 1)[1], self.t.real_default_cap)
                    tgt[b] = min(v, cap)
            elif lin == UNMAPPED:
                tgt.update({b: min(v, self.t.unmapped_cap) for b, v in items.items()})
            elif lin == EXCLUDE:
                tgt.update({b: min(v, self.t.exclude_cap) for b, v in items.items()})
            else:
                item_cap = max(self.t.string_cap, self.t.lineage_cap / len(items))
                tgt.update(waterfill(items, self.t.lineage_cap, item_cap))
        return tgt, rate

    def thresholds(self, shards_remaining: int) -> dict[str, float]:
        """Per-bucket keep probabilities for the next dispatched shard.

        `shards_remaining` counts the pass's shards not yet completed, including in-flight ones.
        Before any shard has completed, everything eligible is kept (warm-up).
        """
        th = {DEFAULT_KEY: 1.0}
        if self.shards_done == 0:
            return th
        horizon = self.shards_done + shards_remaining
        tgt, rate = self.targets(horizon)
        for b, target in tgt.items():
            need = target - self.kept[b]
            expected = rate[b] * shards_remaining
            if need <= 0:
                th[b] = 0.0
            elif expected <= 0:
                th[b] = 1.0
            else:
                th[b] = min(1.0, need / expected)
        return th

    def kept_by_lineage(self) -> dict[str, int]:
        out: collections.Counter = collections.Counter()
        for b, n in self.kept.items():
            out[self.lineage.get(b, UNMAPPED)] += n
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))
