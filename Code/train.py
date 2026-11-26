"""Train the two-head detector and write a run directory. Owner: Modeling.

    # real data (on the group instance, through the shared GPU lock)
    bash Code/scripts/gpu_run.sh python Code/train.py --split ~/of_data/splits/E0/main.parquet --run-id E0_resnet50_s0

    # pipeline check on the synthetic store (no split files needed)
    python Code/train.py --standin --set epochs=2

    # change any setting from configs/train_base.yaml
    python Code/train.py --split ... --set backbone=convnext_tiny --set seed=1 --set two_heads=false

One epoch = every training image once, augmented on the GPU (data/gpu_aug.py). After each epoch the model is scored
on the dev set; the epoch with the best dev AUROC is kept (early stopping). The decision threshold
tau is then fixed on the dev predictions of that epoch and saved in the model card.

Output: <out>/<run_id>/ with config.yaml, metrics.json, preds.parquet (dev set), model.pt, model_card.json.
Test sets are scored afterwards by evaluate.py.

Training recipe: AdamW with two learning rates (backbone 1e-4, heads 1e-3), weight decay 0.05,
one epoch of linear warm-up then cosine decay, bf16 autocast on the GPU, channels-last memory layout.
"""
from __future__ import annotations

import argparse
import math
import random
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from common import paths
from data.gpu_aug import IMAGENET_MEAN, IMAGENET_STD, GpuAugment
from data.memmap_dataset import ViewDataset, make_loader
from data.standin import make_standin_store
from models.backbones import input_stats
from models.heads import build_detector, detector_loss, param_groups
from models.runs import (auroc, predict, preds_table, save_checkpoint, write_json, youden_threshold,
                         append_preds)

DEFAULT_CONFIG = paths.CONFIG_DIR / "train_base.yaml"


def load_config(path: Path, overrides: list[str]) -> dict:
    """Read the YAML config, then apply --set key=value overrides (values parsed as YAML: 2, 1e-4, true, resnet50)."""
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, _, value = item.partition("=")
        if key not in cfg:
            raise KeyError(f"unknown setting {key!r}; known: {sorted(cfg)}")
        parsed = yaml.safe_load(value)
        if isinstance(parsed, str):          # YAML reads "2e-4" (no decimal point) as text, not a number
            try:
                parsed = float(parsed)
            except ValueError:
                pass
        cfg[key] = parsed
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_schedule(steps_per_epoch: int, total_steps: int, warmup_epochs: float):
    """Multiplier on the base learning rate at each step: linear warm-up from 0 to 1, then cosine down to 0.

    Warm-up keeps the first updates small while the new heads are still random, so their large early gradients
    do not wreck the pretrained backbone.
    """
    warmup = max(1, int(warmup_epochs * steps_per_epoch))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return factor


def git_commit() -> str | None:
    """The commit the code ran from, so a result can be traced back to exact code."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=paths.REPO_DIR, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def train(cfg: dict, split_path: Path, run_dir: Path, data_root: Path | None = None,
          views_root: Path | None = None, max_steps: int | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    seed_everything(cfg["seed"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "split": str(split_path)}, sort_keys=False))

    # ---- data -----------------------------------------------------------------------------------------------
    train_ds = ViewDataset(split_path, ["train"], view=cfg["view"], data_root=data_root, views_root=views_root)
    dev_ds = ViewDataset(split_path, ["dev"], view=cfg["view"], classes=train_ds.classes,
                         data_root=data_root, views_root=views_root)
    train_loader = make_loader(train_ds, cfg["batch_size"], train=True, num_workers=cfg["workers"], seed=cfg["seed"])
    dev_loader = make_loader(dev_ds, cfg["eval_batch_size"], train=False, num_workers=cfg["workers"])
    aug = GpuAugment(device=device, seed=cfg["seed"],
                     jpeg_range=(65, 95) if cfg["jpeg_equalize"] else None)
    classes = train_ds.classes if cfg["two_heads"] else []
    print(f"device {device}: {len(train_ds):,} train / {len(dev_ds):,} dev images, view {cfg['view']}, "
          f"{len(classes)} attribution classes")

    # ---- model and optimizer ----------------------------------------------------------------------------------
    model = build_detector(cfg["backbone"], n_classes=len(classes), pretrained=cfg["pretrained"],
                           dropout=cfg["dropout"])
    if input_stats(model.backbone) != (IMAGENET_MEAN, IMAGENET_STD):
        raise ValueError(f"{cfg['backbone']} expects other input statistics than data/gpu_aug.py provides")
    model = model.to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(param_groups(model, cfg["lr_backbone"], cfg["lr_heads"], cfg["weight_decay"]))
    steps_per_epoch = len(train_loader)
    total_steps = min(steps_per_epoch * cfg["epochs"], max_steps or float("inf"))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_schedule(steps_per_epoch, total_steps, cfg["warmup_epochs"]))

    # ---- training loop ----------------------------------------------------------------------------------------
    history, best, step, epochs_without_gain = [], None, 0, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        sums = {"loss": 0.0, "bce": 0.0, "ce": 0.0}
        n_batches = n_images = 0
        t0 = time.perf_counter()
        for images, label, attr, bce_mask, _idx in train_loader:
            x = aug.train(images)
            label = label.to(device, non_blocking=True)
            attr = attr.to(device, non_blocking=True)
            bce_mask = bce_mask.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(x)
            loss, parts = detector_loss(out, label, attr, bce_mask, cfg["attr_weight"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            for k in sums:
                sums[k] += parts[k]
            n_batches += 1
            n_images += images.shape[0]
            step += 1
            if step >= total_steps:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0

        dev = predict(model, dev_loader, aug, device)
        dev_auroc = auroc(dev["label"], dev["p_fake"])
        record = {"epoch": epoch, **{k: v / max(n_batches, 1) for k, v in sums.items()},
                  "dev_auroc": dev_auroc, "images_per_s": n_images / seconds, "seconds": seconds}
        history.append(record)
        print(f"epoch {epoch}: loss {record['loss']:.4f} (bce {record['bce']:.4f}, ce {record['ce']:.4f}) | "
              f"dev AUROC {dev_auroc:.4f} | {record['images_per_s']:,.0f} images/s | {seconds:,.0f} s")

        if best is None or dev_auroc > best["dev_auroc"]:       # keep only the best epoch's weights
            best = {"epoch": epoch, "dev_auroc": dev_auroc, "dev": dev}
            save_checkpoint(run_dir / "model.pt", model, cfg["backbone"], len(classes))
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= cfg["patience"]:
                print(f"early stop: no better dev AUROC for {cfg['patience']} epochs")
                break
        if step >= total_steps:
            break

    # ---- outputs ----------------------------------------------------------------------------------------------
    tau = youden_threshold(best["dev"]["label"], best["dev"]["p_fake"])
    append_preds(run_dir, preds_table(dev_ds, best["dev"], "dev", classes))
    card = {"run_id": run_dir.name, "backbone": cfg["backbone"], "classes": classes, "tau": tau,
            "view": cfg["view"], "eval_crop": aug.crop}
    write_json(run_dir / "model_card.json", card)
    metrics = {
        "run_id": run_dir.name, "best_epoch": best["epoch"], "dev_auroc": best["dev_auroc"], "tau": tau,
        "n_train": len(train_ds), "n_dev": len(dev_ds), "history": history,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "git_commit": git_commit(), "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(run_dir / "metrics.json", metrics)
    print(f"best epoch {best['epoch']}: dev AUROC {best['dev_auroc']:.4f}, tau {tau:.3f} -> {run_dir}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--split", help="split parquet written by etl/make_splits.py")
    source.add_argument("--standin", action="store_true", help="train on a synthetic store (pipeline check only)")
    ap.add_argument("--standin-rows", type=int, default=600, help="rows per synthetic shard (6 shards)")
    ap.add_argument("--run-id", help="name of the run directory (default: built from the settings and time)")
    ap.add_argument("--out", default=str(paths.RUNS_DIR), help="parent folder for run directories")
    ap.add_argument("--max-steps", type=int, help="stop after this many optimizer steps (quick checks)")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config), args.set)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"{'standin' if args.standin else Path(args.split).parent.name}_{cfg['backbone']}" \
                            f"_s{cfg['seed']}_{stamp}"
    run_dir = Path(args.out).expanduser() / run_id

    if args.standin:
        with tempfile.TemporaryDirectory() as tmp:
            split = make_standin_store(tmp, n_shards=6, rows_per_shard=args.standin_rows, seed=cfg["seed"])
            train(cfg, split, run_dir, data_root=Path(tmp), views_root=Path(tmp), max_steps=args.max_steps)
    else:
        train(cfg, Path(args.split).expanduser(), run_dir, max_steps=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
