#!/usr/bin/env python3
"""Train one frozen CinC-2017 candidate using train/development records only.

No locked-test or external record is opened by this script.  The checkpoint metric
is the P5 common-support penalized record-level AUPRC, not a window-level score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset
import yaml


CHANNELS = {"tiny": (8, 16, 32), "small": (16, 32, 64), "medium": (24, 48, 96)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_windows(mat_path: Path, window: int, hop: int, quality_gate: bool) -> list[np.ndarray]:
    signal = np.asarray(loadmat(mat_path)["val"], dtype=np.float32).reshape(-1)
    output = []
    for start in range(0, signal.size - window + 1, hop):
        raw = signal[start : start + window]
        q05, median, q95 = np.quantile(raw, (0.05, 0.5, 0.95), method="linear")
        spread = q95 - q05
        plateau = (np.count_nonzero(raw[1:] == raw[:-1]) / (raw.size - 1)) if raw.size >= 2 else 1.0
        outlier = np.max(np.abs(raw - median)) / (spread + 1e-6)
        invalid = (not np.isfinite(raw).all()) or plateau >= 0.95 or spread <= 1e-6 or outlier > 25
        if quality_gate and invalid:
            continue
        output.append(np.clip((raw - median) / (spread + 1e-6), -8, 8).astype(np.float32))
    return output


class FrozenNet(nn.Module):
    def __init__(self, scale: str) -> None:
        super().__init__()
        c1, c2, c3 = CHANNELS[scale]
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, 5, stride=2, bias=True), nn.ReLU(),
            nn.Conv1d(c1, c2, 5, stride=2, bias=True), nn.ReLU(),
            nn.Conv1d(c2, c3, 5, stride=2, bias=True), nn.ReLU(),
        )
        self.head = nn.Linear(c3, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).mean(dim=-1)).squeeze(-1)


def make_windows(records: list[dict], base: Path, window: int, hop: int, quality: bool) -> tuple[np.ndarray, np.ndarray]:
    values, labels = [], []
    for record in records:
        for item in normalized_windows(base / record["relative_path"], window, hop, quality):
            values.append(item)
            labels.append(record["mapped_label"])
    if not values:
        raise RuntimeError("No valid training windows")
    return np.stack(values)[:, None, :], np.asarray(labels, dtype=np.float32)


def prepare_development(records: list[dict], base: Path, window: int, hop: int, quality: bool):
    """Load and preprocess development records once, outside the epoch loop."""
    windows, record_indices, labels = [], [], []
    rejected = 0
    for record_index, record in enumerate(records):
        pieces = normalized_windows(base / record["relative_path"], window, hop, quality)
        labels.append(record["mapped_label"])
        if not pieces:
            rejected += 1
            continue
        windows.extend(pieces)
        record_indices.extend([record_index] * len(pieces))
    x = torch.from_numpy(np.stack(windows)[:, None, :]) if windows else torch.empty((0, 1, window))
    ridx = torch.as_tensor(record_indices, dtype=torch.long)
    return x, ridx, np.asarray(labels, dtype=np.int64), rejected


@torch.inference_mode()
def development_metric(model: nn.Module, prepared, batch_size: int = 512) -> tuple[float, dict]:
    x, record_indices, labels, rejected = prepared
    model.eval()
    score_sums = torch.zeros(len(labels), dtype=torch.float64)
    score_counts = torch.zeros(len(labels), dtype=torch.long)
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        scores = torch.sigmoid(model(x[start:end])).to(torch.float64)
        score_sums.index_add_(0, record_indices[start:end], scores)
        score_counts.index_add_(0, record_indices[start:end], torch.ones(end - start, dtype=torch.long))
    record_scores = torch.full((len(labels),), -1.0, dtype=torch.float64)
    scorable = score_counts > 0
    record_scores[scorable] = score_sums[scorable] / score_counts[scorable]
    metric = average_precision_score(labels, record_scores.numpy(), pos_label=1)
    return float(metric), {"rejected_records": rejected, "records": len(labels), "development_windows": int(len(x))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--window-s", type=float, choices=(5, 10), required=True)
    p.add_argument("--hop-s", type=float, choices=(2.5, 5), required=True)
    p.add_argument("--preprocessing", choices=("robust_norm", "robust_norm_quality_gate"), required=True)
    p.add_argument("--model", choices=tuple(CHANNELS), required=True)
    p.add_argument("--seed", type=int, default=20260921)
    args = p.parse_args()
    if (args.window_s, args.hop_s) not in {(5, 2.5), (10, 5)}:
        raise ValueError("Only frozen valid window-hop pairs are permitted")
    if args.run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {args.run_dir}")
    os.environ.update({"PYTHONHASHSEED": str(args.seed), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(1)
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    split = yaml.safe_load(args.split.read_text(encoding="utf-8"))
    if manifest.get("view_scope") != ["train", "development"] or split.get("view_scope") != ["train", "development"]:
        raise RuntimeError("Trainer accepts only filtered train/development input views")
    if manifest.get("parent_manifest_sha256") != split.get("parent_manifest_sha256"):
        raise RuntimeError("Filtered view files do not point to the same parent manifest")
    if manifest.get("parent_split_sha256") != split.get("parent_split_sha256"):
        raise RuntimeError("Filtered view files do not point to the same parent split")
    index = {r["record_id"]: r for r in manifest["records"]}
    if set(index) != {x["record_id"] for x in split["assignments"]}:
        raise RuntimeError("Filtered manifest and split view membership differs")
    if any(x["split"] not in {"train", "development"} for x in split["assignments"]):
        raise RuntimeError("Non-train/development assignment found in filtered input")
    train = [index[x["record_id"]] for x in split["assignments"] if x["split"] == "train"]
    development = [index[x["record_id"]] for x in split["assignments"] if x["split"] == "development"]
    record_root = Path(manifest["record_root"])
    window, hop = int(args.window_s * 300), int(args.hop_s * 300)
    quality = args.preprocessing == "robust_norm_quality_gate"
    x, y = make_windows(train, record_root, window, hop, quality)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=64, shuffle=True)
    prepared_development = prepare_development(development, record_root, window, hop, quality)
    model = FrozenNet(args.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_epoch, no_improvement, history = float("-inf"), 0, 0, []
    args.run_dir.mkdir(parents=True)
    for epoch in range(1, 101):
        model.train(); losses = []
        for bx, by in loader:
            optimizer.zero_grad(); loss = loss_fn(model(bx), by); loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        metric, detail = development_metric(model, prepared_development)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "development_common_support_penalized_auprc": metric, **detail})
        print(f"epoch={epoch} train_loss={history[-1]['train_loss']:.6f} development_common_support_penalized_auprc={metric:.6f} rejected_records={detail['rejected_records']}", flush=True)
        if metric > best + 0.0005:
            best, best_epoch, no_improvement = metric, epoch, 0
            torch.save(model.state_dict(), args.run_dir / "best_checkpoint.pt")
        else:
            no_improvement += 1
        if no_improvement >= 10:
            break
    metadata = {"status": "development_only_completed", "locked_test_accessed": False, "external_data_accessed": False,
                "manifest_sha256": digest(args.manifest), "split_sha256": digest(args.split),
                "parent_manifest_sha256": manifest["parent_manifest_sha256"],
                "parent_split_sha256": manifest["parent_split_sha256"], "input_view_scope": manifest["view_scope"],
                "record_root": manifest["record_root"],
                "seed": args.seed,
                "window_s": args.window_s, "hop_s": args.hop_s, "preprocessing": args.preprocessing, "model": args.model,
                "train_records": len(train), "development_records": len(development), "train_windows": int(y.size),
                "best_epoch": best_epoch, "best_development_common_support_penalized_auprc": best, "history": history}
    (args.run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in metadata if k != "history"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
