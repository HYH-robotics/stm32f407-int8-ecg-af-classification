#!/usr/bin/env python3
"""Train-only calibrated symmetric INT8 conversion and development evaluation.

This stage accepts only the filtered train/development view. Calibration uses
unlabeled train windows; model/config ranking uses record-level development AP.
No locked-test or external files are opened.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.io import loadmat
from sklearn.metrics import average_precision_score

from train_cinc2017_candidate import FrozenNet


ROOT = Path(__file__).resolve().parents[1]
VIEW = ROOT / "data/ecg/views/train_dev_v2"
TRAIN_MANIFEST = VIEW / "train_development_manifest.yaml"
TRAIN_SPLIT = VIEW / "train_development_split.yaml"
RUNS = ROOT / "runs"
OUT = ROOT / "artifacts/p6_int8"
N_CAL = 1024
BATCH = 32
EPS = 1e-6

CONFIGS = [
    (10, 5, "robust_norm", "tiny", "P6_W10_H5_robust_norm_tiny_seed20260921_isov1"),
    (10, 5, "robust_norm_quality_gate", "tiny", "P6_W10_H5_quality_gate_tiny_seed20260921_isov1"),
    (5, 2.5, "robust_norm", "tiny", "P6_W5_H2.5_robust_norm_tiny_seed20260921_isov1"),
    (5, 2.5, "robust_norm_quality_gate", "tiny", "P6_W5_H2.5_quality_gate_tiny_seed20260921_isov1"),
    (10, 5, "robust_norm", "small", "P6_W10_H5_robust_norm_small_seed20260921_isov1_retry1"),
    (10, 5, "robust_norm_quality_gate", "small", "P6_W10_H5_quality_gate_small_seed20260921_isov1"),
    (5, 2.5, "robust_norm", "small", "P6_W5_H2.5_robust_norm_small_seed20260921_isov1"),
    (5, 2.5, "robust_norm_quality_gate", "small", "P6_W5_H2.5_quality_gate_small_seed20260921_isov1"),
    (10, 5, "robust_norm", "medium", "P6_W10_H5_robust_norm_medium_seed20260921_isov1"),
    (10, 5, "robust_norm_quality_gate", "medium", "P6_W10_H5_quality_gate_medium_seed20260921_isov1"),
    (5, 2.5, "robust_norm", "medium", "P6_W5_H2.5_robust_norm_medium_seed20260921_isov1"),
    (5, 2.5, "robust_norm_quality_gate", "medium", "P6_W5_H2.5_quality_gate_medium_seed20260921_isov1"),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def round_away_np(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def round_away_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))


def quantize_np(x: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(round_away_np(x.astype(np.float64) / scale), -127, 127).astype(np.int8)


def windows_with_starts(mat_path: Path, win_n: int, hop_n: int, gate: bool):
    raw_signal = np.asarray(loadmat(mat_path)["val"], dtype=np.float32).reshape(-1)
    for start in range(0, raw_signal.size - win_n + 1, hop_n):
        raw = raw_signal[start : start + win_n]
        q05, median, q95 = np.quantile(raw, (0.05, 0.5, 0.95), method="linear")
        spread = float(q95 - q05)
        plateau = float(np.count_nonzero(raw[1:] == raw[:-1]) / (raw.size - 1)) if raw.size >= 2 else 1.0
        outlier = float(np.max(np.abs(raw - median)) / (spread + EPS))
        invalid = (not np.isfinite(raw).all()) or plateau >= 0.95 or spread <= EPS or outlier > 25
        if gate and invalid:
            continue
        norm = np.clip((raw - median) / (spread + EPS), -8, 8).astype(np.float32)
        yield start, raw.copy(), norm


def make_calibration(records: list[dict], root: Path, win_n: int, hop_n: int, gate: bool):
    selected, canonical = [], bytearray()
    for rec in sorted(records, key=lambda r: r["record_id"]):
        for start, raw, norm in windows_with_starts(root / rec["relative_path"], win_n, hop_n, gate):
            raw_bytes = np.ascontiguousarray(raw, dtype="<f4").tobytes()
            raw_hash = sha256_bytes(raw_bytes)
            canonical.extend(f"{rec['record_id']}\t{start}\t{raw_hash}\n".encode("utf-8"))
            selected.append((rec["record_id"], start, raw_hash, norm))
            if len(selected) == N_CAL:
                return selected, sha256_bytes(bytes(canonical))
    return selected, sha256_bytes(bytes(canonical))


def prepare_development(records: list[dict], root: Path, win_n: int, hop_n: int, gate: bool):
    xs, record_indices, labels, rejected = [], [], [], 0
    for index, rec in enumerate(records):
        pieces = [item[2] for item in windows_with_starts(root / rec["relative_path"], win_n, hop_n, gate)]
        labels.append(int(rec["mapped_label"]))
        if not pieces:
            rejected += 1
            continue
        xs.extend(pieces)
        record_indices.extend([index] * len(pieces))
    return (np.stack(xs) if xs else np.empty((0, win_n), np.float32),
            np.asarray(record_indices, dtype=np.int64), np.asarray(labels, dtype=np.int64), rejected)


def state_weight(state, key: str) -> np.ndarray:
    return state[key].detach().cpu().numpy().astype(np.float64)


def calculate_scales(model: FrozenNet, calibration: list[tuple], win_n: int) -> list[float]:
    maxima = np.zeros(5, dtype=np.float64)
    modules = model.features
    for offset in range(0, len(calibration), BATCH):
        a = np.stack([x[3] for x in calibration[offset : offset + BATCH]])[:, None, :]
        x = torch.from_numpy(a.astype(np.float32))
        maxima[0] = max(maxima[0], float(x.abs().max()))
        h = x
        for j, conv_index in enumerate((0, 2, 4), start=1):
            h = modules[conv_index](h)
            h = modules[conv_index + 1](h)
            maxima[j] = max(maxima[j], float(h.abs().max()))
        pooled = h.mean(dim=-1)
        maxima[4] = max(maxima[4], float(pooled.abs().max()))
    return [float(v / 127.0) if v > 0 else (1.0 / 127.0) for v in maxima]


def quantized_parameters(model: FrozenNet, activation_scales: list[float]):
    state = model.state_dict()
    layers = []
    input_scales = [activation_scales[0], activation_scales[1], activation_scales[2], activation_scales[3]]
    output_scales = [activation_scales[1], activation_scales[2], activation_scales[3]]
    for index in range(3):
        wk, bk = f"features.{2*index}.weight", f"features.{2*index}.bias"
        w = state_weight(state, wk)
        flat = w.reshape(w.shape[0], -1)
        maxabs = np.max(np.abs(flat), axis=1)
        sw = np.where(maxabs > 0, maxabs / 127.0, 1.0 / 127.0)
        qw = np.clip(round_away_np(w / sw[:, None, None]), -127, 127).astype(np.int8)
        b = state_weight(state, bk)
        sb = input_scales[index] * sw
        qb = round_away_np(b / sb).astype(np.int64)
        layers.append({"qw": qw, "qb": qb, "sw": sw, "s_in": input_scales[index], "s_out": output_scales[index]})
    w = state_weight(state, "head.weight")
    maxabs = np.max(np.abs(w.reshape(w.shape[0], -1)), axis=1)
    sw = np.where(maxabs > 0, maxabs / 127.0, 1.0 / 127.0)
    qw = np.clip(round_away_np(w / sw[:, None]), -127, 127).astype(np.int8)
    b = state_weight(state, "head.bias")
    qb = round_away_np(b / (activation_scales[4] * sw)).astype(np.int64)
    layers.append({"qw": qw, "qb": qb, "sw": sw, "s_in": activation_scales[4], "s_out": None})
    return layers


@torch.inference_mode()
def run_int8_windows(x_np: np.ndarray, layers, activation_scales: list[float]) -> np.ndarray:
    if len(x_np) == 0:
        return np.empty(0, dtype=np.float64)
    output = []
    for start in range(0, len(x_np), BATCH):
        x = torch.from_numpy(quantize_np(x_np[start : start + BATCH], activation_scales[0])[:, None, :].astype(np.float64))
        for index in range(3):
            layer = layers[index]
            qw = torch.from_numpy(layer["qw"].astype(np.float64))
            qb = torch.from_numpy(layer["qb"].astype(np.float64))
            # Integer-valued float64 convolution exactly emulates int8 x int8 -> int32
            # accumulation for these bounded layer shapes; each accumulator is <2^53.
            acc = F.conv1d(x, qw, qb, stride=2)
            ratio = (layer["s_in"] * layer["sw"]) / layer["s_out"]
            q = round_away_torch(acc * torch.as_tensor(ratio[None, :, None], dtype=torch.float64))
            q = torch.clamp(q, 0, 127)  # ReLU then symmetric int8 saturation.
            x = q
        # Quantized global average pooling uses the frozen activation scale after pooling.
        pooled_q = x.mean(dim=-1)
        pooled_ratio = activation_scales[3] / activation_scales[4]
        pooled_q = torch.clamp(round_away_torch(pooled_q * pooled_ratio), -127, 127)
        layer = layers[3]
        acc = F.linear(pooled_q, torch.from_numpy(layer["qw"].astype(np.float64)), torch.from_numpy(layer["qb"].astype(np.float64)))
        logits = acc * (layer["s_in"] * layer["sw"][0])
        scores = torch.sigmoid(logits)
        output.append(scores.cpu().numpy().reshape(-1))
    return np.concatenate(output)


def evaluate_records(records, x, record_indices, labels, rejected, layers, scales):
    window_scores = run_int8_windows(x, layers, scales)
    sums = np.zeros(len(labels), dtype=np.float64)
    counts = np.zeros(len(labels), dtype=np.int64)
    np.add.at(sums, record_indices, window_scores)
    np.add.at(counts, record_indices, 1)
    scores = np.full(len(labels), -1.0, dtype=np.float64)
    scorable = counts > 0
    scores[scorable] = sums[scorable] / counts[scorable]
    common_ap = float(average_precision_score(labels, scores, pos_label=1))
    scorable_ap = float(average_precision_score(labels[scorable], scores[scorable], pos_label=1))
    return scores, scorable, common_ap, scorable_ap, int(rejected), int(counts.sum())


def main() -> None:
    torch.set_num_threads(1)
    manifest = yaml.safe_load(TRAIN_MANIFEST.read_text(encoding="utf-8"))
    split = yaml.safe_load(TRAIN_SPLIT.read_text(encoding="utf-8"))
    if manifest["view_scope"] != ["train", "development"] or split["view_scope"] != ["train", "development"]:
        raise RuntimeError("INT8 conversion requires strict train/development-only view")
    if manifest["parent_manifest_sha256"] != split["parent_manifest_sha256"] or manifest["parent_split_sha256"] != split["parent_split_sha256"]:
        raise RuntimeError("Filtered input provenance mismatch")
    index = {r["record_id"]: r for r in manifest["records"]}
    assignments = split["assignments"]
    if set(index) != {x["record_id"] for x in assignments} or any(x["split"] not in {"train", "development"} for x in assignments):
        raise RuntimeError("Filtered membership/scope validation failed")
    train = [index[x["record_id"]] for x in assignments if x["split"] == "train"]
    dev = [index[x["record_id"]] for x in assignments if x["split"] == "development"]
    raw_root = Path(manifest["record_root"])
    OUT.mkdir(parents=True, exist_ok=True)
    predictions_dir = ROOT / "results/raw/p6_int8_development_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for window_s, hop_s, preprocessing, model_name, run_name in CONFIGS:
        run = RUNS / run_name
        meta = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
        if meta["locked_test_accessed"] or meta["external_data_accessed"] or meta["input_view_scope"] != ["train", "development"]:
            raise RuntimeError(f"Unsafe candidate metadata: {run_name}")
        quality = preprocessing == "robust_norm_quality_gate"
        win_n, hop_n = int(window_s * 300), int(hop_s * 300)
        calibration, calibration_hash = make_calibration(train, raw_root, win_n, hop_n, quality)
        if len(calibration) != N_CAL:
            raise RuntimeError(f"Expected {N_CAL} calibration windows, got {len(calibration)} for {run_name}")
        net = FrozenNet(model_name)
        net.load_state_dict(torch.load(run / "best_checkpoint.pt", map_location="cpu", weights_only=True))
        weight_hash = sha256_file(run / "best_checkpoint.pt")
        scales = calculate_scales(net, calibration, win_n)
        layers = quantized_parameters(net, scales)
        dev_x, dev_ridx, dev_labels, dev_rejected = prepare_development(dev, raw_root, win_n, hop_n, quality)
        scores, scorable, common_ap, scorable_ap, rejected, dev_windows = evaluate_records(dev, dev_x, dev_ridx, dev_labels, dev_rejected, layers, scales)
        config_id = f"W{window_s}_H{hop_s:g}_{preprocessing}_{model_name}_INT8"
        pred_path = predictions_dir / f"{config_id}.csv"
        with pred_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["record_id", "mapped_label", "score_int8", "quality_flag"])
            for rec, label, score, ok in zip(dev, dev_labels, scores, scorable):
                writer.writerow([rec["record_id"], int(label), f"{score:.17g}" if ok else "", "scorable" if ok else "rejected"])

        arrays = {}
        layer_rows = []
        for i, layer in enumerate(layers):
            arrays[f"w{i}"] = layer["qw"]
            arrays[f"b{i}"] = layer["qb"].astype(np.int32)
            layer_rows.append({"name": f"layer_{i}", "weight_axis": 0, "weight_scales": layer["sw"].tolist(),
                               "input_activation_scale": layer["s_in"], "output_activation_scale": layer["s_out"],
                               "zero_point": 0, "qrange": [-127, 127], "rounding": "round_away_from_zero",
                               "saturation": "after_rounding; ReLU outputs clamped [0,127]" if i < 3 else "after_rounding"})
        param_path = OUT / f"{config_id}.npz"
        np.savez_compressed(param_path, **arrays)
        manifest_path = OUT / f"{config_id}.json"
        qmanifest = {
            "schema_version": "1.0", "config_id": config_id, "status": "calibrated_and_development_evaluated",
            "target_role": "INT8_primary_deployment_candidate", "input_view_scope": ["train", "development"],
            "locked_test_accessed": False, "external_data_accessed": False,
            "train_dev_manifest_sha256": sha256_file(TRAIN_MANIFEST), "train_dev_split_sha256": sha256_file(TRAIN_SPLIT),
            "parent_manifest_sha256": manifest["parent_manifest_sha256"], "parent_split_sha256": manifest["parent_split_sha256"],
            "weights_sha256": weight_hash, "quantized_parameters_sha256": sha256_file(param_path),
            "calibration_count": len(calibration), "calibration_selection": "train only; sort by (record_id,window_start_sample); first 1024 valid raw windows; no labels",
            "calibration_list_sha256": calibration_hash,
            "activation_scales_order": ["input_window", "conv1_relu", "conv2_relu", "conv3_relu", "global_average_pool"],
            "activation_scales": scales, "layers": layer_rows,
            "arithmetic": {"weight_quantization": "symmetric per output channel, zero_point=0, qrange[-127,127], scale=max_abs/127 or 1/127",
                           "activation_quantization": "symmetric per tensor, zero_point=0, qrange[-127,127], scale=max_abs/127 or 1/127",
                           "accumulator": "int32; int8-by-int8 MAC; bias rounded in input_scale*weight_scale units",
                           "requantization": "round_away_from_zero(accumulator*input_scale*weight_scale/output_scale), then saturation; ReLU conv activations clamp to [0,127]",
                           "pooling": "mean integer activation codes, requantized from conv3 scale to pooled activation scale",
                           "score": "float64 dequantized final logit followed by logistic sigmoid in this PC reference"},
            "development": {"records": len(dev), "windows": dev_windows, "rejected_records": rejected,
                            "coverage": (len(dev) - rejected) / len(dev),
                            "common_support_penalized_auprc": common_ap, "scorable_only_auprc": scorable_ap,
                            "predictions_csv": str(pred_path.relative_to(ROOT)).replace("\\", "/"),
                            "predictions_sha256": sha256_file(pred_path)},
            "source_run_metadata_sha256": sha256_file(run / "run_metadata.json"),
            "conversion_script_sha256": sha256_file(Path(__file__)),
            "trainer_sha256": sha256_file(ROOT / "scripts/train_cinc2017_candidate.py"),
        }
        manifest_path.write_text(json.dumps(qmanifest, indent=2), encoding="utf-8")
        rows.append({"config_id": config_id, "window_s": window_s, "hop_s": hop_s, "preprocessing": preprocessing,
                     "model": model_name, "best_epoch_fp32": meta["best_epoch"],
                     "development_common_support_penalized_auprc_int8": common_ap,
                     "development_scorable_only_auprc_int8": scorable_ap,
                     "development_rejected_records": rejected, "development_coverage": (len(dev) - rejected) / len(dev),
                     "development_windows": dev_windows, "calibration_count": len(calibration),
                     "calibration_list_sha256": calibration_hash, "weights_sha256": weight_hash,
                     "quantized_manifest_sha256": sha256_file(manifest_path),
                     "locked_test_accessed": "false", "external_data_accessed": "false"})
        print(f"config={config_id} AP_common={common_ap:.9f} AP_scorable={scorable_ap:.9f} rejected={rejected}", flush=True)

    csv_path = ROOT / "results/raw/int8_design_space_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"status": "development_only_completed", "rows": len(rows), "locked_test_accessed": False,
                      "external_data_accessed": False, "csv": str(csv_path), "csv_sha256": sha256_file(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
