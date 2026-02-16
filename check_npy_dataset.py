from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    raise ImportError(
        "matplotlib is required for visualization. Install with: pip install matplotlib"
    ) from e

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _progress(iterable, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, ncols=100)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_cxt(arr: np.ndarray, expected_channels: int) -> tuple[np.ndarray, str]:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr.shape}")

    r, c = arr.shape
    if r == expected_channels and c != expected_channels:
        return arr.astype(np.float32, copy=False), "C,T"
    if c == expected_channels and r != expected_channels:
        return arr.T.astype(np.float32, copy=False), "T,C->C,T"

    # Fall back to heuristic when dimensions are ambiguous.
    if r <= 16 and c > r:
        return arr.astype(np.float32, copy=False), "heuristic C,T"
    if c <= 16 and r > c:
        return arr.T.astype(np.float32, copy=False), "heuristic T,C->C,T"

    return arr.astype(np.float32, copy=False), "ambiguous treated as C,T"


def _downsample_for_plot(x: np.ndarray, max_points: int) -> np.ndarray:
    if x.shape[-1] <= max_points:
        return x
    stride = int(math.ceil(x.shape[-1] / max_points))
    return x[..., ::stride]


def inspect_dataset(
    manifest_csv: str,
    expected_channels: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    manifest = pd.read_csv(manifest_csv)
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    channel_rms_rows: list[dict[str, Any]] = []

    for _, row in _progress(manifest.iterrows(), total=len(manifest), desc="Inspect npy"):
        rec: dict[str, Any] = {
            "subject_id": str(row.get("subject_id", "")),
            "subject_name": str(row.get("subject_name", "")),
            "stage": str(row.get("stage", "")),
            "session_id": str(row.get("session_id", "")),
            "axis": str(row.get("axis", "")),
            "trial_id": str(row.get("trial_id", "")),
            "path": str(row.get("path", "")),
            "raw_T_manifest": _safe_float(row.get("raw_T")),
            "raw_C_manifest": _safe_float(row.get("raw_C")),
            "duration_s_manifest": _safe_float(row.get("duration_s")),
        }
        p = rec["path"]

        rec["exists"] = os.path.exists(p)
        rec["load_ok"] = False
        rec["shape_0"] = np.nan
        rec["shape_1"] = np.nan
        rec["channels"] = np.nan
        rec["timepoints"] = np.nan
        rec["orientation"] = ""
        rec["dtype"] = ""
        rec["has_nan"] = False
        rec["has_inf"] = False
        rec["min"] = np.nan
        rec["max"] = np.nan
        rec["mean"] = np.nan
        rec["std"] = np.nan
        rec["rms_global"] = np.nan
        rec["error"] = ""

        if not rec["exists"]:
            rec["error"] = "missing_file"
            records.append(rec)
            errors.append(f"Missing file: {p}")
            continue

        try:
            arr = np.load(p)
            rec["dtype"] = str(arr.dtype)
            rec["shape_0"] = int(arr.shape[0]) if arr.ndim >= 1 else np.nan
            rec["shape_1"] = int(arr.shape[1]) if arr.ndim >= 2 else np.nan
            x, orient = _to_cxt(arr, expected_channels=expected_channels)
            rec["orientation"] = orient
            rec["channels"] = int(x.shape[0])
            rec["timepoints"] = int(x.shape[1])

            rec["has_nan"] = bool(np.isnan(x).any())
            rec["has_inf"] = bool(np.isinf(x).any())
            rec["min"] = float(np.nanmin(x))
            rec["max"] = float(np.nanmax(x))
            rec["mean"] = float(np.nanmean(x))
            rec["std"] = float(np.nanstd(x))
            rec["rms_global"] = float(np.sqrt(np.nanmean(np.square(x))))
            rec["load_ok"] = True

            rms_ch = np.sqrt(np.nanmean(np.square(x), axis=1))
            for ch_idx, ch_rms in enumerate(rms_ch):
                channel_rms_rows.append(
                    {
                        "path": p,
                        "subject_id": rec["subject_id"],
                        "stage": rec["stage"],
                        "axis": rec["axis"],
                        "trial_id": rec["trial_id"],
                        "channel_idx": int(ch_idx),
                        "rms": float(ch_rms),
                    }
                )

        except Exception as e:
            rec["error"] = repr(e)
            errors.append(f"Load failure: {p} -> {repr(e)}")

        records.append(rec)

    qc_df = pd.DataFrame(records)
    return qc_df, channel_rms_rows, errors


def plot_length_hist(valid_df: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(valid_df["timepoints"].astype(float), bins=30, edgecolor="black")
    plt.title("Distribution of Timepoints per NPY")
    plt.xlabel("Timepoints")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_stage_axis_counts(valid_df: pd.DataFrame, out_png: str) -> None:
    ct = pd.crosstab(valid_df["stage"], valid_df["axis"]).sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(ct.values, aspect="auto")
    ax.set_title("File Count by Stage and Axis")
    ax.set_xticks(np.arange(ct.shape[1]), labels=ct.columns.tolist())
    ax.set_yticks(np.arange(ct.shape[0]), labels=ct.index.tolist())
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, str(int(ct.values[i, j])), ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_subject_stage_heatmap(valid_df: pd.DataFrame, out_png: str) -> None:
    pivot = (
        valid_df.groupby(["subject_id", "stage"])["path"]
        .count()
        .unstack(fill_value=0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title("File Count by Subject and Stage")
    ax.set_xticks(np.arange(pivot.shape[1]), labels=pivot.columns.tolist())
    ax.set_yticks(np.arange(pivot.shape[0]), labels=pivot.index.tolist())
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_channel_rms(channel_rms_df: pd.DataFrame, out_png: str) -> None:
    if len(channel_rms_df) == 0:
        return
    grouped = [
        channel_rms_df.loc[channel_rms_df["channel_idx"] == ch, "rms"].values
        for ch in sorted(channel_rms_df["channel_idx"].unique().tolist())
    ]
    labels = [f"ch{ch}" for ch in sorted(channel_rms_df["channel_idx"].unique().tolist())]

    plt.figure(figsize=(8, 5))
    plt.boxplot(grouped, tick_labels=labels, showfliers=False)
    plt.title("RMS Distribution per Channel")
    plt.xlabel("Channel")
    plt.ylabel("RMS")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_sample_waveforms(
    valid_df: pd.DataFrame,
    expected_channels: int,
    n_samples: int,
    max_points: int,
    seed: int,
    out_png: str,
) -> None:
    if len(valid_df) == 0 or n_samples <= 0:
        return

    rng = np.random.RandomState(seed)
    pick_n = min(n_samples, len(valid_df))
    pick_idx = rng.choice(np.arange(len(valid_df)), size=pick_n, replace=False)
    picked = valid_df.iloc[pick_idx].reset_index(drop=True)

    ncols = 3
    nrows = int(math.ceil(pick_n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.6 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, (_, row) in enumerate(picked.iterrows()):
        ax = axes[i]
        arr = np.load(row["path"])
        x, _ = _to_cxt(arr, expected_channels=expected_channels)
        x = _downsample_for_plot(x, max_points=max_points)
        t = np.arange(x.shape[1])

        # Stack normalized channels to improve readability.
        for ch in range(x.shape[0]):
            sig = x[ch]
            std = float(np.std(sig))
            if std > 1e-8:
                sig = (sig - np.mean(sig)) / std
            ax.plot(t, sig + ch * 4.0, lw=0.8)

        ax.set_title(
            f"{row['subject_id']} {row['stage']} {row['axis']} trial{row['trial_id']}\n"
            f"C={x.shape[0]}, T={x.shape[1]}",
            fontsize=9,
        )
        ax.set_xlabel("sample index")
        ax.set_yticks([ch * 4.0 for ch in range(x.shape[0])], labels=[f"ch{ch}" for ch in range(x.shape[0])])

    for j in range(pick_n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Random NPY Waveform Samples (channel-normalized for display)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_png, dpi=150)
    plt.close()


def build_session_axis_coverage(qc_df: pd.DataFrame) -> pd.DataFrame:
    ok = qc_df[qc_df["load_ok"] == True].copy()
    if len(ok) == 0:
        return pd.DataFrame()

    coverage = (
        ok.groupby(["session_id", "subject_id", "stage", "axis"])["trial_id"]
        .nunique()
        .rename("n_trials")
        .reset_index()
    )
    return coverage


def compute_summary(qc_df: pd.DataFrame, expected_channels: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    summary["total_manifest_rows"] = int(len(qc_df))
    summary["missing_files"] = int((qc_df["exists"] == False).sum())
    summary["load_failures"] = int(((qc_df["exists"] == True) & (qc_df["load_ok"] == False)).sum())
    summary["valid_files"] = int((qc_df["load_ok"] == True).sum())
    summary["has_nan_files"] = int((qc_df["has_nan"] == True).sum())
    summary["has_inf_files"] = int((qc_df["has_inf"] == True).sum())

    valid = qc_df[qc_df["load_ok"] == True].copy()
    if len(valid) > 0:
        tp = valid["timepoints"].astype(float)
        summary["timepoints"] = {
            "min": float(np.min(tp)),
            "p5": float(np.percentile(tp, 5)),
            "median": float(np.median(tp)),
            "mean": float(np.mean(tp)),
            "p95": float(np.percentile(tp, 95)),
            "max": float(np.max(tp)),
        }
        ch_counts = valid["channels"].astype(int).value_counts().sort_index().to_dict()
        summary["channel_count_distribution"] = {str(k): int(v) for k, v in ch_counts.items()}
        summary["expected_channel_match_ratio"] = float(
            np.mean(valid["channels"].astype(int).values == int(expected_channels))
        )
        summary["subjects"] = int(valid["subject_id"].nunique())
        summary["sessions"] = int(valid["session_id"].nunique())
        summary["stages"] = sorted(valid["stage"].dropna().unique().tolist())
        summary["axes"] = sorted(valid["axis"].dropna().unique().tolist())
    else:
        summary["timepoints"] = {}
        summary["channel_count_distribution"] = {}
        summary["expected_channel_match_ratio"] = 0.0
        summary["subjects"] = 0
        summary["sessions"] = 0
        summary["stages"] = []
        summary["axes"] = []

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Check and visualize converted NPY sEMG dataset.")
    ap.add_argument("--manifest_csv", type=str, default="dataset/manifest.csv")
    ap.add_argument("--labels_csv", type=str, default="dataset/labels.csv")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--expected_channels", type=int, default=4)
    ap.add_argument("--n_waveform_samples", type=int, default=12)
    ap.add_argument("--max_points_per_waveform", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir.strip() or f"dataset_qc/qc_{timestamp}"
    _ensure_dir(out_dir)

    qc_df, channel_rms_rows, errors = inspect_dataset(
        manifest_csv=args.manifest_csv,
        expected_channels=args.expected_channels,
    )
    channel_rms_df = pd.DataFrame(channel_rms_rows)

    summary = compute_summary(qc_df, expected_channels=args.expected_channels)
    session_axis_cov_df = build_session_axis_coverage(qc_df)

    # Optional labels consistency check.
    label_cov_df = pd.DataFrame()
    if os.path.exists(args.labels_csv):
        labels_df = pd.read_csv(args.labels_csv)
        sessions_in_labels = set(labels_df["session_id"].astype(str).tolist())
        sessions_in_qc = set(qc_df["session_id"].astype(str).tolist())
        label_cov_df = pd.DataFrame(
            [
                {"check": "sessions_in_labels", "value": len(sessions_in_labels)},
                {"check": "sessions_in_qc", "value": len(sessions_in_qc)},
                {"check": "sessions_only_in_labels", "value": len(sessions_in_labels - sessions_in_qc)},
                {"check": "sessions_only_in_qc", "value": len(sessions_in_qc - sessions_in_labels)},
            ]
        )

    qc_csv = os.path.join(out_dir, "file_qc.csv")
    rms_csv = os.path.join(out_dir, "channel_rms.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    errors_txt = os.path.join(out_dir, "errors.txt")
    coverage_csv = os.path.join(out_dir, "session_axis_coverage.csv")
    label_cov_csv = os.path.join(out_dir, "label_coverage.csv")

    qc_df.to_csv(qc_csv, index=False)
    channel_rms_df.to_csv(rms_csv, index=False)
    session_axis_cov_df.to_csv(coverage_csv, index=False)
    if len(label_cov_df) > 0:
        label_cov_df.to_csv(label_cov_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(errors_txt, "w", encoding="utf-8") as f:
        if len(errors) == 0:
            f.write("No errors.\n")
        else:
            for e in errors:
                f.write(e + "\n")

    valid_df = qc_df[qc_df["load_ok"] == True].copy()
    if len(valid_df) > 0:
        plot_length_hist(valid_df, os.path.join(out_dir, "plot_length_hist.png"))
        plot_stage_axis_counts(valid_df, os.path.join(out_dir, "plot_stage_axis_counts.png"))
        plot_subject_stage_heatmap(valid_df, os.path.join(out_dir, "plot_subject_stage_heatmap.png"))
        plot_channel_rms(channel_rms_df, os.path.join(out_dir, "plot_channel_rms_box.png"))
        plot_sample_waveforms(
            valid_df,
            expected_channels=args.expected_channels,
            n_samples=args.n_waveform_samples,
            max_points=args.max_points_per_waveform,
            seed=args.seed,
            out_png=os.path.join(out_dir, "plot_waveform_samples.png"),
        )

    print(f"QC output dir: {out_dir}")
    print(f"Saved: {qc_csv}")
    print(f"Saved: {summary_json}")
    print(f"Valid files: {summary['valid_files']}/{summary['total_manifest_rows']}")
    print(f"Missing files: {summary['missing_files']}, Load failures: {summary['load_failures']}")


if __name__ == "__main__":
    main()
