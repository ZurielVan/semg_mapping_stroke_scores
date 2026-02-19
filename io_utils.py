from __future__ import annotations

import io
import os
import re
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch


def _load_npy_or_pt(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".pt") or path.endswith(".pth"):
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        if torch.is_tensor(obj):
            return obj.detach().cpu().numpy()
        if isinstance(obj, np.ndarray):
            return obj
        raise ValueError(f"Unsupported torch content in {path}")
    raise ValueError(f"Unsupported binary format: {path}")


def read_lvm(path: str, encoding: str = "utf-8") -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Best-effort LabVIEW Measurement (.lvm) parser.

    Returns:
      data: np.ndarray shape (T, D) float64
      channel_names: optional list[str] length D

    Notes:
      - LVM formats vary; this parser handles common cases:
          * header terminated by "***End_of_Header***"
          * a column-name line containing "X_Value" or "Time"
          * tab-delimited numeric data
      - Some exports (like your sample) contain *no header* and start directly with numeric rows,
        sometimes preceded by a leading tab which creates an all-NaN first column. We drop all-NaN columns.
    """
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        lines = f.read().splitlines()

    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("***End_of_Header***"):
            start = i + 1
            break

    header_line = None
    if start is None:
        # try to find a column-name line
        for i, line in enumerate(lines[:200]):
            if ("X_Value" in line) or (line.strip().lower().startswith("time")):
                header_line = line
                start = i + 1
                break

    if start is None:
        # fallback: first line with at least 2 float-like tokens
        for i, line in enumerate(lines):
            parts = re.split(r"[\t,; ]+", line.strip())
            if len(parts) < 2:
                continue
            try:
                float(parts[0]); float(parts[1])
                start = i
                break
            except Exception:
                continue

    if start is None:
        raise ValueError(f"Cannot locate numeric data start in LVM: {path}")

    # If header_line not found yet, attempt to use the immediate previous line as header (if non-numeric)
    if header_line is None and start > 0:
        prev = lines[start - 1].strip()
        parts = re.split(r"[\t,; ]+", prev)
        if len(parts) >= 2:
            is_numeric = True
            for p in parts[:2]:
                try:
                    float(p)
                except Exception:
                    is_numeric = False
                    break
            if not is_numeric:
                header_line = prev

    channel_names = None
    if header_line is not None:
        toks = re.split(r"[\t,; ]+", header_line.strip())
        channel_names = [t for t in toks if len(t) > 0]

    data_str = "\n".join(lines[start:])

    # 1) try tab-delimited first (common for LVM)
    arr = np.genfromtxt(io.StringIO(data_str), delimiter="\t")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # 2) if it collapses into 1 column, fallback to whitespace parsing
    if arr.shape[1] == 1:
        arr2 = np.genfromtxt(io.StringIO(data_str), delimiter=None)
        if arr2.ndim == 1:
            arr2 = arr2.reshape(-1, 1)
        if arr2.shape[1] > 1:
            arr = arr2

    # drop all-nan columns (e.g. leading tab)
    if np.isnan(arr).any():
        keep = ~np.all(np.isnan(arr), axis=0)
        arr = arr[:, keep]
        if channel_names is not None and len(channel_names) == keep.shape[0]:
            channel_names = [n for n, k in zip(channel_names, keep.tolist()) if k]

    return arr, channel_names

def extract_emg(
    data: np.ndarray,
    emg_cols: Optional[List[int]] = None,
    time_col: Optional[int] = 0,
    expected_emg_ch: int = 4,
) -> np.ndarray:
    """
    Convert raw data (T,D) into EMG array (C,T) float32.

    This function is intentionally *defensive* because real LVM exports vary.
    It supports:
      - explicit emg_cols (recommended once you confirm channel meanings)
      - heuristic selection when emg_cols is None

    Heuristic (when emg_cols is None):
      1) optionally drop a time column *only if it actually looks like time* (monotonic increasing).
      2) drop any all-NaN or all-zero columns (common: trailing trigger channel or empty column).
      3) if remaining columns > expected_emg_ch: take the first expected_emg_ch columns.
         (In your dataset, EMG channels are typically placed before auxiliary/unused channels.)
      4) if remaining columns == expected_emg_ch: take all.
      5) otherwise: take whatever remains (and you should set emg_cols explicitly).

    Returns:
      emg: (C,T)
    """
    if data.ndim != 2:
        raise ValueError(f"extract_emg expects (T,D), got {data.shape}")

    T, D = data.shape

    if emg_cols is None:
        cols = list(range(D))

        # 1) drop time column only if it looks like time
        if time_col is not None and 0 <= int(time_col) < D:
            tc = data[:, int(time_col)]
            # time should be mostly increasing with low jitter
            diff = np.diff(tc)
            inc_ratio = float(np.mean(diff > 0)) if diff.size > 0 else 0.0
            jitter = float(np.std(diff) / (np.mean(diff) + 1e-8)) if diff.size > 0 else 1e9
            looks_like_time = (inc_ratio > 0.99) and (jitter < 0.2)
            if looks_like_time:
                cols = [c for c in cols if c != int(time_col)]

        # 2) drop all-zero / all-nan columns
        kept = []
        for c in cols:
            col = data[:, c]
            if np.all(np.isnan(col)):
                continue
            if np.allclose(col, 0.0):
                # common: trailing digital/trigger channel stored as zeros
                continue
            kept.append(c)
        cols = kept

        # 3) choose emg columns
        if len(cols) == expected_emg_ch:
            emg_cols = cols
        elif len(cols) > expected_emg_ch:
            emg_cols = cols[:expected_emg_ch]
        else:
            emg_cols = cols  # fallback

    emg = data[:, emg_cols].astype(np.float32)  # (T,C)
    emg = emg.T  # (C,T)
    return emg

def load_emg_timeseries(
    path: str,
    emg_cols: Optional[List[int]] = None,
    time_col: Optional[int] = 0,
    expected_emg_ch: int = 4,
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Unified loader for .npy/.pt and .lvm.

    Returns:
      emg: np.ndarray shape (C,T)
      channel_names: optional list[str] from source (raw), not necessarily EMG-only
    """
    path = str(path)
    if path.endswith(".lvm"):
        raw, names = read_lvm(path)
        emg = extract_emg(raw, emg_cols=emg_cols, time_col=time_col, expected_emg_ch=expected_emg_ch)
        return emg, names

    arr = _load_npy_or_pt(path)
    # expected arr shape could be (C,T) or (T,C)
    if arr.ndim != 2:
        raise ValueError(f"Expect 2D array from {path}, got {arr.shape}")

    # if second dim is small (<=16) we assume (T,C); else if first dim small (<=16) assume (C,T)
    if arr.shape[0] <= 16 and arr.shape[1] > arr.shape[0]:
        emg = arr.astype(np.float32)  # (C,T)
    elif arr.shape[1] <= 16 and arr.shape[0] > arr.shape[1]:
        emg = arr.T.astype(np.float32)  # (C,T)
    else:
        # ambiguous, assume (C,T)
        emg = arr.astype(np.float32)
    return emg, None
