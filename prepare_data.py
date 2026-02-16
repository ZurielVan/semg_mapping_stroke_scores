from __future__ import annotations

import argparse
import ast
import os
import re
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .io_utils import read_lvm, extract_emg

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _norm_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _progress(iterable, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, ncols=100)


def load_ensembled_info_table(path: str) -> pd.DataFrame:
    """
    Read ensembledInfo.csv which uses the first 3 rows as multi-level headers:
      row0: time axis (Info/pre/post/3mf/...)
      row1: category (basicInfo/FMA/MAS/ARAT/WMFT/FIM/...)
      row2: field name (Suject code, W/H, S/E, ...)
    """
    raw = pd.read_csv(path)
    level_time = raw.iloc[0].tolist()
    level_cat = raw.iloc[1].tolist()
    level_name = raw.iloc[2].tolist()
    cols = pd.MultiIndex.from_arrays([level_time, level_cat, level_name])
    df = raw.iloc[3:].copy()
    df.columns = cols
    df.reset_index(drop=True, inplace=True)
    return df


def build_subject_map(df_ens: pd.DataFrame) -> pd.DataFrame:
    subj_code = df_ens[("Info", "basicInfo", "Suject code")].astype(str)
    subj_name = df_ens[("Info", "basicInfo", "Suject name")].astype(str)
    out = pd.DataFrame({
        "subject_id": subj_code,
        "subject_name": subj_name,
        "subject_name_norm": subj_name.map(_norm_name),
    })
    return out


def build_labels(df_ens: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    stages = ["pre", "post", "3mf"]
    rows = []
    for _, row in _progress(df_ens.iterrows(), total=len(df_ens), desc="Build labels"):
        subject_id = str(row[("Info", "basicInfo", "Suject code")])
        subject_name = str(row[("Info", "basicInfo", "Suject name")])
        for st in stages:
            # main labels
            fma_wh = pd.to_numeric(row.get((st, "FMA", "W/H"), np.nan), errors="coerce")
            fma_se = pd.to_numeric(row.get((st, "FMA", "S/E"), np.nan), errors="coerce")
            fma_ue = pd.to_numeric(row.get((st, "FMA", "UE"), np.nan), errors="coerce")

            # aux labels (optional)
            mas_finger = pd.to_numeric(row.get((st, "MAS", "Finger"), np.nan), errors="coerce")
            mas_wrist = pd.to_numeric(row.get((st, "MAS", "Wrist"), np.nan), errors="coerce")
            mas_elbow = pd.to_numeric(row.get((st, "MAS", "Elbow"), np.nan), errors="coerce")

            arat_full = pd.to_numeric(row.get((st, "ARAT", "Full"), np.nan), errors="coerce")
            arat_grasp = pd.to_numeric(row.get((st, "ARAT", "Grasp"), np.nan), errors="coerce")
            arat_grip = pd.to_numeric(row.get((st, "ARAT", "Grip"), np.nan), errors="coerce")
            arat_pinch = pd.to_numeric(row.get((st, "ARAT", "Pinch"), np.nan), errors="coerce")
            arat_gross = pd.to_numeric(row.get((st, "ARAT", "Gross"), np.nan), errors="coerce")

            wmft_score = pd.to_numeric(row.get((st, "WMFT", "Score"), np.nan), errors="coerce")
            wmft_strength = pd.to_numeric(row.get((st, "WMFT", "Strength"), np.nan), errors="coerce")
            wmft_time = pd.to_numeric(row.get((st, "WMFT", "Time"), np.nan), errors="coerce")

            fim_full = pd.to_numeric(row.get((st, "FIM", "Full"), np.nan), errors="coerce")

            session_id = f"{subject_id}_{st}"

            rows.append({
                "session_id": session_id,
                "subject_id": subject_id,
                "subject_name": subject_name,
                "stage": st,
                "FMA_WH": fma_wh,
                "FMA_SE": fma_se,
                "FMA_UE": fma_ue,
                "MAS_Finger": mas_finger,
                "MAS_Wrist": mas_wrist,
                "MAS_Elbow": mas_elbow,
                "ARAT_Full": arat_full,
                "ARAT_Grasp": arat_grasp,
                "ARAT_Grip": arat_grip,
                "ARAT_Pinch": arat_pinch,
                "ARAT_Gross": arat_gross,
                "WMFT_Score": wmft_score,
                "WMFT_Strength": wmft_strength,
                "WMFT_Time": wmft_time,
                "FIM_Full": fim_full,
            })

    labels_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    labels_df.to_csv(out_csv, index=False)
    return labels_df


def _parse_shape(s: str) -> Optional[Tuple[int, int]]:
    if not isinstance(s, str) or s.strip() == "":
        return None
    try:
        tup = ast.literal_eval(s)
        if isinstance(tup, tuple) and len(tup) == 2:
            return int(tup[0]), int(tup[1])
    except Exception:
        return None
    return None


def _axis_trial_from_filename(fn: str) -> Tuple[str, str]:
    fn = str(fn)
    if fn.lower().startswith("horizontaltask"):
        axis = "y"
    elif fn.lower().startswith("verticaltask"):
        axis = "z"
    else:
        # fallback: keep unknown axis
        axis = "unknown"

    m = re.search(r"(\d+)", fn)
    trial_id = m.group(1) if m else "0"
    return axis, trial_id


def resolve_file_path(data_root: str, subject_name: str, time_axis: str, file_name: str) -> str:
    """
    Try multiple conventions. Returns the first existing path.
    If none found, returns a best-guess path (may not exist).
    """
    data_root = str(data_root)
    subject_name = str(subject_name)
    time_axis = str(time_axis)
    file_name = str(file_name)

    cand = os.path.join(data_root, subject_name, time_axis, file_name)
    if os.path.exists(cand):
        return cand

    # case-insensitive time axis folder
    subj_dir = os.path.join(data_root, subject_name)
    if os.path.isdir(subj_dir):
        # search under subject dir for matching file
        for root, _, files in os.walk(subj_dir):
            for f in files:
                if f == file_name:
                    # prefer paths whose root contains time_axis token
                    if time_axis.lower() in root.lower():
                        return os.path.join(root, f)

    # global search (small dataset -> acceptable)
    for root, _, files in os.walk(data_root):
        for f in files:
            if f == file_name and subject_name.lower() in root.lower():
                return os.path.join(root, f)

    return cand


def build_manifest(
    data_info_csv: str,
    subject_map_df: pd.DataFrame,
    data_root: str,
    fs: float,
    out_csv: str,
    convert_to_npy: bool,
    out_npy_dir: str,
    emg_cols: Optional[List[int]],
    time_col: Optional[int],
    expected_emg_ch: int,
    drop_unmatched_subjects: bool,
) -> pd.DataFrame:
    info = pd.read_csv(data_info_csv)
    name2id = {row["subject_name_norm"]: row["subject_id"] for _, row in subject_map_df.iterrows()}
    dropped_unmatched = 0

    rows = []
    for _, r in _progress(info.iterrows(), total=len(info), desc="Build manifest"):
        subject_name = str(r["SubjectName"])
        subject_id = name2id.get(_norm_name(subject_name), None)
        if subject_id is None:
            if drop_unmatched_subjects:
                dropped_unmatched += 1
                continue
            # keep original name as subject_id if mapping missing
            subject_id = subject_name

        time_axis = str(r["TimeAxis"])
        stage = time_axis
        if stage.upper() == "3MFU":
            stage = "3mf"
        stage = stage.lower()

        axis, trial_id = _axis_trial_from_filename(r["FileName"])
        session_id = f"{subject_id}_{stage}"

        src_path = resolve_file_path(data_root, subject_name, time_axis, r["FileName"])

        if convert_to_npy:
            os.makedirs(out_npy_dir, exist_ok=True)
            # read and extract EMG, then save as (C,T)
            if not os.path.exists(src_path):
                raise FileNotFoundError(f"Cannot find raw file: {src_path}")
            raw, ch_names = read_lvm(src_path)
            emg = extract_emg(raw, emg_cols=emg_cols, time_col=time_col, expected_emg_ch=expected_emg_ch)  # (C,T)
            out_name = f"{session_id}_{axis}_trial{trial_id}.npy"
            out_path = os.path.join(out_npy_dir, out_name)
            np.save(out_path, emg.astype(np.float32))
            path = out_path
        else:
            path = src_path

        # raw shape/duration (support both old and new dataInfo formats)
        raw_T = None
        raw_C = None
        duration_s = None

        # New format: explicit columns
        if ("T" in info.columns) and ("C" in info.columns):
            try:
                raw_T = int(r.get("T"))
                raw_C = int(r.get("C"))
            except Exception:
                raw_T, raw_C = None, None
        else:
            shape = _parse_shape(str(r.get("Shape (C,T)", "")))
            if shape is not None:
                raw_T, raw_C = int(shape[0]), int(shape[1])

        if "DurationTime (s)" in info.columns:
            try:
                duration_s = float(r.get("DurationTime (s)"))
            except Exception:
                duration_s = None

        rows.append({
            "subject_id": subject_id,
            "subject_name": subject_name,
            "stage": stage,
            "session_id": session_id,
            "axis": axis,
            "trial_id": trial_id,
            "path": path,
            "fs": fs,
            "raw_T": raw_T,
            "raw_C": raw_C,
            "duration_s": duration_s,
        })


    manifest_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    manifest_df.to_csv(out_csv, index=False)
    if drop_unmatched_subjects and dropped_unmatched > 0:
        print(f"Dropped unmatched rows: {dropped_unmatched}")
    return manifest_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensembled_info_csv", type=str, required=True)
    ap.add_argument("--data_info_csv", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--fs", type=float, default=1200.0, help="Sampling rate (Hz) used for duration sanity checks / metadata")

    ap.add_argument("--convert_to_npy", action="store_true", help="Convert .lvm to .npy for faster training")
    ap.add_argument("--emg_cols", type=str, default="", help="comma-separated indices for EMG columns in raw (T,D). default auto")
    ap.add_argument("--time_col", type=int, default=0)
    ap.add_argument("--expected_emg_ch", type=int, default=4)

    ap.add_argument("--drop_unmatched_subjects", action="store_true", help="Skip rows in dataInfo.csv whose SubjectName cannot be mapped to ensembledInfo subject list")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    emg_cols = None
    if args.emg_cols.strip() != "":
        emg_cols = [int(x) for x in args.emg_cols.split(",")]

    df_ens = load_ensembled_info_table(args.ensembled_info_csv)
    subj_map = build_subject_map(df_ens)
    subj_map.to_csv(os.path.join(args.out_dir, "subject_map.csv"), index=False)

    labels_csv = os.path.join(args.out_dir, "labels.csv")
    labels_df = build_labels(df_ens, labels_csv)

    manifest_csv = os.path.join(args.out_dir, "manifest.csv")
    npy_dir = os.path.join(args.out_dir, "npy") if args.convert_to_npy else ""
    manifest_df = build_manifest(
        data_info_csv=args.data_info_csv,
        subject_map_df=subj_map,
        data_root=args.data_root,
        fs=args.fs,
        out_csv=manifest_csv,
        convert_to_npy=args.convert_to_npy,
        out_npy_dir=npy_dir,
        emg_cols=emg_cols,
        time_col=args.time_col,
        expected_emg_ch=args.expected_emg_ch,
        drop_unmatched_subjects=args.drop_unmatched_subjects,
    )

    print("Saved:")
    print("  ", manifest_csv)
    print("  ", labels_csv)
    print(f"Rows: labels={len(labels_df)}, manifest={len(manifest_df)}")
    if args.convert_to_npy:
        print("  ", npy_dir)


if __name__ == "__main__":
    main()
