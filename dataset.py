from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, OrderedDict

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .io_utils import load_emg_timeseries


@dataclass(frozen=True)
class TrialRecord:
    session_id: str
    subject_id: str
    stage: str
    axis: str          # 'y' or 'z'
    trial_id: str
    path: str


@dataclass
class SessionRecord:
    session_id: str
    subject_id: str
    stage: str
    trials_by_axis: Dict[str, List[TrialRecord]]
    labels: Dict[str, Any]


class _SequenceCache:
    """
    Simple LRU cache for sequences.
    Each DataLoader worker has its own Dataset instance -> own cache.
    """
    def __init__(self, max_items: int = 128):
        self.max_items = max_items
        self._cache = OrderedDict()

    def get(self, key: str):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: Any):
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)


def _per_window_channel_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: (C, T)
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)


def _pad_or_trim(x: torch.Tensor, target_len: int) -> torch.Tensor:
    C, T = x.shape
    if T == target_len:
        return x
    if T > target_len:
        return x[:, :target_len]
    pad_len = target_len - T
    pad = torch.zeros((C, pad_len), dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


def _num_window_starts(T: int, Tw: int, Ts: int) -> int:
    if T <= Tw:
        return 1
    n = (T - Tw) // Ts + 1
    return max(1, int(n))


def _sample_window_starts_train(T: int, Tw: int, Ts: int, K: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Sample K window starts without materializing the full start list.
    """
    n = _num_window_starts(T, Tw, Ts)
    if n == 1:
        return np.zeros((K,), dtype=np.int64)

    replace = n < K
    idx = rng.choice(n, size=K, replace=replace).astype(np.int64)
    return idx * int(Ts)


def _sample_window_starts_eval(T: int, Tw: int, Ts: int, K: int) -> np.ndarray:
    """
    Deterministic K picks over valid start-index space.
    """
    n = _num_window_starts(T, Tw, Ts)
    if n == 1:
        return np.zeros((K,), dtype=np.int64)

    if n >= K:
        idx = np.linspace(0, n - 1, num=K).round().astype(np.int64)
    else:
        idx = (np.arange(K, dtype=np.int64) % n)
    return idx * int(Ts)


class SessionBagDataset(Dataset):
    """
    Hierarchical MIL dataset.
    Each item is a SESSION bag with fixed tensor shapes:

    If return_two_views=False:
      windows:      (A=2, R, K, C=4, Tw)
      window_mask:  (2, R, K)
      trial_mask:   (2, R)
      labels: y_wh, y_se, label_mask

    If return_two_views=True (train for Mean-Teacher):
      windows_1, window_mask_1, trial_mask_1  (student view)
      windows_2, window_mask_2, trial_mask_2  (teacher view)
      labels: y_wh, y_se, label_mask

    Notes:
      - R = trials_per_axis (may be > available, then sample with replacement)
      - K = windows_per_trial sampled per (axis, trial)
      - For val/test, sampling is deterministic to reduce evaluation variance.
    """
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        subjects: List[str],
        Tw_samples: int,
        Ts_samples: int,
        windows_per_trial: int,
        trials_per_axis: int,
        mode: str,  # 'train' | 'val' | 'test'
        supervised_aug=None,
        return_two_views: bool = False,
        require_both_main_labels: bool = True,
        # raw loader config
        emg_cols: Optional[List[int]] = None,
        time_col: Optional[int] = 0,
        expected_emg_ch: int = 4,
        cache_size: int = 64,
    ):
        super().__init__()
        self.subjects = set([str(s) for s in subjects])
        self.Tw = int(Tw_samples)
        self.Ts = int(Ts_samples)
        self.K = int(windows_per_trial)
        self.R = int(trials_per_axis)
        self.mode = mode
        self.supervised_aug = supervised_aug
        self.return_two_views = bool(return_two_views)

        self.emg_cols = emg_cols
        self.time_col = time_col
        self.expected_emg_ch = expected_emg_ch

        self.cache = _SequenceCache(max_items=cache_size)

        # Build label map: session_id -> row dict
        labels_map: Dict[str, Dict[str, Any]] = {}
        for _, row in labels_df.iterrows():
            sid = str(row["session_id"])
            labels_map[sid] = row.to_dict()

        # Group trials by session_id and axis
        grouped = defaultdict(lambda: defaultdict(list))  # session_id -> axis -> [TrialRecord]
        for _, row in manifest_df.iterrows():
            subj = str(row["subject_id"])
            if subj not in self.subjects:
                continue
            sid = str(row["session_id"])
            axis = str(row["axis"])
            tr = TrialRecord(
                session_id=sid,
                subject_id=subj,
                stage=str(row["stage"]),
                axis=axis,
                trial_id=str(row["trial_id"]),
                path=str(row["path"]),
            )
            grouped[sid][axis].append(tr)

        sessions: Dict[str, SessionRecord] = {}
        for sid, axis_map in grouped.items():
            any_axis = next(iter(axis_map.keys()))
            any_trial = axis_map[any_axis][0]
            labels = labels_map.get(sid, {"session_id": sid})
            sessions[sid] = SessionRecord(
                session_id=sid,
                subject_id=any_trial.subject_id,
                stage=any_trial.stage,
                trials_by_axis={ax: axis_map.get(ax, []) for ax in ["y", "z"]},
                labels=labels,
            )

        self.sessions = list(sessions.values())
        if require_both_main_labels:
            kept = []
            for s in self.sessions:
                y_wh = pd.to_numeric(s.labels.get("FMA_WH", np.nan), errors="coerce")
                y_se = pd.to_numeric(s.labels.get("FMA_SE", np.nan), errors="coerce")
                if np.isfinite(y_wh) and np.isfinite(y_se):
                    kept.append(s)
            self.sessions = kept
        self.sessions.sort(key=lambda s: s.session_id)

    def __len__(self) -> int:
        return len(self.sessions)

    def _load_seq_cached(self, path: str) -> np.ndarray:
        cached = self.cache.get(path)
        if cached is not None:
            return cached
        emg, _ = load_emg_timeseries(
            path,
            emg_cols=self.emg_cols,
            time_col=self.time_col,
            expected_emg_ch=self.expected_emg_ch,
        )  # (C,T)
        if emg.shape[0] != self.expected_emg_ch:
            # keep running but warn by raising if severe mismatch
            # In practice you can set expected_emg_ch accordingly.
            pass
        self.cache.put(path, emg)
        return emg

    def _sample_view(self, sess: SessionRecord, rng: np.random.RandomState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A = 2
        R = self.R
        K = self.K
        C = self.expected_emg_ch
        Tw = self.Tw

        windows = torch.zeros((A, R, K, C, Tw), dtype=torch.float32)
        window_mask = torch.zeros((A, R, K), dtype=torch.bool)
        trial_mask = torch.zeros((A, R), dtype=torch.bool)

        axis_to_i = {"y": 0, "z": 1}

        for axis in ["y", "z"]:
            a_i = axis_to_i[axis]
            trials = sess.trials_by_axis.get(axis, [])

            if self.mode == "train":
                # random sample trials
                if len(trials) == 0:
                    chosen = []
                else:
                    tr_n = len(trials)
                    replace = tr_n < R
                    tr_idx = rng.choice(tr_n, size=R, replace=replace).astype(int).tolist()
                    chosen = [trials[i] for i in tr_idx]
            else:
                # deterministic for val/test
                trials_sorted = sorted(trials, key=lambda tr: tr.trial_id)
                if len(trials_sorted) >= R:
                    chosen = trials_sorted[:R]
                else:
                    chosen = trials_sorted + (trials_sorted[-1:] * (R - len(trials_sorted)) if len(trials_sorted) > 0 else [])

            for r_i in range(R):
                if r_i >= len(chosen) or len(chosen) == 0:
                    continue
                tr = chosen[r_i]
                arr = self._load_seq_cached(tr.path)  # (C,T)
                if arr.shape[0] != C:
                    # allow mismatch only if user config expects different C
                    # but keep safe
                    arr = arr[:C, :]
                x = torch.from_numpy(arr).float()  # (C,T)
                T_raw = x.shape[-1]
                if T_raw < Tw:
                    x = _pad_or_trim(x, Tw)
                T = x.shape[-1]

                if self.mode == "train":
                    picked = _sample_window_starts_train(T, Tw, self.Ts, K, rng)
                else:
                    picked = _sample_window_starts_eval(T, Tw, self.Ts, K)

                ws = []
                for st in picked:
                    st = int(st)
                    w = x[:, st:st + Tw]
                    w = _pad_or_trim(w, Tw)
                    w = _per_window_channel_norm(w)
                    if self.supervised_aug is not None and self.mode == "train":
                        w = self.supervised_aug(w)
                    ws.append(w)
                ws = torch.stack(ws, dim=0)  # (K,C,Tw)

                windows[a_i, r_i] = ws
                window_mask[a_i, r_i] = True
                trial_mask[a_i, r_i] = True

        return windows, window_mask, trial_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sess = self.sessions[idx]

        if self.mode == "train":
            rng1 = np.random.RandomState(np.random.randint(0, 2**31 - 1))
            w1, wm1, tm1 = self._sample_view(sess, rng1)
            if self.return_two_views:
                rng2 = np.random.RandomState(np.random.randint(0, 2**31 - 1))
                w2, wm2, tm2 = self._sample_view(sess, rng2)
            else:
                w2 = wm2 = tm2 = None
        else:
            rng = np.random.RandomState(0)  # deterministic
            w1, wm1, tm1 = self._sample_view(sess, rng)
            w2 = wm2 = tm2 = None

        labels = sess.labels
        y_wh = pd.to_numeric(labels.get("FMA_WH", np.nan), errors="coerce")
        y_se = pd.to_numeric(labels.get("FMA_SE", np.nan), errors="coerce")
        y_wh_t = torch.tensor(float(y_wh) if np.isfinite(y_wh) else float("nan"), dtype=torch.float32)
        y_se_t = torch.tensor(float(y_se) if np.isfinite(y_se) else float("nan"), dtype=torch.float32)
        label_mask = torch.tensor(
            [bool(np.isfinite(y_wh)), bool(np.isfinite(y_se))],
            dtype=torch.bool
        )

        out = {
            "y_wh": y_wh_t,
            "y_se": y_se_t,
            "label_mask": label_mask,  # (2,)
            "session_id": sess.session_id,
            "subject_id": sess.subject_id,
            "stage": sess.stage,
        }

        if self.return_two_views and self.mode == "train":
            out.update({
                "windows_1": w1,
                "window_mask_1": wm1,
                "trial_mask_1": tm1,
                "windows_2": w2,
                "window_mask_2": wm2,
                "trial_mask_2": tm2,
            })
        else:
            out.update({
                "windows": w1,
                "window_mask": wm1,
                "trial_mask": tm1,
            })

        return out


class SSLWindowDataset(Dataset):
    """
    Self-supervised: each item returns two augmented views of the same window.
    """
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        subjects: List[str],
        Tw_samples: int,
        windows_per_epoch: int,
        ssl_aug,
        emg_cols: Optional[List[int]] = None,
        time_col: Optional[int] = 0,
        expected_emg_ch: int = 4,
        cache_size: int = 256,
    ):
        super().__init__()
        self.subjects = set([str(s) for s in subjects])
        self.Tw = int(Tw_samples)
        self.windows_per_epoch = int(windows_per_epoch)
        self.ssl_aug = ssl_aug

        self.emg_cols = emg_cols
        self.time_col = time_col
        self.expected_emg_ch = expected_emg_ch

        self.cache = _SequenceCache(max_items=cache_size)

        self.trials: List[TrialRecord] = []
        for _, row in manifest_df.iterrows():
            subj = str(row["subject_id"])
            if subj not in self.subjects:
                continue
            self.trials.append(TrialRecord(
                session_id=str(row["session_id"]),
                subject_id=subj,
                stage=str(row["stage"]),
                axis=str(row["axis"]),
                trial_id=str(row["trial_id"]),
                path=str(row["path"]),
            ))

        if len(self.trials) == 0:
            raise ValueError("SSLWindowDataset got zero trials. Check subjects filter or manifest.")

    def __len__(self) -> int:
        return self.windows_per_epoch

    def _load_seq_cached(self, path: str) -> np.ndarray:
        cached = self.cache.get(path)
        if cached is not None:
            return cached
        emg, _ = load_emg_timeseries(
            path,
            emg_cols=self.emg_cols,
            time_col=self.time_col,
            expected_emg_ch=self.expected_emg_ch,
        )
        self.cache.put(path, emg)
        return emg

    def __getitem__(self, idx: int):
        tr = self.trials[np.random.randint(0, len(self.trials))]
        arr = self._load_seq_cached(tr.path)  # (C,T)
        C, T = arr.shape
        if T <= self.Tw:
            st = 0
        else:
            st = np.random.randint(0, T - self.Tw + 1)
        w = torch.from_numpy(arr[:, st:st+self.Tw]).float()
        w = _pad_or_trim(w, self.Tw)
        w = _per_window_channel_norm(w)

        x1 = self.ssl_aug(w)
        x2 = self.ssl_aug(w)
        return x1, x2
