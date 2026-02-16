import os
import json
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Fully deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def torch_load_compat(path: str, map_location: str | torch.device = "cpu") -> Any:
    """
    PyTorch >=2.6 changed torch.load default to weights_only=True.
    We need weights_only=False for local trusted checkpoints that include
    non-tensor metadata.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # For older PyTorch versions without weights_only argument.
        return torch.load(path, map_location=map_location)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    logits: any shape
    mask: broadcastable to logits, True=valid, False=invalid
    Returns: softmax normalized over valid entries; if all invalid along dim -> all zeros.
    """
    mask = mask.to(dtype=torch.bool)
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~mask, neg_inf)
    probs = torch.softmax(masked_logits, dim=dim)
    probs = probs * mask.to(dtype=probs.dtype)
    denom = probs.sum(dim=dim, keepdim=True)
    probs = torch.where(denom > 0, probs / (denom + 1e-12), torch.zeros_like(probs))
    return probs


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@dataclass
class AverageMeter:
    sum: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)
