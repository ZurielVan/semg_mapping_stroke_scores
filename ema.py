from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator
import copy
import torch
import torch.nn as nn


@torch.no_grad()
def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """
    In-place EMA update: ema_model <- decay * ema_model + (1-decay) * model

    This is the standard Mean-Teacher / Polyak averaging update and is typically
    more efficient than swapping weights each iteration.
    """
    d = float(decay)
    msd = model.state_dict()
    for k, v in ema_model.state_dict().items():
        if k in msd:
            v.copy_(v * d + msd[k].detach() * (1.0 - d))


def clone_as_ema_teacher(model: nn.Module) -> nn.Module:
    """
    Create a teacher model (deep copy) with gradients disabled.
    """
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    return teacher


class ShadowEMA:
    """
    EMA as a parameter shadow (dict). Useful when you don't want a separate teacher model.
    Kept here for completeness; the main training loop uses a real teacher model.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self._init(model)

    def _init(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @contextmanager
    def apply_shadow(self, model: nn.Module) -> Iterator[None]:
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name].data)
        try:
            yield
        finally:
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                p.data.copy_(self.backup[name].data)
            self.backup = {}
