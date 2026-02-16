from typing import Tuple, List
import torch
import random


class RandomAmplitudeScale:
    def __init__(self, scale_range: Tuple[float, float] = (0.7, 1.3), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        scale = random.uniform(*self.scale_range)
        return x * scale


class AddGaussianNoise:
    def __init__(self, sigma_ratio_range: Tuple[float, float] = (0.01, 0.05), p: float = 0.5):
        self.sigma_ratio_range = sigma_ratio_range
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        rms = torch.sqrt(torch.mean(x ** 2) + 1e-8)
        sigma = random.uniform(*self.sigma_ratio_range) * float(rms)
        return x + sigma * torch.randn_like(x)


class RandomTimeShift:
    def __init__(self, max_shift: int = 10, p: float = 0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=-1)


class TimeMask:
    def __init__(self, mask_ratio_range: Tuple[float, float] = (0.0, 0.2), p: float = 0.5):
        self.mask_ratio_range = mask_ratio_range
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        C, T = x.shape
        ratio = random.uniform(*self.mask_ratio_range)
        m = int(T * ratio)
        if m <= 0:
            return x
        start = random.randint(0, max(0, T - m))
        x = x.clone()
        x[:, start:start+m] = 0.0
        return x


class ChannelDropout:
    def __init__(self, drop_p: float = 0.1, p: float = 0.5):
        self.drop_p = drop_p
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        C, T = x.shape
        x = x.clone()
        for c in range(C):
            if random.random() < self.drop_p:
                x[c] = 0.0
        return x


class Compose:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


def build_ssl_augmentations(strength: str = "medium") -> Compose:
    """
    For self-supervised contrastive pretraining: stronger perturbations are acceptable.
    strength: "light" | "medium" | "strong"
    """
    if strength == "light":
        return Compose([
            RandomAmplitudeScale((0.9, 1.1), p=0.8),
            AddGaussianNoise((0.005, 0.02), p=0.8),
            RandomTimeShift(max_shift=5, p=0.6),
            TimeMask((0.0, 0.1), p=0.4),
            ChannelDropout(drop_p=0.1, p=0.2),
        ])
    if strength == "strong":
        return Compose([
            RandomAmplitudeScale((0.7, 1.3), p=0.9),
            AddGaussianNoise((0.01, 0.05), p=0.9),
            RandomTimeShift(max_shift=15, p=0.7),
            TimeMask((0.1, 0.2), p=0.7),
            ChannelDropout(drop_p=0.2, p=0.4),
        ])
    # medium
    return Compose([
        RandomAmplitudeScale((0.8, 1.2), p=0.9),
        AddGaussianNoise((0.01, 0.03), p=0.9),
        RandomTimeShift(max_shift=10, p=0.7),
        TimeMask((0.0, 0.2), p=0.6),
        ChannelDropout(drop_p=0.15, p=0.3),
    ])


def build_sup_augmentations(strength: str = "weak") -> Compose:
    """
    For supervised MIL training: keep augmentations weak to preserve clinical meaning.
    This is especially important when using Mean-Teacher consistency:
      - student sees weak perturbation
      - teacher sees another weak perturbation
    """
    if strength == "none":
        return Compose([])
    if strength == "weak":
        return Compose([
            RandomAmplitudeScale((0.9, 1.1), p=0.5),
            AddGaussianNoise((0.0, 0.01), p=0.5),
            RandomTimeShift(max_shift=3, p=0.3),
        ])
    # medium (use with caution)
    return Compose([
        RandomAmplitudeScale((0.85, 1.15), p=0.6),
        AddGaussianNoise((0.0, 0.02), p=0.6),
        RandomTimeShift(max_shift=5, p=0.5),
        TimeMask((0.0, 0.1), p=0.3),
    ])
