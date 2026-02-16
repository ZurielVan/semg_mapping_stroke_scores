from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedHuberLoss(nn.Module):
    """
    Huber loss on normalized error (y - yhat)/scale, to balance labels with different ranges.
    """
    def __init__(self, delta: float = 0.04):
        super().__init__()
        self.delta = float(delta)

    def forward(self, y: torch.Tensor, yhat: torch.Tensor, scale: float) -> torch.Tensor:
        e = (y - yhat) / float(scale)
        abs_e = torch.abs(e)
        quad = torch.minimum(abs_e, torch.tensor(self.delta, device=e.device, dtype=e.dtype))
        lin = abs_e - quad
        loss = 0.5 * quad**2 + self.delta * lin
        return loss.mean()


class MainFmaLoss(nn.Module):
    """
    Main supervised loss for FMA_WH (0..24) and FMA_SE (0..42).
    Optionally add UE consistency regularizer, where UE = WH + SE (0..66).
    """
    def __init__(self, delta: float = 0.04, lambda_ue: float = 0.1, use_ue: bool = True):
        super().__init__()
        self.huber = NormalizedHuberLoss(delta=delta)
        self.lambda_ue = float(lambda_ue)
        self.use_ue = bool(use_ue)

    def forward(
        self,
        y_wh: torch.Tensor, y_se: torch.Tensor,
        yhat_wh: torch.Tensor, yhat_se: torch.Tensor,
        label_mask: torch.Tensor
    ) -> torch.Tensor:
        # label_mask: (B,2)
        loss = 0.0
        denom = 0.0

        wh_mask = label_mask[:, 0]
        se_mask = label_mask[:, 1]

        if wh_mask.any():
            loss = loss + self.huber(y_wh[wh_mask], yhat_wh[wh_mask], scale=24.0)
            denom += 1.0
        if se_mask.any():
            loss = loss + self.huber(y_se[se_mask], yhat_se[se_mask], scale=42.0)
            denom += 1.0

        if self.use_ue:
            ue_mask = wh_mask & se_mask
            if ue_mask.any():
                y_ue = y_wh + y_se
                yhat_ue = yhat_wh + yhat_se
                loss = loss + self.lambda_ue * self.huber(y_ue[ue_mask], yhat_ue[ue_mask], scale=66.0)
                denom += self.lambda_ue

        return loss / max(1e-8, denom)


class ConsistencyLoss(nn.Module):
    """
    Mean-Teacher consistency between student and teacher predictions.
    Input should already be normalized to comparable scale (e.g., yhat/scale in [0,1]).
    """
    def __init__(self, loss_type: str = "mse", huber_delta: float = 0.02):
        super().__init__()
        self.loss_type = loss_type
        self.huber_delta = float(huber_delta)

    def forward(self, p_student: torch.Tensor, p_teacher: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(p_student, p_teacher)
        if self.loss_type == "smoothl1":
            return F.smooth_l1_loss(p_student, p_teacher, beta=self.huber_delta)
        raise ValueError(f"Unknown consistency loss_type: {self.loss_type}")
