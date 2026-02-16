import torch


@torch.no_grad()
def mae(y: torch.Tensor, yhat: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(y - yhat)).item())


@torch.no_grad()
def rmse(y: torch.Tensor, yhat: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((y - yhat) ** 2)).item())


@torch.no_grad()
def normalized_mae(y: torch.Tensor, yhat: torch.Tensor, scale: float) -> float:
    return float(torch.mean(torch.abs((y - yhat) / scale)).item())


@torch.no_grad()
def within_tolerance(y: torch.Tensor, yhat: torch.Tensor, tol: float) -> float:
    return float((torch.abs(y - yhat) <= tol).float().mean().item())
