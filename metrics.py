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


@torch.no_grad()
def r2_score(y: torch.Tensor, yhat: torch.Tensor) -> float:
    """
    Coefficient of determination (R^2).
    """
    if y.numel() == 0:
        return float("nan")
    ss_res = torch.sum((y - yhat) ** 2)
    y_mean = torch.mean(y)
    ss_tot = torch.sum((y - y_mean) ** 2)
    if float(ss_tot.item()) <= 1e-12:
        return 1.0 if float(ss_res.item()) <= 1e-12 else 0.0
    return float((1.0 - ss_res / ss_tot).item())


@torch.no_grad()
def correlation(y: torch.Tensor, yhat: torch.Tensor) -> float:
    """
    Pearson correlation coefficient.
    """
    if y.numel() < 2:
        return 0.0
    y_c = y - torch.mean(y)
    yhat_c = yhat - torch.mean(yhat)
    denom = torch.sqrt(torch.sum(y_c ** 2) * torch.sum(yhat_c ** 2))
    if float(denom.item()) <= 1e-12:
        return 0.0
    corr = torch.sum(y_c * yhat_c) / denom
    corr = torch.clamp(corr, min=-1.0, max=1.0)
    return float(corr.item())
