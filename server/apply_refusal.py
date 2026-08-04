"""Minimal load-time row projection used by the Frosty VL (Qwen3-VL persona plugin) server."""
import torch


def project_rows(weight: torch.Tensor, direction: torch.Tensor, lam: float) -> torch.Tensor:
    """Apply W <- W - lam * outer(v, v @ W), returning W's original dtype."""
    weight_fp32 = weight.to(torch.float32)
    direction_fp32 = direction.to(device=weight.device, dtype=torch.float32)
    projection = torch.outer(direction_fp32, direction_fp32 @ weight_fp32)
    return (weight_fp32 - lam * projection).to(weight.dtype)
