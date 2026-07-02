import torch

def quantile_per_frame(tensor: torch.Tensor, quantile: float) -> list[float]:
    ret = []
    for f in tensor:
        ret.append(torch.quantile(f, quantile))
    return ret
