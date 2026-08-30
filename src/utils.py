import random
import numpy as np
import torch
from typing import Optional, Dict, Any
import re


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def rbf_kernel(
    x: torch.Tensor,
    sigma: Optional[float] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Build dense RBF kernel:
        K_ij = exp(-||x_i - x_j||^2 / (2 sigma^2))

    Args:
        x: (n, d) float tensor
        sigma: bandwidth. If None, use median heuristic on pairwise distances.
        eps: numerical stability

    Returns:
        K: (n, n) tensor
    """
    if x.ndim != 2:
        raise ValueError(f"x must have shape (n, d), got {tuple(x.shape)}")

    # Squared Euclidean distances
    dist2 = torch.cdist(x, x, p=2) ** 2

    if sigma is None:
        # Median heuristic on off-diagonal distances
        n = x.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
        vals = dist2[mask]
        # Use sqrt(median(dist^2)) as sigma scale
        sigma = torch.sqrt(torch.median(vals).clamp_min(eps)).item()

    sigma2 = max(float(sigma) ** 2, eps)
    K = torch.exp(-dist2 / (2.0 * sigma2))
    return K


def extract_answer_og(solution_str):
    """
    Extracts the numerical answer from a string following '####'.
    Returns a float on success, or -1 on failure.

    Based on verl https://github.com/volcengine/verl/blob/644aaa76bcd2b44920dd835c3b95308db78e959f/examples/data_preprocess/gsm8k.py
    """
    # The regex correctly captures the number part.
    solution = re.search(r"####\s*(-?[0-9.,]+)", solution_str)
    
    if solution is None:
        return -1
    
    # Use the captured group directly.
    # The .split() call was removed as it caused the bug.
    final_solution_str = solution.group(1).replace(",", "").strip()
    
    try:
        # Convert the cleaned string to a float for a consistent return type.
        return float(final_solution_str)
    except ValueError:
        # Handle cases where the captured string can't be converted to a number.
        return -1



