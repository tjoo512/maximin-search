import math
from typing import Dict, Any, Optional, Tuple, List

import numpy as np


def _validate_inputs(
    s: np.ndarray,
    K: np.ndarray,
    lam: float,
    m: int,
) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square 2D numpy array.")
    if s.ndim != 1:
        raise ValueError("s must be a 1D numpy array.")
    if K.shape[0] != s.shape[0]:
        raise ValueError("s and K must have compatible dimensions.")
    if not (0 <= m <= s.shape[0]):
        raise ValueError("m must satisfy 0 <= m <= n.")
    if lam < 0:
        raise ValueError("lam must be nonnegative.")

    # Symmetrize defensively.
    K = 0.5 * (K + K.T)
    return s, K


def set_to_indicator(indices: np.ndarray, n: int) -> np.ndarray:
    q = np.zeros(n, dtype=np.int64)
    q[indices] = 1
    return q


def k_value_from_set(K: np.ndarray, S: np.ndarray) -> float:
    if S.size == 0:
        return 0.0
    return float(K[np.ix_(S, S)].sum())


def original_objective_from_set(
    s: np.ndarray,
    K: np.ndarray,
    lam: float,
    S: np.ndarray,
) -> float:
    if S.size == 0:
        return 0.0
    return float(s[S].sum() - lam * math.sqrt(max(k_value_from_set(K, S), 0.0)))


def _greedy_init_for_fixed_eta(
    s: np.ndarray,
    K: np.ndarray,
    lam: float,
    eta: float,
    m: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Greedy initialization for the fixed-eta objective:
        G_eta(S) = sum_{i in S} s_i - (lam/(2 eta)) * k(S) - (lam/2) * eta

    The marginal gain for adding j to current set S is:
        s_j - alpha * (2 * c_j + K_jj),
    where c_j = sum_{i in S} K_{j,i}.
    """
    n = s.shape[0]
    if m == 0:
        return np.empty(0, dtype=int)

    diagK = np.diag(K)
    alpha = lam / (2.0 * eta)

    selected = np.zeros(n, dtype=bool)
    c = np.zeros(n, dtype=float)  # c[l] = sum_{j in S} K[l, j]
    S = []

    for _ in range(m):
        gains = s - alpha * (2.0 * c + diagK)
        gains[selected] = -np.inf

        # Stable tie-breaking with tiny random perturbation.
        noise = 1e-15 * rng.standard_normal(n)
        j = int(np.argmax(gains + noise))

        S.append(j)
        selected[j] = True
        c += K[:, j]

    return np.array(S, dtype=int)


def _randomized_greedy_init_for_fixed_eta(
    s: np.ndarray,
    K: np.ndarray,
    lam: float,
    eta: float,
    m: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Greedy initialization for the fixed-eta objective:
        G_eta(S) = sum_{i in S} s_i - (lam/(2 eta)) * k(S) - (lam/2) * eta

    The marginal gain for adding j to current set S is:
        s_j - alpha * (2 * c_j + K_jj),
    where c_j = sum_{i in S} K_{j,i}.
    """
    n = s.shape[0]
    if m == 0:
        return np.empty(0, dtype=int)

    diagK = np.diag(K)
    alpha = lam / (2.0 * eta)

    selected = np.zeros(n, dtype=bool)
    c = np.zeros(n, dtype=float)  # c[l] = sum_{j in S} K[l, j]
    S = []

    for i in range(m):
        gains = s - alpha * (2.0 * c + diagK)
        gains[selected] = -np.inf

        # Get indices of the k largest elements
        k = min(m, len(gains) - i)
        topk_idx = np.argpartition(gains, -k)[-k:]

        # Choose random index
        j = np.random.choice(topk_idx)

        S.append(j)
        selected[j] = True
        c += K[:, j]

    return np.array(S, dtype=int)

def _random_init(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    if m == 0:
        return np.empty(0, dtype=int)
    return np.array(rng.choice(n, size=m, replace=False), dtype=int)


def solve_fixed_eta_local_search(
    s: np.ndarray,
    K: np.ndarray,
    lam: float,
    eta: float,
    m: int,
    rng: Optional[np.random.Generator] = None,
    random_state: Optional[object] = None,
    max_passes: int = 100,
    n_restarts: int = 8,
    tol: float = 1e-12,
    use_at_most_constraint: bool = False,
) -> Dict[str, Any]:
    """
    Solve the fixed-eta size-m subset problem with multi-start local 1-swap search.

    Objective for a set S of size m:
        sum_{j in S} s_j - alpha * sum_{i,j in S} K_ij - 0.5 * lam * eta

    where
        alpha = lam / (2 * eta)

    Accepts either:
      - rng: np.random.Generator
      - random_state: int, np.random.RandomState, or np.random.Generator
    """
    s = np.asarray(s, dtype=float)
    K = np.asarray(K, dtype=float)

    if s.ndim != 1:
        raise ValueError("s must be a 1D array.")
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square 2D array.")
    if K.shape[0] != s.shape[0]:
        raise ValueError("K and s must have compatible dimensions.")

    n = s.shape[0]
    if not (0 <= m <= n):
        raise ValueError(f"Require 0 <= m <= n, got m={m}, n={n}.")
    if eta <= 0:
        raise ValueError(f"eta must be positive, got eta={eta}.")
    if n_restarts <= 0:
        raise ValueError(f"n_restarts must be positive, got n_restarts={n_restarts}.")

    # Backward-compatible RNG handling.
    if rng is not None and random_state is not None:
        raise ValueError("Pass only one of `rng` or `random_state`, not both.")

    if rng is None:
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, np.random.Generator):
            rng = random_state
        elif isinstance(random_state, np.random.RandomState):
            # Convert legacy RandomState to Generator via a seed draw.
            rng = np.random.default_rng(int(random_state.randint(0, 2**32 - 1)))
        else:
            # Treat ints / seed-like objects as seeds.
            rng = np.random.default_rng(random_state)

    alpha = lam / (2.0 * eta)
    diagK = np.diag(K)

    def local_search_from_init(S0: np.ndarray) -> Tuple[np.ndarray, float, float]:
        S = np.array(S0, dtype=int, copy=True)
        S = np.unique(S)
        # if S.size != m:
        #     raise ValueError("Initialization must contain exactly m unique indices.")

        selected = np.zeros(n, dtype=bool)
        selected[S] = True

        # c[l] = sum_{j in S} K[l, j]
        c = K[:, S].sum(axis=1)
        kS = float(c[S].sum())  # equals sum_{i,j in S} K_ij
        current_value = float(s[S].sum() - alpha * kS - 0.5 * lam * eta)

        if m == n:
            S = np.sort(S)
            kS = k_value_from_set(K, S)
            current_value = float(s[S].sum() - alpha * kS - 0.5 * lam * eta)
            return S, current_value, kS

        for _ in range(max_passes):
            best_gain = 0.0
            best_out = -1
            best_in = -1

            notS = np.flatnonzero(~selected)
            if notS.size == 0:
                break

            for i in S:
                delta_k = (
                    -2.0 * c[i]
                    + diagK[i]
                    + 2.0 * c[notS]
                    - 2.0 * K[notS, i]
                    + diagK[notS]
                )
                gains = (s[notS] - s[i]) - alpha * delta_k

                if gains.size == 0:
                    continue

                j_pos = int(np.argmax(gains))
                gain = float(gains[j_pos])

                if gain > best_gain + tol:
                    best_gain = gain
                    best_out = int(i)
                    best_in = int(notS[j_pos])

            if best_in < 0:
                break

            selected[best_out] = False
            selected[best_in] = True

            c += K[:, best_in] - K[:, best_out]
            S[S == best_out] = best_in
            current_value += best_gain
            kS = float(c[S].sum())

        S = np.sort(S)
        kS = k_value_from_set(K, S)
        current_value = float(s[S].sum() - alpha * kS - 0.5 * lam * eta)
        return S, current_value, kS

    starts: List[np.ndarray] = [
        _randomized_greedy_init_for_fixed_eta(s, K, lam, eta, m, rng),
        _greedy_init_for_fixed_eta(s, K, lam, eta, m, rng),
    ]
    for _ in range(max(0, n_restarts - 2)):
        starts.append(_random_init(n, m, rng))

    best_S = None
    best_val = -np.inf
    best_k = None

    for S0 in starts:
        S_candidate, val_candidate, k_candidate = local_search_from_init(S0)
        if val_candidate > best_val:
            best_S = S_candidate
            best_val = val_candidate
            best_k = k_candidate

    if best_S is None:
        raise RuntimeError("Local search failed to produce any candidate solution.")

    return {
        "indices": best_S,
        "q": set_to_indicator(best_S, n),
        "fixed_eta_objective": float(best_val),
        "k_value": float(best_k),
    }


def solve_kernel_subset_problem(
    s: np.ndarray,
    K: np.ndarray,
    lam: float,
    m: int,
    *,
    epsilon: float = 0.05,
    n_restarts: int = 3,
    max_passes: int = 50,
    tol: float = 1e-12,
    random_state: Optional[int] = None,
    use_at_most_constraint: bool = False,
) -> Dict[str, Any]:
    r"""
    Approximate solver for

        max_{q in {0,1}^n} s^T q - lam * sqrt(q^T K q)
        s.t. 1^T q = m

    using:
      - geometric discretization of eta in [sqrt(m), m]
      - a practical 1-swap local-search inner solver for each fixed eta

    Parameters
    ----------
    s : (n,) numpy array
        Linear scores.
    K : (n, n) numpy array
        Symmetric PSD kernel matrix, typically dense RBF kernel.
    lam : float
        Nonnegative penalty weight.
    m : int
        Required cardinality.
    epsilon : float, default=0.05
        Geometric grid factor for eta_t = sqrt(m) * (1 + epsilon)^t.
    n_restarts : int, default=3
        Number of inner-solver restarts per eta.
    max_passes : int, default=50
        Max number of local-search passes per restart.
    tol : float, default=1e-12
        Improvement tolerance for accepting swaps.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    result : dict
        Keys include:
          - "indices": selected indices
          - "q": 0/1 indicator vector
          - "objective": original objective value
          - "k_value": q^T K q
          - "eta_used": eta value of the winning candidate
          - "all_candidates": list of candidate summaries
    """

    s, K = _validate_inputs(s, K, lam, m)
    n = s.shape[0]

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    if m == 0:
        return {
            "indices": np.empty(0, dtype=int),
            "q": np.zeros(n, dtype=np.int64),
            "objective": 0.0,
            "k_value": 0.0,
            "eta_used": None,
            "all_candidates": [],
        }

    eta_min = math.sqrt(m)
    eta_max = float(m)

    etas = [eta_min]
    while etas[-1] < eta_max:
        etas.append(min(eta_max, etas[-1] * (1.0 + epsilon)))

    all_candidates = []
    best = None
    best_obj = -np.inf

    # Use independent seeds across eta values for stability.
    base_rng = np.random.default_rng(random_state)
    seeds = base_rng.integers(0, 2**32 - 1, size=len(etas), dtype=np.uint64)

    for eta, seed in zip(etas, seeds):
        inner = solve_fixed_eta_local_search(
            s, K, lam, eta, m, n_restarts=n_restarts,
            max_passes=max_passes, tol=tol, random_state=int(seed),
            use_at_most_constraint=use_at_most_constraint,
        )

        S = inner["indices"]
        obj = original_objective_from_set(s, K, lam, S)
        kS = k_value_from_set(K, S)

        candidate = {
            "eta": float(eta),
            "indices": S,
            "q": inner["q"],
            "objective": float(obj),
            "fixed_eta_objective": float(inner["fixed_eta_objective"]),
            "k_value": float(kS),
        }

        all_candidates.append(candidate)

        if obj > best_obj:
            best_obj = obj
            best = candidate

    return {
        "indices": best["indices"],
        "q": best["q"],
        "objective": best["objective"],
        "k_value": best["k_value"],
        "eta_used": best["eta"],
        "all_candidates": all_candidates,
    }

