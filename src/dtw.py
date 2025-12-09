from __future__ import annotations

from typing import List, Sequence, Callable
import math

# DTW on multivariate feature vectors using Euclidean distance

def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def dtw_distance(
    seq_a,
    seq_b,
    dist_fn=euclidean,
    window=0,
) -> float:
    """
    Compute DTW distance between two multivariate sequences.
    see: https://en.wikipedia.org/wiki/Dynamic_time_warping
    Returns
    - DTW cost (lower means more similar).
    """
    n, m = len(seq_a), len(seq_b)

    # Failsafe for empty sequences
    if n == 0 or m == 0:
        return float("inf")
    
    # Initialize cost matrix with infinities
    inf = float("inf")
    cost = []
    for i in range(n + 1):
        row = []
        for j in range(m + 1):
            row.append(inf)
        cost.append(row)
    cost[0][0] = 0.0

    # Validate window
    if window < 0:
        raise ValueError("window must be a non-negative integer")

    # Sakoe–Chiba band constraint (assumed always given)
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)
        for j in range(j_start, j_end):
            d = dist_fn(seq_a[i - 1], seq_b[j - 1])
            cost[i][j] = d + min(cost[i - 1][j], # insertion
                                cost[i][j - 1], # deletion
                                cost[i - 1][j - 1] # match
                                )

    return cost[n][m]


def to_feature_vectors(features: Sequence[object]) -> List[Sequence[float]]:
    # Expect objects with attributes x,y,vx,vy,pressure
    result: List[Sequence[float]] = []
    for f in features:
        result.append([f.x, f.y, f.vx, f.vy, f.pressure])
    return result
