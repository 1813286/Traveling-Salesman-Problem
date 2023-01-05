from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import numpy as np


def tspDynamic(
    distance_matrix: np.ndarray,
    maxsize: Optional[int] = None,
) -> Tuple[List, float]:
    
    # Get initial set {1, 2, ..., tsp_size} as a frozenset because @lru_cache
    # requires a hashable type
    N = frozenset(range(1, distance_matrix.shape[0]))
    memo: Dict[Tuple, int] = {}

    # Step 1: get minimum distance
    @lru_cache(maxsize=maxsize)
    def dist(ni: int, N: frozenset) -> float:
        if not N:
            return distance_matrix[ni, 0]

        # Store the costs in the form (nj, dist(nj, N))
        costs = [
            (nj, distance_matrix[ni, nj] + dist(nj, N.difference({nj})))
            for nj in N
        ]
        nmin, min_cost = min(costs, key=lambda x: x[1])
        memo[(ni, N)] = nmin
        return min_cost

    best_distance = dist(0, N)

    # Step 2: get path with the minimum distance
    ni = 0  # start at the origin
    solution = [0]
    while N:
        ni = memo[(ni, N)]
        solution.append(ni)
        N = N.difference({ni})

    return solution, best_distance

    