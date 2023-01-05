from random import sample
from timeit import default_timer
from typing import List, Optional, Tuple

import numpy as np

from heuristicTsp.utils import compute_permutation_distance
from heuristicTsp.perturbation_schemes import neighborhood_gen


def solve_tsp_local_search(
    distance_matrix: np.ndarray,
    x0: Optional[List[int]] = None,
    perturbation_scheme: str = "two_opt",
    max_processing_time: Optional[float] = None,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List, float]:
    x, fx = setup(distance_matrix, x0)
    max_processing_time = max_processing_time or np.inf
    if log_file:
        log_file_handler = open(log_file, "w", encoding="utf-8")

    tic = default_timer()
    stop_early = False
    improvement = True
    while improvement and (not stop_early):
        improvement = False
        for n_index, xn in enumerate(neighborhood_gen[perturbation_scheme](x)):
            if default_timer() - tic > max_processing_time:
                warning_msg = "WARNING: Stopping early due to time constraints"
                if log_file:
                    print(warning_msg, file=log_file_handler)
                if verbose:
                    print(warning_msg)
                stop_early = True
                break

            fn = compute_permutation_distance(distance_matrix, xn)

            msg = f"Current value: {fx}; Neighbor: {n_index}"
            if log_file:
                print(msg, file=log_file_handler)
            if verbose:
                print(msg)

            if fn < fx:
                improvement = True
                x, fx = xn, fn
                break  # early stop due to first improvement local search

    return x, fx


def setup(
    distance_matrix: np.ndarray, x0: Optional[List] = None
) -> Tuple[List[int], float]:
    if not x0:
        n = distance_matrix.shape[0]  # number of nodes
        x0 = [0] + sample(range(1, n), n - 1)  # ensure 0 is the first node

    fx0 = compute_permutation_distance(distance_matrix, x0)
    return x0, fx0
