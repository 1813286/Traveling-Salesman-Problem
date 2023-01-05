from timeit import default_timer
from typing import List, Optional, Tuple

import numpy as np
from heuristicTsp.perturbation_schemes import neighborhood_gen
from heuristicTsp.local_search import setup
from heuristicTsp.utils import compute_permutation_distance


def tspHeuristic(
    distance_matrix: np.ndarray,
    x0: Optional[List[int]] = None,
    perturbation_scheme: str = "two_opt",
    alpha: float = 0.9,
    max_processing_time: float = None,
    log_file: Optional[str] = None,
    verbose: bool = False
) -> Tuple[List, float]:
    x, fx = setup(distance_matrix, x0)
    temp = _initial_temperature(distance_matrix, x, fx, perturbation_scheme)
    max_processing_time = max_processing_time or np.inf
    if log_file:
        log_file_handler = open(log_file, "w", encoding="utf-8")

    n = len(x)
    k_inner_min = n  # min inner iterations
    k_inner_max = 10 * n  # max inner iterations
    k_noimprovements = 0  # number of inner loops without improvement

    tic = default_timer()
    stop_early = False
    while (k_noimprovements < 3) and (not stop_early):
        k_accepted = 0  # number of accepted perturbations
        for k in range(k_inner_max):
            if default_timer() - tic > max_processing_time:
                warning_msg = "WARNING: Stopping early due to time constraints"
                if log_file:
                    print(warning_msg, file=log_file_handler)
                if verbose:
                    print(warning_msg)
                stop_early = True
                break

            xn = _perturbation(x, perturbation_scheme)
            fn = compute_permutation_distance(distance_matrix, xn)

            if _acceptance_rule(fx, fn, temp):
                x, fx = xn, fn
                k_accepted += 1
                k_noimprovements = 0

            msg = (
                f"Temperature {temp}. Current value: {fx} "
                f"k: {k + 1}/{k_inner_max} "
                f"k_accepted: {k_accepted}/{k_inner_min} "
                f"k_noimprovements: {k_noimprovements}"
            )
            if log_file:
                print(msg, file=log_file_handler)
            if verbose:
                print(msg)

            if k_accepted >= k_inner_min:
                break

        temp *= alpha  # temperature update
        k_noimprovements += k_accepted == 0

    return x, fx


def _initial_temperature(
    distance_matrix: np.ndarray,
    x: List[int],
    fx: float,
    perturbation_scheme: str,
) -> float:
    # Step 1
    dfx_list = []
    for _ in range(100):
        xn = _perturbation(x, perturbation_scheme)
        fn = compute_permutation_distance(distance_matrix, xn)
        dfx_list.append(fn - fx)

    dfx_mean = np.abs(np.mean(dfx_list))

    # Step 2
    tau0 = 0.5
    return -dfx_mean / np.log(tau0)


def _perturbation(x: List[int], perturbation_scheme: str):
    return next(neighborhood_gen[perturbation_scheme](x))


def _acceptance_rule(fx: float, fn: float, temp: float) -> bool:
    dfx = fn - fx
    return (dfx < 0) or (
        (dfx > 0) and (np.random.rand() <= np.exp(-(fn - fx) / temp))
    )
