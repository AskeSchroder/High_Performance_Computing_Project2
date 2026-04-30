import argparse
from os.path import join
import time

import cupy as cp
import numpy as np


LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
GRID_SIZE = 512


def load_building_ids(load_dir):
    with open(join(load_dir, "building_ids.txt"), "r") as f:
        return f.read().splitlines()


def load_data_to_gpu(load_dir, bid):
    """
    Load one building on the host and transfer it to the GPU.
    File I/O still happens on the CPU, but all later work stays on the GPU.
    """
    u = np.zeros((GRID_SIZE + 2, GRID_SIZE + 2), dtype=np.float64)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(bool)
    return cp.asarray(u), cp.asarray(interior_mask)


def jacobi_cupy(u, interior_mask, max_iter=20_000, atol=1e-4):
    """
    Reference-style Jacobi solver fully on the GPU.
    """
    u = u.copy()

    for _ in range(max_iter):
        u_new = 0.25 * (
            u[1:-1, :-2] +
            u[1:-1, 2:] +
            u[:-2, 1:-1] +
            u[2:, 1:-1]
        )

        u_old_interior = u[1:-1, 1:-1][interior_mask]
        u_new_interior = u_new[interior_mask]

        delta = cp.abs(u_old_interior - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if float(delta) < atol:
            break

    return u


def summary_stats_cupy(u, interior_mask):
    """
    Compute the final summary statistics on the GPU and return Python scalars.
    """
    u_interior = u[1:-1, 1:-1][interior_mask]

    mean_temp = float(cp.mean(u_interior))
    std_temp = float(cp.std(u_interior))
    pct_above_18 = float(cp.mean(u_interior > 18.0) * 100.0)
    pct_below_15 = float(cp.mean(u_interior < 15.0) * 100.0)

    return {
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "pct_above_18": pct_above_18,
        "pct_below_15": pct_below_15,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run wall-heating simulation on first N floorplans using CuPy with GPU-side stats."
    )
    parser.add_argument("N", type=int, nargs="?", default=1, help="Number of floorplans to process")
    parser.add_argument("--max-iter", type=int, default=20_000, help="Maximum Jacobi iterations")
    parser.add_argument("--atol", type=float, default=1e-4, help="Convergence tolerance")
    parser.add_argument("--time", action="store_true", help="Print timing information")
    args = parser.parse_args()

    building_ids = load_building_ids(LOAD_DIR)[:args.N]

    # Warm up CuPy and trigger one-time GPU setup outside the timed region.
    if building_ids:
        first_u0, first_mask = load_data_to_gpu(LOAD_DIR, building_ids[0])
        _ = jacobi_cupy(first_u0, first_mask, max_iter=1, atol=args.atol)
        cp.cuda.Stream.null.synchronize()

    if args.time:
        t0 = time.perf_counter()

    print("building_id,mean_temp,std_temp,pct_above_18,pct_below_15")

    for bid in building_ids:
        u0, interior_mask = load_data_to_gpu(LOAD_DIR, bid)
        u = jacobi_cupy(u0, interior_mask, max_iter=args.max_iter, atol=args.atol)
        stats = summary_stats_cupy(u, interior_mask)

        print(
            f"{bid},"
            f"{stats['mean_temp']},"
            f"{stats['std_temp']},"
            f"{stats['pct_above_18']},"
            f"{stats['pct_below_15']}"
        )

    if args.time:
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        print(f"# Total runtime: {t1 - t0:.3f} seconds")


if __name__ == "__main__":
    main()
