import argparse
import math
import multiprocessing as mp
from os.path import join
import time

import numpy as np
from numba import njit


LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
GRID_SIZE = 512


def load_building_ids(load_dir):
    with open(join(load_dir, "building_ids.txt"), "r") as f:
        return f.read().splitlines()


def load_data(load_dir, bid):
    """
    Loads one building:
      - domain: 512x512 initial temperature grid
      - interior mask: 512x512 boolean mask
    Pads the domain to 514x514 to match the reference implementation.
    """
    u = np.zeros((GRID_SIZE + 2, GRID_SIZE + 2), dtype=np.float64)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(np.bool_)
    return u, interior_mask


@njit(cache=True)
def jacobi_numba(u0, interior_mask, max_iter, atol):
    """
    CPU JIT Jacobi solver using explicit loops and double buffering.
    u0 shape:            (514, 514)
    interior_mask shape: (512, 512)
    """
    u = u0.copy()
    u_new = u0.copy()

    nrows, ncols = u.shape

    for _ in range(max_iter):
        delta = 0.0

        # Row-major access: keep j as inner loop for better cache locality
        for i in range(1, nrows - 1):
            ii = i - 1
            for j in range(1, ncols - 1):
                jj = j - 1

                if interior_mask[ii, jj]:
                    new_val = 0.25 * (
                        u[i, j - 1] +
                        u[i, j + 1] +
                        u[i - 1, j] +
                        u[i + 1, j]
                    )

                    diff = abs(new_val - u[i, j])
                    if diff > delta:
                        delta = diff

                    u_new[i, j] = new_val
                else:
                    u_new[i, j] = u[i, j]

        u, u_new = u_new, u

        if delta < atol:
            break

    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]

    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.mean(u_interior > 18.0) * 100.0
    pct_below_15 = np.mean(u_interior < 15.0) * 100.0

    return {
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "pct_above_18": pct_above_18,
        "pct_below_15": pct_below_15,
    }


def process_building(bid, max_iter, atol):
    u0, interior_mask = load_data(LOAD_DIR, bid)
    u = jacobi_numba(u0, interior_mask, max_iter, atol)
    stats = summary_stats(u, interior_mask)

    return (
        bid,
        stats["mean_temp"],
        stats["std_temp"],
        stats["pct_above_18"],
        stats["pct_below_15"],
    )


def chunk_list(items, n_chunks):
    """
    Split items into n_chunks contiguous chunks with sizes as equal as possible.
    This implements static scheduling.
    """
    n = len(items)
    chunk_size = math.ceil(n / n_chunks)
    return [items[i:i + chunk_size] for i in range(0, n, chunk_size)]


def worker_process_chunk(args):
    """
    Each worker gets one fixed chunk of building IDs and processes them sequentially.
    """
    chunk, max_iter, atol = args
    results = []

    for bid in chunk:
        results.append(process_building(bid, max_iter, atol))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Wall-heating simulation with Numba CPU JIT + static scheduling."
    )
    parser.add_argument("N", type=int, nargs="?", default=1,
                        help="Number of floorplans to process")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes")
    parser.add_argument("--max-iter", type=int, default=20_000,
                        help="Maximum Jacobi iterations")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="Convergence tolerance")
    parser.add_argument("--time", action="store_true",
                        help="Print timing information")
    args = parser.parse_args()

    building_ids = load_building_ids(LOAD_DIR)[:args.N]
    n_workers = min(args.workers, len(building_ids))
    chunks = chunk_list(building_ids, n_workers)

    if args.time:
        t0 = time.perf_counter()

    # Warm-up compile once in the parent process
    first_u0, first_mask = load_data(LOAD_DIR, building_ids[0])
    _ = jacobi_numba(first_u0, first_mask, args.max_iter, args.atol)

    if n_workers == 1:
        chunk_results = [worker_process_chunk((chunks[0], args.max_iter, args.atol))]
    else:
        with mp.Pool(processes=n_workers) as pool:
            chunk_results = pool.map(
                worker_process_chunk,
                [(chunk, args.max_iter, args.atol) for chunk in chunks]
            )

    all_results = [row for chunk in chunk_results for row in chunk]

    print("building_id,mean_temp,std_temp,pct_above_18,pct_below_15")
    for bid, mean_temp, std_temp, pct_above_18, pct_below_15 in all_results:
        print(f"{bid},{mean_temp},{std_temp},{pct_above_18},{pct_below_15}")

    if args.time:
        t1 = time.perf_counter()
        print(f"# Total runtime: {t1 - t0:.3f} seconds")
        print(f"# Workers: {n_workers}")
        print(f"# Buildings: {len(building_ids)}")


if __name__ == "__main__":
    main()