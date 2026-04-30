import argparse
import multiprocessing as mp
from os.path import join
import time

import numpy as np


LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
GRID_SIZE = 512


try:
    profile
except NameError:
    def profile(func):
        return func


def load_building_ids(load_dir):
    with open(join(load_dir, "building_ids.txt"), "r") as f:
        return f.read().splitlines()


def load_data(load_dir, bid):
    u = np.zeros((GRID_SIZE + 2, GRID_SIZE + 2), dtype=np.float64)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(bool)
    return u, interior_mask


@profile
def jacobi(u, interior_mask, max_iter=20_000, atol=1e-4):
    u = np.copy(u)

    for _ in range(max_iter):
        u_new = 0.25 * (
            u[1:-1, :-2] +
            u[1:-1, 2:] +
            u[:-2, 1:-1] +
            u[2:, 1:-1]
        )

        u_old_interior = u[1:-1, 1:-1][interior_mask]
        u_new_interior = u_new[interior_mask]

        delta = np.abs(u_old_interior - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

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
    u = jacobi(u0, interior_mask, max_iter=max_iter, atol=atol)
    stats = summary_stats(u, interior_mask)

    return (
        bid,
        stats["mean_temp"],
        stats["std_temp"],
        stats["pct_above_18"],
        stats["pct_below_15"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Parallel wall-heating simulation with dynamic scheduling over floorplans."
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

    if args.time:
        t0 = time.perf_counter()

    if n_workers == 1:
        all_results = [
            process_building(bid, args.max_iter, args.atol)
            for bid in building_ids
        ]
    else:
        with mp.Pool(processes=n_workers) as pool:
            all_results = pool.starmap(
                process_building,
                [(bid, args.max_iter, args.atol) for bid in building_ids],
                chunksize=1
            )

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