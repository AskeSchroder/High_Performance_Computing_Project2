import argparse
import math
from os.path import join
import time

import numpy as np


LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
GRID_SIZE = 512


_CUDA = None
_JACOBI_STEP_KERNEL = None


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


def jacobi_fixed_numpy(u0, interior_mask, max_iter=20_000):
    """
    Fixed-iteration Jacobi solver using NumPy.
    (No early stopping; matches the CUDA version's behavior.)
    """
    u = np.copy(u0)

    for _ in range(max_iter):
        u_new = 0.25 * (
            u[1:-1, :-2] +   # left
            u[1:-1, 2:] +    # right
            u[:-2, 1:-1] +   # up
            u[2:, 1:-1]      # down
        )

        u[1:-1, 1:-1][interior_mask] = u_new[interior_mask]

    return u


def _require_cuda():
    try:
        from numba import cuda  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Numba CUDA is not available in this Python environment. "
            "Load a CUDA-enabled module/conda env with numba+cuda."
        ) from exc

    from numba import cuda

    if not cuda.is_available():
        raise SystemExit(
            "CUDA is not available (no GPU / no CUDA driver visible). "
            "Run this script on a CUDA-capable node."
        )

    return cuda


def _get_cuda_kernel():
    global _CUDA, _JACOBI_STEP_KERNEL
    if _JACOBI_STEP_KERNEL is not None:
        return _CUDA, _JACOBI_STEP_KERNEL

    cuda = _require_cuda()

    @cuda.jit
    def jacobi_step_kernel(u, u_new, interior_mask):
        """
        One Jacobi iteration.
        u, u_new: (514, 514) float64
        interior_mask: (512, 512) bool
        """
        i, j = cuda.grid(2)  # indices into the 512x512 interior
        if i < GRID_SIZE and j < GRID_SIZE:
            ii = i + 1
            jj = j + 1

            if interior_mask[i, j]:
                u_new[ii, jj] = 0.25 * (
                    u[ii, jj - 1] +
                    u[ii, jj + 1] +
                    u[ii - 1, jj] +
                    u[ii + 1, jj]
                )
            else:
                u_new[ii, jj] = u[ii, jj]

    _CUDA = cuda
    _JACOBI_STEP_KERNEL = jacobi_step_kernel
    return _CUDA, _JACOBI_STEP_KERNEL


def jacobi_fixed_cuda(u0, interior_mask, max_iter=20_000, threadsperblock=(16, 16)):
    """
    Fixed-iteration Jacobi solver using a custom CUDA kernel.

    The kernel performs exactly one Jacobi iteration, so the host repeatedly
    launches the kernel and swaps the two device buffers between iterations.
    """
    if u0.shape != (GRID_SIZE + 2, GRID_SIZE + 2):
        raise ValueError(f"u0 must have shape {(GRID_SIZE + 2, GRID_SIZE + 2)}")
    if interior_mask.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"interior_mask must have shape {(GRID_SIZE, GRID_SIZE)}")

    cuda, jacobi_step_kernel = _get_cuda_kernel()

    d_u = cuda.to_device(u0)
    d_u_new = cuda.to_device(u0)
    d_mask = cuda.to_device(interior_mask)

    blockspergrid = (
        math.ceil(GRID_SIZE / threadsperblock[0]),
        math.ceil(GRID_SIZE / threadsperblock[1]),
    )

    for _ in range(max_iter):
        jacobi_step_kernel[blockspergrid, threadsperblock](d_u, d_u_new, d_mask)
        d_u, d_u_new = d_u_new, d_u

    return d_u.copy_to_host()


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


def process_building(bid, load_dir, max_iter, backend, threadsperblock):
    u0, interior_mask = load_data(load_dir, bid)

    if backend == "cuda":
        u = jacobi_fixed_cuda(
            u0,
            interior_mask,
            max_iter=max_iter,
            threadsperblock=threadsperblock,
        )
    elif backend == "numpy":
        u = jacobi_fixed_numpy(u0, interior_mask, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown backend: {backend}")

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
        description="Wall-heating simulation using fixed-iteration Jacobi (NumPy vs CUDA kernel)."
    )
    parser.add_argument("N", type=int, nargs="?", default=1,
                        help="Number of floorplans to process")
    parser.add_argument("--load-dir", default=LOAD_DIR,
                        help="Directory containing building_ids.txt and the .npy files")
    parser.add_argument("--backend", choices=("cuda", "numpy"), default="cuda",
                        help="Implementation to run (default: cuda)")
    parser.add_argument("--max-iter", type=int, default=20_000,
                        help="Fixed number of Jacobi iterations (no early stopping)")
    parser.add_argument("--tpb-x", type=int, default=16,
                        help="CUDA threads-per-block in X (i dimension)")
    parser.add_argument("--tpb-y", type=int, default=16,
                        help="CUDA threads-per-block in Y (j dimension)")
    parser.add_argument("--time", action="store_true",
                        help="Print timing information")
    args = parser.parse_args()

    load_dir = args.load_dir
    threadsperblock = (args.tpb_x, args.tpb_y)

    building_ids = load_building_ids(load_dir)[:args.N]

    # Warm-up compilation for CUDA outside the timed region.
    if args.backend == "cuda" and building_ids:
        u0, interior_mask = load_data(load_dir, building_ids[0])
        _ = jacobi_fixed_cuda(
            u0,
            interior_mask,
            max_iter=1,
            threadsperblock=threadsperblock,
        )

    if args.time:
        t0 = time.perf_counter()

    all_results = [
        process_building(
            bid,
            load_dir=load_dir,
            max_iter=args.max_iter,
            backend=args.backend,
            threadsperblock=threadsperblock,
        )
        for bid in building_ids
    ]

    if args.time:
        t1 = time.perf_counter()

    print("building_id,mean_temp,std_temp,pct_above_18,pct_below_15")
    for bid, mean_temp, std_temp, pct_above_18, pct_below_15 in all_results:
        print(f"{bid},{mean_temp},{std_temp},{pct_above_18},{pct_below_15}")

    if args.time:
        print(f"# Total runtime: {t1 - t0:.3f} seconds")
        print(f"# Backend: {args.backend}")
        print(f"# Iterations: {args.max_iter}")
        print(f"# Buildings: {len(building_ids)}")


if __name__ == "__main__":
    main()
