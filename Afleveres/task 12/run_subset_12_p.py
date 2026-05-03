import argparse
import math
import multiprocessing as mp
from os.path import join
import time

import numpy as np


LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
GRID_SIZE = 512
PADDED_GRID_SIZE = GRID_SIZE + 2
PADDED_GRID_N = PADDED_GRID_SIZE * PADDED_GRID_SIZE
REDUCE_THREADS = 256


_CUDA = None
_JACOBI_STEP_KERNEL = None
_DELTA_KERNEL = None
_STATS_KERNEL = None


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


def choose_block_size(delta, atol, large_block, medium_block, small_block, near_block):
    delta_ratio = delta / atol
    if delta_ratio > 1_000.0:
        return large_block
    if delta_ratio > 100.0:
        return medium_block
    if delta_ratio > 10.0:
        return small_block
    return near_block


def jacobi_adaptive_numpy(
    u0,
    interior_mask,
    max_iter=20_000,
    atol=1e-4,
    large_block=2_500,
    medium_block=1_500,
    small_block=500,
    near_block=50,
):
    """
    Adaptive-block Jacobi solver using NumPy.
    """
    u = np.copy(u0)
    iterations_done = 0
    block_size = large_block

    while iterations_done < max_iter:
        remaining = max_iter - iterations_done
        this_block = min(block_size, remaining)
        delta = np.nan

        for _ in range(this_block):
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

        iterations_done += this_block

        if delta < atol:
            break

        block_size = choose_block_size(
            float(delta),
            atol,
            large_block=large_block,
            medium_block=medium_block,
            small_block=small_block,
            near_block=near_block,
        )

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


def _get_cuda_tools():
    global _CUDA, _JACOBI_STEP_KERNEL, _DELTA_KERNEL, _STATS_KERNEL
    if _JACOBI_STEP_KERNEL is not None:
        return (
            _CUDA,
            _JACOBI_STEP_KERNEL,
            _DELTA_KERNEL,
            _STATS_KERNEL,
        )

    cuda = _require_cuda()

    @cuda.jit
    def jacobi_step_kernel(u, u_new, interior_mask):
        """
        One Jacobi iteration.
        u, u_new: (514, 514) float64
        interior_mask: (512, 512) bool
        """
        i, j = cuda.grid(2)
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

    @cuda.jit
    def delta_kernel(u, u_new, interior_mask, delta_out):
        smax = cuda.shared.array(REDUCE_THREADS, dtype=np.float64)
        tid = cuda.threadIdx.x
        stride = cuda.blockDim.x

        local_max = 0.0
        for idx in range(tid, GRID_SIZE * GRID_SIZE, stride):
            i = idx // GRID_SIZE
            j = idx % GRID_SIZE
            if interior_mask[i, j]:
                ii = i + 1
                jj = j + 1
                diff = u_new[ii, jj] - u[ii, jj]
                if diff < 0.0:
                    diff = -diff
                if diff > local_max:
                    local_max = diff

        smax[tid] = local_max
        cuda.syncthreads()

        offset = cuda.blockDim.x // 2
        while offset > 0:
            if tid < offset and smax[tid + offset] > smax[tid]:
                smax[tid] = smax[tid + offset]
            cuda.syncthreads()
            offset //= 2

        if tid == 0:
            delta_out[0] = smax[0]

    @cuda.jit
    def stats_kernel(u, interior_mask, stats_out):
        ssum = cuda.shared.array(REDUCE_THREADS, dtype=np.float64)
        ssum_sq = cuda.shared.array(REDUCE_THREADS, dtype=np.float64)
        sabove = cuda.shared.array(REDUCE_THREADS, dtype=np.float64)
        sbelow = cuda.shared.array(REDUCE_THREADS, dtype=np.float64)
        scount = cuda.shared.array(REDUCE_THREADS, dtype=np.float64)

        tid = cuda.threadIdx.x
        stride = cuda.blockDim.x

        local_sum = 0.0
        local_sum_sq = 0.0
        local_above = 0.0
        local_below = 0.0
        local_count = 0.0

        for idx in range(tid, GRID_SIZE * GRID_SIZE, stride):
            i = idx // GRID_SIZE
            j = idx % GRID_SIZE
            if interior_mask[i, j]:
                val = u[i + 1, j + 1]
                local_sum += val
                local_sum_sq += val * val
                local_count += 1.0
                if val > 18.0:
                    local_above += 1.0
                if val < 15.0:
                    local_below += 1.0

        ssum[tid] = local_sum
        ssum_sq[tid] = local_sum_sq
        sabove[tid] = local_above
        sbelow[tid] = local_below
        scount[tid] = local_count
        cuda.syncthreads()

        offset = cuda.blockDim.x // 2
        while offset > 0:
            if tid < offset:
                ssum[tid] += ssum[tid + offset]
                ssum_sq[tid] += ssum_sq[tid + offset]
                sabove[tid] += sabove[tid + offset]
                sbelow[tid] += sbelow[tid + offset]
                scount[tid] += scount[tid + offset]
            cuda.syncthreads()
            offset //= 2

        if tid == 0:
            total = ssum[0]
            total_sq = ssum_sq[0]
            above = sabove[0]
            below = sbelow[0]
            count = scount[0]
            mean = total / count
            variance = total_sq / count - mean * mean
            if variance < 0.0:
                variance = 0.0
            stats_out[0] = mean
            stats_out[1] = math.sqrt(variance)
            stats_out[2] = above / count * 100.0
            stats_out[3] = below / count * 100.0

    _CUDA = cuda
    _JACOBI_STEP_KERNEL = jacobi_step_kernel
    _DELTA_KERNEL = delta_kernel
    _STATS_KERNEL = stats_kernel
    return (
        _CUDA,
        _JACOBI_STEP_KERNEL,
        _DELTA_KERNEL,
        _STATS_KERNEL,
    )


def jacobi_adaptive_cuda(
    u0,
    interior_mask,
    max_iter=20_000,
    atol=1e-4,
    large_block=2_500,
    medium_block=1_500,
    small_block=500,
    near_block=50,
    threadsperblock=(16, 16),
    return_device=False,
    workspace=None,
):
    """
    Adaptive-block Jacobi solver using the custom CUDA kernel from Task 8.
    Delta is computed on the GPU at block boundaries, and only one scalar is
    copied back to the CPU for the convergence decision.
    """
    if u0.shape != (GRID_SIZE + 2, GRID_SIZE + 2):
        raise ValueError(f"u0 must have shape {(GRID_SIZE + 2, GRID_SIZE + 2)}")
    if interior_mask.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"interior_mask must have shape {(GRID_SIZE, GRID_SIZE)}")

    cuda, jacobi_step_kernel, delta_kernel, _ = _get_cuda_tools()

    if workspace is None:
        d_u = cuda.to_device(u0)
        d_u_new = cuda.to_device(u0)
        d_mask = cuda.to_device(interior_mask)
        d_delta = cuda.device_array(1, dtype=np.float64)
    else:
        d_u = workspace["d_u"]
        d_u_new = workspace["d_u_new"]
        d_mask = workspace["d_mask"]
        d_delta = workspace["d_delta"]
        d_u.copy_to_device(u0)
        d_u_new.copy_to_device(u0)
        d_mask.copy_to_device(interior_mask)

    blockspergrid = (
        math.ceil(GRID_SIZE / threadsperblock[0]),
        math.ceil(GRID_SIZE / threadsperblock[1]),
    )
    iterations_done = 0
    block_size = large_block

    while iterations_done < max_iter:
        remaining = max_iter - iterations_done
        this_block = min(block_size, remaining)
        delta = np.nan

        for iteration_in_block in range(this_block):
            jacobi_step_kernel[blockspergrid, threadsperblock](d_u, d_u_new, d_mask)

            if iteration_in_block == this_block - 1:
                delta_kernel[1, REDUCE_THREADS](d_u, d_u_new, d_mask, d_delta)
                delta = float(d_delta.copy_to_host()[0])

            d_u, d_u_new = d_u_new, d_u

        iterations_done += this_block

        if delta < atol:
            break

        block_size = choose_block_size(
            delta,
            atol,
            large_block=large_block,
            medium_block=medium_block,
            small_block=small_block,
            near_block=near_block,
        )

    if return_device:
        return d_u, d_mask

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


def init_cuda_workspace():
    cuda, _, _, _ = _get_cuda_tools()

    return {
        "d_u": cuda.device_array((PADDED_GRID_SIZE, PADDED_GRID_SIZE), dtype=np.float64),
        "d_u_new": cuda.device_array((PADDED_GRID_SIZE, PADDED_GRID_SIZE), dtype=np.float64),
        "d_mask": cuda.device_array((GRID_SIZE, GRID_SIZE), dtype=np.bool_),
        "d_delta": cuda.device_array(1, dtype=np.float64),
        "d_stats": cuda.device_array(4, dtype=np.float64),
    }


def summary_stats_cuda(d_u, d_mask, workspace=None):
    cuda, _, _, stats_kernel = _get_cuda_tools()
    if workspace is None:
        d_stats = cuda.device_array(4, dtype=np.float64)
    else:
        d_stats = workspace["d_stats"]

    stats_kernel[1, REDUCE_THREADS](d_u, d_mask, d_stats)
    stats = d_stats.copy_to_host()

    return {
        "mean_temp": float(stats[0]),
        "std_temp": float(stats[1]),
        "pct_above_18": float(stats[2]),
        "pct_below_15": float(stats[3]),
    }


def process_building(
    bid,
    load_dir,
    max_iter,
    atol,
    large_block,
    medium_block,
    small_block,
    near_block,
    backend,
    threadsperblock,
    workspace=None,
):
    u0, interior_mask = load_data(load_dir, bid)

    if backend == "cuda":
        d_u, d_mask = jacobi_adaptive_cuda(
            u0,
            interior_mask,
            max_iter=max_iter,
            atol=atol,
            large_block=large_block,
            medium_block=medium_block,
            small_block=small_block,
            near_block=near_block,
            threadsperblock=threadsperblock,
            return_device=True,
            workspace=workspace,
        )
        stats = summary_stats_cuda(d_u, d_mask, workspace=workspace)
    elif backend == "numpy":
        u = jacobi_adaptive_numpy(
            u0,
            interior_mask,
            max_iter=max_iter,
            atol=atol,
            large_block=large_block,
            medium_block=medium_block,
            small_block=small_block,
            near_block=near_block,
        )
        stats = summary_stats(u, interior_mask)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return (
        bid,
        stats["mean_temp"],
        stats["std_temp"],
        stats["pct_above_18"],
        stats["pct_below_15"],
    )


def process_building_chunk(
    gpu_id,
    building_ids,
    load_dir,
    max_iter,
    atol,
    large_block,
    medium_block,
    small_block,
    near_block,
    backend,
    threadsperblock,
):
    if backend == "cuda":
        from numba import cuda

        cuda.select_device(gpu_id)
        workspace = None
        if building_ids:
            workspace = init_cuda_workspace()
            u0, interior_mask = load_data(load_dir, building_ids[0])
            _ = jacobi_adaptive_cuda(
                u0,
                interior_mask,
                max_iter=1,
                atol=atol,
                large_block=1,
                medium_block=1,
                small_block=1,
                near_block=1,
                threadsperblock=threadsperblock,
                workspace=workspace,
            )
    else:
        workspace = None

    return [
        process_building(
            bid,
            load_dir=load_dir,
            max_iter=max_iter,
            atol=atol,
            large_block=large_block,
            medium_block=medium_block,
            small_block=small_block,
            near_block=near_block,
            backend=backend,
            threadsperblock=threadsperblock,
            workspace=workspace,
        )
        for bid in building_ids
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Wall-heating simulation using adaptive-block Jacobi (NumPy vs CUDA kernel)."
    )
    parser.add_argument("N", type=int, nargs="?", default=1,
                        help="Number of floorplans to process")
    parser.add_argument("--load-dir", default=LOAD_DIR,
                        help="Directory containing building_ids.txt and the .npy files")
    parser.add_argument("--backend", choices=("cuda", "numpy"), default="cuda",
                        help="Implementation to run (default: cuda)")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs to use for the CUDA backend")
    parser.add_argument("--max-iter", type=int, default=20_000,
                        help="Maximum number of Jacobi iterations")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="Convergence tolerance")
    parser.add_argument("--large-block", type=int, default=2_500,
                        help="Block size when far from convergence")
    parser.add_argument("--medium-block", type=int, default=1_500,
                        help="Block size for intermediate delta / atol")
    parser.add_argument("--small-block", type=int, default=500,
                        help="Block size when getting close")
    parser.add_argument("--near-block", type=int, default=50,
                        help="Block size when very close to convergence")
    parser.add_argument("--tpb-x", type=int, default=16,
                        help="CUDA threads-per-block in X (i dimension)")
    parser.add_argument("--tpb-y", type=int, default=16,
                        help="CUDA threads-per-block in Y (j dimension)")
    parser.add_argument("--verify", action="store_true",
                        help="Run a small correctness check: CUDA vs NumPy on one building")
    parser.add_argument("--verify-tol", type=float, default=1e-6,
                        help="Max abs-diff tolerance for --verify")
    parser.add_argument("--time", action="store_true",
                        help="Print timing information")
    args = parser.parse_args()

    load_dir = args.load_dir
    threadsperblock = (args.tpb_x, args.tpb_y)

    building_ids = load_building_ids(load_dir)[:args.N]

    if args.verify:
        bid = building_ids[0] if building_ids else None
        if bid is None:
            raise SystemExit("No buildings selected (N=0).")

        u0, interior_mask = load_data(load_dir, bid)
        u_cuda = jacobi_adaptive_cuda(
            u0,
            interior_mask,
            max_iter=args.max_iter,
            atol=args.atol,
            large_block=args.large_block,
            medium_block=args.medium_block,
            small_block=args.small_block,
            near_block=args.near_block,
            threadsperblock=threadsperblock,
            return_device=False,
        )
        u_numpy = jacobi_adaptive_numpy(
            u0,
            interior_mask,
            max_iter=args.max_iter,
            atol=args.atol,
            large_block=args.large_block,
            medium_block=args.medium_block,
            small_block=args.small_block,
            near_block=args.near_block,
        )

        max_abs_diff = float(np.max(np.abs(u_cuda - u_numpy)))
        print(f"# Verify building {bid}: max_abs_diff={max_abs_diff:.6e}")
        if max_abs_diff > args.verify_tol:
            raise SystemExit(
                f"Verification failed: max_abs_diff {max_abs_diff:.6e} > tol {args.verify_tol:.6e}"
            )
        print("# Verification passed")
        return

    if args.backend == "cuda" and building_ids and args.num_gpus == 1:
        u0, interior_mask = load_data(load_dir, building_ids[0])
        workspace = init_cuda_workspace()
        _ = jacobi_adaptive_cuda(
            u0,
            interior_mask,
            max_iter=1,
            atol=args.atol,
            large_block=1,
            medium_block=1,
            small_block=1,
            near_block=1,
            threadsperblock=threadsperblock,
            workspace=workspace,
        )
    else:
        workspace = None

    if args.time:
        t0 = time.perf_counter()

    if args.backend == "cuda" and args.num_gpus > 1 and building_ids:
        worker_count = min(args.num_gpus, len(building_ids))
        chunks = [
            building_ids[i::worker_count]
            for i in range(worker_count)
            if building_ids[i::worker_count]
        ]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=worker_count) as pool:
            chunk_results = pool.starmap(
                process_building_chunk,
                [
                    (
                        gpu_id,
                        chunk,
                        load_dir,
                        args.max_iter,
                        args.atol,
                        args.large_block,
                        args.medium_block,
                        args.small_block,
                        args.near_block,
                        args.backend,
                        threadsperblock,
                    )
                    for gpu_id, chunk in enumerate(chunks)
                ],
            )
        order = {bid: idx for idx, bid in enumerate(building_ids)}
        all_results = [item for chunk in chunk_results for item in chunk]
        all_results.sort(key=lambda row: order[row[0]])
    else:
        all_results = [
            process_building(
                bid,
                load_dir=load_dir,
                max_iter=args.max_iter,
                atol=args.atol,
                large_block=args.large_block,
                medium_block=args.medium_block,
                small_block=args.small_block,
                near_block=args.near_block,
                backend=args.backend,
                threadsperblock=threadsperblock,
                workspace=workspace,
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
        if args.backend == "cuda":
            print(f"# GPUs: {min(args.num_gpus, len(building_ids)) if building_ids else 0}")
        print(f"# Max iterations: {args.max_iter}")
        print(
            "# Adaptive blocks:"
            f" large={args.large_block}"
            f" medium={args.medium_block}"
            f" small={args.small_block}"
            f" near={args.near_block}"
        )
        print(f"# Buildings: {len(building_ids)}")


if __name__ == "__main__":
    main()
