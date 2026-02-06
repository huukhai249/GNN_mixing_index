#!/usr/bin/env python3
"""
Optimized Generalized_Nearest_Neighbor.py

- Fast neighbor queries via scipy.spatial.cKDTree (preferred) or sklearn (fallback).
- Saves CSV with columns: time,gnn_mixing_index
- If settings file contains "Yes" for plots, saves PNG of GNN vs time
- Robust settings parsing and edge-case handling
"""
from edempy import Deck
import numpy as np
import os
import re
import logging
import math

# plotting
import matplotlib
matplotlib.use("Agg")  # safe for headless
import matplotlib.pyplot as plt

# Try to import fast KDTree backends
_KD_BACKEND = None
KDTree = None
SKNearest = None
try:
    from scipy.spatial import cKDTree as KDTree
    _KD_BACKEND = "scipy"
except Exception:
    KDTree = None
    try:
        from sklearn.neighbors import NearestNeighbors as SKNearest
        _KD_BACKEND = "sklearn"
    except Exception:
        SKNearest = None
        _KD_BACKEND = None

logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_settings_file(path):
    """
    Parse settings file robustly. Returns:
      (start_time: float or None, end_time: float or None, N_nb: int or None, plots_raw: str or None)
    """
    start_time = None
    end_time = None
    N_nb = None
    plots_raw = None
    try:
        with open(path, "r") as fh:
            text = fh.read()
        # Extract numeric tokens
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if len(nums) >= 3:
            start_time = float(nums[0])
            end_time = float(nums[1])
            try:
                N_nb = int(float(nums[2]))
            except Exception:
                N_nb = int(nums[2])
        # Find 'Yes'/'No' for plots (case-insensitive)
        if re.search(r"\bYes\b", text, flags=re.IGNORECASE):
            plots_raw = "Yes"
        elif re.search(r"\bNo\b", text, flags=re.IGNORECASE):
            plots_raw = "No"
        else:
            # try key=value style
            m = re.search(r"(plots|plot)\s*[:=]\s*([\"']?)(\w+)\2", text, flags=re.IGNORECASE)
            if m:
                plots_raw = m.group(3)
    except Exception as e:
        logging.warning("Failed to parse settings file %s: %s", path, e)
    return start_time, end_time, N_nb, plots_raw


def calculate_gnn_mixing_index(fractions_B_of_A: np.ndarray, fractions_A_of_B: np.ndarray, NA: int, NB: int) -> float:
    """
    GNN mixing index = (sum over A of fraction of B neighbors)/NA + (sum over B of fraction of A neighbors)/NB
    If NA==0 or NB==0 -> return 0.0 (and log a warning).
    """
    if NA <= 0 or NB <= 0:
        logging.warning("NA=%s or NB=%s -> GNN Mixing Index cannot be calculated; returning 0.", NA, NB)
        return 0.0
    sum_A = np.sum(fractions_B_of_A) if fractions_B_of_A.size > 0 else 0.0
    sum_B = np.sum(fractions_A_of_B) if fractions_A_of_B.size > 0 else 0.0
    gnn = float(sum_A / NA + sum_B / NB)
    logging.debug("sum_A=%s sum_B=%s NA=%s NB=%s -> gnn=%s", sum_A, sum_B, NA, NB, gnn)
    return gnn


def compute_fractions(positions: np.ndarray, types_int: np.ndarray, k_neighbors: int):
    """
    Compute per-particle neighbor fractions.
    Returns (fractions_B_of_A, fractions_A_of_B)
    - fractions_B_of_A: for each A particle, fraction of its k neighbors that are B
    - fractions_A_of_B: for each B particle, fraction of its k neighbors that are A
    """
    N = positions.shape[0]
    if N == 0:
        return np.array([]), np.array([])

    k = min(k_neighbors, max(0, N - 1))
    if k == 0:
        return np.array([]), np.array([])

    if _KD_BACKEND == "scipy" and KDTree is not None:
        tree = KDTree(positions)
        # Try n_jobs; some scipy versions don't support it
        try:
            dist, idx = tree.query(positions, k=k + 1, n_jobs=-1)
        except TypeError:
            dist, idx = tree.query(positions, k=k + 1)
        idx = np.atleast_2d(idx)
        self_idx = np.arange(N)[:, None]
        mask = idx != self_idx
        try:
            neighbor_idx = idx[mask].reshape(N, k)
        except Exception:
            neighbor_idx = np.zeros((N, k), dtype=int)
            for i in range(N):
                row = idx[i]
                neighbors = row[row != i][:k]
                if neighbors.size < k:
                    pad_size = k - neighbors.size
                    pad = np.repeat(neighbors[-1] if neighbors.size > 0 else (0 if i != 0 else 1), pad_size)
                    neighbors = np.concatenate([neighbors, pad])
                neighbor_idx[i] = neighbors

    elif _KD_BACKEND == "sklearn" and SKNearest is not None:
        # sklearn returns self as first neighbor when n_neighbors=k+1; exclude self
        knn = SKNearest(n_neighbors=k + 1, algorithm='auto', metric='euclidean', n_jobs=-1)
        knn.fit(positions)
        dist, idx = knn.kneighbors(positions)
        idx = np.atleast_2d(idx)
        self_idx = np.arange(N)[:, None]
        mask = idx != self_idx
        try:
            neighbor_idx = idx[mask].reshape(N, k)
        except Exception:
            neighbor_idx = np.zeros((N, k), dtype=int)
            for i in range(N):
                row = idx[i]
                neighbors = row[row != i][:k]
                if neighbors.size < k:
                    pad_size = k - neighbors.size
                    pad = np.repeat(neighbors[-1] if neighbors.size > 0 else (0 if i != 0 else 1), pad_size)
                    neighbors = np.concatenate([neighbors, pad])
                neighbor_idx[i] = neighbors

    else:
        # Brute-force fallback
        logging.warning("No KD-tree backend available (scipy/sklearn). Using brute-force neighbor search (O(N^2)).")
        diff = positions[:, None, :] - positions[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        neighbor_idx = np.argpartition(dist_sq, k, axis=1)[:, :k]

    neighbor_types = types_int[neighbor_idx]  # shape (N, k)
    is_A = (types_int == 0)
    is_B = (types_int == 1)

    if np.any(is_A):
        rows_A = np.where(is_A)[0]
        if rows_A.size > 0:
            nb_B_counts_A = np.sum(neighbor_types[rows_A] == 1, axis=1)
            fractions_B_of_A = nb_B_counts_A.astype(float) / float(k)
        else:
            fractions_B_of_A = np.array([])
    else:
        fractions_B_of_A = np.array([])

    if np.any(is_B):
        rows_B = np.where(is_B)[0]
        if rows_B.size > 0:
            nb_A_counts_B = np.sum(neighbor_types[rows_B] == 0, axis=1)
            fractions_A_of_B = nb_A_counts_B.astype(float) / float(k)
        else:
            fractions_A_of_B = np.array([])
    else:
        fractions_A_of_B = np.array([])

    return fractions_B_of_A, fractions_A_of_B


def process_dem_file(dem_path: str, settings_path: str):
    logging.info("-------------------------------------------------------")
    logging.info("Loading: %s", dem_path)
    logging.info("-------------------------------------------------------")

    deck = Deck(dem_path)

    if not os.path.exists(settings_path):
        logging.info("No settings file found at %s - skipping.", settings_path)
        return

    start_time, end_time, N_nb, plots_raw = parse_settings_file(settings_path)
    if start_time is None or end_time is None or N_nb is None:
        logging.warning("Settings missing or unparsable in %s - need start_time, end_time, N_nb", settings_path)
        return

    logging.info("start_time: %s", start_time)
    logging.info("end_time: %s", end_time)
    logging.info("N_nb: %s", N_nb)
    logging.info("plots: %s", plots_raw)

    do_plots = False
    if isinstance(plots_raw, str) and plots_raw.strip().lower().startswith("y"):
        do_plots = True

    # Find nearest timesteps by time
    tvals = np.array(deck.timestepValues)
    start_tstep = int(np.abs(tvals - start_time).argmin())
    end_tstep = int(np.abs(tvals - end_time).argmin())
    if end_tstep <= start_tstep:
        logging.warning("end_tstep <= start_tstep -> nothing to do.")
        return

    times_out = tvals[start_tstep:end_tstep]
    num_steps = len(times_out)
    GNN_Mixing_Index = np.zeros(num_steps)

    for idx_t, t_tstep in enumerate(range(start_tstep, end_tstep)):
        time_val = deck.timestepValues[t_tstep]
        logging.info("Processing timestep: %d Time: %s s...", t_tstep, time_val)

        # Collect positions and types
        positions_list = []
        types_list = []
        for ptype in deck.timestep[t_tstep].h5ParticleTypes:
            num = deck.timestep[t_tstep].particle[ptype].numParticles
            if num <= 0:
                continue
            positions = deck.timestep[t_tstep].particle[ptype].getPositions()
            positions_list.append(np.asarray(positions))
            if ptype == '0':
                types_list.append(np.zeros(num, dtype=np.int8))
            else:
                types_list.append(np.ones(num, dtype=np.int8))

        if len(positions_list) == 0:
            logging.info("No particles at this timestep.")
            GNN_Mixing_Index[idx_t] = 0.0
            continue

        positions_all = np.vstack(positions_list)
        types_all = np.concatenate(types_list).astype(np.int8)
        N_total = positions_all.shape[0]

        NA = int(np.sum(types_all == 0))
        NB = int(np.sum(types_all == 1))
        logging.info("Total particles: %d (A=%d, B=%d)", N_total, NA, NB)

        if N_total <= 1:
            logging.info("Not enough particles to compute neighbors.")
            GNN_Mixing_Index[idx_t] = 0.0
            continue

        # Compute neighbor fractions
        fractions_B_of_A, fractions_A_of_B = compute_fractions(positions_all, types_all, N_nb)

        # Compute GNN mixing index
        GNN_Mixing_Index[idx_t] = calculate_gnn_mixing_index(fractions_B_of_A, fractions_A_of_B, NA, NB)

    # Save CSV by time
    base = os.path.splitext(os.path.basename(dem_path))[0]
    out_csv = os.path.join(os.path.dirname(dem_path), f"{base}_GNN_Mixing_Index_by_time.csv")
    try:
        with open(out_csv, "w", newline="") as fh:
            fh.write("time,gnn_mixing_index\n")
            for t_val, val in zip(times_out, GNN_Mixing_Index):
                fh.write(f"{t_val:.9f},{val:.12g}\n")
        logging.info("Saved time-based results to %s", out_csv)
    except Exception as e:
        logging.warning("Failed to save CSV: %s", e)

    # Plot if requested
    if do_plots:
        try:
            plt.figure(figsize=(8, 4.5))
            plt.plot(times_out, GNN_Mixing_Index, marker='o', linestyle='-')
            plt.xlabel("Time (s)")
            plt.ylabel("GNN Mixing Index")
            plt.title(f"GNN Mixing Index vs Time ({base})")
            plt.grid(True)
            plt.tight_layout()
            out_png = os.path.join(os.path.dirname(dem_path), f"{base}_GNN_vs_time.png")
            plt.savefig(out_png, dpi=200)
            plt.close()
            logging.info("Saved plot to %s", out_png)
        except Exception as e:
            logging.warning("Failed to create/save plot: %s", e)


def main():
    # Walk current directory for .dem files (same behavior as original)
    for root, dirs, files in os.walk(os.curdir):
        for file in files:
            if file.endswith(".dem"):
                dem_path = os.path.join(root, file)
                settings_path = os.path.join(root, "GNNMixingIdx_settings.txt")
                process_dem_file(dem_path, settings_path)


if __name__ == "__main__":
    main()