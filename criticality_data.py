"""Proprio-only data layer for the critical-frame classifier.

Pure helpers (normalize_state, make_windows) are import-light and unit-tested.
extract_episode_arrays / CriticalityWindowDataset (added later) depend on the
heavy LeRobot stack and are smoke-verified, not unit-tested.
"""
import numpy as np


def normalize_state(state, state_min, state_max):
    """Per-channel min-max normalize to [-1, 1] with clamping.
    state: [..., C]; state_min/max: [C]. Returns same shape as state."""
    state = np.asarray(state, dtype=np.float64)
    state_min = np.asarray(state_min, dtype=np.float64)
    state_max = np.asarray(state_max, dtype=np.float64)
    rng = np.maximum(state_max - state_min, 1e-8)
    clamped = np.clip(state, state_min, state_max)
    return (clamped - state_min) / rng * 2.0 - 1.0


def make_windows(state_traj, crit, W, causal=True):
    """Slide a length-W window over one episode trajectory.

    state_traj: [T, C] (already normalized). crit: [T] bool.
    causal=True  -> window indices [t-W+1 .. t]   (label = crit[t], last col = t)
    causal=False -> window indices [t-W//2 .. t-W//2+W-1] (centered, offline only)
    Edge frames are clamped to [0, T-1] (edge-padding).
    Returns X [T, C, W], y [T] (uint8/bool).
    """
    state_traj = np.asarray(state_traj, dtype=np.float64)
    crit = np.asarray(crit).astype(bool)
    T, C = state_traj.shape
    half = W // 2
    windows = np.empty((T, C, W), dtype=np.float64)
    for t in range(T):
        if causal:
            idx = np.arange(t - W + 1, t + 1)
        else:
            idx = np.arange(t - half, t - half + W)
        idx = np.clip(idx, 0, T - 1)
        windows[t] = state_traj[idx].T   # [C, W]
    return windows, crit.copy()


import json
import os

import torch


def extract_episode_arrays(data_paths, milestones_jsonl, cache_path,
                           subdataset_ep_offsets=None, force=False):
    """Iterate the LeRobot dataset once (no video decode) and build per-episode
    raw-state trajectories + critical-window labels. Cached to cache_path (npz).

    Returns dict: ep_idx -> {'state': [T,14] float32, 'crit': [T] bool,
                             'windows': [(cs,ce), ...]}.
    """
    if os.path.exists(cache_path) and not force:
        blob = np.load(cache_path, allow_pickle=True)
        return blob["episodes"].item()

    from viva_dataset import VivaDatasetDSM
    ds = VivaDatasetDSM(
        data_paths=list(data_paths),
        milestones_jsonl=milestones_jsonl,
        subdataset_ep_offsets=subdataset_ep_offsets,
        skip_video_decoding=True,
        dual_slope_r=1.0,            # irrelevant; we never read value here
    )

    episodes = {}
    n = len(ds.lerobot_dataset)
    for i in range(n):
        sample = ds.lerobot_dataset[i]
        state = sample.get("observation.state")
        state = state.float().numpy() if isinstance(state, torch.Tensor) \
            else np.asarray(state, dtype=np.float32)
        fi = int(sample["frame_index"])
        ep_local = int(sample["episode_index"])
        sub = ds._get_subdataset_index(i)
        ep = (subdataset_ep_offsets[sub] + ep_local) if subdataset_ep_offsets else ep_local
        cws = ds._episode_critical_windows.get(ep, [])
        crit = any(cs <= fi < ce for cs, ce in cws)
        rec = episodes.setdefault(ep, {"state": {}, "windows": cws})
        rec["state"][fi] = (state, crit)

    out = {}
    for ep, rec in episodes.items():
        T = max(rec["state"]) + 1
        state_arr = np.zeros((T, 14), dtype=np.float32)
        crit_arr = np.zeros(T, dtype=bool)
        for fi, (s, c) in rec["state"].items():
            state_arr[fi] = s
            crit_arr[fi] = c
        out[ep] = {"state": state_arr, "crit": crit_arr, "windows": rec["windows"]}

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, episodes=np.array(out, dtype=object))
    return out


def compute_state_stats(episodes, ep_ids):
    """Per-channel min/max over the given episodes' raw states."""
    states = np.concatenate([episodes[e]["state"] for e in ep_ids], axis=0)
    return states.min(axis=0), states.max(axis=0)


class CriticalityWindowDataset(torch.utils.data.Dataset):
    """Flattened (window, label) dataset over a set of episodes.

    Normalization stats are supplied (fit on the TRAIN split, reused for val).
    """

    def __init__(self, episodes, ep_ids, state_min, state_max, W, causal=True):
        self.state_min = np.asarray(state_min, dtype=np.float64)
        self.state_max = np.asarray(state_max, dtype=np.float64)
        xs, ys = [], []
        for e in ep_ids:
            traj = normalize_state(episodes[e]["state"], self.state_min, self.state_max)
            X, y = make_windows(traj, episodes[e]["crit"], W, causal=causal)
            xs.append(X)
            ys.append(y)
        self.X = np.concatenate(xs, axis=0).astype(np.float32)   # [N, 14, W]
        self.y = np.concatenate(ys, axis=0).astype(np.float32)   # [N]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])
