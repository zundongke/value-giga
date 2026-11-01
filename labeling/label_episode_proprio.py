"""Label one episode with milestone + critical-step from proprioception only (no VLM).

Critical events are detected from gripper-width threshold crossings on
observation.state channels 6 (left) and 13 (right), plus optional arm-velocity
dwell detection for press/contact events.

Milestone stages are derived from the critical events:
  - Each critical event boundary increments the stage counter.
  - stage_label[t] = number of critical-event-ends that have occurred by frame t.

This produces the same JSON format as label_episode.py (VLM version), so the
same visualization code works on both.

Usage:
  python label_episode_proprio.py --task task_a --session 2026-05-09-v2 --episode 9
  python label_episode_proprio.py --task aloha_coffee --episode 0
  python label_episode_proprio.py --task aloha_towel --episode 5
  python label_episode_proprio.py --task aloha_candy --episode 3

Output: ./labels_proprio/<task>/ep009.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------

TASKS = {
    "task_a": {
        "data_root": "/vePFS-East/comp_robot/zundong/Task_A/base/base",
        "has_sessions": True,
        "src_fps": 30.0,
        "task_text": "Flatten and fold the cloth.",
        "gripper_channels": [6, 13],
        "arm_channels": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
    },
    "aloha_coffee": {
        "data_root": "/vePFS-East/comp_robot/zundong/ViVa_pic/data/lerobot/aloha_static_coffee",
        "has_sessions": False,
        "src_fps": 50.0,
        "task_text": (
            "Place the coffee capsule inside the capsule container, then "
            "place the cup onto the center of the cup tray, then push the "
            "'Hot Water' and 'Travel Mug' buttons."
        ),
        "gripper_channels": [6, 13],
        "arm_channels": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
    },
    "aloha_towel": {
        "data_root": "/vePFS-East/comp_robot/zundong/ViVa_pic/data/lerobot/aloha_static_towel",
        "has_sessions": False,
        "src_fps": 50.0,
        "task_text": "Pick up a piece of paper towel and place it on the spilled liquid.",
        "gripper_channels": [6, 13],
        "arm_channels": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
    },
    "aloha_candy": {
        "data_root": "/vePFS-East/comp_robot/zundong/ViVa_pic/data/lerobot/aloha_static_candy",
        "has_sessions": False,
        "src_fps": 50.0,
        "task_text": "Pick up the candy and unwrap it.",
        "gripper_channels": [6, 13],
        "arm_channels": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
    },
}

# ---------------------------------------------------------------------------
# Detection parameters
# ---------------------------------------------------------------------------

WINDOW_HALF = 1       # ±1 frame around each transition
MIN_HOLD = 8          # ignore close/open pairs held < 8 frames (chatter filter)
DWELL_MIN_RUN = 10    # minimum consecutive low-speed frames to count as dwell
DWELL_PCTILE = 12     # velocity percentile threshold for dwell detection
MAX_DWELL = 3         # max dwell events per episode
GRIPPER_MIN_RANGE = 0.02  # skip channel if range < this (never opened/closed)


# ---------------------------------------------------------------------------
# Gripper crossing detection
# ---------------------------------------------------------------------------

def crossings(g: np.ndarray, thr: float):
    closed = g < thr
    changes = np.diff(closed.astype(int))
    closes = (np.where(changes == 1)[0] + 1).tolist()
    opens = (np.where(changes == -1)[0] + 1).tolist()
    return closes, opens


def filter_chatter(closes: list, opens: list, min_hold: int):
    events = sorted([(c, "c") for c in closes] + [(o, "o") for o in opens])
    keep = []
    last_c = None
    for f, k in events:
        if k == "c":
            last_c = f
        else:
            if last_c is not None and (f - last_c) >= min_hold:
                keep.append(last_c)
                keep.append(f)
            last_c = None
    return keep


# ---------------------------------------------------------------------------
# Dwell detection (arm velocity)
# ---------------------------------------------------------------------------

def detect_dwells(state: np.ndarray, arm_ch: list, T: int,
                  grip_events: list) -> list[int]:
    vel = np.linalg.norm(np.diff(state[:, arm_ch], axis=0), axis=1)
    vel = np.concatenate([[vel[0]], vel])
    smoothed = np.convolve(vel, np.ones(5) / 5, mode="same")
    thr = np.percentile(smoothed, DWELL_PCTILE)
    low = smoothed < thr

    runs = []
    i = 0
    while i < T:
        if low[i]:
            j = i
            while j < T and low[j]:
                j += 1
            if j - i >= DWELL_MIN_RUN:
                mid = (i + j) // 2
                runs.append((mid, float(smoothed[mid])))
            i = j
        else:
            i += 1

    # Keep only dwells that are far from gripper events and away from edges
    dwells = [
        t for t, _ in sorted(runs, key=lambda x: x[1])
        if 15 < t < T - 10 and all(abs(t - ge) > 20 for ge in grip_events)
    ]
    return sorted(dwells[:MAX_DWELL])


# ---------------------------------------------------------------------------
# Label one episode
# ---------------------------------------------------------------------------

def label_episode(cfg: dict, session: str | None, episode: int,
                  enable_dwell: bool = True):
    root = Path(cfg["data_root"])
    if cfg["has_sessions"]:
        root = root / session
    pq = root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"parquet not found: {pq}")

    df = pd.read_parquet(pq)
    state = np.stack(df["observation.state"].to_numpy())
    T = len(state)

    # --- Gripper events ---
    grip_events = []
    for ch in cfg["gripper_channels"]:
        g = state[:, ch]
        if g.max() - g.min() < GRIPPER_MIN_RANGE:
            continue
        thr = (g.min() + g.max()) / 2
        c, o = crossings(g, thr)
        grip_events.extend(filter_chatter(c, o, MIN_HOLD))
    grip_events = sorted(set(grip_events))

    # --- Dwell events ---
    dwell_events = []
    if enable_dwell:
        dwell_events = detect_dwells(state, cfg["arm_channels"], T, grip_events)

    # --- Combine all events ---
    # event_kind: 0 = gripper (reliable), 1 = dwell (lower confidence)
    all_events = sorted(
        [(t, 0) for t in grip_events] + [(t, 1) for t in dwell_events]
    )

    # --- Build critical mask and windows ---
    critical_mask = np.zeros(T, dtype=bool)
    critical_windows = []
    event_kinds = []
    event_ends = []

    for t, kind in all_events:
        a = max(0, t - WINDOW_HALF)
        b = min(T, t + WINDOW_HALF + 1)
        critical_mask[a:b] = True
        critical_windows.append([a, b])
        event_kinds.append(kind)
        event_ends.append(b - 1)

    # --- Build milestone (stage) label ---
    # stage = cumulative count of critical events completed
    stage = np.zeros(T, dtype=int)
    for end in sorted(event_ends):
        stage[end:] += 1

    # --- is_milestone: frames where stage transitions ---
    is_milestone = np.zeros(T, dtype=bool)
    milestone_cuts = []
    if len(stage) > 0:
        changes = np.where(np.diff(stage) > 0)[0] + 1
        for c in changes:
            is_milestone[c] = True
            milestone_cuts.append(int(c))

    n_stages = int(stage.max()) + 1
    stage_names = [f"stage_{i}" for i in range(n_stages)]

    return {
        "episode_id": episode,
        "session": session,
        "total_frames": T,
        "src_fps": cfg["src_fps"],
        "task": cfg["task_text"],
        "label_source": "proprioception",
        "stages": stage_names,
        "milestone_cuts": milestone_cuts,
        "critical_windows": critical_windows,
        "event_kinds": event_kinds,
        "num_gripper_events": len(grip_events),
        "num_dwell_events": len(dwell_events),
        "per_frame": {
            "stage": stage.tolist(),
            "is_milestone": is_milestone.tolist(),
            "is_critical": critical_mask.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Label one episode with milestone + critical-step from proprioception."
    )
    ap.add_argument("--task", required=True, choices=list(TASKS.keys()))
    ap.add_argument("--session", default=None,
                    help="session name (required for task_a)")
    ap.add_argument("--episode", required=True, type=int)
    ap.add_argument("--no-dwell", action="store_true",
                    help="disable arm-velocity dwell detection (gripper only)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: ./labels_proprio/<task>/)")
    args = ap.parse_args()

    cfg = TASKS[args.task]
    if cfg["has_sessions"] and not args.session:
        raise SystemExit(f"--session is required for task '{args.task}'")

    print(f"labeling episode {args.episode} from proprioception ...")
    result = label_episode(cfg, args.session, args.episode,
                           enable_dwell=not args.no_dwell)

    out_dir = args.out_dir or Path("labels_proprio") / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ep{args.episode:03d}.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    T = result["total_frames"]
    n_crit = sum(result["per_frame"]["is_critical"])
    n_ms = len(result["milestone_cuts"])
    n_stages = int(max(result["per_frame"]["stage"])) + 1
    print(f"  T={T}  gripper_events={result['num_gripper_events']}  "
          f"dwell_events={result['num_dwell_events']}")
    print(f"  critical windows: {len(result['critical_windows'])}  "
          f"critical frames: {n_crit}/{T} ({100*n_crit/T:.1f}%)")
    print(f"  milestones: {n_ms}  stages: {n_stages}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
