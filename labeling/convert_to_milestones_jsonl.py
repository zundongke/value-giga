"""Convert label JSONs (from label_episode.py / label_episode_proprio.py)
to the milestones JSONL format consumed by MM/ViVa's viva_dataset.py.

Usage:
  python convert_to_milestones_jsonl.py \
      --label-dir labels_vlm_all \
      --out milestones_vlm_all.jsonl

  python convert_to_milestones_jsonl.py \
      --label-dir labels_proprio_all \
      --out milestones_proprio_all.jsonl

Output format (one JSON line per episode):
  {"episode_idx": 0, "task": "task_a", "total_frames": 2075,
   "milestones": [
     {"start_frame": 0, "critical_frame": 80, "label": "approach",
      "critical_windows": [[96, 112], [208, 224]]},
     {"start_frame": 80, "critical_frame": 400, "label": "flatten",
      "critical_windows": [[448, 464]]},
     ...
   ]}
"""

import argparse
import json
from pathlib import Path


def convert_episode(label: dict, task_name: str, global_ep_idx: int) -> dict:
    T = label["total_frames"]
    stages = label["stages"]
    cuts = label["milestone_cuts"]
    crit_windows = label["critical_windows"]

    # Build stage boundaries: [0, cut0, cut1, ..., cutN-1]
    boundaries = [0] + cuts
    n_stages = len(boundaries)

    milestones = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < n_stages else T
        stage_label = stages[i] if i < len(stages) else f"stage_{i}"

        # Critical windows that fall within this stage
        stage_cws = [
            [a, b] for a, b in crit_windows
            if a >= start and a < end
        ]

        # critical_frame = first critical window start in this stage,
        # or midpoint between start and end if no critical windows
        if stage_cws:
            critical_frame = stage_cws[0][0]
        else:
            critical_frame = (start + end) // 2

        milestones.append({
            "start_frame": start,
            "critical_frame": critical_frame,
            "label": stage_label,
            "critical_windows": stage_cws,
        })

    return {
        "episode_idx": global_ep_idx,
        "task": task_name,
        "total_frames": T,
        "milestones": milestones,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", required=True, type=Path,
                    help="directory with task subdirs containing ep*.json")
    ap.add_argument("--out", required=True, type=Path,
                    help="output JSONL path")
    ap.add_argument("--task-order", nargs="+",
                    default=["task_a", "aloha_coffee", "aloha_towel", "aloha_candy"],
                    help="task subdirectory order (determines global episode indices)")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    global_idx = 0
    total = 0

    with open(args.out, "w") as out_f:
        for task_name in args.task_order:
            task_dir = args.label_dir / task_name
            if not task_dir.exists():
                print(f"skip {task_name} (not found)")
                continue
            jsons = sorted(task_dir.glob("ep*.json"))
            for jp in jsons:
                label = json.load(open(jp))
                row = convert_episode(label, task_name, global_idx)
                out_f.write(json.dumps(row) + "\n")
                global_idx += 1
                total += 1
            print(f"{task_name}: {len(jsons)} episodes (global {global_idx - len(jsons)}..{global_idx - 1})")

    print(f"\nwrote {args.out}  ({total} episodes)")


if __name__ == "__main__":
    main()
