"""Evaluate a trained criticality classifier vs the gripper-velocity baseline.

Reports frame-level ROC/PR-AUC, segment-level detection rate, and the same
metrics for the gripper baseline on the held-out (val) episodes from config.json.
"""
import argparse
import json
import os

import numpy as np
import torch

from criticality_data import extract_episode_arrays, normalize_state, make_windows
from criticality_model import CriticalityTCN
from criticality_metrics import (roc_auc, average_precision,
                                 segment_detection_rate, gripper_velocity_score)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--data_paths", nargs="+", required=True)
    p.add_argument("--milestones_jsonl", required=True)
    p.add_argument("--subdataset_ep_offsets", type=int, nargs="*", default=None)
    p.add_argument("--seg_thresh", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = json.load(open(os.path.join(args.exp_dir, "config.json")))
    stats = np.load(os.path.join(args.exp_dir, "norm_stats.npz"))
    smin, smax = stats["state_min"], stats["state_max"]
    W, causal = cfg["W"], bool(cfg["causal"])
    val_ids = cfg["val_ids"]

    episodes = extract_episode_arrays(
        args.data_paths, args.milestones_jsonl,
        os.path.join(cfg.get("exp_root", "data/exps"), "_cache",
                     cfg["exp_name"] + "_episodes.npz"),
        subdataset_ep_offsets=args.subdataset_ep_offsets)

    ckpt = torch.load(os.path.join(args.exp_dir, "best.pt"), map_location=args.device)
    model = CriticalityTCN(14, cfg["hidden"], cfg["levels"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_prob, all_base, all_lab = [], [], []
    seg_model, seg_base, n_win = 0.0, 0.0, 0
    for e in val_ids:
        traj_raw = episodes[e]["state"]
        crit = episodes[e]["crit"]
        windows = episodes[e]["windows"]
        X, y = make_windows(normalize_state(traj_raw, smin, smax), crit, W, causal)
        with torch.no_grad():
            prob = torch.sigmoid(
                model(torch.tensor(X, dtype=torch.float32, device=args.device))
            ).cpu().numpy()
        base = gripper_velocity_score(traj_raw)
        all_prob.append(prob); all_base.append(base); all_lab.append(y)
        if windows:
            sm = segment_detection_rate(prob, windows, args.seg_thresh)
            # baseline segment rate uses a per-episode median-based threshold
            sb = segment_detection_rate(base, windows, float(np.median(base) + base.std()))
            seg_model += sm * len(windows); seg_base += sb * len(windows)
            n_win += len(windows)

    prob = np.concatenate(all_prob); base = np.concatenate(all_base); lab = np.concatenate(all_lab)
    res = {
        "model_roc_auc": roc_auc(prob, lab),
        "model_pr_auc": average_precision(prob, lab),
        "baseline_roc_auc": roc_auc(base, lab),
        "baseline_pr_auc": average_precision(base, lab),
        "model_seg_detect": seg_model / max(n_win, 1),
        "baseline_seg_detect": seg_base / max(n_win, 1),
        "n_val_frames": int(len(lab)), "n_val_windows": n_win,
    }
    print(json.dumps(res, indent=2))
    with open(os.path.join(args.exp_dir, "eval.json"), "w") as f:
        json.dump(res, f, indent=2)

    if args.wandb:
        import wandb
        run = wandb.init(project="viva-criticality", name=cfg["exp_name"] + "-eval",
                         config=cfg)
        run.log(res); run.finish()

    verdict = "BEATS" if res["model_pr_auc"] > res["baseline_pr_auc"] else "DOES NOT BEAT"
    print(f"\nModel {verdict} gripper baseline on PR-AUC "
          f"({res['model_pr_auc']:.3f} vs {res['baseline_pr_auc']:.3f}).")


if __name__ == "__main__":
    main()
