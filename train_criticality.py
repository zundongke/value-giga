"""Train the proprio causal-TCN critical-frame classifier.

Example:
  PY=/vePFS-East/comp_robot/shock/conda_envs/viva/bin/python
  $PY train_criticality.py \
    --data_paths /dev/shm/task_a_0509v2 \
    --milestones_jsonl data/milestones_task_a_0509v2.jsonl \
    --exp_name crit-clf-taskA-proprioW16-0530.1200
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from criticality_data import (extract_episode_arrays, compute_state_stats,
                              CriticalityWindowDataset)
from criticality_model import CriticalityTCN
from criticality_metrics import roc_auc, average_precision


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_paths", nargs="+", required=True)
    p.add_argument("--milestones_jsonl", required=True)
    p.add_argument("--exp_name", required=True)
    p.add_argument("--exp_root", default="data/exps")
    p.add_argument("--subdataset_ep_offsets", type=int, nargs="*", default=None)
    p.add_argument("--W", type=int, default=16)
    p.add_argument("--causal", type=int, default=1)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    exp_dir = os.path.join(args.exp_root, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    episodes = extract_episode_arrays(
        args.data_paths, args.milestones_jsonl,
        os.path.join(args.exp_root, "_cache", args.exp_name + "_episodes.npz"),
        subdataset_ep_offsets=args.subdataset_ep_offsets)

    ep_ids = sorted(episodes)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(ep_ids)
    n_val = max(1, int(len(ep_ids) * args.val_frac))
    val_ids, train_ids = ep_ids[:n_val], ep_ids[n_val:]

    smin, smax = compute_state_stats(episodes, train_ids)
    np.savez(os.path.join(exp_dir, "norm_stats.npz"), state_min=smin, state_max=smax)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "train_ids": train_ids, "val_ids": val_ids}, f, indent=2)

    causal = bool(args.causal)
    train_ds = CriticalityWindowDataset(episodes, train_ids, smin, smax, args.W, causal)
    val_ds = CriticalityWindowDataset(episodes, val_ids, smin, smax, args.W, causal)

    n_pos = float(train_ds.y.sum())
    n_neg = float(len(train_ds.y) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=args.device)
    print(f"train N={len(train_ds)} pos={int(n_pos)} pos_weight={pos_weight.item():.2f} "
          f"| val N={len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=1024, shuffle=False)

    model = CriticalityTCN(14, args.hidden, args.levels).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="viva-criticality", name=args.exp_name, config=vars(args))

    best_pr = -1.0
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(args.device), yb.to(args.device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                logits = model(xb.to(args.device))
                scores.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(yb.numpy())
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
        va_auc = roc_auc(scores, labels)
        va_pr = average_precision(scores, labels)
        print(f"epoch {epoch:03d} loss={loss.item():.4f} val_roc_auc={va_auc:.4f} val_pr_auc={va_pr:.4f}")
        if run:
            run.log({"epoch": epoch, "train_loss": loss.item(),
                     "val_roc_auc": va_auc, "val_pr_auc": va_pr})
        if va_pr > best_pr:
            best_pr = va_pr
            torch.save({"model": model.state_dict(),
                        "args": vars(args), "val_pr_auc": va_pr, "epoch": epoch},
                       os.path.join(exp_dir, "best.pt"))

    print(f"best val_pr_auc={best_pr:.4f} -> {exp_dir}/best.pt")
    if run:
        run.summary["best_val_pr_auc"] = best_pr
        run.finish()


if __name__ == "__main__":
    main()
