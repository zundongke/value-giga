"""Visualize ViVa's predicted value with a generative-entropy confidence band.

No GT needed (works on held-out 0519v2): top panel shows predicted value (ensemble
mean) with a +/- value_ensemble_std band; bottom shows the entropy (ensemble std)
per frame. Wide band / high entropy = low confidence.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="0519v2 ep0 (r10, K=8)")
    args = ap.parse_args()

    d = np.load(args.npz)
    v = d["value"].astype(np.float64)
    ens = d["value_ensemble_std"].astype(np.float64)
    fse = d["fs_ensemble_std"].astype(np.float64) if "fs_ensemble_std" in d else None
    x = np.arange(len(v))

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    a0, a1 = axes
    a0.fill_between(x, np.clip(v - ens, 0, 1), np.clip(v + ens, 0, 1),
                    color="C0", alpha=0.25, label="±1 ensemble std (uncertainty)")
    a0.plot(x, v, color="C0", lw=1.2, label="predicted value (ensemble mean)")
    a0.set_ylabel("value"); a0.set_ylim(-0.05, 1.05)
    a0.legend(loc="upper left", fontsize=8); a0.grid(alpha=0.25)
    a0.set_title(f"ViVa value + generative-entropy confidence band — {args.tag}")

    a1.plot(x, ens, color="C3", lw=1.0, label="value entropy (ensemble std)")
    if fse is not None:
        a1.plot(x, fse, color="C2", lw=0.8, alpha=0.7, label="future-state entropy")
    # flag the least-confident frames (top 10% entropy)
    thr = np.quantile(ens, 0.90)
    a1.axhline(thr, color="0.5", ls="--", lw=0.8, label=f"90th pct ({thr:.3f})")
    for xs in x[ens >= thr]:
        a1.axvspan(xs, xs + 1, color="red", alpha=0.10, lw=0)
    a1.set_xlabel("frame"); a1.set_ylabel("entropy")
    a1.legend(loc="upper left", fontsize=8); a1.grid(alpha=0.25)
    a1.set_title("Per-frame uncertainty (red = top-10% least-confident frames)")

    fig.tight_layout(); fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}  | value[{v.min():.3f},{v.max():.3f}] "
          f"entropy[{ens.min():.4f},{ens.max():.4f}] mean={ens.mean():.4f}")


if __name__ == "__main__":
    main()
