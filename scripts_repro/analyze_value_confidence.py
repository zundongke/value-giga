"""Does value-latent dispersion (value_std) predict value error? (confidence test)

Premise: the value frame is trained to be a CONSTANT field; low dispersion at
inference should mean the denoiser is confident => low |pred - GT|. We test the
rank-correlation between dispersion and error, plus a binned calibration curve.
A useful confidence signal needs a clearly POSITIVE dispersion-vs-error relation.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_and_viz_video import compute_gt_value_curve  # noqa


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--milestones", default="data/milestones_task_a_0509v2.jsonl")
    ap.add_argument("--episode", type=int, default=2)
    ap.add_argument("--dual_slope_r", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    d = np.load(args.npz)
    value = d["value"].astype(np.float64)
    T = len(value)
    gt = np.asarray(compute_gt_value_curve(args.episode, T, args.milestones, args.dual_slope_r),
                    dtype=np.float64)
    gt = gt[:T]
    err = np.abs(value - gt)

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    # candidate confidence/uncertainty signals (higher = expected MORE error)
    signals = {}
    signals["intra_value_std"] = d["value_std"].astype(np.float64)
    if "value_ensemble_std" in d and float(np.nanmax(d["value_ensemble_std"])) > 0:
        signals["value_ensemble_std"] = d["value_ensemble_std"].astype(np.float64)
    if "fs_ensemble_std" in d and float(np.nanmax(d["fs_ensemble_std"])) > 0:
        signals["fs_ensemble_std"] = d["fs_ensemble_std"].astype(np.float64)
    if "value_ensemble_std" in signals and "fs_ensemble_std" in signals:
        signals["value+state (z-sum)"] = z(signals["value_ensemble_std"]) + z(signals["fs_ensemble_std"])

    K = int(d["num_samples"]) if "num_samples" in d else 1
    print(f"[{args.tag}] T={T}  K={K}  mean|err|={err.mean():.4f}")
    print(f"  {'signal':<22} {'Spearman(sig,err)':>18} {'Pearson':>9}")
    for name, sig in signals.items():
        print(f"  {name:<22} {spearman(sig, err):>+18.3f} {float(np.corrcoef(sig, err)[0,1]):>+9.3f}")

    # plot the ensemble value signal if present, else intra
    std = signals.get("value_ensemble_std", signals["intra_value_std"])
    rho = spearman(std, err); pear = float(np.corrcoef(std, err)[0, 1])

    # Calibration: bin frames by dispersion decile, show mean error per bin
    order = np.argsort(std)
    nb = 10
    edges = [order[int(i * T / nb):int((i + 1) * T / nb)] for i in range(nb)]
    bin_std = np.array([std[ix].mean() for ix in edges])
    bin_err = np.array([err[ix].mean() for ix in edges])
    print("  decile  mean_std  mean_|err|")
    for i in range(nb):
        print(f"   {i:2d}     {bin_std[i]:.4f}    {bin_err[i]:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(std, err, s=4, alpha=0.25, color="C0")
    ax[0].set_xlabel("value_std (dispersion)"); ax[0].set_ylabel("|pred - GT| value error")
    ax[0].set_title(f"{args.tag}: per-frame  (Spearman={rho:+.3f}, Pearson={pear:+.3f})")
    ax[0].grid(alpha=0.25)
    ax[1].plot(bin_std, bin_err, "o-", color="C3")
    ax[1].set_xlabel("mean dispersion (by decile)"); ax[1].set_ylabel("mean |error|")
    ax[1].set_title("Calibration: dispersion decile vs error\n(monotone rising = usable confidence)")
    ax[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
