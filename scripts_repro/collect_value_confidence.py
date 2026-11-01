"""8-GPU collection of per-frame predicted value + latent dispersion (value_std).

Tests the "similarity = confidence" idea: does low dispersion of the value latent
frame predict low value error? Saves an npz with value, value_std, frame_index.
GT value + correlation analysis happen in analyze_value_confidence.py.

Usage:
  torchrun --nproc_per_node=8 scripts_repro/collect_value_confidence.py \
    --checkpoint .../checkpoint_step_7000 --config config/...yaml \
    --data_path /dev/shm/task_a_0509v2 --t5_embedding data/t5_task_a_0509v2.pt \
    --episode 2 --out data/exps/comparison_viz/valconf_0509v2_ep2_r10.npz
"""
import argparse, logging, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import build_episode_table, infer_single_frame, load_model, load_state_txt
from viva_dataset import VivaDataset, compute_dataset_state_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--t5_embedding", required=True)
    p.add_argument("--state_txt", default=None)
    p.add_argument("--episode", type=int, default=2)
    p.add_argument("--num_inference_steps", type=int, default=1)
    p.add_argument("--num_samples", type=int, default=1,
                   help=">1 enables multi-sample generative-entropy (value_ensemble_std)")
    p.add_argument("--vlb_entropy", action="store_true",
                   help="Use DR-style trajectory-VLB entropy on self-generated samples")
    p.add_argument("--num_score_levels", type=int, default=6,
                   help="Noise levels for the VLB entropy estimate (when --vlb_entropy)")
    p.add_argument("--num_noise_draws", type=int, default=1,
                   help="MC noise draws averaged per level in the VLB estimate (when --vlb_entropy)")
    p.add_argument("--frame_limit", type=int, default=None,
                   help="Only collect frames [0, frame_limit). Default: full episode.")
    p.add_argument("--frames_npz", default=None,
                   help="OOD-label npz (frame_index,n_dims_oor): score only the labeled OOD "
                        "frames plus --neg_per sampled in-dist frames (cheap AUC on separated set)")
    p.add_argument("--neg_per", type=int, default=400,
                   help="Number of in-distribution frames to sample as negatives (with --frames_npz)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rank = int(os.environ["LOCAL_RANK"]); world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank); device = f"cuda:{rank}"
    dist.init_process_group(backend="nccl", init_method="env://", device_id=torch.device(device))

    config = OmegaConf.load(args.config)
    if args.state_txt and os.path.exists(args.state_txt):
        state_stats = load_state_txt(args.state_txt)
    elif config.dataset.get("compute_state_stats", False):
        state_stats = compute_dataset_state_stats([args.data_path])
    else:
        state_stats = None

    dataset = VivaDataset(
        data_paths=[args.data_path],
        video_height=config.common.video_height, video_width=config.common.video_width,
        state_stats=state_stats,
        cam_key_high=getattr(config.dataset, "cam_key_high", "observation.images.cam_high"),
        cam_key_left=getattr(config.dataset, "cam_key_left", "observation.images.cam_left_wrist"),
        cam_key_right=getattr(config.dataset, "cam_key_right", "observation.images.cam_right_wrist"),
    )
    episode_table = build_episode_table(dataset)
    ep_info = next(ep for ep in episode_table if ep["episode_index"] == args.episode)
    ep_len = ep_info["episode_length"]
    global_start = ep_info["global_start"]
    if args.frames_npz:
        z = np.load(args.frames_npz)
        lab = z["n_dims_oor"] > 0
        fidx = z["frame_index"]
        neg = np.random.default_rng(0).choice(
            fidx[~lab], size=min(args.neg_per, int((~lab).sum())), replace=False)
        frames = np.sort(np.concatenate([fidx[lab], neg])).astype(int)
    else:
        n_frames = min(args.frame_limit, ep_len) if args.frame_limit else ep_len
        frames = np.arange(n_frames, dtype=int)
    n_frames = len(frames)

    per = (n_frames + world - 1) // world
    s, e = rank * per, min((rank + 1) * per, n_frames)
    if rank == 0:
        logger.info(f"ep={args.episode} ep_len={ep_len} collect={n_frames} per_rank={per}")

    model = load_model(args.checkpoint, config, device)
    t5 = torch.load(args.t5_embedding, map_location="cpu") if os.path.exists(args.t5_embedding) else None

    # Actual future states from the rollout parquet, normalized like the model's output,
    # for the grounded DR-likelihood residual ||pred_future - s[t+offset]||.
    import pyarrow.parquet as pq
    offset = int(getattr(config.dataset, "future_offset", 50))
    pqf = os.path.join(args.data_path, f"data/chunk-000/episode_{args.episode:06d}.parquet")
    states_raw = np.asarray(pq.read_table(pqf, columns=["observation.state"]).column("observation.state").to_pylist(), dtype=np.float64)
    smin = np.array(dataset.state_min, dtype=np.float64); smax = np.array(dataset.state_max, dtype=np.float64)
    states_norm = (np.clip(states_raw, smin, smax) - smin) / (smax - smin + 1e-8) * 2 - 1   # [ep_len,14] in [-1,1]

    val = np.zeros(e - s, dtype=np.float32)
    std = np.zeros(e - s, dtype=np.float32)
    ens = np.zeros(e - s, dtype=np.float32)
    fse = np.zeros(e - s, dtype=np.float32)
    fres = np.full(e - s, np.nan, dtype=np.float32)   # actual-future residual
    for i, f in enumerate(int(x) for x in frames[s:e]):
        r = infer_single_frame(model, dataset, global_start + f, ep_info, t5,
                               device=device, num_inference_steps=args.num_inference_steps,
                               num_samples=args.num_samples,
                               vlb_entropy=args.vlb_entropy, num_score_levels=args.num_score_levels,
                               num_noise_draws=args.num_noise_draws)
        val[i] = r["prediction"]; std[i] = r["value_std"]
        ens[i] = r["value_ensemble_std"]; fse[i] = r["fs_ensemble_std"]
        pf = np.asarray(r.get("pred_future_state", [np.nan] * 14), dtype=np.float64)
        if f + offset < ep_len and np.isfinite(pf).all():
            fres[i] = float(np.linalg.norm(pf - states_norm[f + offset]))
    logger.info(f"[Rank {rank}] frames [{s},{e}) done")

    gather = lambda: [None] * world
    all_val, all_std, all_ens, all_fse, all_fres = gather(), gather(), gather(), gather(), gather()
    dist.all_gather_object(all_val, val); dist.all_gather_object(all_std, std)
    dist.all_gather_object(all_ens, ens); dist.all_gather_object(all_fse, fse)
    dist.all_gather_object(all_fres, fres)
    if rank == 0:
        value = np.concatenate(all_val); value_std = np.concatenate(all_std)
        value_ensemble_std = np.concatenate(all_ens)
        fs_ensemble_std = np.concatenate(all_fse)
        fs_actual_residual = np.concatenate(all_fres)
        assert len(value) == n_frames
        np.savez(args.out, value=value, value_std=value_std,
                 value_ensemble_std=value_ensemble_std, fs_ensemble_std=fs_ensemble_std,
                 fs_actual_residual=fs_actual_residual, future_offset=np.int64(offset),
                 frame_index=frames, ep_len=np.int64(ep_len),
                 num_samples=np.int64(args.num_samples))
        logger.info(f"wrote {args.out}: value[{value.min():.3f},{value.max():.3f}] "
                    f"fs_ens[{np.nanmin(fs_ensemble_std):.4f},{np.nanmax(fs_ensemble_std):.4f}] "
                    f"fs_resid[{np.nanmin(fs_actual_residual):.4f},{np.nanmax(fs_actual_residual):.4f}] "
                    f"K={args.num_samples} offset={offset}")
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
