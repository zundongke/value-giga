"""
Evaluate ViVa on Task_A and render a video per episode:
  top row:   [top_head | hand_left | hand_right]
  bottom row: value curve (predicted vs ground-truth DSM) + metrics overlay

Usage:
  python scripts_repro/eval_and_viz_video.py \
    --checkpoint data/exps/viva-TaskA0509-DSM-0526.2230/viva-TaskA0509-DSM-0526.2230/checkpoint_step_7000 \
    --config config/train_viva-TaskA0509-DSM-0526.2230.yaml \
    --data_path /dev/shm/task_a_0509v2 \
    --t5_embedding data/t5_task_a_0509v2.pt \
    --episodes 0 1 2 \
    --output_dir data/exps/viva-TaskA0509-DSM-0526.2230/viz
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from viva_model import VivaModel
from viva_dataset import VivaDataset, compute_dataset_state_stats
from inference import load_model, build_episode_table

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def compute_gt_value_curve(episode_idx, episode_length, milestones_jsonl, dual_slope_r=3.0):
    """Compute ground-truth DSM value curve for an episode."""
    episode_milestones = {}
    episode_critical_windows = {}
    if milestones_jsonl and os.path.exists(milestones_jsonl):
        with open(milestones_jsonl) as f:
            for line in f:
                row = json.loads(line.strip())
                ep = int(row["episode_idx"])
                starts = []
                all_cws = []
                for m in row["milestones"]:
                    starts.append(int(m["start_frame"]))
                    for cw in m.get("critical_windows", []):
                        all_cws.append((int(cw[0]), int(cw[1])))
                episode_milestones[ep] = sorted(starts)
                if all_cws:
                    episode_critical_windows[ep] = sorted(all_cws)

    T = episode_length
    cws = episode_critical_windows.get(episode_idx)
    if cws is None:
        return np.linspace(0, 1, T)

    event_frames = sorted({
        f for cs, ce in cws
        for f in range(max(0, cs), min(T, ce))
    })
    M = len(event_frames)
    r = dual_slope_r
    denom = (r - 1.0) * M + T
    d_ord = 1.0 / denom
    d_crit = r / denom
    inc = np.full(T, d_ord)
    for ef in event_frames:
        inc[ef] = d_crit
    curve = np.zeros(T)
    if T > 1:
        curve[1:] = np.cumsum(inc[1:])
        if curve[-1] > 0:
            curve /= curve[-1]
    return curve


@torch.no_grad()
def infer_episode(model, dataset, ep_info, t5_embedding, device, num_inference_steps=1,
                  sample_stride=1):
    """Run inference on an episode, returning predictions and camera frames."""
    global_start = ep_info["global_start"]
    ep_len = ep_info["episode_length"]
    dtype = torch.bfloat16

    predictions = []
    frames_top = []
    frames_left = []
    frames_right = []
    frame_indices = []

    for i in tqdm(range(0, ep_len, sample_stride), desc=f"ep{ep_info['episode_index']}", leave=False):
        global_idx = global_start + i
        sample = dataset.lerobot_dataset[global_idx]

        frame_index = sample['frame_index'].item() if isinstance(sample['frame_index'], torch.Tensor) else sample['frame_index']
        frame_indices.append(frame_index)

        # State
        state_raw = sample.get('observation.state', None)
        if state_raw is not None:
            state = state_raw.float() if isinstance(state_raw, torch.Tensor) else torch.tensor(state_raw, dtype=torch.float32)
            state = torch.clamp(state, dataset.state_min, dataset.state_max)
            state = (state - dataset.state_min) / (dataset.state_max - dataset.state_min + 1e-8)
            state = state * 2 - 1
        else:
            state = torch.zeros(14)

        # Images
        cam_high_raw = sample.get(dataset.cam_key_high, None)
        cam_left_raw = sample.get(dataset.cam_key_left, None)
        cam_right_raw = sample.get(dataset.cam_key_right, None)

        cam_high = dataset._process_image(cam_high_raw) if cam_high_raw is not None else torch.zeros(3, dataset.video_height, dataset.video_width)
        cam_left = dataset._process_image(cam_left_raw) if cam_left_raw is not None else torch.zeros(3, dataset.video_height, dataset.video_width)
        cam_right = dataset._process_image(cam_right_raw) if cam_right_raw is not None else torch.zeros(3, dataset.video_height, dataset.video_width)

        # Store raw frames for visualization (before model input processing)
        def to_uint8(t):
            return (t.clamp(0, 1) * 255).permute(1, 2, 0).byte().numpy()
        frames_top.append(to_uint8(cam_high))
        frames_left.append(to_uint8(cam_left))
        frames_right.append(to_uint8(cam_right))

        # Model inference
        state_b = state.unsqueeze(0).to(device=device, dtype=dtype)
        cam_high_b = cam_high.unsqueeze(0).to(device=device, dtype=dtype)
        cam_left_b = cam_left.unsqueeze(0).to(device=device, dtype=dtype)
        cam_right_b = cam_right.unsqueeze(0).to(device=device, dtype=dtype)

        if t5_embedding is not None:
            t5_emb = t5_embedding.unsqueeze(0).to(device=device, dtype=dtype)
        else:
            t5_emb = torch.zeros(1, 4, 4096, device=device, dtype=dtype)

        output = model.predict_value(
            state_b, cam_left_b, cam_right_b, cam_high_b,
            t5_emb, num_inference_steps=num_inference_steps,
        )
        predictions.append(output['value'][0].float().item())

    return {
        'predictions': predictions,
        'frame_indices': frame_indices,
        'frames_top': frames_top,
        'frames_left': frames_left,
        'frames_right': frames_right,
    }


def render_video(result, gt_curve, ep_info, output_path, fps=10):
    """Render a video: top row = 3 cameras, bottom = value curve."""
    import cv2

    predictions = result['predictions']
    frame_indices = result['frame_indices']
    frames_top = result['frames_top']
    frames_left = result['frames_left']
    frames_right = result['frames_right']
    ep_len = ep_info['episode_length']

    cam_h, cam_w = frames_top[0].shape[:2]
    plot_w = cam_w * 3
    plot_h = cam_h

    total_w = cam_w * 3
    total_h = cam_h + plot_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (total_w, total_h))

    for idx in range(len(predictions)):
        # Top row: 3 cameras side by side
        top_row = np.concatenate([frames_top[idx], frames_left[idx], frames_right[idx]], axis=1)

        # Bottom row: value curve plot
        fig, ax = plt.subplots(1, 1, figsize=(plot_w / 100, plot_h / 100), dpi=100)
        ax.plot(np.arange(ep_len), gt_curve, 'b-', alpha=0.5, linewidth=1.5, label='GT (DSM)')
        ax.plot(frame_indices[:idx+1], predictions[:idx+1], 'r-', linewidth=2, label='Predicted')
        ax.scatter([frame_indices[idx]], [predictions[idx]], c='red', s=60, zorder=5)
        ax.set_xlim(0, ep_len)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Frame', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.set_title(f'Episode {ep_info["episode_index"]}  |  Frame {frame_indices[idx]}/{ep_len}  |  '
                      f'Pred={predictions[idx]:.3f}  GT={gt_curve[frame_indices[idx]]:.3f}',
                      fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(pad=0.5)

        fig.canvas.draw()
        plot_img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)

        # Resize plot to match width
        plot_img = cv2.resize(plot_img, (total_w, plot_h))

        # Combine
        frame = np.concatenate([top_row, plot_img], axis=0)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    logger.info(f"Video saved: {output_path} ({len(predictions)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_path", type=str, nargs='+', required=True)
    parser.add_argument("--t5_embedding", type=str, default=None)
    parser.add_argument("--episodes", type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument("--output_dir", type=str, default="viz_output")
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--sample_stride", type=int, default=5,
                        help="Sample every N-th frame (default 5 = 6fps from 30fps)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    # State stats
    state_stats = None
    if config.dataset.get("compute_state_stats", False):
        logger.info("Computing state statistics...")
        state_stats = compute_dataset_state_stats(args.data_path)

    # Load model
    model = load_model(args.checkpoint, config, args.device)

    # T5 embedding
    t5_embedding = None
    if args.t5_embedding and os.path.exists(args.t5_embedding):
        t5_embedding = torch.load(args.t5_embedding, map_location='cpu')

    # Load dataset
    dataset = VivaDataset(
        data_paths=args.data_path,
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        state_stats=state_stats,
        cam_key_high=getattr(config.dataset, 'cam_key_high', 'observation.images.cam_high'),
        cam_key_left=getattr(config.dataset, 'cam_key_left', 'observation.images.cam_left_wrist'),
        cam_key_right=getattr(config.dataset, 'cam_key_right', 'observation.images.cam_right_wrist'),
    )

    episode_table = build_episode_table(dataset)

    milestones_jsonl = getattr(config.dataset, 'milestones_jsonl', None)
    dual_slope_r = getattr(config.dataset, 'dual_slope_r', 3.0)

    for ep_idx in args.episodes:
        if ep_idx >= len(episode_table):
            logger.warning(f"Episode {ep_idx} out of range (max {len(episode_table)-1}), skipping")
            continue

        ep_info = episode_table[ep_idx]
        logger.info(f"Processing episode {ep_idx} ({ep_info['episode_length']} frames, stride={args.sample_stride})")

        gt_curve = compute_gt_value_curve(
            ep_idx, ep_info['episode_length'], milestones_jsonl, dual_slope_r)

        result = infer_episode(
            model, dataset, ep_info, t5_embedding, args.device,
            num_inference_steps=args.num_inference_steps,
            sample_stride=args.sample_stride,
        )

        output_path = Path(args.output_dir) / f"eval_ep{ep_idx:03d}.mp4"
        render_video(result, gt_curve, ep_info, output_path, fps=args.fps)

        # Also save a summary plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(np.arange(ep_info['episode_length']), gt_curve, 'b-', alpha=0.6, label='GT (DSM)')
        ax.plot(result['frame_indices'], result['predictions'], 'r.-', markersize=3, label='Predicted')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Value')
        ax.set_title(f'Episode {ep_idx} — Value Prediction vs Ground Truth')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(Path(args.output_dir) / f"eval_ep{ep_idx:03d}_curve.png", dpi=150)
        plt.close(fig)
        logger.info(f"Curve plot saved: {args.output_dir}/eval_ep{ep_idx:03d}_curve.png")

    logger.info("Done!")


if __name__ == "__main__":
    main()
