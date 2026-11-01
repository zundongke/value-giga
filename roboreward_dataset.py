# RoboReward adapter for ViVa.
#
# Each RoboReward example is (video_mp4, task_text, reward in {1..5}).
# Mapping to ViVa's 7-frame layout [blank, state, cam_left, cam_right, cam_high, future_state, value]:
#   - state, future_state       -> zeros (RoboReward has no proprioception)
#   - cam_left/cam_right/cam_high -> frames sampled from the single video
#   - value                      -> ViVa paper formula (T - t - 1) / (T - 1)
#                                   for the sampled frame index t in a video of length T
#   - reward score is NOT used as a label here (kept for optional filtering)

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _resize_pad(image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """image: float CHW in [0,1]. Letterbox-resize to target."""
    c, h, w = image.shape
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w),
                          mode='bilinear', align_corners=False).squeeze(0)
    pad_h, pad_w = target_h - new_h, target_w - new_w
    return F.pad(image, (pad_w // 2, pad_w - pad_w // 2,
                         pad_h // 2, pad_h - pad_h // 2), value=0)


class RoboRewardDataset(Dataset):
    """
    ViVa-compatible dataset for the RoboReward corpus.

    Args:
        data_root: directory containing train/, val/, test/ and metadata.jsonl
        split: 'train' | 'val' | 'test'
        t5_table_path: path to .pt holding {task_str: Tensor[L, 4096]} for T5 embeddings
        video_height/width: ViVa target resolution
        cam_temporal_stride: 0 puts the same frame in all 3 cam slots (default).
            >0 puts frames at (t-stride, t, t+stride), giving the cam slots temporal
            spread. Useful only for ablations; ViVa's 3 slots are different views,
            not different times.
        min_score: discard rows with reward strictly below this. None keeps all.
        max_samples: subsample for smoke runs.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        t5_table_path: Optional[str] = None,
        video_height: int = 384,
        video_width: int = 320,
        cam_temporal_stride: int = 0,
        min_score: Optional[int] = None,
        max_samples: Optional[int] = None,
        min_file_size_bytes: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.video_height = video_height
        self.video_width = video_width
        self.cam_temporal_stride = cam_temporal_stride

        meta_path = self.data_root / split / "metadata.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.jsonl missing: {meta_path}")

        rows: List[Dict] = []
        skipped_lfs = 0
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if min_score is not None and int(d.get("reward", 1)) < min_score:
                    continue
                if min_file_size_bytes is not None:
                    p = self.data_root / split / d["file_name"]
                    try:
                        if os.path.getsize(p) < min_file_size_bytes:
                            skipped_lfs += 1
                            continue
                    except OSError:
                        skipped_lfs += 1
                        continue
                rows.append(d)
        if skipped_lfs:
            logger.info(f"  Skipped {skipped_lfs} rows (file size below "
                        f"{min_file_size_bytes} bytes — likely unfetched LFS pointers)")

        if max_samples is not None and max_samples > 0 and max_samples < len(rows):
            rng = random.Random(42)
            rng.shuffle(rows)
            rows = rows[:max_samples]

        self.rows = rows
        self.split_dir = self.data_root / split

        if t5_table_path is None or not os.path.exists(t5_table_path):
            raise FileNotFoundError(
                f"T5 table not found: {t5_table_path}. "
                "Run extract_t5_roboreward.py first.")

        logger.info(f"Loading T5 table (mmap): {t5_table_path}")
        try:
            self.t5_table: Dict[str, torch.Tensor] = torch.load(
                t5_table_path, map_location="cpu",
                mmap=True, weights_only=True)
        except Exception as e:
            logger.warning(f"mmap load failed ({e}); falling back to full load")
            self.t5_table = torch.load(
                t5_table_path, map_location="cpu", weights_only=False)
        logger.info(f"  T5 entries: {len(self.t5_table)}")

        # Quick check: every task in this split has an embedding.
        missing = [r["task"] for r in self.rows if r["task"] not in self.t5_table]
        if missing:
            raise KeyError(
                f"{len(missing)} task strings missing from T5 table. "
                f"First missing: {missing[0][:80]!r}")

        logger.info(f"RoboReward[{split}]: {len(self.rows)} samples, "
                    f"video={video_height}x{video_width}, "
                    f"cam_stride={cam_temporal_stride}, "
                    f"min_score={min_score}")

    def __len__(self) -> int:
        return len(self.rows)

    def _read_frame(self, vr, frame_idx: int, num_frames: int) -> torch.Tensor:
        idx = max(0, min(num_frames - 1, frame_idx))
        arr = vr[idx].asnumpy()  # HWC uint8
        img = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return _resize_pad(img, self.video_height, self.video_width)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        video_path = self.split_dir / row["file_name"]
        task = row["task"]
        score = int(row.get("reward", 1))

        try:
            import decord
            decord.bridge.set_bridge("native")
            vr = decord.VideoReader(str(video_path), num_threads=1)
            T = len(vr)
            if T < 1:
                raise RuntimeError(f"empty video: {video_path}")

            t = random.randint(0, T - 1)
            stride = max(0, int(self.cam_temporal_stride))
            if stride > 0:
                t_left = max(0, t - stride)
                t_right = min(T - 1, t + stride)
            else:
                t_left = t_right = t

            cam_left = self._read_frame(vr, t_left, T)
            cam_high = self._read_frame(vr, t, T)
            cam_right = self._read_frame(vr, t_right, T)

            value = (T - t - 1) / (T - 1) if T > 1 else 0.5
            value_normalized = value * 2 - 1

            t5_emb = self.t5_table[task]  # [L, 4096], cpu, bf16 or fp32
            if not isinstance(t5_emb, torch.Tensor):
                t5_emb = torch.tensor(t5_emb)

            return {
                "cam_high": cam_high,
                "cam_left_wrist": cam_left,
                "cam_right_wrist": cam_right,
                "state": torch.zeros(14, dtype=torch.float32),
                "future_state": torch.zeros(14, dtype=torch.float32),
                "value": torch.tensor(float(value), dtype=torch.float32),
                "value_normalized": torch.tensor(float(value_normalized), dtype=torch.float32),
                "frame_idx": int(t),
                "episode_length": int(T),
                "future_frame_idx": int(t),
                "task_id": 0,  # not used; T5 carried per sample below
                "t5_embedding": t5_emb.float(),  # [L, 4096]
                "score": int(score),
            }
        except Exception as e:
            logger.warning(f"RoboReward error idx={idx} ({video_path}): {e}")
            dummy = torch.zeros(3, self.video_height, self.video_width)
            return {
                "cam_high": dummy.clone(),
                "cam_left_wrist": dummy.clone(),
                "cam_right_wrist": dummy.clone(),
                "state": torch.zeros(14),
                "future_state": torch.zeros(14),
                "value": torch.tensor(0.5),
                "value_normalized": torch.tensor(0.0),
                "frame_idx": 0,
                "episode_length": 1,
                "future_frame_idx": 0,
                "task_id": 0,
                "t5_embedding": torch.zeros(4, 4096),
                "score": 0,
            }


def roboreward_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # Pad T5 embeddings to the longest in this batch.
    max_len = max(b["t5_embedding"].shape[0] for b in batch)
    dim = batch[0]["t5_embedding"].shape[1]
    t5_padded = []
    for b in batch:
        emb = b["t5_embedding"]
        if emb.shape[0] < max_len:
            pad = torch.zeros(max_len - emb.shape[0], dim, dtype=emb.dtype)
            emb = torch.cat([emb, pad], dim=0)
        t5_padded.append(emb)
    t5_stacked = torch.stack(t5_padded, dim=0)  # [B, L_max, 4096]

    return {
        "cam_high": torch.stack([b["cam_high"] for b in batch]),
        "cam_left_wrist": torch.stack([b["cam_left_wrist"] for b in batch]),
        "cam_right_wrist": torch.stack([b["cam_right_wrist"] for b in batch]),
        "state": torch.stack([b["state"] for b in batch]),
        "future_state": torch.stack([b["future_state"] for b in batch]),
        "value": torch.stack([b["value"] for b in batch]),
        "value_normalized": torch.stack([b["value_normalized"] for b in batch]),
        "frame_idx": torch.tensor([b["frame_idx"] for b in batch]),
        "episode_length": torch.tensor([b["episode_length"] for b in batch]),
        "future_frame_idx": torch.tensor([b["future_frame_idx"] for b in batch]),
        "task_id": torch.zeros(len(batch), dtype=torch.long),
        "t5_embedding": t5_stacked,
        "score": torch.tensor([b["score"] for b in batch], dtype=torch.long),
    }
