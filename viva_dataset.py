# Viva Dataset — dual_slope_milestone
#
# Value: Δ_ord for ordinary frames, r·Δ_ord for critical frames.
#        Normalized so v[0]=0, v[T-1]=1, strictly monotonically increasing.
# Future state: ordinary → state[t + future_offset]
#               critical → state[milestone_frame] (stage-end boundary)
#
# Reads a milestones JSONL with per-episode stage boundaries + critical_windows.
# Each line: {"episode_idx": N, "milestones": [{"start_frame": ..., "critical_windows": [...]}]}
#
# 7-frame structure: [blank, state, cam_left, cam_right, cam_high, future_state, value]

import bisect
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import logging
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger(__name__)


class _LocalLeRobotSubDataset:
    def __init__(self, root: str):
        import pyarrow.parquet as pq

        self.root = Path(root)
        info_path = self.root / "meta" / "info.json"
        episodes_path = self.root / "meta" / "episodes.jsonl"
        tasks_path = self.root / "meta" / "tasks.jsonl"

        with open(info_path) as f:
            info = json.load(f)

        episodes = []
        with open(episodes_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    episodes.append({"length": int(row["length"])})

        tasks = {0: "unknown"}
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    task_id = int(row.get("task_index", row.get("task_id", 0)))
                    tasks[task_id] = row.get("task", row.get("prompt", str(row)))

        self.dataset = SimpleNamespace(
            root=self.root,
            meta=SimpleNamespace(info=info, episodes=episodes, tasks=tasks),
        )
        self._episode_lengths = [ep["length"] for ep in episodes]
        self._episode_offsets = np.cumsum([0] + self._episode_lengths).tolist()
        self._table_cache = {}
        self._video_cache = {}
        self._pq = pq

    def __len__(self):
        return int(self._episode_offsets[-1])

    def _episode_for_index(self, idx: int):
        ep_idx = bisect.bisect_right(self._episode_offsets, idx) - 1
        frame_idx = idx - self._episode_offsets[ep_idx]
        return ep_idx, frame_idx

    def _table(self, episode_index: int):
        table = self._table_cache.get(episode_index)
        if table is None:
            chunk_idx = episode_index // 1000
            path = self.root / f"data/chunk-{chunk_idx:03d}" / f"episode_{episode_index:06d}.parquet"
            table = self._pq.read_table(path)
            self._table_cache[episode_index] = table
        return table

    def _read_video_frame(self, video_key: str, episode_index: int, frame_index: int):
        chunk_idx = episode_index // 1000
        path = self.root / f"videos/chunk-{chunk_idx:03d}" / video_key / f"episode_{episode_index:06d}.mp4"
        cache_key = (str(path), video_key)

        reader = self._video_cache.get(cache_key)
        if reader is None:
            try:
                from decord import VideoReader, cpu
                reader = ("decord", VideoReader(str(path), ctx=cpu(0)))
            except Exception:
                import cv2
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open video: {path}")
                reader = ("cv2", cap)
            self._video_cache[cache_key] = reader

        kind, obj = reader
        if kind == "decord":
            frame = obj[int(frame_index)].asnumpy()
        else:
            import cv2
            obj.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = obj.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_index} from {path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(frame)

    def __getitem__(self, idx: int):
        ep_idx, local_frame_idx = self._episode_for_index(int(idx))
        table = self._table(ep_idx)
        row = table.slice(local_frame_idx, 1).to_pydict()

        def scalar(name, default=0):
            values = row.get(name)
            return default if values is None else values[0]

        sample = {
            "observation.state": scalar("observation.state", [0.0] * 14),
            "frame_index": int(scalar("frame_index", local_frame_idx)),
            "episode_index": int(scalar("episode_index", ep_idx)),
            "task_index": int(scalar("task_index", 0)),
        }

        for key in (
            "observation.images.top_head",
            "observation.images.hand_left",
            "observation.images.hand_right",
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ):
            # LeRobot datasets differ on video directory naming: Kai0 uses the
            # full feature key, while older ViVa-compatible exports use suffixes.
            for video_key in (key, key.split(".")[-1]):
                path = self.root / f"videos/chunk-{ep_idx // 1000:03d}" / video_key / f"episode_{ep_idx:06d}.mp4"
                if path.exists():
                    sample[key] = self._read_video_frame(video_key, ep_idx, sample["frame_index"])
                    break
        return sample


class _LocalLeRobotDataset:
    def __init__(self, configs):
        self.datasets = [_LocalLeRobotSubDataset(c["data_path"]) for c in configs]
        self.cumulative_sizes = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int):
        ds_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        prev = 0 if ds_idx == 0 else self.cumulative_sizes[ds_idx - 1]
        return self.datasets[ds_idx][idx - prev]


def _load_lerobot_dataset(configs):
    try:
        from giga_datasets.datasets.dataset import load_dataset
        return load_dataset(configs)
    except ModuleNotFoundError as exc:
        if exc.name != "giga_datasets":
            raise
        logger.warning("giga_datasets is not installed; using local LeRobot parquet/mp4 loader")
        return _LocalLeRobotDataset(configs)


class VivaDatasetDSM(Dataset):

    def __init__(
        self,
        data_paths: Optional[List[str]] = None,
        video_height: int = 384,
        video_width: int = 320,
        state_stats: Optional[Dict] = None,
        skip_video_decoding: bool = False,
        max_samples: Optional[int] = None,
        future_offset: int = 30,
        task_configs: Optional[List[Dict]] = None,
        milestones_jsonl: Optional[str] = None,
        subdataset_ep_offsets: Optional[List[int]] = None,
        dual_slope_r: float = 3.0,
        dual_slope_event_def: str = "window",
        cam_key_high: str = "observation.images.cam_high",
        cam_key_left: str = "observation.images.cam_left_wrist",
        cam_key_right: str = "observation.images.cam_right_wrist",
    ):
        self.video_height = video_height
        self.video_width = video_width
        self.max_samples = max_samples
        self.future_offset = future_offset
        self.dual_slope_r = dual_slope_r
        self.cam_key_high = cam_key_high
        self.cam_key_left = cam_key_left
        self.cam_key_right = cam_key_right
        self.dual_slope_event_def = dual_slope_event_def
        self._subdataset_ep_offsets = subdataset_ep_offsets

        # -- Load milestones JSONL --
        # _episode_milestones: ep_idx → [(start_frame, ...)]  for stage boundaries
        # _episode_critical_windows: ep_idx → [(start, end_excl)]  for critical windows
        self._episode_milestones: Dict[int, List[int]] = {}
        self._episode_critical_windows: Dict[int, List[tuple]] = {}
        if milestones_jsonl is not None:
            with open(milestones_jsonl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    ep = int(row["episode_idx"])
                    starts = []
                    all_cws = []
                    for m in row["milestones"]:
                        starts.append(int(m["start_frame"]))
                        for cw in m.get("critical_windows", []):
                            all_cws.append((int(cw[0]), int(cw[1])))
                    self._episode_milestones[ep] = sorted(starts)
                    if all_cws:
                        self._episode_critical_windows[ep] = sorted(all_cws)
            logger.info(f"Loaded milestones for {len(self._episode_milestones)} episodes")
            logger.info(f"Loaded critical_windows for {len(self._episode_critical_windows)} episodes")
        else:
            logger.warning("No milestones_jsonl provided — value will be linear")

        # -- Value curve cache --
        self._value_curves: Dict[int, np.ndarray] = {}

        # -- Multi-task data paths --
        self._subdataset_to_task_id = None
        self.task_configs = task_configs
        if task_configs is not None:
            all_data_paths = []
            self._subdataset_to_task_id = {}
            sub_ds_counter = 0
            for task_idx, tc in enumerate(task_configs):
                for dp in tc['data_paths']:
                    all_data_paths.append(dp)
                    self._subdataset_to_task_id[sub_ds_counter] = task_idx
                    sub_ds_counter += 1
            data_paths = all_data_paths
            logger.info(f"Multi-task: {len(task_configs)} tasks, {len(data_paths)} sub-datasets")
        else:
            if data_paths is None:
                raise ValueError("Either data_paths or task_configs must be provided")

        # -- State normalization --
        if state_stats is not None:
            self.state_min = torch.tensor(state_stats['state_min'], dtype=torch.float32)
            self.state_max = torch.tensor(state_stats['state_max'], dtype=torch.float32)
        else:
            self.state_min = torch.tensor([-3.0] * 14, dtype=torch.float32)
            self.state_max = torch.tensor([3.0] * 14, dtype=torch.float32)

        # -- Load LeRobot datasets --
        configs = []
        for input_dir in data_paths:
            configs.append({
                "_class_name": "LeRobotDataset",
                "data_path": str(input_dir),
                "skip_video_decoding": skip_video_decoding,
                "tolerance_s": 100.0,
            })
        self.lerobot_dataset = _load_lerobot_dataset(configs)
        logger.info(f"Total samples: {len(self.lerobot_dataset)}")

        if not hasattr(self.lerobot_dataset, 'cumulative_sizes'):
            cumsum = []
            total = 0
            for ds in self.lerobot_dataset.datasets:
                total += len(ds)
                cumsum.append(total)
            self.lerobot_dataset.cumulative_sizes = cumsum

        # -- Episode lengths per subdataset --
        self._subdataset_episode_lengths = []
        max_episode_length = 0
        for dataset in self.lerobot_dataset.datasets:
            sub_lengths = []
            meta = dataset.dataset.meta
            for episode_idx in range(meta.info["total_episodes"]):
                length = int(meta.episodes[episode_idx]['length'])
                sub_lengths.append(length)
                max_episode_length = max(max_episode_length, length)
            self._subdataset_episode_lengths.append(sub_lengths)
        self.max_episode_length = max_episode_length

        # -- Sampling --
        self._total_samples = len(self.lerobot_dataset)
        self._sample_indices = None
        if self.max_samples is not None and self.max_samples > 0:
            self._effective_length = min(self.max_samples, self._total_samples)
            if self._effective_length < self._total_samples:
                rng = np.random.RandomState(42)
                self._sample_indices = rng.choice(
                    self._total_samples, size=self._effective_length, replace=False)
        else:
            self._effective_length = self._total_samples

        logger.info(f"DSM dataset ready: {self._effective_length} samples, "
                    f"r={dual_slope_r}, event_def={dual_slope_event_def}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_subdataset_index(self, global_idx: int) -> int:
        return bisect.bisect_right(self.lerobot_dataset.cumulative_sizes, global_idx)

    def __len__(self) -> int:
        return self._effective_length

    def _process_image(self, img) -> torch.Tensor:
        import torch.nn.functional as F
        if isinstance(img, torch.Tensor):
            if img.dim() == 3:
                image = img.float() if img.shape[0] == 3 else img.permute(2, 0, 1).float()
            else:
                image = img.float()
            if image.max() > 1.0:
                image = image / 255.0
        elif hasattr(img, 'convert'):
            img_np = np.array(img.convert('RGB')).astype(np.float32) / 255.0
            image = torch.from_numpy(img_np).permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        c, h, w = image.shape
        target_h, target_w = self.video_height, self.video_width
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w),
                              mode='bilinear', align_corners=False).squeeze(0)
        pad_h, pad_w = target_h - new_h, target_w - new_w
        image = F.pad(image, (pad_w // 2, pad_w - pad_w // 2,
                              pad_h // 2, pad_h - pad_h // 2), value=0)
        return image

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        state = torch.clamp(state, self.state_min, self.state_max)
        state_normalized = (state - self.state_min) / (self.state_max - self.state_min + 1e-8)
        return state_normalized * 2 - 1

    def _load_state(self, sample) -> torch.Tensor:
        state_raw = sample.get('observation.state', None)
        if state_raw is not None:
            if isinstance(state_raw, torch.Tensor):
                return self._normalize_state(state_raw.float())
            return self._normalize_state(torch.tensor(state_raw, dtype=torch.float32))
        return torch.zeros(14)

    # ------------------------------------------------------------------
    # Value: dual_slope_milestone
    #
    #   inc[t] = Δ_crit  if t is a critical event frame
    #            Δ_ord   otherwise
    #
    #   Δ_ord  = 1 / ((r-1)·M + T)
    #   Δ_crit = r / ((r-1)·M + T)
    #
    #   v[0] = 0,  v[t] = Σ_{k=1..t} inc[k],  normalized so v[T-1] = 1
    # ------------------------------------------------------------------

    def _compute_value(self, frame_idx: int, episode_length: int,
                       episode_idx: int) -> tuple:
        """Returns (value_in_0_1, is_critical)."""
        cws = self._episode_critical_windows.get(episode_idx)
        if cws is None:
            # No critical windows → linear increasing
            if episode_length <= 1:
                return 0.0, False
            return frame_idx / (episode_length - 1), False

        T = int(episode_length)
        curve = self._value_curves.get(episode_idx)
        if curve is None or len(curve) != T:
            event_frames = sorted({
                    f for cs, ce in cws
                    for f in range(max(0, cs), min(T, ce))
                })

            M = len(event_frames)
            r = self.dual_slope_r
            denom = (r - 1.0) * M + T
            d_ord = 1.0 / denom
            d_crit = r / denom
            inc = np.full(T, d_ord, dtype=np.float64)
            for ef in event_frames:
                inc[ef] = d_crit
            curve = np.zeros(T, dtype=np.float64)
            if T > 1:
                curve[1:] = np.cumsum(inc[1:])
                if curve[-1] > 0:
                    curve /= curve[-1]
            self._value_curves[episode_idx] = curve

        v = float(curve[max(0, min(T - 1, int(frame_idx)))])
        in_crit = any(cs <= frame_idx < ce for cs, ce in cws)
        return v, in_crit

    # ------------------------------------------------------------------
    # Future state: milestone_during_critical
    #
    #   ordinary → state[t + future_offset]  (sliding window)
    #   critical → state[milestone_frame]    (stage-end boundary)
    # ------------------------------------------------------------------

    def _compute_future_frame(self, frame_index: int, episode_length: int,
                              episode_idx: int) -> int:
        default = min(frame_index + self.future_offset, episode_length - 1)

        cws = self._episode_critical_windows.get(episode_idx)
        starts = self._episode_milestones.get(episode_idx)
        if cws is None or starts is None:
            return default

        # Is frame inside any critical window?
        in_cw = False
        for cs, ce in cws:
            if cs <= frame_index < ce:
                in_cw = True
                break

        if not in_cw:
            return default

        # Critical frame → find the milestone (stage-end) for current stage
        stage_idx = 0
        for k, s in enumerate(starts):
            if frame_index >= s:
                stage_idx = k
        stage_end = starts[stage_idx + 1] if stage_idx + 1 < len(starts) else episode_length
        milestone_frame = min(stage_end - 1, episode_length - 1)
        return milestone_frame

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            actual_idx = idx
            if self._sample_indices is not None:
                actual_idx = int(self._sample_indices[idx])

            sample = self.lerobot_dataset[actual_idx]

            sub_ds_idx = self._get_subdataset_index(actual_idx)
            frame_index = sample['frame_index'].item() if isinstance(sample['frame_index'], torch.Tensor) else sample['frame_index']
            episode_index_local = sample['episode_index'].item() if isinstance(sample['episode_index'], torch.Tensor) else sample['episode_index']
            episode_length = self._subdataset_episode_lengths[sub_ds_idx][episode_index_local]

            if self._subdataset_ep_offsets:
                episode_index = self._subdataset_ep_offsets[sub_ds_idx] + episode_index_local
            else:
                episode_index = episode_index_local

            # -- Images --
            cam_high_raw = sample.get(self.cam_key_high, None)
            cam_left_raw = sample.get(self.cam_key_left, None)
            cam_right_raw = sample.get(self.cam_key_right, None)
            zero_img = torch.zeros(3, self.video_height, self.video_width)
            cam_high = self._process_image(cam_high_raw) if cam_high_raw is not None else zero_img.clone()
            cam_left_wrist = self._process_image(cam_left_raw) if cam_left_raw is not None else zero_img.clone()
            cam_right_wrist = self._process_image(cam_right_raw) if cam_right_raw is not None else zero_img.clone()

            # -- State --
            state_normalized = self._load_state(sample)

            # -- Value (dual slope milestone) --
            value, is_critical = self._compute_value(frame_index, episode_length, episode_index)

            # -- Future state (milestone during critical) --
            future_frame_idx = self._compute_future_frame(frame_index, episode_length, episode_index)
            delta = future_frame_idx - frame_index
            if delta > 0:
                future_sample = self.lerobot_dataset[actual_idx + delta]
                future_state_normalized = self._load_state(future_sample)
            else:
                future_state_normalized = state_normalized.clone()

            task_id = self._subdataset_to_task_id[sub_ds_idx] if self._subdataset_to_task_id else 0

            return {
                'cam_high': cam_high,
                'cam_left_wrist': cam_left_wrist,
                'cam_right_wrist': cam_right_wrist,
                'state': state_normalized,
                'future_state': future_state_normalized,
                'value': torch.tensor(value, dtype=torch.float32),
                'value_normalized': torch.tensor(float(value) * 2 - 1, dtype=torch.float32),
                'is_critical': torch.tensor(is_critical, dtype=torch.bool),
                'frame_idx': frame_index,
                'episode_length': episode_length,
                'future_frame_idx': future_frame_idx,
                'task_id': task_id,
            }

        except Exception as e:
            logger.warning(f"Error loading sample idx={idx}: {e}")
            zero_img = torch.zeros(3, self.video_height, self.video_width)
            return {
                'cam_high': zero_img.clone(),
                'cam_left_wrist': zero_img.clone(),
                'cam_right_wrist': zero_img.clone(),
                'state': torch.zeros(14),
                'future_state': torch.zeros(14),
                'value': torch.tensor(0.0),
                'value_normalized': torch.tensor(-1.0),
                'is_critical': torch.tensor(False, dtype=torch.bool),
                'frame_idx': 0,
                'episode_length': 1,
                'future_frame_idx': 0,
                'task_id': 0,
            }


def dsm_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return {
        'cam_high': torch.stack([b['cam_high'] for b in batch]),
        'cam_left_wrist': torch.stack([b['cam_left_wrist'] for b in batch]),
        'cam_right_wrist': torch.stack([b['cam_right_wrist'] for b in batch]),
        'state': torch.stack([b['state'] for b in batch]),
        'future_state': torch.stack([b['future_state'] for b in batch]),
        'value': torch.stack([b['value'] for b in batch]),
        'value_normalized': torch.stack([b['value_normalized'] for b in batch]),
        'is_critical': torch.stack([b['is_critical'] for b in batch]),
        'frame_idx': torch.tensor([b['frame_idx'] for b in batch]),
        'episode_length': torch.tensor([b['episode_length'] for b in batch]),
        'future_frame_idx': torch.tensor([b['future_frame_idx'] for b in batch]),
        'task_id': torch.tensor([b['task_id'] for b in batch], dtype=torch.long),
    }


def compute_dataset_state_stats(data_paths):
    """Compute per-channel min/max directly from LeRobot parquet, without decoding video."""
    import pyarrow.parquet as pq

    state_min = None
    state_max = None
    for data_path in data_paths:
        parquet_files = sorted((Path(data_path) / "data").glob("chunk-*/episode_*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No LeRobot episode parquet files found under {data_path}")
        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file, columns=["observation.state"])
            states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
            if states.size == 0:
                continue
            if states.ndim != 2 or states.shape[1] != 14:
                raise ValueError(f"Expected 14-dimensional observation.state in {parquet_file}, got {states.shape}")
            lo = states.min(axis=0)
            hi = states.max(axis=0)
            state_min = lo if state_min is None else np.minimum(state_min, lo)
            state_max = hi if state_max is None else np.maximum(state_max, hi)
    if state_min is None:
        return {'state_min': [-3.0] * 14, 'state_max': [3.0] * 14}
    return {
        'state_min': state_min.tolist(),
        'state_max': state_max.tolist(),
    }


VivaDataset = VivaDatasetDSM
viva_collate_fn = dsm_collate_fn
