# ViVa in mm-value

ViVa is a video-generative value model for robot manipulation. It conditions a Wan2.2-TI2V-5B backbone on three camera views, a 14-dimensional robot state, and a precomputed T5 task embedding. It predicts a scalar task-progress value and a future robot state. This repository contains the model, LeRobot data pipeline, training and inference entry points, milestone labels, and uncertainty analysis code.

The implementation is adapted from [ViVa: A Video-Generative Value Model for Robot Reinforcement Learning](https://arxiv.org/abs/2604.08168). The `wan/` directory contains upstream Wan code, with one import-time CUDA initialization fix in `wan/modules/t5.py`; the pretrained Wan weights must be downloaded separately.

## Layout

| Path | Purpose |
| --- | --- |
| `viva_model.py`, `wan_model.py`, `wan/` | Generative model and Wan backbone |
| `viva_dataset.py` | LeRobot loading and progress/future-state targets |
| `train.py`, `inference.py`, `visualization.py` | Train, annotate episodes, and visualize results |
| `config/train_example.yaml`, `config/train_aloha_example.yaml` | Portable single- and multi-task training templates |
| `labeling/` | Stage/critical-window labels and conversion scripts |
| `scripts_repro/` | Value-confidence analysis |
| `criticality_*.py`, `train_criticality.py`, `eval_criticality.py` | Separate proprioception-only critical-frame classifier |

## Install

Requires Linux, CUDA GPUs, and Python 3.11. The 5B backbone is too large for a typical CPU workstation.

```bash
conda create -n viva python=3.11 -y
conda activate viva
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir weights/Wan2.2-TI2V-5B
```

Model weights, datasets, checkpoints, generated media, and text embeddings are excluded from Git. Place them locally under the paths referenced by your config.

## Prepare data and train

1. Prepare a LeRobot dataset with three camera streams and `observation.state` (14 values per frame). Update the camera keys in `config/train_example.yaml` if needed.
2. Generate or provide a T5 embedding for the task description. `get_text_embedding.py` accepts `--txt_list` (a file listing text files) and `--device cuda:0`; it writes embeddings under `data/t5_embedding/`.
3. Copy `config/train_example.yaml` to a local config. Set `dataset.data_paths`, `dataset.t5_embedding_path`, and any model paths that differ from the defaults.
4. For milestone-aware training, set `dataset.milestones_jsonl` to a file whose `episode_idx` matches the dataset episode order. For multiple datasets with local episode numbering, also set `dataset.subdataset_ep_offsets` to the global starting episode index of each dataset. See `labeling/milestones_vlm_all.jsonl` for the schema. `labeling/convert_to_milestones_jsonl.py` converts the per-episode labels in `labeling/labels_vlm_all/`.

```bash
cp config/train_example.yaml config/train_local.yaml
# Edit config/train_local.yaml before running.
accelerate launch --num_processes 8 --mixed_precision bf16 \
  train.py --config config/train_local.yaml
```

Without `milestones_jsonl`, the target value increases linearly from 0 to 1 over each episode. With milestones and critical windows, the value rises faster in critical windows; `dual_slope_r` controls that slope ratio. During critical windows the future-state target is the end of the current stage; elsewhere it is `future_offset` frames ahead. The main model predicts a continuous value and future state, not a discrete stage name.

## Infer and inspect

```bash
python inference.py \
  --checkpoint /path/to/checkpoint_step_7000 \
  --config config/train_local.yaml \
  --data_path /path/to/lerobot_dataset \
  --t5_embedding /path/to/task_embedding.pt \
  --num_gpus 1 --num_inference_steps 1

python visualization.py \
  --checkpoint /path/to/checkpoint_step_7000 \
  --config config/train_local.yaml \
  --data_path /path/to/lerobot_dataset \
  --t5_embedding /path/to/task_embedding.pt \
  --episode 0
```

`inference.py` writes `prediction` (0–1 progress) and `model_prediction` (`prediction × (episode_length − 1)`) **into the input dataset's Parquet files**. Use a working copy of the dataset if the original must stay unchanged. Use the training set's state min/max for inference. If inference data differs from training data, pass the saved training statistics with `--state_txt`; otherwise, `compute_state_stats: true` computes statistics from the inference dataset.

## Uncertainty / entropy proxies

`VivaModel.predict_value(..., num_samples=K)` produces `value_ensemble_std` and `future_state_ensemble_std` across K independent generations. `VivaModel.estimate_generative_entropy(...)` computes a denoising-residual score at multiple noise levels. These are uncertainty proxies, not calibrated Shannon entropies. The batch `inference.py` command does not expose these modes; use `scripts_repro/collect_value_confidence.py` for per-frame collection and `scripts_repro/analyze_value_confidence.py` for analysis. The collection script requires `torchrun` and a checkpoint.

## Included labels

`labeling/labels_vlm_all/` contains stage and critical-window labels for 30 Task A and 150 ALOHA episodes. `labeling/milestones_vlm_all.jsonl` contains their converted milestone records. `data/milestones_task_a_0509v2.jsonl` is a Task A-only variant. These labels are examples tied to specific episode ordering; they are not automatically applicable to a new dataset.

## Verify

```bash
python -m compileall -q .
pip install -r requirements-dev.txt
python -m pytest tests/test_criticality.py
```

The test suite covers the separate critical-frame classifier. Full ViVa training and inference require a compatible dataset, pretrained weights, and CUDA GPUs.
