# Stage and critical-window labels

`labels_vlm_all/<task>/epNNN.json` contains one episode's original labels. Key fields are:

- `stages`: ordered stage names;
- `milestone_cuts`: frame indices where a new stage begins;
- `critical_windows`: `[start, end)` frame intervals;
- `per_frame.stage`: stage index for every frame.

The 180 included episodes comprise 30 `task_a` and 50 each of `aloha_coffee`, `aloha_towel`, and `aloha_candy`. The annotations were produced with a VLM and need review before using them as ground truth in a new setting.

`milestones_vlm_all.jsonl` is the training-friendly conversion. Every line has an `episode_idx` and a `milestones` list. A milestone has `start_frame`, `label`, and `critical_windows`. The `episode_idx` is global and depends on the order in which datasets are loaded: Task A first, then coffee, towel, candy. Match that order before using the combined file in a training config.

To rebuild it from the original JSON labels:

```bash
python labeling/convert_to_milestones_jsonl.py \
  --label-dir labeling/labels_vlm_all \
  --out labeling/milestones_vlm_all.jsonl
```

`label_episode.py` and `label_episode_proprio.py` are the original label-generation scripts. Their dataset and model paths are environment-specific; edit those paths for your environment before running them.
