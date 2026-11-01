"""Label one episode with both milestone stages and critical-step windows via VLM.

Outputs a single JSON file per episode with per-frame annotations:
  - stage (int): which milestone stage this frame belongs to
  - is_milestone (bool): whether this frame is a milestone transition frame
  - is_critical (bool): whether this frame falls in a critical-step window

Usage:
  DASHSCOPE_API_KEY=sk-... python label_episode.py \
      --task task_a --session 2026-05-09-v2 --episode 9

  DASHSCOPE_API_KEY=sk-... python label_episode.py \
      --task aloha_coffee --episode 0

  DASHSCOPE_API_KEY=sk-... python label_episode.py \
      --task aloha_towel --episode 5

Output (default: ./labels/ep{NNN}.json):
  {
    "episode_id": 9,
    "total_frames": 1949,
    "src_fps": 30.0,
    "task": "Flatten and fold the cloth.",
    "stages": ["approach", "flatten", "fold", "press", "return"],
    "milestone_cuts": [105, 750, 1620, 1845],
    "critical_windows": [[143, 146], [185, 188], ...],
    "per_frame": {
      "stage": [0, 0, ..., 4, 4],
      "is_milestone": [false, ..., true, ..., false],
      "is_critical": [false, ..., true, ..., false]
    }
  }
"""

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------

TASKS = {
    "task_a": {
        "data_root": "/vePFS-East/comp_robot/zundong/Task_A/base/base",
        "has_sessions": True,
        "cam": "top_head",
        "src_fps": 30.0,
        "task_text": "Flatten and fold the cloth.",
        "stages": ["approach", "flatten", "fold", "press", "return"],
        "transitions": [
            "approach->flatten", "flatten->fold",
            "fold->press", "press->return",
        ],
        "milestone_prompt": """\
You are watching a bimanual robot demonstration video of the task:
"{task}"

The video has been subsampled into {n} frames in chronological order.
Identify the SAMPLE INDEX (0..{nm1}) at which each of the four stage
transitions FIRST occurs. The 5 stages are:

  0 approach : both arms moving toward the cloth, NO contact yet
  1 flatten  : arms in contact with cloth, smoothing it flat on the table
               (no lifting / folding yet)
  2 fold     : at least one gripper has grasped a cloth edge AND begun
               lifting/folding it over onto another part of the cloth
  3 press    : the fold has been laid down; grippers press / smooth the
               folded cloth in place (or release and lightly tap)
  4 return   : grippers have left the cloth and arms are retracting

Reply with EXACTLY this JSON (no prose, no code fences):
{{"transitions": {{"approach->flatten": <int>, "flatten->fold": <int>,
                  "fold->press": <int>, "press->return": <int>}}}}

The 4 indices must be strictly increasing in 0..{nm1}.""",
        "critical_prompt": """\
You are watching a bimanual robot demonstration video of the task:
"{task}"

The video has been subsampled into {n} frames in chronological order
(sample indices 0..{nm1}).

Your job: identify the precise INSTANTS of PRECISION-MANIPULATION CONTACT.

Expected precision-contact instants for this task (multiple of EACH may occur):
  * LEFT-gripper closes (pinches) on cloth.
  * RIGHT-gripper closes (pinches) on cloth.
  * LEFT-gripper releases the cloth.
  * RIGHT-gripper releases the cloth.
  * Either gripper PRESSES DOWN on the folded cloth (each separate press = its own event).
  * Either gripper RE-GRASPS for adjustment.

Sustained holding (the 5-50 samples WHILE a gripper holds cloth) is NOT critical.
Only the start (close) and end (open) of each holding period are critical.

ORDINARY (NOT critical), regardless of speed:
  * Sliding a gripper across the cloth to smooth or flatten it.
  * Slow approach toward the cloth (only the actual pinch instant is critical).
  * Carrying a held-cloth-edge through the air.
  * Gripper transport through free space.
  * Approach / retract motions.

Each contact instant should span AT MOST 2 SAMPLES (1-2 wide).

BUDGET: between 2 and 12 windows. Most episodes have 4-10.

Reply with EXACTLY this JSON (no prose, no code fences):
{{"windows": [[<start_sample>, <end_sample>], ...]}}

Constraints:
  - Each [start, end] pair satisfies 0 <= start <= end <= {nm1}
  - end - start MUST be 0 or 1 (i.e. 1 or 2 samples wide)
  - Sorted by start_sample
  - Total window count: between 2 and 12 (inclusive)""",
    },
    "aloha_coffee": {
        "data_root": "/vePFS-East/comp_robot/zundong/ViVa_pic/data/lerobot/aloha_static_coffee",
        "has_sessions": False,
        "cam": "observation.images.cam_high",
        "src_fps": 50.0,
        "task_text": (
            "Place the coffee capsule inside the capsule container, then "
            "place the cup onto the center of the cup tray, then push the "
            "'Hot Water' and 'Travel Mug' buttons."
        ),
        "stages": [
            "approach_capsule", "insert_capsule", "place_cup",
            "press_buttons", "done",
        ],
        "transitions": [
            "approach_capsule->insert_capsule", "insert_capsule->place_cup",
            "place_cup->press_buttons", "press_buttons->done",
        ],
        "milestone_prompt": """\
You are watching a bimanual robot demonstration video of the task:
"{task}"

The video has been subsampled into {n} frames in chronological order.
Identify the SAMPLE INDEX (0..{nm1}) at which each of the four stage
transitions FIRST occurs. The 5 stages are:

  0 approach_capsule : the grippers are moving toward the coffee CAPSULE; NO contact yet.
  1 insert_capsule   : at least one gripper has GRASPED the capsule AND is
                       moving it toward / inserting it into the capsule container.
  2 place_cup        : the capsule is released; a gripper has grasped the
                       cup and is moving / placing it onto the cup tray.
  3 press_buttons    : the cup is released; a gripper is pressing the
                       'Hot Water' or 'Travel Mug' button on the machine.
  4 done             : the buttons have been pressed and the grippers are
                       retracting away from the machine.

Reply with EXACTLY this JSON (no prose, no code fences):
{{"transitions": {{"approach_capsule->insert_capsule": <int>,
                 "insert_capsule->place_cup": <int>,
                 "place_cup->press_buttons": <int>,
                 "press_buttons->done": <int>}}}}

The 4 indices must be strictly increasing in 0..{nm1}.""",
        "critical_prompt": """\
You are watching a bimanual robot demonstration video of the task:
"{task}"

The video has been subsampled into {n} frames in chronological order
(sample indices 0..{nm1}).

Your job: identify the precise INSTANTS of PRECISION-MANIPULATION CONTACT.

For this task, the expected precision-contact instants are exactly:
  1. The moment the gripper CLOSES around the coffee capsule.
  2. The moment the capsule is RELEASED into the capsule container slot.
  3. The moment the gripper CLOSES around the cup.
  4. The moment the cup is RELEASED onto the cup tray.
  5. The moment the 'Hot Water' button is DEPRESSED.
  6. The moment the 'Travel Mug' button is DEPRESSED.

ORDINARY (NOT critical), regardless of speed:
  * Slow lowering of the capsule toward the slot (only the actual entry instant is critical).
  * Slow lowering of the cup toward the tray (only the actual touch-down instant is critical).
  * Slow approach toward a button (only the actual depression instant is critical).
  * Gripper transport through free space.
  * Approach / retract motions.

Each contact instant should span AT MOST 2 SAMPLES.

Reply with EXACTLY this JSON (no prose, no code fences):
{{"windows": [[<start_sample>, <end_sample>], ...]}}

Constraints:
  - Each [start, end] pair satisfies 0 <= start <= end <= {nm1}
  - end - start should be 0 or 1 (i.e. 1 or 2 samples wide)
  - Windows must be sorted by start_sample and non-overlapping
  - Aim for 4 to 6 windows total""",
    },
    "aloha_towel": {
        "data_root": "/vePFS-East/comp_robot/zundong/ViVa_pic/data/lerobot/aloha_static_towel",
        "has_sessions": False,
        "cam": "observation.images.cam_high",
        "src_fps": 50.0,
        "task_text": "Pick up a piece of paper towel and place it on the spilled liquid.",
        "stages": ["approach", "grasp", "carry", "release", "return"],
        "transitions": [
            "approach->grasp", "grasp->carry",
            "carry->release", "release->return",
        ],
        "milestone_prompt": """\
You are watching a robot demonstration video of the task:
"{task}"

The video has been subsampled into {n} frames in chronological order.
Identify the SAMPLE INDEX (0..{nm1}) at which each of the four stage
transitions FIRST occurs. The transitions are:

  approach -> grasp   : the moment the gripper closes around the paper towel
  grasp    -> carry   : the moment the towel leaves the table surface
  carry    -> release : the moment the towel is placed and the gripper begins to open
  release  -> return  : the moment the gripper, now empty, starts retracting

Reply with EXACTLY this JSON (no prose, no code fences):
{{"transitions": {{"approach->grasp": <int>, "grasp->carry": <int>,
                  "carry->release": <int>, "release->return": <int>}}}}

The 4 indices must be strictly increasing in 0..{nm1}.""",
        "critical_prompt": """\
You are watching a robot demonstration video of the task:
"{task}"

The video has been subsampled into {n} frames in chronological order
(sample indices 0..{nm1}).

Your job: identify the precise INSTANTS of PRECISION-MANIPULATION CONTACT.

Expected precision-contact instants:
  * Gripper CLOSES around the paper towel (grasp).
  * Gripper RELEASES the towel onto the spill (release).
  * Any RE-GRASP or adjustment contact.

Sustained holding (while carrying the towel) is NOT critical.
Only the start (close) and end (open) of each holding period are critical.

ORDINARY (NOT critical):
  * Slow approach toward the towel.
  * Carrying the towel through the air.
  * Gripper transport through free space.
  * Approach / retract motions.

Each contact instant should span AT MOST 2 SAMPLES (1-2 wide).

BUDGET: between 2 and 6 windows.

Reply with EXACTLY this JSON (no prose, no code fences):
{{"windows": [[<start_sample>, <end_sample>], ...]}}

Constraints:
  - Each [start, end] pair satisfies 0 <= start <= end <= {nm1}
  - end - start MUST be 0 or 1 (i.e. 1 or 2 samples wide)
  - Sorted by start_sample
  - Total window count: between 2 and 6 (inclusive)""",
    },
    "aloha_candy": {
        "data_root": "/vePFS-East/comp_robot/zundong/ViVa_pic/data/lerobot/aloha_static_candy",
        "has_sessions": False,
        "cam": "observation.images.cam_high",
        "src_fps": 50.0,
        "task_text": "Pick up the candy and unwrap it.",
        "stages": ["approach", "grasp", "manipulate", "unwrap", "return"],
        "transitions": [
            "approach->grasp", "grasp->manipulate",
            "manipulate->unwrap", "unwrap->return",
        ],
        "milestone_prompt": """\
You are watching a bimanual robot demonstration video of the task:
"{task}"

The video has been subsampled into {n} frames in chronological order.
Identify the SAMPLE INDEX (0..{nm1}) at which each of the four stage
transitions FIRST occurs. The 5 stages are:

  0 approach    : grippers moving toward the candy, NO contact yet
  1 grasp       : at least one gripper has closed around the candy
  2 manipulate  : both grippers are working to unwrap — pulling wrapper apart
  3 unwrap      : wrapper is separated from the candy body
  4 return      : grippers retract away from the candy

Reply with EXACTLY this JSON (no prose, no code fences):
{{"transitions": {{"approach->grasp": <int>, "grasp->manipulate": <int>,
                  "manipulate->unwrap": <int>, "unwrap->return": <int>}}}}

The 4 indices must be strictly increasing in 0..{nm1}.""",
        "critical_prompt": """\
You are watching a bimanual robot demonstration video of the task:
"{task}"

The video has been subsampled into {n} frames in chronological order
(sample indices 0..{nm1}).

Your job: identify the precise INSTANTS of PRECISION-MANIPULATION CONTACT.

Expected precision-contact instants:
  * LEFT-gripper closes on the candy or wrapper.
  * RIGHT-gripper closes on the candy or wrapper.
  * Either gripper releases the candy/wrapper.
  * Either gripper RE-GRASPS for a different grip angle.
  * The moment the wrapper tears/separates.

Sustained holding is NOT critical. Only the start (close) and end (open)
of each holding period are critical.

ORDINARY (NOT critical):
  * Slow approach toward the candy.
  * Holding the candy steady (only the grasp/release endpoints are critical).
  * Gripper transport through free space.

Each contact instant should span AT MOST 2 SAMPLES (1-2 wide).

BUDGET: between 2 and 10 windows.

Reply with EXACTLY this JSON (no prose, no code fences):
{{"windows": [[<start_sample>, <end_sample>], ...]}}

Constraints:
  - Each [start, end] pair satisfies 0 <= start <= end <= {nm1}
  - end - start MUST be 0 or 1 (i.e. 1 or 2 samples wide)
  - Sorted by start_sample
  - Total window count: between 2 and 10 (inclusive)""",
    },
}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-vl-max-latest"


def encode_frame(frame: np.ndarray, max_side: int = 384) -> str:
    img = Image.fromarray(frame)
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)), Image.BILINEAR
        )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def parse_json_response(txt: str) -> dict:
    clean = txt
    if "```" in clean:
        for p in clean.split("```"):
            if "{" in p:
                clean = p
                break
    start = clean.find("{")
    end = clean.rfind("}")
    if start == -1:
        raise ValueError(f"no JSON object in VLM response: {txt[:300]}")
    if end == -1 or end <= start:
        # Truncated response — try to close the JSON
        clean = clean[start:]
        # Find last complete array element: ...], and close ]}
        last_bracket = clean.rfind("],")
        if last_bracket != -1:
            clean = clean[:last_bracket + 1] + "]}"
        else:
            raise ValueError(f"truncated JSON, cannot recover: {txt[:300]}")
    else:
        clean = clean[start:end + 1]
    return json.loads(clean)


def load_video(cfg: dict, session: str | None, episode: int) -> np.ndarray:
    root = Path(cfg["data_root"])
    if cfg["has_sessions"]:
        root = root / session
    mp4 = root / "videos" / "chunk-000" / cfg["cam"] / f"episode_{episode:06d}.mp4"
    if not mp4.exists():
        raise FileNotFoundError(f"video not found: {mp4}")
    return iio.imread(str(mp4))


def subsample(frames: np.ndarray, src_fps: float, sample_fps: float):
    T = frames.shape[0]
    stride = max(1, int(round(src_fps / sample_fps)))
    idx = np.arange(0, T, stride, dtype=np.int64)
    return frames[idx], idx, stride


def vlm_call(client, model: str, prompt: str, frames: np.ndarray,
             max_side: int, max_tokens: int = 400) -> tuple[str, float]:
    content = [{"type": "text", "text": prompt}]
    for fr in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_frame(fr, max_side)},
        })
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip(), time.time() - t0


# ---------------------------------------------------------------------------
# Milestone labeling
# ---------------------------------------------------------------------------

def label_milestones(client, model: str, cfg: dict, sampled_frames: np.ndarray,
                     sample_idx: np.ndarray, max_side: int):
    n = len(sampled_frames)
    prompt = cfg["milestone_prompt"].format(
        task=cfg["task_text"], n=n, nm1=n - 1,
    )
    txt, dt = vlm_call(client, model, prompt, sampled_frames, max_side, 200)
    parsed = parse_json_response(txt)
    trans = parsed["transitions"]
    cuts_sample = [int(trans[k]) for k in cfg["transitions"]]

    if not all(0 <= c <= n - 1 for c in cuts_sample):
        raise ValueError(f"cuts out of range [0,{n-1}]: {cuts_sample}")
    if not all(cuts_sample[i] < cuts_sample[i + 1] for i in range(3)):
        raise ValueError(f"cuts not monotonic: {cuts_sample}")

    cuts_frame = [int(sample_idx[s]) for s in cuts_sample]
    return cuts_frame, dt


# ---------------------------------------------------------------------------
# Critical-step labeling
# ---------------------------------------------------------------------------

def label_critical(client, model: str, cfg: dict, sampled_frames: np.ndarray,
                   sample_idx: np.ndarray, stride: int, T: int, max_side: int,
                   retries: int = 2, max_windows: int = 12):
    n = len(sampled_frames)
    prompt = cfg["critical_prompt"].format(
        task=cfg["task_text"], n=n, nm1=n - 1,
    )
    last_err = None
    for attempt in range(1 + retries):
        txt, dt = vlm_call(client, model, prompt, sampled_frames, max_side, 800)
        try:
            parsed = parse_json_response(txt)
            windows_sample = sorted([int(a), int(b)] for a, b in parsed["windows"])
        except Exception as e:
            last_err = e
            print(f"  attempt {attempt+1} parse failed: {e}")
            print(f"  raw response: {txt[:400]}")
            continue

        # Filter out-of-range windows
        windows_sample = [[a, b] for a, b in windows_sample
                          if 0 <= a <= b <= n - 1]
        if not windows_sample:
            print(f"  attempt {attempt+1}: no valid windows after filtering")
            continue

        # If VLM flooded with too many windows, subsample keeping spread
        if len(windows_sample) > max_windows:
            print(f"  VLM returned {len(windows_sample)} windows, capping to {max_windows}")
            step = len(windows_sample) / max_windows
            windows_sample = [windows_sample[int(i * step)]
                              for i in range(max_windows)]

        windows_frame = []
        for a, b in windows_sample:
            fa = int(sample_idx[a])
            fb = int(sample_idx[b]) + stride if b < n - 1 else T
            windows_frame.append([fa, min(fb, T)])
        return windows_frame, dt

    raise ValueError(f"critical labeling failed after {1+retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Label one episode with milestone + critical-step via VLM."
    )
    ap.add_argument("--task", required=True, choices=list(TASKS.keys()))
    ap.add_argument("--session", default=None,
                    help="session name (required for task_a, e.g. 2026-05-09-v2)")
    ap.add_argument("--episode", required=True, type=int)
    ap.add_argument("--sample-fps", type=float, default=4.0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--max-side", type=int, default=384)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: ./labels/<task>/)")
    args = ap.parse_args()

    cfg = TASKS[args.task]
    if cfg["has_sessions"] and not args.session:
        raise SystemExit(f"--session is required for task '{args.task}'")

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("set DASHSCOPE_API_KEY first")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    # Load and subsample video
    print(f"loading episode {args.episode} ...")
    frames = load_video(cfg, args.session, args.episode)
    T = frames.shape[0]
    sampled, sample_idx, stride = subsample(frames, cfg["src_fps"], args.sample_fps)
    print(f"  T={T} frames @ {cfg['src_fps']}fps, subsampled to {len(sampled)} @ {args.sample_fps}fps")

    # Step 1: milestone
    print("labeling milestones ...")
    cuts_frame, dt_ms = label_milestones(
        client, args.model, cfg, sampled, sample_idx, args.max_side,
    )
    print(f"  cuts={cuts_frame}  ({dt_ms:.1f}s)")

    # Step 2: critical-step
    print("labeling critical steps ...")
    crit_windows, dt_cr = label_critical(
        client, args.model, cfg, sampled, sample_idx, stride, T, args.max_side,
    )
    print(f"  windows={crit_windows}  ({dt_cr:.1f}s)")

    # Build per-frame arrays
    stage = np.zeros(T, dtype=int)
    for i, c in enumerate(cuts_frame):
        stage[c:] = i + 1

    is_milestone = np.zeros(T, dtype=bool)
    for c in cuts_frame:
        is_milestone[c] = True

    is_critical = np.zeros(T, dtype=bool)
    for a, b in crit_windows:
        is_critical[a:b] = True

    # Write JSON
    out_dir = args.out_dir or Path("labels") / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ep{args.episode:03d}.json"

    result = {
        "episode_id": args.episode,
        "session": args.session,
        "total_frames": T,
        "src_fps": cfg["src_fps"],
        "sample_fps": args.sample_fps,
        "model": args.model,
        "task": cfg["task_text"],
        "stages": cfg["stages"],
        "milestone_cuts": cuts_frame,
        "critical_windows": crit_windows,
        "vlm_time_milestone_s": round(dt_ms, 2),
        "vlm_time_critical_s": round(dt_cr, 2),
        "per_frame": {
            "stage": stage.tolist(),
            "is_milestone": is_milestone.tolist(),
            "is_critical": is_critical.tolist(),
        },
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    n_crit = int(is_critical.sum())
    print(f"\nwrote {out_path}")
    print(f"  stages: {len(cfg['stages'])}  milestone frames: {len(cuts_frame)}")
    print(f"  critical windows: {len(crit_windows)}  critical frames: {n_crit}/{T} ({100*n_crit/T:.1f}%)")


if __name__ == "__main__":
    main()
