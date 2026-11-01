"""Unit tests for the critical-frame classifier (pure functions + model)."""
import numpy as np
import torch

from criticality_metrics import roc_auc, average_precision

def test_roc_auc_perfect():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert abs(roc_auc(scores, labels) - 1.0) < 1e-9

def test_roc_auc_reversed():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 1, 1])
    assert abs(roc_auc(scores, labels) - 0.0) < 1e-9

def test_roc_auc_ties_chance():
    # all equal scores -> 0.5 regardless of labels
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 1, 0, 1])
    assert abs(roc_auc(scores, labels) - 0.5) < 1e-9

def test_roc_auc_degenerate_nan():
    assert np.isnan(roc_auc(np.array([0.1, 0.2]), np.array([0, 0])))

def test_average_precision_perfect():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert abs(average_precision(scores, labels) - 1.0) < 1e-9

def test_average_precision_known():
    # ranking: pos, neg, pos  -> AP = (1.0 + 2/3) / 2
    scores = np.array([0.9, 0.8, 0.7])
    labels = np.array([1, 0, 1])
    assert abs(average_precision(scores, labels) - (1.0 + 2.0/3.0) / 2.0) < 1e-9

from criticality_metrics import segment_detection_rate, gripper_velocity_score

def test_segment_detection_rate():
    # two GT windows: [1,3) and [5,7). prob fires inside the first only.
    prob = np.array([0.0, 0.9, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0])
    windows = [(1, 3), (5, 7)]
    assert segment_detection_rate(prob, windows, thresh=0.5) == 0.5

def test_segment_detection_rate_no_windows_nan():
    assert np.isnan(segment_detection_rate(np.zeros(5), [], thresh=0.5))

def test_gripper_velocity_score_shape_and_signal():
    # state_traj [T,14]; only gripper ch 6 moves at t=2
    T = 5
    traj = np.zeros((T, 14))
    traj[2, 6] = 1.0  # spike -> large |Δ| at t=2 and t=3
    score = gripper_velocity_score(traj, gripper_channels=(6, 13))
    assert score.shape == (T,)
    assert score[2] > score[0]

from criticality_data import normalize_state, make_windows

def test_normalize_state_range():
    s = np.array([[0.0, 5.0], [10.0, 0.0]])
    smin = np.array([0.0, 0.0]); smax = np.array([10.0, 10.0])
    out = normalize_state(s, smin, smax)
    # 0 -> -1, 10 -> +1, 5 -> 0
    assert np.allclose(out, np.array([[-1.0, 0.0], [1.0, -1.0]]))

def test_normalize_state_clamps():
    s = np.array([[-5.0, 15.0]])
    out = normalize_state(s, np.array([0.0, 0.0]), np.array([10.0, 10.0]))
    assert np.allclose(out, np.array([[-1.0, 1.0]]))

def test_make_windows_causal_shapes_and_padding():
    traj = np.arange(5 * 2).reshape(5, 2).astype(float)  # T=5, C=2
    crit = np.array([0, 0, 1, 1, 0]).astype(bool)
    X, y = make_windows(traj, crit, W=3, causal=True)
    assert X.shape == (5, 2, 3)   # [T, C, W]
    assert y.tolist() == [0, 0, 1, 1, 0]
    # frame 0 window left-padded with frame 0: cols all == traj[0]
    assert np.allclose(X[0, :, 0], traj[0]) and np.allclose(X[0, :, 2], traj[0])
    # frame 2 causal window = frames [0,1,2], last col == traj[2]
    assert np.allclose(X[2, :, 2], traj[2])
    assert np.allclose(X[2, :, 0], traj[0])

def test_make_windows_symmetric():
    traj = np.arange(5 * 1).reshape(5, 1).astype(float)
    crit = np.zeros(5, dtype=bool)
    X, y = make_windows(traj, crit, W=4, causal=False)
    assert X.shape == (5, 1, 4)

from criticality_model import CriticalityTCN, StreamingCriticalityDetector

def test_tcn_forward_shape():
    m = CriticalityTCN(in_ch=14, hidden=32, levels=3)
    x = torch.randn(4, 14, 16)
    out = m(x)
    assert out.shape == (4,)

def test_tcn_causal_last_step_independent_of_padding():
    # zeroing the earliest window frames must not blow up; output finite
    m = CriticalityTCN(in_ch=2, hidden=8, levels=2).eval()
    x = torch.randn(1, 2, 8)
    with torch.no_grad():
        a = m(x)
        x2 = x.clone(); x2[:, :, 0] = 999.0   # earliest frame perturbed
        b = m(x2)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()

def test_streaming_detector_left_pads_and_returns_prob():
    m = CriticalityTCN(in_ch=2, hidden=8, levels=2).eval()
    smin = np.zeros(2); smax = np.ones(2)
    det = StreamingCriticalityDetector(m, smin, smax, W=4, device="cpu")
    det.reset()
    p = det.predict(np.array([0.5, 0.5]))   # first frame, buffer left-padded
    assert 0.0 <= p <= 1.0
    for _ in range(5):
        p = det.predict(np.array([0.5, 0.5]))
    assert 0.0 <= p <= 1.0
