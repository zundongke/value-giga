"""Pure-numpy classification metrics + gripper baseline. No sklearn dependency."""
import numpy as np


def roc_auc(scores, labels):
    """Rank-based ROC-AUC (Mann-Whitney U), tie-averaged. NaN if one class absent."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0  # 1-based, tie-averaged
        i = j + 1
    auc = (ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(scores, labels):
    """Area under precision-recall curve (AP). NaN if no positives."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(~labels_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def segment_detection_rate(prob, windows, thresh=0.5):
    """Fraction of GT critical windows [cs,ce) in which prob exceeds thresh at
    least once. NaN if no windows."""
    prob = np.asarray(prob, dtype=np.float64)
    if len(windows) == 0:
        return float("nan")
    hits = 0
    for cs, ce in windows:
        seg = prob[max(0, cs):min(len(prob), ce)]
        if seg.size and seg.max() >= thresh:
            hits += 1
    return hits / len(windows)


def gripper_velocity_score(state_traj, gripper_channels=(6, 13)):
    """Heuristic per-frame criticality score = max abs first-difference over the
    gripper channels. Threshold-free (used directly for AUC). Returns [T]."""
    state_traj = np.asarray(state_traj, dtype=np.float64)
    g = state_traj[:, list(gripper_channels)]            # [T, n_grip]
    dv = np.abs(np.diff(g, axis=0, prepend=g[:1]))       # [T, n_grip], dv[0]=0
    return dv.max(axis=1)                                # [T]
