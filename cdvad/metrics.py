"""Frame-level metrics with an explicit granularity decomposition.

The field reports *pooled* frame-level AUC/AP: every anomalous frame is ranked
against every normal frame in the whole test split, mixing within-video and
cross-video comparisons.  Song & Lee (arXiv:2608.21854) and Abdulaziz &
Bondarev (arXiv:2608.11985) show this is dominated by cross-video ranking and
does not resolve the margins the field reports.

    Micro-AUROC = w * Within-AUROC + (1 - w) * Cross-AUROC

with ``w`` = fraction of anomalous-normal frame pairs that come from the same
video.  We reproduce that identity here and add a **capacity control** (a
video-constant scorer) that upper-bounds what the metric can credit without any
temporal localisation.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _fpr_at_tpr(y: np.ndarray, s: np.ndarray, tpr_target: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve_safe(y, s)
    ok = np.where(tpr >= tpr_target)[0]
    return float(fpr[ok[0]]) if len(ok) else 1.0


def roc_curve_safe(y: np.ndarray, s: np.ndarray):
    """roc_curve with an explicit, dependency-stable signature."""
    from sklearn.metrics import roc_curve
    return roc_curve(y, s)


def per_video(y: np.ndarray, s: np.ndarray) -> float:
    """AUC restricted to pairs from a single video; 0.5 if degenerate."""
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def decompose(y_list: Sequence[np.ndarray],
              s_list: Sequence[np.ndarray]) -> Dict[str, float]:
    """Split pooled AUC into within-video and cross-video parts.

    Returns
    -------
    micro   : pooled AUC over the concatenation (what papers report)
    within  : pair-weighted mean of per-video AUCs (true localisation)
    cross   : residual AUC implied by the micro/within decomposition
    macro   : unweighted mean of per-video AUCs
    w       : fraction of anomalous-normal pairs that are within-video
    """
    keep = [(y, s) for y, s in zip(y_list, s_list)
            if y.sum() > 0 and y.sum() < len(y)]
    # Pooled AUC must use EVERY video, including the all-normal ones -- they
    # supply the large majority of negative frames.  Only the per-video
    # aggregation needs the non-degenerate filter.
    Y = np.concatenate([y for y, _ in zip(y_list, s_list)])
    S = np.concatenate([s for _, s in zip(y_list, s_list)])
    if not keep or Y.sum() == 0 or Y.sum() == len(Y):
        return dict(micro=float("nan"), within=float("nan"),
                    cross=float("nan"), macro=float("nan"), w=float("nan"))

    npos, nneg = float(Y.sum()), float((1 - Y).sum())
    total_pairs = npos * nneg

    within_pairs, within_conc, macros = 0.0, 0.0, []
    for y, s in keep:
        p, n = float(y.sum()), float((1 - y).sum())
        a = roc_auc_score(y, s)
        macros.append(a)
        within_pairs += p * n
        within_conc += a * p * n

    micro = float(roc_auc_score(Y, S))
    within = within_conc / within_pairs if within_pairs else float("nan")
    cross_pairs = total_pairs - within_pairs
    cross = ((micro * total_pairs - within_conc) / cross_pairs
             if cross_pairs > 0 else float("nan"))
    return dict(micro=micro, within=float(within), cross=float(cross),
                macro=float(np.mean(macros)), w=within_pairs / total_pairs)


def capacity_control(y_list: Sequence[np.ndarray],
                     s_list: Sequence[np.ndarray]) -> Dict[str, float]:
    """Replace every frame score by its video mean.

    This destroys all temporal information.  Whatever pooled AUC survives is
    the credit the metric gives purely for ranking *videos*, and is therefore
    an upper bound on what a scene-level (domain-level) shortcut can buy.
    """
    const = [np.full_like(s, float(np.mean(s))) for s in s_list]
    return decompose(y_list, const)


def video_level_auc(y_list: Sequence[np.ndarray],
                    s_list: Sequence[np.ndarray]) -> float:
    """AUC of the video-mean score against video-level labels (1 = has anomaly)."""
    sm, ym = [], []
    for y, s in zip(y_list, s_list):
        sm.append(float(np.mean(s)))
        ym.append(1.0 if y.sum() > 0 else 0.0)
    if len(set(ym)) < 2:
        return float("nan")
    return float(roc_auc_score(np.array(ym), np.array(sm)))


def full_report(y_list: Sequence[np.ndarray],
                s_list: Sequence[np.ndarray],
                prefix: str = "") -> Dict[str, float]:
    """Pooled AUC/AP + decomposition + capacity control, in one call."""
    Y = np.concatenate([y for y, _ in zip(y_list, s_list)])
    S = np.concatenate([s for _, s in zip(y_list, s_list)]).astype(np.float64)
    d = decompose(y_list, s_list)
    cap = capacity_control(y_list, s_list)
    out = {
        "auc": float(roc_auc_score(Y, S)),
        "ap": float(average_precision_score(Y, S)),
        "fpr95": _fpr_at_tpr(Y.astype(int), S),
    }
    out.update({f"within_{k}": v for k, v in d.items() if k != "micro"})
    out["micro_recheck"] = d["micro"]
    out.update({f"const_{k}": v for k, v in cap.items()})
    out["video_auc"] = video_level_auc(y_list, s_list)
    if prefix:
        out = {f"{prefix}{k}": v for k, v in out.items()}
    return out


def domain_auc(y_by_video: np.ndarray,
               s_by_video: np.ndarray) -> float:
    """AUC of a video-level score for discriminating two domains.

    This is the quantity that makes a cross-domain number untrustworthy: if a
    scorer can tell domains apart on *normal* footage alone, then any pooled
    AUC measured across a domain boundary partly rewards that shortcut.
    """
    y = np.asarray(y_by_video).astype(int)
    s = np.asarray(s_by_video, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def within_normal_pair_auc(scores_a: List[np.ndarray],
                           scores_b: List[np.ndarray]) -> float:
    """AUC separating *normal* videos of two domains using video-mean scores.

    The cleanest shortcut probe: no anomalies are involved, so any AUC above
    0.5 is pure domain/recording leakage through the scoring function.
    """
    sa = np.array([float(np.mean(s)) for s in scores_a])
    sb = np.array([float(np.mean(s)) for s in scores_b])
    y = np.concatenate([np.zeros(len(sa)), np.ones(len(sb))])
    s = np.concatenate([sa, sb])
    if len(sa) == 0 or len(sb) == 0:
        return float("nan")
    return float(roc_auc_score(y, s))