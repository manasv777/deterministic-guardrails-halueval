"""Classification metrics, cluster bootstrap CIs, and McNemar's test. Positive class = "hallucinated"."""

from __future__ import annotations

import numpy as np
from scipy import stats


def _counts(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """[tp, fp, tn, fn] with hallucinated as the positive class (inputs are boolean 'is hallucinated')."""
    return np.array([
        np.sum(gold & pred), np.sum(~gold & pred), np.sum(~gold & ~pred), np.sum(gold & ~pred),
    ])


def _from_counts(c: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized metrics from count arrays shaped [..., 4]."""
    tp, fp, tn, fn = (c[..., i].astype(float) for i in range(4))
    n = tp + fp + tn + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        frr = np.where(fp + tn > 0, fp / (fp + tn), 0.0)
    return {"accuracy": (tp + tn) / n, "precision": precision, "recall": recall, "f1": f1,
            "false_refusal_rate": frr}


def scores(gold: np.ndarray, pred: np.ndarray) -> dict:
    c = _counts(gold, pred)
    out = {k: float(v) for k, v in _from_counts(c).items()}
    out["confusion"] = {"tp": int(c[0]), "fp": int(c[1]), "tn": int(c[2]), "fn": int(c[3])}
    return out


def cluster_bootstrap_ci(gold: np.ndarray, preds: dict[str, np.ndarray], clusters: np.ndarray,
                         n_resamples: int, seed: int, alpha: float = 0.05) -> dict:
    """Percentile CIs from a bootstrap that resamples *questions* (clusters), not items.

    Each HaluEval question contributes a correct and a hallucinated item that share a
    knowledge passage, so items are not independent; resampling whole questions keeps
    that pairing intact. The same resampled question sets are used for every config,
    which also gives paired CIs for differences between configs.
    """
    uniq, inv = np.unique(clusters, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_resamples, len(uniq)))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    per_cfg: dict[str, dict[str, np.ndarray]] = {}
    for key, pred in preds.items():
        per_cluster = np.zeros((len(uniq), 4))
        for j, (g, p) in enumerate(zip(gold, pred)):
            per_cluster[inv[j]] += _counts(np.array([g]), np.array([p]))
        per_cfg[key] = _from_counts(per_cluster[draws].sum(axis=1))

    out: dict = {"method": "cluster (question-level) percentile bootstrap", "n_resamples": n_resamples,
                 "seed": seed, "configs": {}, "differences": {}}
    for key, m in per_cfg.items():
        out["configs"][key] = {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))]
                               for k, v in m.items()}
    for a in preds:
        for b in preds:
            if a == b:
                continue
            for metric in ("accuracy", "f1"):
                d = per_cfg[a][metric] - per_cfg[b][metric]
                out["differences"][f"{a}-{b}:{metric}"] = [float(np.percentile(d, lo)), float(np.percentile(d, hi))]
    return out


def mcnemar(gold: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar's test on paired correctness. b = A right & B wrong, c = A wrong & B right."""
    ok_a, ok_b = pred_a == gold, pred_b == gold
    b = int(np.sum(ok_a & ~ok_b))
    c = int(np.sum(~ok_a & ok_b))
    n = b + c
    exact_p = float(stats.binomtest(b, n, 0.5).pvalue) if n else 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n if n else 0.0
    return {
        "a_right_b_wrong": b,
        "a_wrong_b_right": c,
        "n_discordant": n,
        "exact_p": exact_p,
        "chi2_cc": float(chi2),
        "chi2_cc_p": float(stats.chi2.sf(chi2, 1)) if n else 1.0,
        "test": "exact binomial McNemar (two-sided); chi-square with continuity correction also reported",
    }
