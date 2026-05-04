"""PPI++ estimators: standard (unweighted) and importance-weighted variants.

These are the core statistical estimators of Adaptive AutoEval.
Both functions accept either 1-D (single model) or 2-D (M models)
arrays, enabling simultaneous evaluation of multiple models.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def ppi_unweighted(
    phi_lab: np.ndarray,
    syn_lab: np.ndarray,
    syn_unl: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standard PPI++ estimator (no importance weighting).

    Combines true labels on a small labeled set with synthetic
    annotator predictions on a large unlabeled set to reduce
    variance while maintaining unbiasedness -- *under the
    assumption that labeled and unlabeled data are i.i.d.*

    When this assumption is violated (covariate shift), use
    :func:`ppi_weighted` instead.

    Args:
        phi_lab: True loss values on labeled data.
            Shape ``(n,)`` for a single model or ``(n, M)`` for M models.
        syn_lab: Synthetic annotator predictions on labeled data.
            Same shape as ``phi_lab``.
        syn_unl: Synthetic annotator predictions on unlabeled data.
            Shape ``(N,)`` or ``(N, M)``.

    Returns:
        Tuple of:
          - mu_hat: Point estimate of target performance.
            Shape ``()`` or ``(M,)``.
          - var_hat: Estimated variance of the estimator.
            Same shape as ``mu_hat``.

    Example::

        mu, var = ppi_unweighted(phi_lab, syn_lab, syn_unl)
        z = 1.645  # 90% CI
        ci = (mu - z * np.sqrt(var), mu + z * np.sqrt(var))
    """
    n = phi_lab.shape[0]
    N = syn_unl.shape[0]

    cov_num = np.mean(
        (phi_lab - phi_lab.mean(0)) * (syn_lab - syn_lab.mean(0)), axis=0
    )
    var_full = (n / N) * syn_unl.var(0) + syn_lab.var(0)
    lambd = np.clip(
        np.where(var_full > 1e-12, cov_num / var_full, 1.0), 0.0, 1.0
    )
    mu_hat = lambd * syn_unl.mean(0) + (phi_lab - lambd * syn_lab).mean(0)
    resid = phi_lab - lambd * syn_lab
    var_hat = resid.var(0) / n + lambd ** 2 * syn_unl.var(0) * (n / N) / n

    return mu_hat, var_hat


def ppi_weighted(
    phi_lab: np.ndarray,
    syn_lab: np.ndarray,
    syn_unl: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptive AutoEval: importance-weighted PPI++ estimator.

    Extends PPI++ to handle unknown covariate shift by reweighting
    the labeled data to match the target distribution.  Importance
    weights are typically estimated with
    :func:`adaptive_autoeval.weights.learn_importance_weights`.

    The optimal mixing parameter lambda is chosen to minimize
    asymptotic variance using weighted covariances, accounting for
    the variance inflation from non-uniform weights.

    Args:
        phi_lab: True loss values on labeled data.
            Shape ``(n,)`` for a single model or ``(n, M)`` for M models.
        syn_lab: Synthetic annotator predictions on labeled data.
            Same shape as ``phi_lab``.
        syn_unl: Synthetic annotator predictions on unlabeled data.
            Shape ``(N,)`` or ``(N, M)``.
        weights: Importance weights for labeled points.
            Shape ``(n,)``. Should have mean 1.0 (will be
            normalized internally if not).

    Returns:
        Tuple of:
          - mu_hat: Importance-weighted point estimate of target performance.
            Shape ``()`` or ``(M,)``.
          - var_hat: Estimated variance of the weighted estimator.
            Same shape as ``mu_hat``. Accounts for variance inflation
            from non-uniform weighting via squared weights.

    Example::

        weights = learn_importance_weights(feat_lab, feat_unl)
        mu, var = ppi_weighted(phi_lab, syn_lab, syn_unl, weights)

    Note:
        Outperforms oracle weights (true density ratio) in finite
        samples due to implicit regularization from logistic
        regression. See Section 6.1 of the paper.
    """
    n = phi_lab.shape[0]
    N = syn_unl.shape[0]

    # Normalize weights to mean 1
    w = weights / weights.mean()

    # Apply weights to labeled losses
    phi_w = w[:, None] * phi_lab if phi_lab.ndim > 1 else w * phi_lab

    # Weighted covariance for optimal lambda
    cov_num = np.mean(
        (phi_w - phi_w.mean(0)) * (syn_lab - syn_lab.mean(0)), axis=0
    )
    var_full = (n / N) * syn_unl.var(0) + syn_lab.var(0)
    lambd = np.clip(
        np.where(var_full > 1e-12, cov_num / var_full, 1.0), 0.0, 1.0
    )

    # Weighted PPI++ estimate
    mu_hat = lambd * syn_unl.mean(0) + (phi_w - lambd * syn_lab).mean(0)

    # Variance: squared weights account for inflation from non-uniform weighting
    resid = (w[:, None] if phi_lab.ndim > 1 else w) * (
        phi_lab - lambd * syn_lab
    )
    var_hat = resid.var(0) / n + lambd ** 2 * syn_unl.var(0) * (n / N) / n

    return mu_hat, var_hat