"""Importance weight estimation via discriminative classification.

Implements the density ratio estimation approach from Section 4.1
of the paper: train a logistic regression classifier to distinguish
labeled (source) from unlabeled (target) data, then compute the
importance weight as the odds ratio of the predicted probabilities.

The key insight (following Sugiyama et al., 2007; Bickel et al., 2009)
is that the density ratio can be estimated without separately estimating
each density::

    w(x) = p_target(x) / p_source(x)
          = P(unlabeled | x) / P(labeled | x)   [by Bayes' theorem]
          ≈ (1 - g(x)) / g(x)                   [g = P(labeled | x)]

Platt scaling calibration and probability clipping stabilize the
estimates, and importantly, the implicit regularization from logistic
regression produces smoother weights with better finite-sample
coverage than raw oracle density ratios (see paper Section 6.1).
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def learn_importance_weights(
    feat_lab: np.ndarray,
    feat_unl: np.ndarray,
    clip_lo: float = 0.01,
    clip_hi: float = 0.99,
    max_unlabeled_ratio: int = 5,
    random_state: int = 0,
) -> np.ndarray:
    """Estimate importance weights w(x) = p_target(x) / p_source(x).

    Trains a logistic regression classifier with Platt scaling to
    distinguish labeled (source) from unlabeled (target) data.
    The estimated weight for each labeled point is the odds ratio
    of the classifier's predicted probability::

        w(x_i) = P(unlabeled | x_i) / P(labeled | x_i)
               = (1 - g(x_i)) / g(x_i)

    where g(x) = P(labeled | x) is the classifier output.

    The unlabeled set is subsampled to ``max_unlabeled_ratio * n``
    to balance the binary classification problem and control
    training time.

    Args:
        feat_lab: Feature matrix for labeled (source) data.
            Shape ``(n, d)``.
        feat_unl: Feature matrix for unlabeled (target) data.
            Shape ``(N, d)``.  Typically ``N >> n``.
        clip_lo: Lower probability clipping bound.
            Clips ``P(unlabeled | x)`` to ``[clip_lo, clip_hi]``
            before computing the odds ratio, preventing extreme
            weights from high-confidence classifier predictions.
            Default ``0.01``.
        clip_hi: Upper probability clipping bound. Default ``0.99``.
        max_unlabeled_ratio: Subsample unlabeled set to at most
            ``max_unlabeled_ratio * n`` points for class balance.
            Default ``5``.
        random_state: Random seed for reproducibility. Default ``0``.

    Returns:
        weights: Normalized importance weights for labeled points.
            Shape ``(n,)``.  Normalized so that ``weights.mean() == 1.0``.

    Example::

        from adaptive_autoeval import learn_importance_weights, ppi_weighted

        weights = learn_importance_weights(feat_lab, feat_unl)
        mu, var = ppi_weighted(phi_lab, syn_lab, syn_unl, weights)

    Note:
        The implicit regularization from logistic regression with
        Platt scaling produces smoother weights than the raw oracle
        density ratio, yielding better finite-sample coverage.
        See paper Section 6.1 and the discussion in Section 7.
    """
    n = len(feat_lab)
    N = len(feat_unl)

    # Subsample unlabeled set to balance classes
    if N > max_unlabeled_ratio * n:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(N, size=max_unlabeled_ratio * n, replace=False)
        feat_unl = feat_unl[idx]

    # Construct binary classification dataset
    # labeled = class 0, unlabeled = class 1
    X = np.vstack([feat_lab, feat_unl])
    y = np.array([0] * n + [1] * len(feat_unl))

    # Standardize features
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    # Logistic regression with Platt scaling calibration
    clf = CalibratedClassifierCV(
        LogisticRegression(max_iter=300, C=1.0, random_state=random_state),
        cv=3,
        method="sigmoid",
    )
    clf.fit(Xs, y)

    # Estimate density ratio: P(unlabeled|x) / P(labeled|x)
    p_unl = np.clip(
        clf.predict_proba(sc.transform(feat_lab))[:, 1],
        clip_lo,
        clip_hi,
    )
    weights = p_unl / (1.0 - p_unl)

    # Normalize to mean 1 for numerical stability
    return weights / weights.mean()