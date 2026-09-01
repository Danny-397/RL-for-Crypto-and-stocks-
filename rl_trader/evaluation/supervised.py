"""Supervised baselines: would a simpler learner have found the signal?

The existing baselines are all rule-based or trivial — buy-and-hold, a moving
average crossover, random, flat. Beating those says the agent is not *useless*.
It does not answer the objection a skeptical reader actually raises:

    "Maybe there is exploitable structure and PPO is just bad at finding it."

The surrogate-data test argues against that indirectly, by showing the agent
does no better on real prices than on the same returns shuffled. This attacks it
directly: fit an ordinary supervised model on the **same 28 features**, over the
**same training split**, and trade its predictions through the **same
environment** with the same costs. If a linear model cannot extract an edge
either, two unrelated method classes agree, and "there is nothing here to find"
becomes much harder to argue with. If it *beats* the agent, that is a finding
too — a more interesting one — and it belongs in the write-up either way.

Both models are implemented here rather than imported. Partly to avoid adding a
dependency to a project whose point is that it built its own algorithm, and
partly because a closed-form ridge solution and a fifteen-line logistic fit are
easier for a reader to audit than a call into a library.

No leakage
----------
Weights are fit on the training split only. The features arriving here were
already scaled by a scaler fit on that same split (see ``data_loader``), and the
prediction at bar *t* uses only features observable at *t* to forecast the move
into *t+1*. That is the same information set the RL agent has.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# Ridge penalty. Large enough that 28 correlated technical features cannot be
# inverted into an overfit solution on a few hundred training bars, small enough
# that a genuine linear relationship would survive. Not tuned against the test
# split -- tuning it there is the exact error this file exists to check for.
RIDGE_LAMBDA = 10.0

# Logistic fit. Full-batch gradient descent: the design is tiny and convergence
# is not interesting enough to warrant anything cleverer.
LOGIT_STEPS = 400
LOGIT_LR = 0.5


def _design(features: np.ndarray, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Features at *t* against the log return from *t* to *t+1*.

    The last bar is dropped because its forward return is not observable, which
    is the whole point -- a model that keeps it is reading the future.
    """
    x = np.asarray(features, dtype=np.float64)[:-1]
    p = np.asarray(prices, dtype=np.float64)
    y = np.log(np.maximum(p[1:], 1e-12) / np.maximum(p[:-1], 1e-12))
    n = min(len(x), len(y))
    return x[:n], y[:n]


def _with_bias(x: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((len(x), 1))])


def fit_ridge(features: np.ndarray, prices: np.ndarray,
              lam: float = RIDGE_LAMBDA) -> Tuple[np.ndarray, float]:
    """Closed-form ridge regression of next-bar log return on the features.

    Returns the weight vector and the training-set standard deviation of the
    target, which is used to scale a prediction into a position: a forecast of
    one standard deviation asks for full exposure.
    """
    x, y = _design(features, prices)
    if len(x) < 2:
        return np.zeros(features.shape[1] + 1), 1.0
    xb = _with_bias(x)
    # The bias column is left unpenalised; shrinking the intercept toward zero
    # would bias the forecast rather than regularise it.
    penalty = lam * np.eye(xb.shape[1])
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(xb.T @ xb + penalty, xb.T @ y)
    scale = float(y.std()) or 1.0
    return weights, scale


def fit_logistic(features: np.ndarray, prices: np.ndarray,
                 steps: int = LOGIT_STEPS, lr: float = LOGIT_LR) -> np.ndarray:
    """Logistic regression on the *direction* of the next bar.

    Trained by full-batch gradient descent on the log-loss. Direction is the
    easier target -- a model that cannot beat a coin flip on the sign is not
    going to be rescued by predicting the magnitude.
    """
    x, y = _design(features, prices)
    if len(x) < 2:
        return np.zeros(features.shape[1] + 1)
    xb = _with_bias(x)
    labels = (y > 0).astype(np.float64)
    weights = np.zeros(xb.shape[1])
    n = len(xb)
    for _ in range(steps):
        z = np.clip(xb @ weights, -30, 30)
        probs = 1.0 / (1.0 + np.exp(-z))
        grad = xb.T @ (probs - labels) / n
        grad[:-1] += (RIDGE_LAMBDA / n) * weights[:-1]   # L2, intercept exempt
        weights -= lr * grad
    return weights


def ridge_action(weights: np.ndarray, scale: float):
    """Trade the forecast: a one-sigma prediction asks for full exposure."""
    def action(env) -> float:
        row = np.asarray(env.data.features[env.t], dtype=np.float64)
        pred = float(np.append(row, 1.0) @ weights)
        return float(np.clip(pred / max(scale, 1e-12), -1.0, 1.0))
    return action


def logistic_action(weights: np.ndarray):
    """Map the probability of an up-move onto a position in [-1, 1]."""
    def action(env) -> float:
        row = np.asarray(env.data.features[env.t], dtype=np.float64)
        z = float(np.clip(np.append(row, 1.0) @ weights, -30, 30))
        prob = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(2.0 * prob - 1.0, -1.0, 1.0))
    return action


def supervised_policies(train_features: np.ndarray,
                        train_prices: np.ndarray) -> Dict[str, object]:
    """Fit both models on the training split and return their action functions.

    Returns an empty mapping when there is too little training data to fit
    anything, rather than returning a degenerate model that would quietly
    report a flat 0% and look like a result.
    """
    if train_features is None or len(train_prices) < 60:
        return {}
    ridge_w, scale = fit_ridge(train_features, train_prices)
    logit_w = fit_logistic(train_features, train_prices)
    return {
        "ridge_forecast": ridge_action(ridge_w, scale),
        "logistic_direction": logistic_action(logit_w),
    }


def train_accuracy(train_features: np.ndarray, train_prices: np.ndarray,
                   weights: Optional[np.ndarray] = None) -> float:
    """In-sample directional accuracy of the logistic fit.

    Reported alongside the returns because it separates two very different
    failures: a model that cannot fit its own training data at all, and one that
    fits it well and still earns nothing out of sample. Only the second is
    evidence about the market.
    """
    x, y = _design(train_features, train_prices)
    if len(x) < 2:
        return float("nan")
    if weights is None:
        weights = fit_logistic(train_features, train_prices)
    z = np.clip(_with_bias(x) @ weights, -30, 30)
    predicted_up = (1.0 / (1.0 + np.exp(-z))) > 0.5
    return float((predicted_up == (y > 0)).mean())
