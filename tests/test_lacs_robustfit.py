"""Regression for the LACS IRLS solver weighting.

_robustfit must perform weighted least squares weighted by the bisquare weights
w, i.e. its returned coefficients must satisfy the w-weighted normal equations
    X^T W X beta = X^T W y,  W = diag(w).
Scaling the lstsq design/RHS by w instead of sqrt(w) minimizes sum(w^2 r^2)
and returns the w^2-weighted solution, which violates this property.
"""

import numpy as np

from trizod.lacs.lacs import _robustfit

# Deterministic dataset (linear + heavy-tailed noise) chosen so that many points
# carry intermediate bisquare weights, where w vs w^2 diverges strongly.
X_DATA = np.array(
    [
        -1.9481,
        -1.7816,
        -1.4493,
        -1.1691,
        -1.1566,
        -1.0759,
        -1.0289,
        -0.7607,
        -0.7495,
        -0.5415,
        -0.4348,
        -0.3797,
        -0.168,
        -0.1459,
        0.3484,
        0.5317,
        0.7004,
        0.7788,
        1.1594,
        1.1607,
        1.2058,
        1.3595,
        2.0978,
        3.0601,
        3.0749,
        3.279,
        3.3078,
        3.5501,
        3.6546,
        3.7033,
    ]
)
Y_DATA = np.array(
    [
        1.1536,
        -0.123,
        0.0764,
        -3.2262,
        -0.8256,
        3.6162,
        0.2637,
        1.8704,
        0.3339,
        0.9236,
        1.4256,
        3.5366,
        1.3108,
        -4.2362,
        1.3208,
        4.0322,
        2.7837,
        1.6644,
        2.5819,
        2.6773,
        2.5274,
        2.3252,
        3.7062,
        5.0906,
        4.5135,
        2.7373,
        4.6041,
        4.5849,
        3.6859,
        4.5873,
    ]
)


def test_robustfit_solves_w_weighted_least_squares():
    intercept, slope, weights = _robustfit(X_DATA, Y_DATA)

    X = np.column_stack([np.ones(len(X_DATA)), X_DATA])
    W = np.diag(weights)
    beta_w = np.linalg.solve(X.T @ W @ X, X.T @ W @ Y_DATA)

    # Returned coefficients must be the w-weighted (not w^2-weighted) WLS fit.
    assert np.allclose([intercept, slope], beta_w, atol=1e-5)
