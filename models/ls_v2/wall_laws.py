# SPDX-License-Identifier: Apache-2.0
"""Smooth near-wall approximations for a stationary, hydraulically smooth wall.

The composite velocity law is a Reichardt-type approximation calibrated to
recover the Launder-Spalding logarithmic intercept for large y-plus. It is not
the exact implicit Spalding law and is not a low-Re turbulence closure.
"""

from sympy import exp, log


C_MU = 0.09
KAPPA = 0.4187
E = 9.793


def composite_u_plus(y_plus):
    """Return U+; approaches y+ near zero and log(E*y+)/kappa at large y+."""
    intercept = log(E / KAPPA) / KAPPA
    return (
        log(1 + KAPPA * y_plus) / KAPPA
        + intercept * (
            1 - exp(-y_plus / 11.0)
            - (y_plus / 11.0) * exp(-y_plus / 3.0)
        )
    )


def blended_epsilon(k, nu, distance, y_plus):
    """OpenFOAM epsilonWallFunction EXPONENTIAL blending of the two limits."""
    epsilon_viscous = 2 * nu * k / distance**2
    epsilon_log = C_MU**0.75 * k**1.5 / (KAPPA * distance)
    gamma = 0.001 * y_plus**4 / (1 + y_plus)
    return (
        epsilon_viscous * exp(-gamma)
        + epsilon_log * exp(-1 / (gamma + 1.0e-20))
    )
