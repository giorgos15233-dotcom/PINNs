# SPDX-License-Identifier: Apache-2.0
"""Steady 2-D incompressible standard RANS k-epsilon (no wall functions).

This file implements the complete symmetric Boussinesq stress divergence.
Pressure p is the mean static pressure, not p + 2*k/3.
Reference scales nondimensionalize the PDE residuals; they are fixed, not
adaptive or learned loss weights.

IMPORTANT: Standard high-Re k-epsilon is not a low-Re near-wall closure.
No-slip + k=0 does not remove that model limitation.
"""
from sympy import Function, Symbol
from physicsnemo.sym.eq.pde import PDE

C_MU = 0.09


class kEpsilonInit(PDE):
    """Initialization toward freestream values, using the actual inlet data."""

    def __init__(self, u_inlet=1.0, k_inlet=1.5e-6, ep_inlet=2.025e-8):
        x, y = Symbol("x"), Symbol("y")
        u, v = Function("u")(x, y), Function("v")(x, y)
        p = Function("p")(x, y)
        k, ep = Function("k")(x, y), Function("ep")(x, y)

        self.equations = {
            "u_init": (u - u_inlet) / u_inlet,
            "v_init": v / u_inlet,
            "p_init": p / (u_inlet**2),
            "k_init": (k - k_inlet) / k_inlet,
            "ep_init": (ep - ep_inlet) / ep_inlet,
        }


class kEpsilon(PDE):
    """Standard Launder-Spalding k-epsilon with physically complete RANS stress."""

    def __init__(
        self, nu=1e-6, rho=1.0, k_ref=1e-3, ep_ref=1e-2,
        chord=1.0, u_ref=1.0, sigma_k=1.0, sigma_ep=1.3,
        C_ep1=1.44, C_ep2=1.92,
    ):
        x, y = Symbol("x"), Symbol("y")
        u, v = Function("u")(x, y), Function("v")(x, y)
        p = Function("p")(x, y)
        k, ep = Function("k")(x, y), Function("ep")(x, y)

        ux, uy = u.diff(x), u.diff(y)
        vx, vy = v.diff(x), v.diff(y)

        # Positivity comes from the exponential network outputs. Tiny numerical
        # guards are many orders smaller than the prescribed inlet k and ep.
        nu_t = C_MU * k**2 / (ep + 1e-20)
        nu_eff = nu + nu_t

        # P_k = 2*nu_t*S_ij*S_ij, including both symmetric shear terms.
        production = nu_t * (2*ux**2 + 2*vy**2 + (uy + vx)**2)

        # Mean static pressure p: isotropic Reynolds-stress correction
        # +(2/3)*grad(k) must be retained explicitly.
        momentum_x = (
            u*ux + v*uy + p.diff(x)/rho + 2*k.diff(x)/3
            - (2*nu_eff*ux).diff(x)
            - (nu_eff*(uy + vx)).diff(y)
        )
        momentum_y = (
            u*vx + v*vy + p.diff(y)/rho + 2*k.diff(y)/3
            - (nu_eff*(uy + vx)).diff(x)
            - (2*nu_eff*vy).diff(y)
        )

        transport_k = (
            u*k.diff(x) + v*k.diff(y)
            - ((nu + nu_t/sigma_k)*k.diff(x)).diff(x)
            - ((nu + nu_t/sigma_k)*k.diff(y)).diff(y)
            - production + ep
        )
        transport_ep = (
            u*ep.diff(x) + v*ep.diff(y)
            - ((nu + nu_t/sigma_ep)*ep.diff(x)).diff(x)
            - ((nu + nu_t/sigma_ep)*ep.diff(y)).diff(y)
            - (C_ep1*production - C_ep2*ep)*ep/(k + 1e-20)
        )

        self.equations = {
            "continuity": (ux + vy)/(u_ref/chord),
            "momentum_x": momentum_x/(u_ref**2/chord),
            "momentum_y": momentum_y/(u_ref**2/chord),
            "k_equation": transport_k/(k_ref*u_ref/chord),
            "ep_equation": transport_ep/(ep_ref*u_ref/chord),
        }
