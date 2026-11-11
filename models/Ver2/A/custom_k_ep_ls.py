# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Steady incompressible Navier-Stokes and standard k-epsilon equations.

The pressure variable is dimensional pressure. The momentum equations therefore
use grad(p) / rho. The turbulence variables k and ep are specific turbulence
kinetic energy and its dissipation rate, respectively.
"""

from sympy import Abs, Function, Symbol, log, sqrt

from physicsnemo.sym.eq.pde import PDE


class kEpsilon(PDE):
    """High-Reynolds-number standard k-epsilon model in 2D steady form."""

    name = "kEpsilon"

    def __init__(
        self,
        nu=1.0e-6,
        rho=1.0,
        C_mu=0.09,
        sigma_k=1.0,
        sigma_ep=1.3,
        C_ep1=1.44,
        C_ep2=1.92,
        eps_regularization=1.0e-10,
    ):
        if nu <= 0.0:
            raise ValueError("nu must be positive")
        if rho <= 0.0:
            raise ValueError("rho must be positive")

        x, y = Symbol("x"), Symbol("y")
        u = Function("u")(x, y)
        v = Function("v")(x, y)
        p = Function("p")(x, y)
        k = Function("k")(x, y)
        ep = Function("ep")(x, y)

        ux, uy = u.diff(x), u.diff(y)
        vx, vy = v.diff(x), v.diff(y)
        px, py = p.diff(x), p.diff(y)
        kx, ky = k.diff(x), k.diff(y)
        epx, epy = ep.diff(x), ep.diff(y)

        nu_t = C_mu * k**2 / (ep + eps_regularization)
        nu_t_x = nu_t.diff(x)
        nu_t_y = nu_t.diff(y)

        # This is 2 * nu_t * S_ij * S_ij for an incompressible 2D velocity field.
        production_k = nu_t * (2.0 * ux**2 + 2.0 * vy**2 + (uy + vx) ** 2)

        nu_eff_k = nu + nu_t / sigma_k
        nu_eff_ep = nu + nu_t / sigma_ep

        self.equations = {
            "continuity": ux + vy,
            "momentum_x": (
                u * ux
                + v * uy
                + px / rho
                - (nu + nu_t) * (u.diff(x, 2) + u.diff(y, 2))
                - nu_t_x * ux
                - nu_t_y * uy
            ),
            "momentum_y": (
                u * vx
                + v * vy
                + py / rho
                - (nu + nu_t) * (v.diff(x, 2) + v.diff(y, 2))
                - nu_t_x * vx
                - nu_t_y * vy
            ),
            "k_equation": (
                u * kx
                + v * ky
                - nu_eff_k * (k.diff(x, 2) + k.diff(y, 2))
                - (nu_t_x / sigma_k) * kx
                - (nu_t_y / sigma_k) * ky
                - production_k
                + ep
            ),
            "ep_equation": (
                u * epx
                + v * epy
                - nu_eff_ep * (ep.diff(x, 2) + ep.diff(y, 2))
                - (nu_t_x / sigma_ep) * epx
                - (nu_t_y / sigma_ep) * epy
                - (C_ep1 * production_k - C_ep2 * ep)
                * ep
                / (k + eps_regularization)
            ),
        }


class kEpsilonLSWF(PDE):
    """Launder-Spalding wall-function residuals evaluated off the wall.

    The supplied normal_x/normal_y values are expected to be the geometry's
    inward-pointing normals (as returned by the user's AirfoilInChannel class).
    They are flipped here to point from the solid wall into the fluid.
    """

    name = "kEpsilonLSWF"

    def __init__(
        self,
        nu=1.0e-6,
        rho=1.0,
        C_mu=0.09,
        kappa=0.4187,
        E=9.793,
        eps_regularization=1.0e-10,
        log_regularization=1.0e-12,
    ):
        if nu <= 0.0:
            raise ValueError("nu must be positive")
        if rho <= 0.0:
            raise ValueError("rho must be positive")

        x, y = Symbol("x"), Symbol("y")
        u = Function("u")(x, y)
        v = Function("v")(x, y)
        k = Function("k")(x, y)
        ep = Function("ep")(x, y)
        u_tau = Function("u_tau")(x, y)
        normal_distance = Function("normal_distance")(x, y)

        # AirfoilInChannel returns the negative of the outward wall normal.
        normal_x = -Symbol("normal_x")
        normal_y = -Symbol("normal_y")

        nu_t = C_mu * k**2 / (ep + eps_regularization)
        y_plus = u_tau * normal_distance / nu
        log_law = log(Abs(E * y_plus) + log_regularization)

        u_tau_from_k = C_mu**0.25 * sqrt(k)
        ep_from_wall_function = C_mu**0.75 * k**1.5 / (kappa * normal_distance)

        # Decompose velocity into wall-normal and wall-parallel components.
        u_normal = u * normal_x + v * normal_y
        u_parallel_x = u - u_normal * normal_x
        u_parallel_y = v - u_normal * normal_y
        u_parallel_magnitude = sqrt(u_parallel_x**2 + u_parallel_y**2 + 1.0e-24)
        u_parallel_target = (u_tau / kappa) * log_law

        # The tangent-projected velocity gradient in the wall-normal direction.
        ux, uy = u.diff(x), u.diff(y)
        vx, vy = v.diff(x), v.diff(y)
        du_parallel_x_dx = ux - (ux * normal_x + vx * normal_y) * normal_x
        du_parallel_y_dx = vx - (ux * normal_x + vx * normal_y) * normal_y
        du_parallel_x_dy = uy - (uy * normal_x + vy * normal_y) * normal_x
        du_parallel_y_dy = vy - (uy * normal_x + vy * normal_y) * normal_y
        du_dsdf_x = du_parallel_x_dx * normal_x + du_parallel_x_dy * normal_y
        du_dsdf_y = du_parallel_y_dx * normal_x + du_parallel_y_dy * normal_y

        # Physical shear stresses. The same rho factor appears on both sides.
        shear_x = rho * (nu + nu_t) * du_dsdf_x
        shear_y = rho * (nu + nu_t) * du_dsdf_y
        shear_x_target = rho * u_tau * u_parallel_x * kappa / log_law
        shear_y_target = rho * u_tau * u_parallel_y * kappa / log_law

        # Included for compatibility/diagnostics. With u_tau=C_mu**(1/4)*sqrt(k),
        # k_wf is an algebraic identity, not an independent physical boundary condition.
        k_from_u_tau = u_tau**2 / sqrt(C_mu)

        self.equations = {
            "velocity_wall_normal_wf": u_normal,
            "velocity_wall_parallel_wf": u_parallel_magnitude - u_parallel_target,
            "k_wf": k - k_from_u_tau,
            "ep_wf": ep - ep_from_wall_function,
            "wall_shear_stress_x_wf": shear_x - shear_x_target,
            "wall_shear_stress_y_wf": shear_y - shear_y_target,
            # Exporting u_tau_from_k is useful for inspection; it is not a loss residual.
            "u_tau_from_k_check": u_tau - u_tau_from_k,
        }
