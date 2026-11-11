# SPDX-License-Identifier: Apache-2.0
"""Steady incompressible high-Re standard k-epsilon + equilibrium L-S wall functions.

Launder-Spalding formulas adapted from NVIDIA PhysicsNeMo turbulent-channel
example, examples/turbulent_channel/2d/custom_k_ep_ls.py.
This airfoil adaptation uses the local tangent, complete RANS stress tensor,
and explicitly documented residual reference scales. All lambda weights = 1.
"""
from sympy import Function, Symbol, Max, log, exp
from physicsnemo.sym.eq.pde import PDE

CMU = 0.09
KAPPA = 0.4187
E = 9.793


class kEpsilon(PDE):
    name = 'kEpsilon'

    def __init__(self, nu=1e-6, rho=1.0, k_ref=1e-3, ep_ref=1e-2,
                 c=1.0, u_ref=1.0, sigma_k=1.0, sigma_ep=1.3,
                 C_ep1=1.44, C_ep2=1.92):
        x, y = Symbol('x'), Symbol('y')
        u, v = Function('u')(x, y), Function('v')(x, y)
        p = Function('p')(x, y)
        k, ep = Function('k')(x, y), Function('ep')(x, y)
        ux, uy, vx, vy = u.diff(x), u.diff(y), v.diff(x), v.diff(y)
        nu_t = CMU * k**2 / (ep + 1e-20)
        prod = nu_t * (2 * ux**2 + 2 * vy**2 + (uy + vx)**2)
        mu_eff = nu + nu_t
        # Mean static pressure p (NOT modified pressure): include 2/3 grad(k).
        residual_x = (u*ux + v*uy + p.diff(x)/rho + 2*k.diff(x)/3
                      - (2*mu_eff*ux).diff(x) - (mu_eff*(uy + vx)).diff(y))
        residual_y = (u*vx + v*vy + p.diff(y)/rho + 2*k.diff(y)/3
                      - (mu_eff*(uy + vx)).diff(x) - (2*mu_eff*vy).diff(y))
        rk = (u*k.diff(x) + v*k.diff(y)
              - ((nu + nu_t/sigma_k)*k.diff(x)).diff(x)
              - ((nu + nu_t/sigma_k)*k.diff(y)).diff(y) - prod + ep)
        re = (u*ep.diff(x) + v*ep.diff(y)
              - ((nu + nu_t/sigma_ep)*ep.diff(x)).diff(x)
              - ((nu + nu_t/sigma_ep)*ep.diff(y)).diff(y)
              - (C_ep1*prod - C_ep2*ep)*ep/(k + 1e-20))
        self.equations = {
            'continuity': (ux + vy) / (u_ref/c),
            'momentum_x': residual_x / (u_ref*u_ref/c),
            'momentum_y': residual_y / (u_ref*u_ref/c),
            'k_equation': rk / (k_ref*u_ref/c),
            'ep_equation': re / (ep_ref*u_ref/c),
        }


class kEpsilonLSWF(PDE):
    name = 'kEpsilonLSWF'

    def __init__(self, nu=1e-6, rho=1.0, u_ref=1.0, ep_ref=1e-2,
                 shear_ref=1e-3, k_ref=1e-3,
                 yplus_min=30.0, yplus_max=300.0):
        if not (0.0 < yplus_min < yplus_max):
            raise ValueError('Require 0 < yplus_min < yplus_max')
        x, y = Symbol('x'), Symbol('y')
        u, v = Function('u')(x, y), Function('v')(x, y)
        k, ep = Function('k')(x, y), Function('ep')(x, y)
        # Geometry uses normal pointing from fluid toward the solid; flip it.
        nx, ny = -Symbol('normal_x'), -Symbol('normal_y')
        tx, ty = Symbol('tangent_x'), Symbol('tangent_y')
        d = Symbol('normal_distance')
        utau = CMU**0.25 * k_ref**0.5 * exp(Symbol('k_star')/2)
        yp = utau*d/nu
        loglaw = log(E*yp)
        # Work directly in log(y+) so the validity penalty retains a finite
        # derivative with respect to k_star even when k is extremely small.
        # This is a soft inequality, not a clipped y+ in the wall law.
        log_yp = log(CMU**0.25 * k_ref**0.5 * d / nu) + Symbol('k_star')/2
        yplus_low = Max(log(yplus_min) - log_yp, 0)
        yplus_high = Max(log_yp - log(yplus_max), 0)
        un = u*nx + v*ny
        ut = u*tx + v*ty
        nu_t = CMU*k**2/ep
        ux, uy, vx, vy = u.diff(x), u.diff(y), v.diff(x), v.diff(y)
        # Tangential traction per unit density: t · [2(nu+nu_t)S] · n
        shear_t = (nu + nu_t) * (
            tx * (2*ux*nx + (uy + vx)*ny)
            + ty * ((uy + vx)*nx + 2*vy*ny)
        )
        # In equilibrium, U_t = u_tau*loglaw/KAPPA makes the original
        # Launder-Spalding shear target exactly u_tau**2. Using that identity
        # avoids division by loglaw near E*y+ = 1 during optimization.
        tau_target = utau**2
        ep_target = CMU**0.75 * k_ref**1.5 * exp(1.5*Symbol('k_star'))/(KAPPA*d)
        self.equations = {
            'velocity_wall_normal_wf': un/u_ref,
            'velocity_wall_parallel_wf': (ut - utau*loglaw/KAPPA)/u_ref,
            'ep_wf': (ep - ep_target)/ep_ref,
            'wall_shear_stress_wf': (shear_t - tau_target)/shear_ref,
            # Enforce the modeled log-layer's admissible y+ interval without
            # replacing the actual y+ by an artificial clipped value.
            'y_plus_low_wf': yplus_low,
            'y_plus_high_wf': yplus_high,
            # No independent k_wf residual; utau is *derived* from k.
        }


class InitialProfiles(PDE):
    name = 'InitialProfiles'

    def __init__(self, u_inf=1.0, rho=1.0, k_in=1.5e-6, ep_in=2.025e-8):
        x, y = Symbol('x'), Symbol('y')
        u, v, p = Function('u')(x,y), Function('v')(x,y), Function('p')(x,y)
        # Log-space initialization: q=log(variable/reference). Gradients
        # remain well-conditioned at the very small free-stream k and epsilon.
        self.equations = {
            'u_init': (u - u_inf)/u_inf,
            'v_init': v/u_inf,
            'p_init': p/(rho*u_inf*u_inf),
            'k_init': Symbol('k_star') - Symbol('k_seed_log'),
            'ep_init': Symbol('ep_star') - Symbol('ep_seed_log'),
        }
