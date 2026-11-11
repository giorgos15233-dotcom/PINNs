# SPDX-License-Identifier: Apache-2.0
"""Steady incompressible RANS k-epsilon with a composite near-wall closure.

The wall flux uses a smooth Reichardt-type viscous/log profile calibrated to
the Launder-Spalding log intercept. At P, prescribe normal flow and tangential
stress (Robin condition), not independent tangential Dirichlet and Neumann
conditions for the same momentum equation. All explicit lambda weights = 1.
"""
from sympy import Function, Symbol, Max, sqrt, exp
from physicsnemo.sym.eq.pde import PDE
from wall_laws import C_MU as CMU, KAPPA, E, composite_u_plus, blended_epsilon



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
        u_plus = composite_u_plus(yp)
        un = u*nx + v*ny
        ut = u*tx + v*ty
        nu_t = CMU*k**2/ep
        ux, uy, vx, vy = u.diff(x), u.diff(y), v.diff(x), v.diff(y)
        # Tangential traction per unit density: t · [2(nu+nu_t)S] · n
        shear_t = (nu + nu_t) * (
            tx * (2*ux*nx + (uy + vx)*ny)
            + ty * ((uy + vx)*nx + 2*vy*ny)
        )
        # Wall traction from the off-wall velocity (not an imposed positive
        # traction): tau = nu * U_t / d * (y_plus / U_plus).
        # The low-y+ limit is nu*U_t/d; the log-law limit is
        # u_tau*U_t/U_plus. Reverse flow therefore reverses the traction.
        tau_target = (nu * ut / d) * (yp / (u_plus + 1.0e-12))
        ep_target = blended_epsilon(k, nu, d, yp)
        # The high-Re k-equation uses approximately zero normal k gradient at
        # the wall; impose it at the truncated-domain boundary P.
        k_n = nx*k.diff(x) + ny*k.diff(y)
        self.equations = {
            'velocity_wall_normal_wf': un/u_ref,
            # This is a DIAGNOSTIC, not an enforced boundary loss. Applying
            # both a prescribed wall velocity and a wall traction overconstrains
            # the tangential momentum equation.
            'velocity_wall_parallel_wf': (
                sqrt(ut**2 + 1.0e-16) - utau*u_plus
            )/u_ref,
            'ep_wf': (ep - ep_target)/ep_ref,
            'wall_shear_stress_wf': (shear_t - tau_target)/shear_ref,
            'k_normal_gradient_wf': k_n / (k_ref / d),
            # These are monitoring outputs only; y+ is never forced into a
            # log-layer interval at a stagnation or separation point.
            'y_plus_low_wf': Max(yplus_min - yp, 0),
            'y_plus_high_wf': Max(yp - yplus_max, 0),
            'wall_velocity_t': ut,
            'wall_velocity_n': un,
            'wall_tau_model': tau_target,
            'wall_uplus_model': u_plus,
            'wall_ep_target': ep_target,
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
