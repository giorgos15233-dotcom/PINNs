# SPDX-License-Identifier: Apache-2.0
"""NACA0012 PINN: three-stage high-Re RANS k-epsilon / equilibrium L-S wall model.

Keep original six constraint names and 28 inferencers. No internal arbitrary
loss weights: every lambda_weighting is 1 and the sum aggregator is used.
See RUN_NOTES.md for limitations, usage, and source correspondence.
"""
import numpy as np
import torch
from omegaconf import OmegaConf
from sympy import Eq, Function, Or, Symbol, exp

import physicsnemo.sym
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint, PointwiseConstraint, PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.geometry import Parameter, Parameterization
from physicsnemo.sym.geometry.primitives_2d import Channel2D, Line
from physicsnemo.sym.hydra import PhysicsNeMoConfig, instantiate_arch
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.utils.io import InferencerPlotter

from custom_airfoil_geometry import AirfoilInChannel
from custom_k_ep_ls import kEpsilon, kEpsilonLSWF, InitialProfiles
from wall_geometry import make_wall_points

C_MU = 0.09
RESIDUALS = ('continuity', 'momentum_x', 'momentum_y', 'k_equation', 'ep_equation')
WALL_RESIDUALS = (
    'velocity_wall_normal_wf', 'velocity_wall_parallel_wf',
    'ep_wf', 'wall_shear_stress_wf', 'y_plus_low_wf', 'y_plus_high_wf',
)


def calculate_delta_y(u_inf, chord, nu, target_y_plus=80.0, thickness_ratio=0.12):
    Re = u_inf*chord/nu
    cf = 1.328/np.sqrt(Re) if Re < 5e5 else (
        (0.074/Re**0.2 - 1742.0/Re) if Re <= 1e7
        else 0.455/np.log10(Re)**2.58
    )
    cf *= 1 + 2*thickness_ratio + 60*thickness_ratio**4
    u_tau_est = u_inf * np.sqrt(cf/2)
    return float(target_y_plus*nu/u_tau_est), float(u_tau_est)


def zero_outvar(names, n=None):
    if n is None:
        return {s: 0.0 for s in names}
    return {s: np.zeros((n, 1), dtype=np.float32) for s in names}


def unity_weights(outvar, n=None):
    if n is None:
        return {s: 1.0 for s in outvar}
    return {s: np.ones((n, 1), dtype=np.float32) for s in outvar}


@physicsnemo.sym.main(config_path='conf', config_name='config')
def run(cfg: PhysicsNeMoConfig) -> None:
    x, y = Symbol('x'), Symbol('y')
    stage = str(OmegaConf.select(cfg, 'custom.stage', default='physics'))
    if stage not in ('initialize', 'mixed', 'physics'):
        raise ValueError('stage must be initialize, mixed or physics')
    yplus_min = float(OmegaConf.select(cfg, 'custom.yplus_min', default=30.0))
    yplus_max = float(OmegaConf.select(cfg, 'custom.yplus_max', default=300.0))
    if not (0.0 < yplus_min < yplus_max):
        raise ValueError('Require 0 < custom.yplus_min < custom.yplus_max')

    u_inf, chord, rho, Re = 1.0, 1.0, 1.0, 1e6
    nu = u_inf*chord/Re
    delta_y, u_tau_est = calculate_delta_y(u_inf, chord, nu)
    k_in = 1.5 * (u_inf * 0.001)**2
    ep_in = C_MU*k_in**2/(10.0*nu)
    # Reference scales are RESIDUAL nondimensionalization, NOT loss weights.
    k_ref, ep_ref, shear_ref = 1e-3, 1e-2, u_tau_est**2
    # Near-wall log-layer seed, NOT a permanent boundary condition.
    k_wall_seed = u_tau_est**2 / np.sqrt(C_MU)
    ep_wall_seed = C_MU**0.75 * k_wall_seed**1.5 / (0.4187*delta_y)
    print(f'Stage: {stage}; Re={Re:g}, delta_y={delta_y:.8g}m, '
          f'estimated u_tau={u_tau_est:.6g}m/s')
    parameterization = Parameterization({
        Parameter('alpha'): (0.0, 0.0),
        Parameter('camber'): (0.0, 0.0),
        Parameter('thickness'): (0.12, 0.12),
    })

    # Original geometry and naming; no rectangular far-field decomposition.
    channel_length = 15.0/2
    channel_height = 10.0/2
    x_fraction = 0.3
    xmin, xmax = -channel_length*x_fraction, channel_length*(1-x_fraction)
    ymin, ymax = -channel_height/2, channel_height/2
    channel = Channel2D((xmin, ymin), (xmax, ymax))
    inlet_geom = Line((xmin, ymin), (xmin, ymax), normal=1)
    outlet_geom = Line((xmax, ymin), (xmax, ymax), normal=1)

    near_field_channel_length = 1.6
    near_field_channel_height = 1.0
    x_fraction_near_field = 0.2
    near_xmin = -near_field_channel_length*x_fraction_near_field
    near_xmax = near_field_channel_length*(1-x_fraction_near_field)
    near_ymin, near_ymax = -0.5, 0.5
    volume_near_field = AirfoilInChannel(
        x_min=near_xmin, x_max=near_xmax,
        y_min=near_ymin, y_max=near_ymax,
        params=parameterization, include_channel_boundary=False,
    )
    custom_channel_length, custom_channel_height, x_fraction_custom = 1.2, 0.4, 0.2
    volume_custom_field = AirfoilInChannel(
        x_min=-custom_channel_length*x_fraction_custom,
        x_max=custom_channel_length*(1-x_fraction_custom),
        y_min=-custom_channel_height/2, y_max=custom_channel_height/2,
        params=parameterization, include_channel_boundary=False,
    )

    eq = kEpsilon(nu=nu, rho=rho, k_ref=k_ref, ep_ref=ep_ref,
                  c=chord, u_ref=u_inf)
    wf = kEpsilonLSWF(nu=nu, rho=rho, u_ref=u_inf,
                      ep_ref=ep_ref, shear_ref=shear_ref, k_ref=k_ref,
                      yplus_min=yplus_min, yplus_max=yplus_max)
    init = InitialProfiles(u_inf=u_inf, rho=rho, k_in=k_in, ep_in=ep_in)
    input_keys = [Key('x'), Key('y'), Key('alpha'), Key('camber'), Key('thickness')]
    flow_net = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("u"), Key("v")],
        cfg=cfg.arch.fourier,
    )
    pressure_net = instantiate_arch(input_keys=input_keys, output_keys=[Key('p')],
                                    cfg=cfg.arch.fully_connected)
    k_net = instantiate_arch(
        input_keys=input_keys, output_keys=[Key('k_star')],
        cfg=cfg.arch.fourier,
    )
    ep_net = instantiate_arch(
        input_keys=input_keys, output_keys=[Key('ep_star')],
        cfg=cfg.arch.fourier,
    )

    # Dimensioned positive outputs k_ref*exp(qk), ep_ref*exp(qep).
    # Log-space inlet/initialization losses keep gradients nonvanishing.
    # Avoid PhysicsNeMo's potentially regularized sqrt in exported u_tau.
    nodes = (
        eq.make_nodes() + wf.make_nodes() + init.make_nodes()
        + [Node.from_sympy(k_ref*exp(Symbol('k_star')), 'k')]
        + [Node.from_sympy(ep_ref*exp(Symbol('ep_star')), 'ep')]
        + [Node.from_sympy(C_MU**0.25*k_ref**0.5*exp(Symbol('k_star')/2), 'u_tau')]
        + [Node.from_sympy(Symbol('u_tau')*Symbol('normal_distance')/nu,
                           'y_plus_wall')]
        + [flow_net.make_node(name='flow_network')]
        + [pressure_net.make_node(name='pressure_network')]
        + [k_net.make_node(name='k_network')]
        + [ep_net.make_node(name='ep_network')]
    )
    domain = Domain()
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=inlet_geom,
        outvar={'u': u_inf, 'v': 0.0,
                'k_star': float(np.log(k_in/k_ref)),
                'ep_star': float(np.log(ep_in/ep_ref))},
        lambda_weighting={'u': 1.0, 'v': 1.0, 'k_star': 1.0, 'ep_star': 1.0},
        batch_size=cfg.batch_size.inlet, parameterization=parameterization,
    )
    domain.add_constraint(inlet, 'inlet')
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=outlet_geom, outvar={'p': 0.0},
        lambda_weighting={'p': 1.0},
        batch_size=cfg.batch_size.outlet, parameterization=parameterization,
    )
    domain.add_constraint(outlet, 'outlet')

    # Original single top_bot name, horizontal channel boundaries only.
    u, v, p = Function('u')(x,y), Function('v')(x,y), Function('p')(x,y)
    k, ep = Function('k')(x,y), Function('ep')(x,y)
    bc_nodes = nodes + [
        Node.from_sympy(u.diff(y), 'u_y_bc'),
        Node.from_sympy(p.diff(y), 'p_y_bc'),
        Node.from_sympy(k.diff(y), 'k_y_bc'),
        Node.from_sympy(ep.diff(y), 'ep_y_bc'),
    ]
    tb = zero_outvar(('v', 'u_y_bc', 'p_y_bc', 'k_y_bc', 'ep_y_bc'))
    domain.add_constraint(PointwiseBoundaryConstraint(
        nodes=bc_nodes, geometry=channel,
        criteria=Or(Eq(y, ymax), Eq(y, ymin)),
        outvar=tb, lambda_weighting=unity_weights(tb),
        batch_size=cfg.batch_size.top_bot, parameterization=parameterization,
    ), 'top_bot')

    # The original Warp SDF is a union of 501 thickness disks. This function
    # uses THAT SAME SDF to place P, not an unrelated analytic NACA curve.
    wall_invar = make_wall_points(
        volume_near_field, parameterization, cfg.batch_size.airfoil, delta_y,
    )

    # Static rejection of points below P. This matters because the custom
    # AirfoilInChannel.sample_interior accepts 'criteria' but does NOT apply it
    # to the returned Warp samples. A criteria=... would silently be ineffective.
    near_raw = volume_near_field.sample_interior(
        max(4*cfg.batch_size.interior_near_field, 10000),
        rand_seed=17, parameterization=parameterization,
    )
    acceptable = np.asarray(near_raw['sdf']).reshape(-1) >= delta_y
    enough = int(np.count_nonzero(acceptable))
    if enough < cfg.batch_size.interior_near_field:
        raise RuntimeError(f'Only {enough} resolved near-field points; '
                           'increase near-field sampling pool')
    near_input = {
        key: np.asarray(near_raw[key], dtype=np.float32)[acceptable]
        for key in ('x', 'y', 'alpha', 'camber', 'thickness')
    }
    sd = np.asarray(near_raw['sdf'], dtype=np.float64)[acceptable]
    k_profile = k_in + (k_wall_seed-k_in)*np.exp(-sd/0.07)
    ep_profile = ep_in + (ep_wall_seed-ep_in)*np.exp(-sd/0.035)
    near_input['k_seed_log'] = np.log(k_profile/k_ref).astype(np.float32)
    near_input['ep_seed_log'] = np.log(ep_profile/ep_ref).astype(np.float32)
    # Every collocation point used for the near-field PDE is >= delta_y.
    print(f'Resolved near-field collocation: {enough}; '
          f'filtered {len(acceptable)-enough} below P')
    near_N = len(near_input['x'])

    # Sample far-field points outside the near-field box.
    far_raw = channel.sample_interior(
        max(cfg.batch_size.interior_far_field, 10000),
        criteria=Or(
            x <= near_xmin,
            x >= near_xmax,
            y <= near_ymin,
            y >= near_ymax,
        ),
        quasirandom=True,
        parameterization=parameterization,
    )

    far_input = {
        key: np.asarray(far_raw[key], dtype=np.float32)
        for key in ('x', 'y', 'alpha', 'camber', 'thickness')
    }

    # Initialize the far-field turbulence to free-stream values.
    far_N = len(far_input['x'])

    far_input['k_seed_log'] = np.full(
        (far_N, 1),
        np.log(k_in / k_ref),
        dtype=np.float32,
    )

    far_input['ep_seed_log'] = np.full(
        (far_N, 1),
        np.log(ep_in / ep_ref),
        dtype=np.float32,
    )

    if stage in ('initialize', 'mixed'):

        # Near-field initialization.
        near_init_names = (
            ('u_init', 'v_init', 'p_init', 'k_init', 'ep_init')
            if stage == 'initialize'
            else ('k_init', 'ep_init')
        )

        near_targets = zero_outvar(near_init_names, near_N)

        domain.add_constraint(
            PointwiseConstraint.from_numpy(
                nodes=nodes,
                invar=near_input,
                outvar=near_targets,
                lambda_weighting=unity_weights(near_targets, near_N),
                batch_size=cfg.batch_size.interior_near_field,
            ),
            'interior_near_init',
        )

        # Far-field initialization.
        far_init_names = (
            ('u_init', 'v_init', 'p_init', 'k_init', 'ep_init')
            if stage == 'initialize'
            else ('u_init', 'v_init', 'k_init', 'ep_init')
        )

        far_targets = zero_outvar(far_init_names, far_N)

        domain.add_constraint(
            PointwiseConstraint.from_numpy(
                nodes=nodes,
                invar=far_input,
                outvar=far_targets,
                lambda_weighting=unity_weights(far_targets, far_N),
                batch_size=cfg.batch_size.interior_far_field,
            ),
            'interior_far_init',
        )
        # Extra initialization-only samples at P prevent the wall-layer seed
        # from being underrepresented in uniform volume collocation.
        seed_invar = dict(wall_invar)
        nwall = len(seed_invar['x'])
        seed_invar['k_seed_log'] = np.full((nwall,1), np.log(k_wall_seed/k_ref), np.float32)
        seed_invar['ep_seed_log'] = np.full((nwall,1), np.log(ep_wall_seed/ep_ref), np.float32)
        wall_init_targets = zero_outvar(('k_init', 'ep_init'), nwall)
        domain.add_constraint(PointwiseConstraint.from_numpy(
            nodes=nodes, invar=seed_invar, outvar=wall_init_targets,
            lambda_weighting=unity_weights(wall_init_targets,nwall),
            batch_size=cfg.batch_size.airfoil,
        ), 'wall_initialization')

    if stage in ('mixed', 'physics'):
        wv = zero_outvar(WALL_RESIDUALS, len(wall_invar['x']))
        domain.add_constraint(PointwiseConstraint.from_numpy(
            nodes=nodes, invar=wall_invar, outvar=wv,
            lambda_weighting=unity_weights(wv, len(wall_invar['x'])),
            batch_size=cfg.batch_size.airfoil,
        ), 'airfoil')

        domain.add_constraint(PointwiseInteriorConstraint(
            nodes=nodes, geometry=channel, outvar=zero_outvar(RESIDUALS),
            lambda_weighting=unity_weights(RESIDUALS),
            batch_size=cfg.batch_size.interior_far_field,
            criteria=Or(x <= near_xmin, x >= near_xmax,
                        y <= near_ymin, y >= near_ymax),
            parameterization=parameterization,
        ), 'interior_far_field')
        near_targets = zero_outvar(RESIDUALS, near_N)
        domain.add_constraint(PointwiseConstraint.from_numpy(
            nodes=nodes, invar=near_input, outvar=near_targets,
            lambda_weighting=unity_weights(near_targets, near_N),
            batch_size=cfg.batch_size.interior_near_field,
        ), 'interior_near_field')

    # 28 original inferencer titles. Boundary_x is the actual solid surface;
    # airfoil is the offset constraint P. Diagnostic 'AirfoilWallAtP' is extra.
    inferencer_id = 0

    def add_inferencer(alpha_inference=0.0, camber_inference=0.0,
                       thickness_inference=0.12, n_interior_points=20000,
                       n_custom_points=300000, n_boundary_points=512):
        nonlocal inferencer_id
        pp = Parameterization({
            Parameter('alpha'): (alpha_inference, alpha_inference),
            Parameter('camber'): (camber_inference, camber_inference),
            Parameter('thickness'): (thickness_inference, thickness_inference),
        })
        near = volume_near_field.sample_interior(
            n_interior_points, rand_seed=inferencer_id, parameterization=pp)
        bd = volume_near_field.sample_boundary(
            n_boundary_points, rand_seed=inferencer_id, parameterization=pp)
        far = channel.sample_interior(n_interior_points,
                                      quasirandom=True, parameterization=pp)
        tight = volume_custom_field.sample_interior(
            n_custom_points, rand_seed=inferencer_id, parameterization=pp)
        fields = ['u','v','p','k','ep','u_tau']
        for title, samples, add_sdf in (
            ('custom_tight_field',tight,True), ('near_field',near,True),
            ('boundary',bd,False), ('far_field',far,False),
        ):
            keys = ['x','y','alpha','camber','thickness']
            invar = {key: samples[key] for key in keys}
            names = list(fields)
            if add_sdf:
                invar['sdf'] = samples['sdf']
                names.append('sdf')
            if title == 'boundary':
                invar['normal_x'], invar['normal_y'] = samples['normal_x'], samples['normal_y']
                names += ['normal_x', 'normal_y']
            domain.add_inferencer(PointwiseInferencer(
                nodes=nodes, invar=invar, output_names=names,
                batch_size=4096, plotter=InferencerPlotter(),
            ), f'{title}_{inferencer_id+1}')
        inferencer_id += 1

    add_inferencer()
#    rng = np.random.default_rng(42)
#    for _ in range(6):
#        add_inferencer(
#            alpha_inference=rng.uniform(0.0,0.0),
#            camber_inference=rng.uniform(0.0,0.0),
#            thickness_inference=rng.uniform(0.12,0.12),
#        )

    # Dedicated P-point diagnostics. PointwiseInferencer evaluates under
    # torch.no_grad(), so DO NOT request wall_shear_stress_wf (or any PDE
    # residual that requires velocity derivatives) from this inferencer.
    # Wall residuals remain in the "airfoil" training constraint and its VTP.
    domain.add_inferencer(PointwiseInferencer(
        nodes=nodes, invar=wall_invar,
        output_names=['u','v','p','k','ep','u_tau','y_plus_wall',
                      'normal_distance', 'normal_x','normal_y',
                      'tangent_x','tangent_y'],
        batch_size=512, plotter=InferencerPlotter(),
    ), 'AirfoilWallAtP')

    # Log actual physical values rather than judging the wall state only from
    # aggregate losses. No spatial derivatives are required by this monitor.
    domain.add_monitor(PointwiseMonitor(
        invar=wall_invar,
        output_names=['k','ep','u_tau','y_plus_wall'],
        metrics={
            'wall_k_mean': lambda var: torch.mean(var['k']),
            'wall_ep_mean': lambda var: torch.mean(var['ep']),
            'wall_u_tau_mean': lambda var: torch.mean(var['u_tau']),
            'wall_y_plus_min': lambda var: torch.min(var['y_plus_wall']),
            'wall_y_plus_mean': lambda var: torch.mean(var['y_plus_wall']),
            'wall_y_plus_max': lambda var: torch.max(var['y_plus_wall']),
            'wall_y_plus_below_log_layer_fraction': lambda var:
                torch.mean((var['y_plus_wall'] < yplus_min).float()),
            'wall_y_plus_above_log_layer_fraction': lambda var:
                torch.mean((var['y_plus_wall'] > yplus_max).float()),
        },
        nodes=nodes,
    ))

    print(f'All lambda weights exactly 1; phase={stage}; '
          f'k_in={k_in:.5e}; ep_in={ep_in:.5e}')
    print(f'Log-layer validity guard (soft): {yplus_min:g} <= y+ <= {yplus_max:g}')
    Solver(cfg, domain).solve()


if __name__ == '__main__':
    run()
