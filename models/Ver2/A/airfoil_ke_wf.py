# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""PINN solver for a 2D airfoil using the standard k-epsilon model.

The Launder-Spalding wall relations are enforced at points displaced from the
airfoil by delta_y into the fluid. The interior PDE losses are suppressed within
the same wall distance, as required by a wall-function formulation.
"""

import numpy as np
from sympy import Function, Max, Min, Symbol, exp, log, sqrt

import physicsnemo.sym
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.geometry.parameterization import Parameter, Parameterization
from physicsnemo.sym.geometry.primitives_2d import Line, Rectangle
from physicsnemo.sym.hydra import PhysicsNeMoConfig, instantiate_arch
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.solver import Solver

from custom_airfoil_geometry import AirfoilInChannel
from custom_k_ep_ls import kEpsilon, kEpsilonLSWF


C_MU = 0.09
TURBULENCE_INTENSITY = 1.0e-3
TURBULENT_VISCOSITY_RATIO = 10.0


def calculate_delta_y(
    u_inf: float,
    chord: float,
    nu: float,
    target_y_plus: float = 80.0,
    thickness_ratio: float = 0.12,
) -> float:
    """Estimate one constant first-cell wall distance for a target mean y+.

    This uses a flat-plate average skin-friction correlation with an airfoil
    form-factor correction. It is a global estimate; local y+ will vary along
    the airfoil and should be checked after training.
    """
    if u_inf <= 0.0 or chord <= 0.0 or nu <= 0.0 or target_y_plus <= 0.0:
        raise ValueError("u_inf, chord, nu, and target_y_plus must be positive")

    reynolds = u_inf * chord / nu
    if reynolds < 5.0e5:
        cf_flat_plate = 1.328 / np.sqrt(reynolds)
    elif reynolds <= 1.0e7:
        cf_flat_plate = 0.074 / reynolds**0.2 - 1742.0 / reynolds
    else:
        cf_flat_plate = 0.455 / (np.log10(reynolds) ** 2.58)

    form_factor = 1.0 + 2.0 * thickness_ratio + 60.0 * thickness_ratio**4
    cf_airfoil = cf_flat_plate * form_factor
    u_tau_est = u_inf * np.sqrt(cf_airfoil / 2.0)
    if not np.isfinite(u_tau_est) or u_tau_est <= 0.0:
        raise ValueError("Could not estimate a positive friction velocity")

    return float(target_y_plus * nu / u_tau_est)


def _add_zero_equation_constraint(
    domain,
    nodes,
    geometry,
    name,
    batch_size,
    lambda_weighting=None,
    parameterization=None,
    sdf_wall_distance=None,
):
    """Add a steady k-epsilon PDE constraint to an interior geometry."""
    x, y = Symbol("x"), Symbol("y")
    outvar = {
        "continuity": 0.0,
        "momentum_x": 0.0,
        "momentum_y": 0.0,
        "k_equation": 0.0,
        "ep_equation": 0.0,
    }
    kwargs = {}
    if lambda_weighting is not None:
        kwargs["lambda_weighting"] = lambda_weighting
    if parameterization is not None:
        kwargs["parameterization"] = parameterization
    if sdf_wall_distance is not None:
        kwargs["lambda_weighting"] = {
            key: Max(Symbol("sdf") - sdf_wall_distance, 0.0)
            for key in outvar
        }

    constraint = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geometry,
        outvar=outvar,
        batch_size=batch_size,
        **kwargs,
    )
    domain.add_constraint(constraint, name)


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # Flow and geometry parameters. The airfoil chord is 1 m.
    u_inf = 1.0
    chord = 1.0
    reynolds = 1.0e6
    nu = u_inf * chord / reynolds
    rho = 1.0
    target_y_plus = 80.0
    delta_y = calculate_delta_y(
        u_inf=u_inf,
        chord=chord,
        nu=nu,
        target_y_plus=target_y_plus,
        thickness_ratio=0.12,
    )

    # Parameter values are fixed for this run but remain explicit network inputs.
    parameterization = Parameterization(
        {
            Parameter("alpha"): (0.0, 0.0),
            Parameter("camber"): (0.0, 0.0),
            Parameter("thickness"): (0.12, 0.12),
        }
    )

    # Full far-field dimensions: length 7.5 m and height 5 m.
    x_min, x_max = -2.5, 5.0
    y_min, y_max = -2.5, 2.5

    # Near-field box: length 1.6 m, height 1.0 m, enclosing the airfoil.
    near_x_min, near_x_max = -0.2, 1.4
    near_y_min, near_y_max = -0.5, 0.5

    # Fine sampling box used for output/inference around the airfoil.
    custom_x_min, custom_x_max = -0.1, 1.1
    custom_y_min, custom_y_max = -0.2, 0.2

    full_geometry = AirfoilInChannel(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        params=parameterization,
        include_channel_boundary=False,
        chord=chord,
    )
    near_geometry = AirfoilInChannel(
        x_min=near_x_min,
        x_max=near_x_max,
        y_min=near_y_min,
        y_max=near_y_max,
        params=parameterization,
        include_channel_boundary=False,
        chord=chord,
    )
    custom_geometry = AirfoilInChannel(
        x_min=custom_x_min,
        x_max=custom_x_max,
        y_min=custom_y_min,
        y_max=custom_y_max,
        params=parameterization,
        include_channel_boundary=False,
        chord=chord,
    )

    # The near-field box contains the whole airfoil. Partition the remaining
    # fluid domain into four ordinary rectangles so criteria filtering is not
    # delegated to AirfoilInChannel.sample_interior().
    far_geometries = [
        Rectangle((x_min, y_min), (near_x_min, y_max)),
        Rectangle((near_x_max, y_min), (x_max, y_max)),
        Rectangle((near_x_min, near_y_max), (near_x_max, y_max)),
        Rectangle((near_x_min, y_min), (near_x_max, near_y_min)),
    ]

    inlet_geometry = Line((x_min, y_min), (x_min, y_max), normal=1)
    outlet_geometry = Line((x_max, y_min), (x_max, y_max), normal=1)
    top_geometry = Line((x_min, y_max), (x_max, y_max), normal=1)
    bottom_geometry = Line((x_min, y_min), (x_max, y_min), normal=1)

    k_inlet = 1.5 * (u_inf * TURBULENCE_INTENSITY) ** 2
    ep_inlet = C_MU * k_inlet**2 / (TURBULENT_VISCOSITY_RATIO * nu)

    k_ep_model = kEpsilon(nu=nu, rho=rho)
    k_ep_wf = kEpsilonLSWF(nu=nu, rho=rho)

    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("alpha"), Key("camber"), Key("thickness")],
        output_keys=[Key("u"), Key("v")],
        frequencies=("axis", [i / 2.0 for i in range(8)]),
        frequencies_params=("axis", [i / 2.0 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )
    pressure_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    k_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("alpha"), Key("camber"), Key("thickness")],
        output_keys=[Key("k_star")],
        frequencies=("axis", [i / 2.0 for i in range(8)]),
        frequencies_params=("axis", [i / 2.0 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )
    ep_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("alpha"), Key("camber"), Key("thickness")],
        output_keys=[Key("ep_star")],
        frequencies=("axis", [i / 2.0 for i in range(8)]),
        frequencies_params=("axis", [i / 2.0 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )

    x, y = Symbol("x"), Symbol("y")
    k_positive = Min(log(1.0 + exp(Symbol("k_star"))), 20.0) + 1.0e-10
    ep_positive = Min(log(1.0 + exp(Symbol("ep_star"))), 180.0) + 1.0e-10

    nodes = (
        k_ep_model.make_nodes()
        + k_ep_wf.make_nodes()
        + [Node.from_sympy(k_positive, "k")]
        + [Node.from_sympy(ep_positive, "ep")]
        + [Node.from_sympy(C_MU**0.25 * sqrt(Symbol("k")), "u_tau")]
        + [flow_net.make_node(name="flow_network")]
        + [pressure_net.make_node(name="pressure_network")]
        + [k_net.make_node(name="k_network")]
        + [ep_net.make_node(name="ep_network")]
    )

    domain = Domain()

    # Inlet values use one consistent free-stream turbulence specification.
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_geometry,
        outvar={"u": u_inf, "v": 0.0, "k": k_inlet, "ep": ep_inlet},
        batch_size=cfg.batch_size.inlet,
        parameterization=parameterization,
    )
    domain.add_constraint(inlet, "Inlet")

    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_geometry,
        outvar={"p": 0.0},
        batch_size=cfg.batch_size.outlet,
        parameterization=parameterization,
    )
    domain.add_constraint(outlet, "Outlet")

    # Top/bottom use symmetry/slip conditions: zero normal velocity and zero
    # normal gradients of tangential velocity, pressure, k, and epsilon.
    u_fun = Function("u")(x, y)
    p_fun = Function("p")(x, y)
    k_fun = Function("k")(x, y)
    ep_fun = Function("ep")(x, y)
    derivative_nodes = [
        Node.from_sympy(u_fun.diff(y), "u_y_bc"),
        Node.from_sympy(p_fun.diff(y), "p_y_bc"),
        Node.from_sympy(k_fun.diff(y), "k_y_bc"),
        Node.from_sympy(ep_fun.diff(y), "ep_y_bc"),
    ]
    boundary_nodes = nodes + derivative_nodes
    top_bottom_outvar = {
        "v": 0.0,
        "u_y_bc": 0.0,
        "p_y_bc": 0.0,
        "k_y_bc": 0.0,
        "ep_y_bc": 0.0,
    }
    top_bottom_weights = {key: 1.0 for key in top_bottom_outvar}

    top = PointwiseBoundaryConstraint(
        nodes=boundary_nodes,
        geometry=top_geometry,
        outvar=top_bottom_outvar,
        lambda_weighting=top_bottom_weights,
        batch_size=cfg.batch_size.top_bot,
        parameterization=parameterization,
    )
    bottom = PointwiseBoundaryConstraint(
        nodes=boundary_nodes,
        geometry=bottom_geometry,
        outvar=top_bottom_outvar,
        lambda_weighting=top_bottom_weights,
        batch_size=cfg.batch_size.top_bot,
        parameterization=parameterization,
    )
    domain.add_constraint(top, "Top")
    domain.add_constraint(bottom, "Bottom")

    # The wall-function points must be off-wall, not on the solid boundary.
    wall_points = near_geometry.sample_boundary(
        n_points=cfg.batch_size.airfoil,
        rand_seed=23,
        parameterization=parameterization,
    )
    wall_x = np.asarray(wall_points["x"], dtype=np.float32)
    wall_y = np.asarray(wall_points["y"], dtype=np.float32)
    # This geometry returns normal_x/normal_y pointing into the solid, so minus
    # those normals shifts the coordinates delta_y into the fluid.
    wall_nx = np.asarray(wall_points["normal_x"], dtype=np.float32)
    wall_ny = np.asarray(wall_points["normal_y"], dtype=np.float32)
    wall_invar = {
        "x": wall_x - delta_y * wall_nx,
        "y": wall_y - delta_y * wall_ny,
        "normal_x": wall_nx,
        "normal_y": wall_ny,
        "normal_distance": np.full_like(wall_x, delta_y, dtype=np.float32),
        "alpha": np.asarray(wall_points["alpha"], dtype=np.float32),
        "camber": np.asarray(wall_points["camber"], dtype=np.float32),
        "thickness": np.asarray(wall_points["thickness"], dtype=np.float32),
    }
    wall_outvar = {
        "velocity_wall_normal_wf": np.zeros_like(wall_x),
        "velocity_wall_parallel_wf": np.zeros_like(wall_x),
        "k_wf": np.zeros_like(wall_x),
        "ep_wf": np.zeros_like(wall_x),
        "wall_shear_stress_x_wf": np.zeros_like(wall_x),
        "wall_shear_stress_y_wf": np.zeros_like(wall_x),
    }
    wall_constraint = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=wall_invar,
        outvar=wall_outvar,
        batch_size=cfg.batch_size.airfoil,
        lambda_weighting={
            "velocity_wall_normal_wf": np.full_like(wall_x, 100.0),
            "velocity_wall_parallel_wf": np.full_like(wall_x, 100.0),
            # k_wf is retained for compatibility; in this formulation it is an
            # algebraic identity and therefore is not an independent condition.
            "k_wf": np.full_like(wall_x, 0.0),
            "ep_wf": np.full_like(wall_x, 1.0),
            "wall_shear_stress_x_wf": np.full_like(wall_x, 100.0),
            "wall_shear_stress_y_wf": np.full_like(wall_x, 100.0),
        },
    )
    domain.add_constraint(wall_constraint, "AirfoilWallFunction")

    # Apply the equations over the near field while excluding the unresolved
    # wall layer (sdf <= delta_y) through zero loss weights.
    _add_zero_equation_constraint(
        domain=domain,
        nodes=nodes,
        geometry=near_geometry,
        name="InteriorNearField",
        batch_size=cfg.batch_size.interior_near_field,
        parameterization=parameterization,
        sdf_wall_distance=delta_y,
    )

    far_batch = max(256, int(cfg.batch_size.interior_far_field / len(far_geometries)))
    for i, far_geometry in enumerate(far_geometries):
        _add_zero_equation_constraint(
            domain=domain,
            nodes=nodes,
            geometry=far_geometry,
            name=f"InteriorFarField_{i}",
            batch_size=far_batch,
            parameterization=parameterization,
        )

    # Fixed inference points are only for monitoring and output, not training.
    inference_invar = custom_geometry.sample_interior(
        n_points=5000,
        rand_seed=123,
        parameterization=parameterization,
    )
    domain.add_inferencer(
        PointwiseInferencer(
            nodes=nodes,
            invar=inference_invar,
            output_names=["u", "v", "p", "k", "ep", "u_tau"],
            batch_size=4096,
        ),
        "AirfoilNearFieldInference",
    )

    print(f"Re = {reynolds:.6g}")
    print(f"Estimated delta_y for target mean y+ = {target_y_plus:g}: {delta_y:.8e} m")
    print(f"Inlet k = {k_inlet:.8e} m^2/s^2")
    print(f"Inlet epsilon = {ep_inlet:.8e} m^2/s^3")

    solver = Solver(cfg, domain)
    solver.solve()


if __name__ == "__main__":
    run()
