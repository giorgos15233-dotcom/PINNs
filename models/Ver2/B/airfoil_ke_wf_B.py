# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""PINN solver for a 2D airfoil using steady incompressible RANS k-epsilon.

The Launder-Spalding wall-function residuals are evaluated at points displaced
from the airfoil into the fluid. Interior PDE losses in the near-wall layer are
suppressed using the same wall distance.
"""

import numpy as np
from sympy import Eq, Function, Max, Min, Or, Symbol, exp, log, sqrt

import physicsnemo.sym
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.geometry import Parameter, Parameterization
from physicsnemo.sym.geometry.primitives_2d import Channel2D, Line
from physicsnemo.sym.hydra import PhysicsNeMoConfig, instantiate_arch
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.utils.io import InferencerPlotter

from custom_airfoil_geometry import AirfoilInChannel
from custom_k_ep_ls import kEpsilon, kEpsilonLSWF


C_MU = 0.09
TURBULENCE_INTENSITY = 1.0e-3
TURBULENT_VISCOSITY_RATIO = 10.0

# Set True only when an additional off-wall diagnostic inferencer is needed.
# The default preserves the original 28 inferencer names and output paths.
ENABLE_WALL_DIAGNOSTICS = False


def naca0012_inward_normals(wall_x, wall_y, chord=1.0, thickness_ratio=0.12):
    """Return unit normals pointing into the solid for a closed NACA 00xx airfoil.

    This matches the analytic NACA 4-digit thickness polynomial used by the
    original AirfoilInChannel geometry. It is valid for camber=0 and alpha=0.
    The geometry sampler may supply vertical normals; those must not be used
    for the curved-wall shear and displacement calculations.
    """
    x_array = np.asarray(wall_x, dtype=np.float64)
    wall_y = np.asarray(wall_y, dtype=np.float64)
    if x_array.shape != wall_y.shape:
        raise ValueError("wall_x and wall_y must have the same shape")
    x_c = x_array / chord
    if not np.all(np.isfinite(x_c)) or np.any(x_c < 0.0) or np.any(x_c > 1.0):
        raise ValueError("Wall samples are outside the NACA airfoil chord")

    x_safe = np.clip(x_c, 1.0e-7, 1.0)
    yt = 5.0 * thickness_ratio * chord * (
        0.2969 * np.sqrt(x_c)
        - 0.1260 * x_c
        - 0.3516 * x_c**2
        + 0.2843 * x_c**3
        - 0.1036 * x_c**4
    )
    max_error = float(np.max(np.abs(np.abs(wall_y) - yt)))
    if max_error > 1.0e-4 * chord:
        raise ValueError(
            f"Wall samples do not fit the closed symmetric NACA profile "
            f"(max thickness mismatch={max_error:g}). "
            "Check alpha/camber and the geometry thickness polynomial."
        )

    slope = 5.0 * thickness_ratio * (
        0.2969 / (2.0 * np.sqrt(x_safe))
        - 0.1260
        - 0.7032 * x_c
        + 0.8529 * x_c**2
        - 4.0 * 0.1036 * x_c**3
    )
    side = np.where(wall_y >= 0.0, 1.0, -1.0)
    magnitude = np.sqrt(slope**2 + 1.0)
    # The inward normal has the same x-component on both surfaces.
    nx_inward = slope / magnitude
    ny_inward = -side / magnitude
    # Preserve the original (N, 1) or (N,) shape to avoid NumPy broadcasting
    # to an accidental (N, N) array when constructing the shifted P points.
    return nx_inward.astype(np.float32), ny_inward.astype(np.float32)


def calculate_delta_y(
    u_inf: float,
    chord: float,
    nu: float,
    target_y_plus: float = 80.0,
    thickness_ratio: float = 0.12,
) -> float:
    """Estimate a constant first-cell wall distance for a target mean y+.

    The estimate uses an average flat-plate skin-friction correlation with an
    airfoil form-factor correction. Local y+ will vary along the airfoil.
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


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    ##########
    # Geometry
    ##########

    x, y = Symbol("x"), Symbol("y")  # Physical coordinates

    # A range of min/max values for the airfoil parameters
    # fmt: off
    parameterization = Parameterization({
        Parameter("alpha"): (0.0, 0.0),       # Angle of attack, in radians
        Parameter("camber"): (0.0, 0.0),      # Airfoil camber, as a fraction of chord
        Parameter("thickness"): (0.12, 0.12), # Airfoil thickness, as a fraction of chord
    })
    # fmt: on

    # Full computational channel; retain the original Channel2D representation.
    channel_length = 15.0 / 2
    channel_height = 10.0 / 2
    x_fraction = 0.3
    channel_x_min = -channel_length * x_fraction
    channel_x_max = channel_length * (1 - x_fraction)
    channel_y_min = -channel_height / 2
    channel_y_max = channel_height / 2

    channel = Channel2D(
        (channel_x_min, channel_y_min),
        (channel_x_max, channel_y_max),
    )
    inlet_geom = Line(
        (channel_x_min, channel_y_min),
        (channel_x_min, channel_y_max),
        normal=1,
    )
    outlet_geom = Line(
        (channel_x_max, channel_y_min),
        (channel_x_max, channel_y_max),
        normal=1,
    )

    near_field_channel_length = 1.6
    near_field_channel_height = 1.0
    x_fraction_near_field = 0.2
    volume_near_field = AirfoilInChannel(
        x_min=-near_field_channel_length * x_fraction_near_field,
        y_min=-near_field_channel_height / 2,
        x_max=near_field_channel_length * (1 - x_fraction_near_field),
        y_max=near_field_channel_height / 2,
        params=parameterization,
        include_channel_boundary=False,
    )

    custom_channel_length = 1.2
    custom_channel_height = 0.4
    x_fraction_custom = 0.2
    volume_custom_field = AirfoilInChannel(
        x_min=-custom_channel_length * x_fraction_custom,
        y_min=-custom_channel_height / 2,
        x_max=custom_channel_length * (1 - x_fraction_custom),
        y_max=custom_channel_height / 2,
        params=parameterization,
        include_channel_boundary=False,
    )

    ###########
    # Equations
    ###########
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
    print(f"Computed Target Delta_Y for Wall Functions (Re={reynolds:e}): {delta_y:e} m")

    k_inlet = 1.5 * (u_inf * TURBULENCE_INTENSITY) ** 2
    ep_inlet = C_MU * k_inlet**2 / (TURBULENT_VISCOSITY_RATIO * nu)

    k_ep_model = kEpsilon(nu=nu, rho=rho)
    k_ep_wf = kEpsilonLSWF(nu=nu, rho=rho)

    ###########
    # Networks
    ###########
    input_keys_list = [
        Key("x"),
        Key("y"),
        Key("alpha"),
        Key("camber"),
        Key("thickness"),
    ]

    flow_net = instantiate_arch(
        input_keys=input_keys_list,
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
        input_keys=input_keys_list,
        output_keys=[Key("k_star")],
        frequencies=("axis", [i / 2.0 for i in range(8)]),
        frequencies_params=("axis", [i / 2.0 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )
    ep_net = instantiate_arch(
        input_keys=input_keys_list,
        output_keys=[Key("ep_star")],
        frequencies=("axis", [i / 2.0 for i in range(8)]),
        frequencies_params=("axis", [i / 2.0 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )

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

    #############
    # Constraints
    #############
    domain = Domain()

    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_geom,
        outvar={"u": u_inf, "v": 0.0, "k": k_inlet, "ep": ep_inlet},
        lambda_weighting={"u": 10.0, "v": 10.0, "k": 1.0, "ep": 1.0},
        batch_size=cfg.batch_size.inlet,
        parameterization=parameterization,
    )
    domain.add_constraint(inlet, "inlet")

    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_geom,
        outvar={"p": 0.0},
        lambda_weighting={"p": 10.0},
        batch_size=cfg.batch_size.outlet,
        parameterization=parameterization,
    )
    domain.add_constraint(outlet, "outlet")

    # Keep the original single top_bot constraint name, but sample only the
    # horizontal channel edges and impose symmetry/slip conditions there.
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
    top_bot_outvar = {
        "v": 0.0,
        "u_y_bc": 0.0,
        "p_y_bc": 0.0,
        "k_y_bc": 0.0,
        "ep_y_bc": 0.0,
    }
    top_bot = PointwiseBoundaryConstraint(
        nodes=nodes + derivative_nodes,
        geometry=channel,
        criteria=Or(Eq(y, channel_y_max), Eq(y, channel_y_min)),
        outvar=top_bot_outvar,
        lambda_weighting={key: 1.0 for key in top_bot_outvar},
        batch_size=cfg.batch_size.top_bot,
        parameterization=parameterization,
    )
    domain.add_constraint(top_bot, "top_bot")

    # Wall-function points are displaced from the physical airfoil boundary
    # into the fluid. The constraint keeps its original title: "airfoil".
    wall_points = volume_near_field.sample_boundary(
        n_points=cfg.batch_size.airfoil,
        rand_seed=23,
        parameterization=parameterization,
    )
    wall_x = np.asarray(wall_points["x"], dtype=np.float32)
    wall_y = np.asarray(wall_points["y"], dtype=np.float32)
    # Analytical normals below assume NACA0012, alpha=0, camber=0, chord=1.
    # Validate them against the sampled profile before displacing the wall points.
    # The original AirfoilInChannel sampler gives vertical NACA wall normals.
    # Compute the curved NACA0012 normals while preserving the original sample points.
    wall_nx, wall_ny = naca0012_inward_normals(wall_x, wall_y, chord=chord)
    normal_lengths = np.sqrt(wall_nx**2 + wall_ny**2)
    if not np.allclose(normal_lengths, 1.0, atol=1.0e-6):
        raise ValueError("Airfoil surface normals are not unit length")
    if np.max(np.abs(wall_nx)) < 0.05:
        raise ValueError("Airfoil wall normals are still vertical")
    print(
        "Wall normals: nx=[%.5f, %.5f], ny=[%.5f, %.5f]"
        % (wall_nx.min(), wall_nx.max(), wall_ny.min(), wall_ny.max())
    )

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
    airfoil_wf = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=wall_invar,
        outvar=wall_outvar,
        batch_size=cfg.batch_size.airfoil,
        lambda_weighting={
            "velocity_wall_normal_wf": np.full_like(wall_x, 100.0),
            "velocity_wall_parallel_wf": np.full_like(wall_x, 100.0),
            # k_wf is an algebraic identity with u_tau=C_mu**(1/4)*sqrt(k).
            "k_wf": np.full_like(wall_x, 0.0),
            "ep_wf": np.full_like(wall_x, 1.0),
            "wall_shear_stress_x_wf": np.full_like(wall_x, 100.0),
            "wall_shear_stress_y_wf": np.full_like(wall_x, 100.0),
        },
    )
    domain.add_constraint(airfoil_wf, "airfoil")

    # Optional diagnostics at exactly the same off-wall points P. The pressure
    # is available from pressure_network; no pressure wall BC is imposed.
    # This block adds no outputs by default, preserving existing automations.
    if ENABLE_WALL_DIAGNOSTICS:
        diagnostic_nodes = nodes + [
            Node.from_sympy(
                Symbol("u_tau") * Symbol("normal_distance") / nu,
                "y_plus_wall",
            ),
            Node.from_sympy(
                Symbol("u_tau") - C_MU**0.25 * sqrt(Symbol("k")),
                "u_tau_consistency",
            ),
        ]
        domain.add_inferencer(
            PointwiseInferencer(
                nodes=diagnostic_nodes,
                invar=wall_invar,
                output_names=[
                    "u", "v", "p", "k", "ep", "u_tau", "y_plus_wall",
                    "u_tau_consistency", "velocity_wall_normal_wf",
                    "velocity_wall_parallel_wf", "ep_wf",
                    "wall_shear_stress_x_wf", "wall_shear_stress_y_wf",
                ],
                batch_size=cfg.batch_size.airfoil,
                plotter=InferencerPlotter(),
            ),
            "AirfoilWallAtP",
        )

    # Interior PDE equations in the far field: one Channel2D constraint, as
    # in the original script. The Or criteria remove the near-field rectangle.
    interior_far_field = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={
            "continuity": 0,
            "momentum_x": 0,
            "momentum_y": 0,
            "k_equation": 0,
            "ep_equation": 0,
        },
        batch_size=cfg.batch_size.interior_far_field,
        criteria=Or(
            x <= -near_field_channel_length * x_fraction_near_field,
            x >= near_field_channel_length * (1 - x_fraction_near_field),
            y <= -near_field_channel_height / 2,
            y >= near_field_channel_height / 2,
        ),
        parameterization=parameterization,
    )
    domain.add_constraint(interior_far_field, "interior_far_field")

    # Interior PDE equations in the near field. Suppress their losses within
    # delta_y of the airfoil, where the wall-function model replaces wall-resolved
    # equations.
    interior_near_field = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=volume_near_field,
        outvar={
            "continuity": 0,
            "momentum_x": 0,
            "momentum_y": 0,
            "k_equation": 0,
            "ep_equation": 0,
        },
        batch_size=cfg.batch_size.interior_near_field,
        lambda_weighting={
            # Zero in the unresolved wall layer, increasing to one at 2*delta_y.
            key: Min(Max((Symbol("sdf") - delta_y) / delta_y, 0.0), 1.0)
            for key in ("continuity", "momentum_x", "momentum_y", "k_equation", "ep_equation")
        },
        parameterization=parameterization,
    )
    domain.add_constraint(interior_near_field, "interior_near_field")

    ##############
    # Inferencers
    ##############
    # Preserve the original inferencer names, sampling geometries, output fields,
    # and plotter so existing post-processing/automation paths remain compatible.
    inferencer_id: int = 0

    def add_inferencer(
        alpha_inference: float = 0.0,
        camber_inference: float = 0.0,
        thickness_inference: float = 0.12,
        n_interior_points: int = 20000,
        n_custom_points: int = 300000,
        n_boundary_points: int = 512,
    ) -> None:
        nonlocal inferencer_id

        inference_parameterization = Parameterization(
            {
                Parameter("alpha"): (alpha_inference, alpha_inference),
                Parameter("camber"): (camber_inference, camber_inference),
                Parameter("thickness"): (thickness_inference, thickness_inference),
            }
        )

        sample_near_field = volume_near_field.sample_interior(
            n_interior_points,
            rand_seed=inferencer_id,
            parameterization=inference_parameterization,
        )
        sample_boundary = volume_near_field.sample_boundary(
            n_boundary_points,
            rand_seed=inferencer_id,
            parameterization=inference_parameterization,
        )
        # Keep boundary_N locations and output fields unchanged, but correct
        # the normals reported for the curved NACA0012 surface.
        sample_boundary["normal_x"], sample_boundary["normal_y"] = (
            naca0012_inward_normals(
                sample_boundary["x"], sample_boundary["y"], chord=chord
            )
        )
        sample_far_field = channel.sample_interior(
            n_interior_points,
            quasirandom=True,
            parameterization=inference_parameterization,
        )
        sample_custom_field = volume_custom_field.sample_interior(
            n_custom_points,
            rand_seed=inferencer_id,
            parameterization=inference_parameterization,
        )

        grid_inference = PointwiseInferencer(
            nodes=nodes,
            invar={
                "x": sample_custom_field["x"],
                "y": sample_custom_field["y"],
                "alpha": alpha_inference * np.ones_like(sample_custom_field["y"]),
                "camber": camber_inference * np.ones_like(sample_custom_field["y"]),
                "thickness": thickness_inference * np.ones_like(sample_custom_field["y"]),
                "sdf": sample_custom_field["sdf"],
            },
            output_names=["u", "v", "p", "k", "ep", "u_tau", "sdf"],
            batch_size=n_custom_points,
            plotter=InferencerPlotter(),
        )
        domain.add_inferencer(grid_inference, f"custom_tight_field_{inferencer_id + 1}")

        grid_inference = PointwiseInferencer(
            nodes=nodes,
            invar={
                "x": sample_near_field["x"],
                "y": sample_near_field["y"],
                "alpha": alpha_inference * np.ones_like(sample_near_field["y"]),
                "camber": camber_inference * np.ones_like(sample_near_field["y"]),
                "thickness": thickness_inference * np.ones_like(sample_near_field["y"]),
                "sdf": sample_near_field["sdf"],
            },
            output_names=["u", "v", "p", "k", "ep", "u_tau", "sdf"],
            batch_size=n_interior_points,
            plotter=InferencerPlotter(),
        )
        domain.add_inferencer(grid_inference, f"near_field_{inferencer_id + 1}")

        grid_inference = PointwiseInferencer(
            nodes=nodes,
            invar={
                "x": sample_boundary["x"],
                "y": sample_boundary["y"],
                "normal_x": sample_boundary["normal_x"],
                "normal_y": sample_boundary["normal_y"],
                "alpha": alpha_inference * np.ones_like(sample_boundary["y"]),
                "camber": camber_inference * np.ones_like(sample_boundary["y"]),
                "thickness": thickness_inference * np.ones_like(sample_boundary["y"]),
            },
            output_names=["u", "v", "p", "k", "ep", "u_tau", "normal_x", "normal_y"],
            batch_size=n_boundary_points,
            plotter=InferencerPlotter(),
        )
        domain.add_inferencer(grid_inference, f"boundary_{inferencer_id + 1}")

        grid_inference = PointwiseInferencer(
            nodes=nodes,
            invar={
                "x": sample_far_field["x"],
                "y": sample_far_field["y"],
                "alpha": alpha_inference * np.ones_like(sample_far_field["y"]),
                "camber": camber_inference * np.ones_like(sample_far_field["y"]),
                "thickness": thickness_inference * np.ones_like(sample_far_field["y"]),
            },
            output_names=["u", "v", "p", "k", "ep", "u_tau"],
            batch_size=n_interior_points,
            plotter=InferencerPlotter(),
        )
        domain.add_inferencer(grid_inference, f"far_field_{inferencer_id + 1}")

        inferencer_id += 1

    add_inferencer(alpha_inference=0.0, camber_inference=0.0, thickness_inference=0.12)

    # Keep the original seven inference sets (one fixed set and six repeated
    # parameter sets); each run receives a distinct sampling seed.
    rng = np.random.default_rng(42)
    for _ in range(6):
        alpha_inference = rng.uniform(0.0, 0.0)
        camber_inference = rng.uniform(0.0, 0.0)
        thickness_inference = rng.uniform(0.12, 0.12)
        add_inferencer(
            alpha_inference=alpha_inference,
            camber_inference=camber_inference,
            thickness_inference=thickness_inference,
        )

    print(f"Re = {reynolds:.6g}")
    print(f"Estimated delta_y for target mean y+ = {target_y_plus:g}: {delta_y:.8e} m")
    print(f"Inlet k = {k_inlet:.8e} m^2/s^2")
    print(f"Inlet epsilon = {ep_inlet:.8e} m^2/s^3")

    solver = Solver(cfg, domain)
    solver.solve()


if __name__ == "__main__":
    run()
