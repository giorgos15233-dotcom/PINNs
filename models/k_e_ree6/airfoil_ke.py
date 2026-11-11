# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sympy import Symbol, Or, Max, Min, exp, log

import physicsnemo.sym
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Line, Channel2D
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.geometry import Parameterization, Parameter
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.utils.io import InferencerPlotter

from custom_airfoil_geometry import AirfoilInChannel
# Importing the custom k-epsilon models and initialization
from custom_k_ep_ls import kEpsilonInit, kEpsilon

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    ##########
    # Geometry
    ##########

    x, y = Symbol("x"), Symbol("y")  # Physical coordinates

    # A range of min/max values for the airfoil parameters
    # fmt: off
    parameterization = Parameterization({
        Parameter("alpha"): (0.0, 0.0),     # Angle of attack, in radians
        Parameter("camber"): (0.0, 0.0),     # Airfoil camber, as a fraction of chord
        Parameter("thickness"): (0.12, 0.12),  # Airfoil thickness, as a fraction of chord
    })
    # fmt: on

    channel_length = 15.0 / 2
    channel_height = 10.0 / 2
    x_fraction = 0.3

    channel = Channel2D(
        (-channel_length * x_fraction, -channel_height / 2),
        (channel_length * (1 - x_fraction), channel_height / 2),
    )
    inlet_geom = Line(
        (-channel_length * x_fraction, -channel_height / 2),
        (-channel_length * x_fraction, channel_height / 2),
        normal=1,
    )
    outlet_geom = Line(
        (channel_length * (1 - x_fraction), -channel_height / 2),
        (channel_length * (1 - x_fraction), channel_height / 2),
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

    ###########
    # Equations
    ###########
    u_inlet = 1.0
    chord = 1.0
    reynolds_number = 1e6
    nu = u_inlet * chord / reynolds_number  # Kinematic viscosity

    # Initialize the RANS k-epsilon turbulence model and its initialization phase
    k_ep_init = kEpsilonInit(nu=nu, rho=1.0)
    k_ep_model = kEpsilon(nu=nu, rho=1.0)

    ###########
    # Networks
    ###########
    # Decoupled architecture: Separate networks for velocity, pressure, and turbulence variables.
    # This prevents the different scales and gradients from interfering with each other (feature bleed).
    input_keys_list = [Key("x"), Key("y"), Key("alpha"), Key("camber"), Key("thickness")]

    flow_net = instantiate_arch(
        input_keys=input_keys_list,
        output_keys=[Key("u"), Key("v")],
        cfg=cfg.arch.fully_connected,
    )
    p_net = instantiate_arch(
        input_keys=input_keys_list,
        output_keys=[Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    
    # Note: Networks output k_star and ep_star. Positivity constraints are applied below.
    k_net = instantiate_arch(
        input_keys=input_keys_list,
        output_keys=[Key("k_star")],
        cfg=cfg.arch.fully_connected,
    )
    ep_net = instantiate_arch(
        input_keys=input_keys_list,
        output_keys=[Key("ep_star")],
        cfg=cfg.arch.fully_connected,
    )

    #############
    # Nodes
    #############
    # Unroll the computational graph. Apply the Softplus transform to k_star and ep_star
    # to strictly enforce positive values for k and ep (critical for k-epsilon stability).
    nodes: list = (
        k_ep_init.make_nodes()
        + k_ep_model.make_nodes()
        + [Node.from_sympy(Min(log(1 + exp(Symbol("k_star"))) + 1e-4, 20), "k")]
        + [Node.from_sympy(Min(log(1 + exp(Symbol("ep_star"))) + 1e-4, 180), "ep")]
        + [flow_net.make_node(name="flow_network")]
        + [p_net.make_node(name="p_network")]
        + [k_net.make_node(name="k_network")]
        + [ep_net.make_node(name="ep_network")]
    )

    #############
    # Constraints
    #############

    domain = Domain()

    # ==========================================
    # Calculate inlet freestream conditions for k and ep
    # Using Viscosity Ratio (R_T) approach for consistent physics across Reynolds numbers
    # ==========================================
    I = 0.001  # Turbulence Intensity of 0.1% (realistic for external flow/wind tunnels)
    R_T = 10.0 # Turbulent Viscosity Ratio (mu_t / mu). Best practice: 1 to 10
    C_mu = 0.09 # Empirical constant from the k-epsilon model, used in calculating ep from k and R_T, Launder & Spalding (1974)

    # 1. Calculate free-stream turbulent kinetic energy (k)
    k_inlet_val = 1.5 * (u_inlet * I) ** 2

    # 2. Calculate free-stream dissipation rate (ep) based on desired R_T and current nu
    # Since nu_t = R_T * nu, and nu_t = C_mu * (k^2 / ep)
    # Therefore: ep = C_mu * k^2 / (R_T * nu)
    ep_inlet_val = C_mu * (k_inlet_val ** 2) / (R_T * nu)

    # Inlet
    inlet_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_geom,
        outvar={"u": u_inlet, "v": 0, "k": k_inlet_val, "ep": ep_inlet_val},
        batch_size=cfg.batch_size.inlet,
        parameterization=parameterization,
    )
    domain.add_constraint(inlet_bc, "inlet")

    # Outlet
    outlet_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_geom,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        parameterization=parameterization,
    )
    domain.add_constraint(outlet_bc, "outlet")

    # Freestream (channel top and bottom)
    top_bot_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={"u": u_inlet, "v": 0, "k": k_inlet_val, "ep": ep_inlet_val},
        batch_size=cfg.batch_size.top_bot,
        parameterization=parameterization,
    )
    domain.add_constraint(top_bot_bc, "top_bot")

    # No-slip condition on the airfoil surface (Low-Re / Wall-resolving approach)
    airfoil_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=volume_near_field,
        outvar={"u": 0, "v": 0, "k": 0},
        batch_size=cfg.batch_size.airfoil,
        parameterization=parameterization,
    )
    domain.add_constraint(airfoil_bc, "airfoil")

    # Interior initialization constraint (helps the RANS model find a stable starting field)
    interior_init = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=volume_near_field,
        outvar={"u_init": 0, "v_init": 0, "p_init": 0, "k_init": 0, "ep_init": 0},
        batch_size=cfg.batch_size.interior_near_field,
        parameterization=parameterization,
    )
    domain.add_constraint(interior_init, "interior_init")

    # Interior constraints for the far-field
    interior_far_field = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "k_equation": 0, "ep_equation": 0},
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

    # Interior constraints for the near-field (with SDF weighting)
    interior_near_field = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=volume_near_field,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "k_equation": 0, "ep_equation": 0},
        batch_size=cfg.batch_size.interior_near_field,
        lambda_weighting={
            "continuity": 2 * Max(Symbol("sdf"), 0),
            "momentum_x": 2 * Max(Symbol("sdf"), 0),
            "momentum_y": 2 * Max(Symbol("sdf"), 0),
            "k_equation": 2 * Max(Symbol("sdf"), 0),
            "ep_equation": 2 * Max(Symbol("sdf"), 0),
        },
        parameterization=parameterization,
    )
    domain.add_constraint(interior_near_field, "interior_near_field")

    ##############
    # Inferencers
    ##############
    inferencer_id: int = 0

    def add_inferencer(
        alpha_inference: float = 0.0,
        camber_inference: float = 0.0,
        thickness_inference: float = 0.12,
        n_interior_points: int = 20000,
        n_boundary_points: int = 512,
    ) -> None:
        """
        Adds an inferencer to the domain.
        """
        nonlocal inferencer_id

        parameterization = Parameterization(
            {
                Parameter("alpha"): (alpha_inference, alpha_inference),
                Parameter("camber"): (camber_inference, camber_inference),
                Parameter("thickness"): (thickness_inference, thickness_inference),
            }
        )

        sample_near_field = volume_near_field.sample_interior(
            n_interior_points,
            rand_seed=inferencer_id,
            parameterization=parameterization,
        )
        sample_boundary = volume_near_field.sample_boundary(
            n_boundary_points,
            rand_seed=inferencer_id,
            parameterization=parameterization,
        )
        sample_far_field = channel.sample_interior(
            n_interior_points,
            quasirandom=True,
            parameterization=parameterization,
        )
        
        # Output names updated to record turbulent kinetics
        grid_inference = PointwiseInferencer(
            nodes=nodes,
            invar={
                "x": sample_near_field["x"],
                "y": sample_near_field["y"],
                "alpha": alpha_inference * np.ones_like(sample_near_field["y"]),
                "camber": camber_inference * np.ones_like(sample_near_field["y"]),
                "thickness": thickness_inference * np.ones_like(sample_near_field["y"]),
            },
            output_names=["u", "v", "p", "k", "ep"],
            batch_size=n_interior_points,
            plotter=InferencerPlotter(),
        )
        domain.add_inferencer(grid_inference, f"near_field_{inferencer_id + 1}")

        grid_inference = PointwiseInferencer(
            nodes=nodes,
            invar={
                "x": sample_boundary["x"],
                "y": sample_boundary["y"],
                "alpha": alpha_inference * np.ones_like(sample_boundary["y"]),
                "camber": camber_inference * np.ones_like(sample_boundary["y"]),
                "thickness": thickness_inference * np.ones_like(sample_boundary["y"]),
            },
            output_names=["u", "v", "p", "k", "ep"],
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
            output_names=["u", "v", "p", "k", "ep"],
            batch_size=n_interior_points,
            plotter=InferencerPlotter(),
        )
        domain.add_inferencer(grid_inference, f"far_field_{inferencer_id + 1}")

        inferencer_id += 1

    add_inferencer(alpha_inference=0.0, camber_inference=0.0, thickness_inference=0.12)

    # Random inferencers
    rng = np.random.default_rng(42)

    for i in range(6):
        alpha_inference = rng.uniform(0.0, 0.0)
        camber_inference = rng.uniform(0.0, 0.0)
        thickness_inference = rng.uniform(0.12, 0.12)
        add_inferencer(
            alpha_inference=alpha_inference,
            camber_inference=camber_inference,
            thickness_inference=thickness_inference,
        )

    ##############
    # Solver
    #############
    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()
