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

import torch
import math
import numpy as np
from sympy import Symbol, Or, Max, Min, exp, log, Number, sqrt
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

# Importing the Standard Wall Function class alongside the initialization and base k-epsilon PDEs
from custom_k_ep_ls import kEpsilonInit, kEpsilon, kEpsilonLSWF

class NormalDistanceEvaluator(torch.nn.Module):
    def __init__(self, dy):
        super().__init__()
        self.dy = dy
            
    def forward(self, invar):
        return {"normal_distance": torch.full_like(invar["x"], self.dy)}


def calculate_delta_y(u_inf: float, chord: float, nu: float, target_y_plus: float = 30.0, thickness_ratio: float = 0.12) -> float:
    """
    Calculates the physical wall distance (Delta Y) required to hit a target y+
    using empirical skin friction relationships for flat plates and the Hoerner form factor.
    """
    reynolds = (u_inf * chord) / nu
    
    # 1. Evaluate Flat Plate Skin Friction Coefficient (Cf)
    if reynolds < 5e5:
        # Laminar (Blasius)
        cf_flat = 1.328 / math.sqrt(reynolds)
    elif 5e5 <= reynolds <= 1e7:
        # Transitional (Mixed Laminar/Turbulent)
        cf_flat = (0.074 / (reynolds ** 0.2)) - (1742.0 / reynolds)
    else:
        # Fully Turbulent (Schlichting Log-Law)
        cf_flat = 0.455 / (math.log10(reynolds) ** 2.58)
        
    # 2. Apply Hoerner Form Factor for Airfoil Geometry
    form_factor = 1.0 + 2.0 * thickness_ratio + 60.0 * (thickness_ratio ** 4)
    cf_airfoil = cf_flat * form_factor
    
    # 3. Calculate Friction Velocity (u_tau)
    u_tau_approx = u_inf * math.sqrt(cf_airfoil / 2.0)
    
    # 4. Calculate required physical distance (Delta y)
    delta_y = (target_y_plus * nu) / u_tau_approx
    
    return float(delta_y)



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

    custom_channel_length = 1.2
    custom_channel_height = 0.4
    x_fraction_custom = 0.2  # Το αφήνουμε 0.2 για να έχει 20% του μήκους μπροστά από την αεροτομή

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
    u_inlet = 1.0
    chord = 1.0
    reynolds_number = 1e6 # Will scale from 1e6 to 1e7 for <1e6, seperation bubbles and transitional detatchments dont work with y+ > 1
    nu = u_inlet * chord / reynolds_number  # Kinematic viscosity

    # Initialize the RANS k-epsilon models, including the Wall Functions PDE
    k_ep_init = kEpsilonInit(nu=nu, rho=1.0)
    k_ep_model = kEpsilon(nu=nu, rho=1.0)
    k_ep_wf = kEpsilonLSWF(nu=nu, rho=1.0)


    # 1. Calculate required physical distance for y+ = 30
    delta_y = calculate_delta_y(u_inlet, chord, nu, target_y_plus=80.0, thickness_ratio=0.12)
    print(f"Computed Target Delta_Y for Wall Functions (Re={reynolds_number:e}): {delta_y:e} m")

    ###########
    # Networks
    ###########
    # Decoupled architecture: Separate networks for velocity, pressure, turbulence, and friction velocity.
    input_keys_list = [Key("x"), Key("y"), Key("alpha"), Key("camber"), Key("thickness")]

    flow_net = instantiate_arch(
        input_keys=input_keys_list,
        output_keys=[Key("u"), Key("v")],
        cfg=cfg.arch.fourier,
    )
    p_net = instantiate_arch(
        input_keys=input_keys_list,
        output_keys=[Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    k_net = instantiate_arch(
        input_keys=input_keys_list,
        output_keys=[Key("k_star")],
        cfg=cfg.arch.fourier,
    )
    ep_net = instantiate_arch(
        input_keys=input_keys_list,
        output_keys=[Key("ep_star")],
        cfg=cfg.arch.fourier,
    )


    normal_dist_node = Node(
        inputs=["x"], 
        outputs=["normal_distance"], 
        evaluate=NormalDistanceEvaluator(delta_y), 
        name="normal_distance_node"
    )
    

    #############
    # Nodes
    #############
    # Unroll the computational graph including all PDEs and positivity constraints.
    nodes: list = (
        k_ep_init.make_nodes()
        + k_ep_model.make_nodes()
        + k_ep_wf.make_nodes()
        + [Node.from_sympy(Min(log(1 + exp(Symbol("k_star"))) + 1e-4, 20), "k")]
        + [Node.from_sympy(Min(log(1 + exp(Symbol("ep_star"))) + 1e-4, 180), "ep")]
        + [flow_net.make_node(name="flow_network")]
        + [p_net.make_node(name="p_network")]
        + [k_net.make_node(name="k_network")]
        + [ep_net.make_node(name="ep_network")]
        + [Node.from_sympy((0.09**0.25) * sqrt(Max(Symbol("k"), 0.0)), "u_tau")]
        + [normal_dist_node]
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
    C_mu = 0.09 # Empirical constant from the k-epsilon model

    # 1. Calculate free-stream turbulent kinetic energy (k)
    k_inlet_val = 1.5 * (u_inlet * I) ** 2

    # 2. Calculate free-stream dissipation rate (ep)
    ep_inlet_val = C_mu * (k_inlet_val ** 2) / (R_T * nu)

    # Inlet
    inlet_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_geom,
        outvar={"u": u_inlet, "v": 0, "k": k_inlet_val, "ep": ep_inlet_val},
        lambda_weighting={"u": 10, "v": 10, "k": 1, "ep": 1},
        batch_size=cfg.batch_size.inlet,
        parameterization=parameterization,
    )
    domain.add_constraint(inlet_bc, "inlet")

    # Outlet
    outlet_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_geom,
        outvar={"p": 0},
        lambda_weighting={"p": 10},
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

    # ==========================================
    # Wall Function Boundary Condition
    # ==========================================
      
    # 2. Create specific parameterization that tricks the PDE into evaluating at offset delta_y
    wf_parameterization = Parameterization({
        Parameter("alpha"): (0.0, 0.0),
        Parameter("camber"): (0.0, 0.0),
        Parameter("thickness"): (0.12, 0.12),
        Parameter("normal_distance"): (delta_y, delta_y),
    })

    # 3. Apply Wall Functions to the Airfoil Surface
    airfoil_wf = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=volume_near_field,
        outvar={
            "velocity_wall_normal_wf": 0,
            "velocity_wall_parallel_wf": 0,
            "ep_wf": 0,
            "wall_shear_stress_x_wf": 0,
            "wall_shear_stress_y_wf": 0,
        },
        lambda_weighting={
            "velocity_wall_normal_wf": 100,
            "velocity_wall_parallel_wf": 100,
            "ep_wf": 1,
            "wall_shear_stress_x_wf": 100,
            "wall_shear_stress_y_wf": 100,
        },
        batch_size=cfg.batch_size.airfoil,
        parameterization=parameterization,
    )
    domain.add_constraint(airfoil_wf, "airfoil")

    # Interior initialization constraint
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
        lambda_weighting={
            "continuity": 100,
            "momentum_x": 1000,
            "momentum_y": 1000,
            "k_equation": 10,
            "ep_equation": 1,
        },
        criteria=Or(
            x <= -near_field_channel_length * x_fraction_near_field,
            x >= near_field_channel_length * (1 - x_fraction_near_field),
            y <= -near_field_channel_height / 2,
            y >= near_field_channel_height / 2,
        ),
        parameterization=parameterization,
    )
    domain.add_constraint(interior_far_field, "interior_far_field")

    # Interior constraints for the near-field
    interior_near_field = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=volume_near_field,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "k_equation": 0, "ep_equation": 0},
        batch_size=cfg.batch_size.interior_near_field,
        lambda_weighting={
            "continuity": 100 * Max(Symbol("sdf"), 0),
            "momentum_x": 1000 * Max(Symbol("sdf"), 0),
            "momentum_y": 1000 * Max(Symbol("sdf"), 0),
            "k_equation": 10 * Max(Symbol("sdf"), 0),
            "ep_equation": 1 * Max(Symbol("sdf"), 0),
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
        n_custom_points = 300000,
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
        
        sample_custom_field = volume_custom_field.sample_interior(
            n_custom_points,
            rand_seed=inferencer_id,
            parameterization=parameterization,
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
            output_names=["u", "v", "p", "k", "ep", "u_tau","sdf"],
            batch_size=n_custom_points,
            plotter=InferencerPlotter(),
        )
        domain.add_inferencer(grid_inference, f"custom_tight_field_{inferencer_id + 1}")


        # Output names updated to record turbulent kinetics and friction velocity
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
            output_names=["u", "v", "p", "k", "ep", "u_tau","sdf"],
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
