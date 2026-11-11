# Curved-wall equilibrium treatment

The airfoil sampling uses a Warp signed-distance geometry derived from NVIDIA's NACA example. The D adaptation projects surface candidates onto the discrete airfoil, constructs P along the outward normal and checks the offset using the signed distance. Near-field interior collocation is filtered outside the unresolved strip. [NVIDIA airfoil geometry](https://github.com/NVIDIA/physicsnemo-sym/tree/main/examples/airfoil_pinn/naca)

The standard k–ε closure supplies an eddy viscosity for the mean momentum equations. In D the stress divergence includes spatially varying effective viscosity and the turbulent kinetic energy gradient for mean static pressure. [Launder and Spalding (1974)](https://doi.org/10.1016/0045-7825(74)90029-2)

The equilibrium wall conditions follow the channel example's friction velocity, logarithmic tangential velocity and dissipation relations. D adapts the flat-wall formulation to a local airfoil tangent. The wall shear target is uτ² under the unit-density convention. Off-wall evaluation is motivated by intermediate-boundary wall treatments. [NVIDIA channel implementation](https://github.com/NVIDIA/physicsnemo-sym/tree/main/examples/turbulent_channel/2d) · [Utyuzhnikov (2005)](https://doi.org/10.1002/fld.873)

The turbulence outputs use positive exponential transforms with reference scales. Residuals are normalized before aggregation, so equal explicit weights can still produce different numerical contributions. D adds soft lower and upper penalties on log y⁺ at P. The target interval is 30–300. An estimated friction correlation sets the initial offset, while predicted k determines the actual y⁺. The local tangent is oriented for attached flow. Stagnation, reverse flow and separation limit the equilibrium assumption.

External Cp and Cf distributions would provide an independent check in addition to integrated coefficients. Benchmark comparisons require matching Reynolds number, transition conditions and trailing-edge geometry. [NASA Turbulence Modeling Resource, NACA 0012](https://tmbwg.github.io/turbmodels/naca0012_val.html)
