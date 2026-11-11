# Saved airfoil fields

The boundary plots trace the sampled upper and lower surfaces separately. Near-field maps show the exported points at their original coordinates. Color limits use the full range of each field and case. Read each color bar when comparing models. The symbol ε corresponds to the VTP array ep.

| Case | Boundary profiles | Near-field maps |
|---|---|---|
| navier-stokes | [Boundary](navier-stokes/boundary.png) | [Near field](navier-stokes/near_field.png) |
| k-epsilon | [Boundary](k-epsilon/boundary.png) | [Near field](k-epsilon/near_field.png) |
| k-epsilon_wf_1 | [Boundary](k-epsilon_wf_1/boundary.png) | [Near field](k-epsilon_wf_1/near_field.png) |
| Ver2_A | Unavailable | [Near field](Ver2_A/near_field.png) |
| Ver2_B | [Boundary](Ver2_B/boundary.png) | [Near field](Ver2_B/near_field.png) |
| Ver2_D_yplus | [Boundary](Ver2_D_yplus/boundary.png) | [Near field](Ver2_D_yplus/near_field.png) |

Navier–Stokes exports p, u and v. The turbulence cases also export k and ep. A supplies a near-field snapshot. Open the adjacent VTP files in ParaView for point values and coordinates. For the three v1 models, surface.csv provides a tabular export of geometry, pressure and velocity. [File checksums](manifest.csv)
