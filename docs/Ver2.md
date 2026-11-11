# Ver2 wall-function experiments

| Attempt | Saved step | Mean y⁺ | Cd from gradients | Cd from surface uτ² |
|---|---:|---:|---:|---:|
| A | 400,000 | 1.02 | unavailable | unavailable |
| B | 115,000 | 1.02 | 0.009337 | 0.009413 |
| D_yplus | unknown | 50.95 | 0.062927 | 0.060131 |

A provides a near-field export and wall constraints. The comparable surface-pressure integral requires a separate boundary export. At the last checkpoint the aggregated loss was 2.68×10⁻⁵, while all 512 sampled wall points had y⁺ near 1.02.

B reached an aggregated loss of 1.44×10⁻³. At the 512 surface points k was 10⁻¹⁰. The friction share was 0.47% using gradients and approximately zero using surface uτ². The saved A and B scripts request conf/ while the archived YAML resides in config/. A also references an external geometry module, and B's archived turbulence module has a _B suffix absent from its import.

In D_yplus, all 512 P points were within the specified wall-coordinate interval and all had negative streamwise velocity. The wall-function integration gave pressure drag 0.06308 and signed friction drag −0.00295. The export lacks a numbered checkpoint and loss history. The archived script enables one inference set while the output folder contains seven numbered sets. The comparison uses boundary_1 and near_field_1. [Measurements](../results/Ver2_comparison.csv) · [Stage commands](../models/Ver2/D_yplus/RUN_NOTES.md)
