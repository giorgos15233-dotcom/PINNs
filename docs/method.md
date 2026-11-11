# Evaluating the airfoil models

The thesis evaluates convergence, symmetry at zero incidence, total drag at matching Reynolds number, and the pressure and viscous contributions to drag. Aggregated training loss is useful within a run. Comparing its absolute magnitude across models also requires accounting for changes in residuals, scales and weights.

With unit density, free-stream speed and chord, each section force coefficient is twice its integrated force per unit span. The gradient extraction uses the saved ViscF_3_lowq_2 ParaView pipeline with the viscosity correction 50000/Re from the analysis workbook. The wall-function extraction uses the surface pressure vector and the signed projection of uτ² along the local velocity in the utau_2 pipeline. The two extractions use different lift signs. Symmetry is assessed with |Cl|.

The friction percentage is 100 Cd,friction / Cd. Signed drag error is 100(Cd / 0.0065 − 1). Pressure and friction sum to total drag before rounding. The benchmark gives total drag for the NACA 0012 at zero incidence and Re = 10⁶. [Sheldahl and Klimas, Table 2](https://www.osti.gov/servlets/purl/6548367)

The field step identifies the saved inference output. The final logged scalar can occur later, at step 200,100 for standard k–ε and 44,600 for wall-function k–ε. Each model's published forces refer to its field step. [Results](../results/v1.csv) · [Training histories](../figures/loss.png)
