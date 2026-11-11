# Running the three D_yplus stages

Run these commands sequentially from this directory in the GPU training environment. All three invocations use the same network_dir configured in conf/config.yaml. Start a fresh experiment with a clean checkpoint directory. training.max_steps gives the absolute endpoint for each stage. Each tee invocation appends standard output to logloss.log.

```bash
python airfoil_ke_wf.py custom.stage=initialize training.max_steps=3000 | tee -a logloss.log
python airfoil_ke_wf.py custom.stage=mixed training.max_steps=16000 | tee -a logloss.log
python airfoil_ke_wf.py custom.stage=physics training.max_steps=400000 | tee -a logloss.log
```

During initialization the model fits the initial profiles. The mixed stage retains the near-field k and ε seed residuals and releases the near-field velocity and pressure seeds. Check the wall residuals and the AirfoilWallValidity monitor before interpreting a stage transition. The original run notes describe extending the mixed stage and inspecting a short physics segment if the monitors are still changing. The published D field has an unknown step. These endpoints describe the training procedure.

The monitor records k, ε, uτ, and the minimum, mean and maximum y⁺, together with fractions outside the modeled range. Inspect the streamwise velocity and shear direction at P as well as the wall-coordinate range. [Method and sources](SOURCES_AND_METHOD.md)
