import os
import numpy as np
import matplotlib.pyplot as plt
import glob

# Identify the checkpoint path. 
# Modulus automatically creates an 'outputs/<config_name>/network_checkpoint/' folder.
# Adjust this string if your Hydra outputs are saved in a slightly different path format.
checkpoint_dir = "./outputs/config/network_checkpoint/"
inferencer_dir = os.path.join(checkpoint_dir, "inferencers")

# We will visualize the near-field of the airfoil to see the boundary layer and wake.
# Look for the first near_field inferencer output (e.g., near_field_1.npz)
data_files = glob.glob(os.path.join(inferencer_dir, "near_field_*.npz"))

if not data_files:
    print(f"Error: Could not find any .npz files in {inferencer_dir}.")
    print("Have you finished training the model?")
    exit()

# Load the first found file
target_file = data_files[0]
print(f"Plotting results from: {target_file}")

data = np.load(target_file, allow_pickle=True)
data = np.atleast_1d(data.f.arr_0)[0]

# Extract variables
x = data["x"].flatten()
y = data["y"].flatten()
u = data["u"].flatten()
v = data["v"].flatten()
p = data["p"].flatten()
k = data["k"].flatten()
ep = data["ep"].flatten()

# Calculate velocity magnitude
vel_mag = np.sqrt(u**2 + v**2)

# Set up the plot (2x2 grid for Velocity, Pressure, TKE, and Epsilon)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Helper function to plot scattered data as 2D contours
def plot_scatter_contour(ax, var, title, cmap="turbo"):
    # Using scatter instead of tricontourf since the point cloud is unstructured
    sc = ax.scatter(x, y, c=var, cmap=cmap, s=2, alpha=0.8)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal', 'box') # Keeps the airfoil geometry from stretching

# 1. Velocity Magnitude
plot_scatter_contour(axs[0, 0], vel_mag, "Velocity Magnitude ($|V|$)", cmap="turbo")

# 2. Pressure
plot_scatter_contour(axs[0, 1], p, "Pressure ($p$)", cmap="jet")

# 3. Turbulent Kinetic Energy (k)
plot_scatter_contour(axs[1, 0], k, "Turbulent Kinetic Energy ($k$)", cmap="inferno")

# 4. Turbulence Dissipation Rate (epsilon)
plot_scatter_contour(axs[1, 1], ep, "Turbulence Dissipation ($\epsilon$)", cmap="magma")

plt.tight_layout()

# Save the figure
output_img = "airfoil_turbulent_results.png"
plt.savefig(output_img, dpi=300)
print(f"Results successfully saved to {output_img}")
# Uncomment the line below if you want the window to pop up when the script runs
# plt.show()