# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data Loading and Previewing
#
# This notebook loads the TCVX21 GPI data in the format of MATLAB `.mat` file containing:
# - `brt_arr (Nr × Nz × Nt)`
# - `r_arr (Nr × Nz)` [m]
# - `z_arr (Nr × Nz)` [m]
# - `t_window (Nt,)`
#
# It then generates a frame-by-frame video over time.

# %%
# Import necessary libraries for numerical operations, plotting, animation, and MATLAB file loading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io as sio


# %% [markdown]
# ## Set file paths and parameters

# %%
# Set file paths and parameters for video generation
# mat_file: Path to the MATLAB .mat file containing the data
# out_mp4: Output filename for the generated video
mat_file = "/home/yinwang/SPC/TCV-X21/1.experimental_data/GPI_data/GPI_TCVX21/70336_1.5750_1.5854.mat"  # <-- change this
out_mp4 = "brt_video_70336.mp4"

# Video settings
fps = 5  # Frames per second for the output video
stride = 5  # Step size for frame selection (skip frames to speed up)
cmap = "viridis"  # Colormap for visualization

# Color limits for the plot (set to None for automatic scaling)
cmin = None
cmax = None


# %% [markdown]
# ## Load MATLAB data

# %%
# Load data from the MATLAB file (non-v7.3, use scipy.io.loadmat)
data = sio.loadmat(mat_file)

brt_arr = np.asarray(data["brt_arr"])
r_arr = np.asarray(data["r_arr"])
z_arr = np.asarray(data["z_arr"])
t_window = np.asarray(data["t_window"]).squeeze()

# Print shapes for verification
print("brt_arr:", brt_arr.shape)
print("r_arr:", r_arr.shape)
print("z_arr:", z_arr.shape)
print("t_window:", t_window.shape)


# %% [markdown]
# ## Validate dimensions

# %%
# Check that the shapes of the loaded arrays are consistent
Nr, Nz, Nt = brt_arr.shape

assert r_arr.shape == (Nr, Nz), "r_arr shape mismatch"
assert z_arr.shape == (Nr, Nz), "z_arr shape mismatch"
assert t_window.shape[0] == Nt, "t_window length mismatch"

print("Shapes are consistent.")


# %% [markdown]
# ## Select frames

# %%
# Optional: Select video start and stop time
# Set your desired start and stop time in seconds (or the unit of t_window)
t_start = 1.576  # video start time
t_stop = 1.5761  # video stop time (default: last frame)

# Find the indices corresponding to start and stop time
t_start_idx = np.searchsorted(t_window, t_start)
t_stop_idx = np.searchsorted(t_window, t_stop)

# Select frames within the time window
frame_indices = np.arange(t_start_idx, t_stop_idx, stride)
print(
    f"Frames to render: {len(frame_indices)} (from t={t_window[t_start_idx]:.6g} to t={t_window[t_stop_idx-1]:.6g})"
)


# %% [markdown]
# ## Determine color limits

# %%
# Determine color limits for visualization
# If cmin/cmax are not set, use percentiles from the data for automatic scaling
if cmin is None or cmax is None:
    sample = brt_arr[:, :, frame_indices]
    vmin = np.nanpercentile(sample, 1)
    vmax = np.nanpercentile(sample, 99)
else:
    vmin, vmax = cmin, cmax

print("Color limits:", vmin, vmax)


# %% [markdown]
# ## Preview first frame

# %%
# Plot a single frame for preview
# k: index of the frame to plot (first selected frame)
k = frame_indices[0]
plt.figure()
plt.pcolormesh(
    r_arr, z_arr, brt_arr[:, :, k], shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
)
plt.xlabel("R [m]")
plt.ylabel("Z [m]")
plt.gca().set_aspect("equal")  # Ensure aspect ratio is equal
plt.title(f"Frame k={k}, t={t_window[k]:.6g}")
plt.colorbar(label="brt")
plt.show()


# %% [markdown]
# ## Create animation

# %%
# Set up the animation for video generation
# Create a figure and axis for plotting
fig, ax = plt.subplots()
ax.set_xlabel("R [m]")
ax.set_ylabel("Z [m]")
ax.set_aspect("equal")  # Ensure aspect ratio is equal

# Initialize the mesh plot with the first frame
mesh = ax.pcolormesh(
    r_arr,
    z_arr,
    brt_arr[:, :, frame_indices[0]],
    shading="auto",
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
)
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("brt")

# Add a text box to display the current time
# time_text will be updated in each frame
time_text = ax.text(
    0.02,
    0.98,
    "",
    transform=ax.transAxes,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
)

# Function to update the plot for each frame in the animation
def update(i):
    k = frame_indices[i]
    mesh.set_array(brt_arr[:, :, k].ravel(order="C"))
    ax.set_title(f"Frame {k+1}/{Nt}")
    time_text.set_text(f"t = {t_window[k]:.6g}")
    return mesh, time_text


# Create the animation object
ani = animation.FuncAnimation(
    fig, update, frames=len(frame_indices), interval=1000 / fps
)
plt.close(fig)


# %% [markdown]
# ## Watch Video

# %%
# Display the animation directly in the notebook (no file saving required)
from IPython.display import HTML

HTML(ani.to_jshtml())

# %% [markdown]
# ## Save video (requires ffmpeg)

# %%
# Save the animation to an MP4 video file
# This requires ffmpeg to be installed and available in the system PATH
writer = animation.FFMpegWriter(fps=fps)
ani.save(out_mp4, writer=writer, dpi=150)
print("Saved:", out_mp4)


# %%
