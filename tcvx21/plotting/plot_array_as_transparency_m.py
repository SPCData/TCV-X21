"""
A helpful function which allows us to plot an array with the transparency set by the value
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_array_as_transparency(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    alphas,
    cmap=plt.cm.Greys,
    intensity=0.3,
    clip=True,
    invert=False,
):
    """
    Displays a 2d field given by 'alphas' as a semi-transparent field, where the min-values are transparent and the
    max-values are opaque.

    Particularly useful for adding a grey region around contour plots, which can be done with
    limits={'xmin': grid.r_s.min(), 'xmax': grid.r_s.max(), 'ymin': grid.z_s.min(), 'ymax':grid.z_s.max()}
    plot_array_as_transparency(axes, limits, alphas=np.isnan(shaped_data))
    where shaped_data is the z-values that you pass to contour

    invert switches the max and min transparencies
    clip=True will mean that masked/NaN alphas are assumed to have a value of '1.0'
    """

    limits = dict(xmin=x.min(), xmax=x.max(), ymin=y.min(), ymax=y.max())

    alphas = np.array(alphas, dtype=float)

    # Normalize to the range [0, 1]
    alphas -= alphas.min()
    alphas /= alphas.max()
    if invert:
        alphas *= -1.0
        alphas += 1.0

    assert alphas.min() == 0.0 and alphas.max() == 1.0

    on_grid = np.ones_like(alphas) * intensity

    colors = Normalize(0.0, 1.0, clip=clip)(on_grid)
    colors = cmap(colors)

    colors[..., -1] = alphas

    return ax.imshow(
        colors,
        extent=(limits["xmin"], limits["xmax"], limits["ymin"], limits["ymax"]),
        origin="lower",
    )
