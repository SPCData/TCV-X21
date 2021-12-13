"""
Routines for adding labels to plots
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText

from tcvx21.units_m import Quantity
from tcvx21.observable_c.observable_m import Observable

# Locations for legend or anchored text
locations = {
    "best": 0,
    "upper right": 1,
    "upper left": 2,
    "lower left": 3,
    "lower right": 4,
    "right": 5,
    "center left": 6,
    "center right": 7,
    "lower center": 8,
    "upper center": 9,
    "center": 10,
}


def add_y_zero_line(ax, level=0.0):
    """Mark the 0-line"""
    ax.axhline(level, color="k", linestyle="-", linewidth=0.5)


def add_x_zero_line(ax):
    """Mark the separatrix"""
    ax.axvline(0.0, color="k", linestyle="-", linewidth=0.5)


def add_twinx_label(ax, text: str, labelpad=20, visible=True):
    """Adds a right-hand-side label with the field direction"""
    new_ax = ax.twinx()
    new_ax.set_yticks([])
    new_ax.set_ylabel(text, rotation=270, labelpad=labelpad)

    if not visible:
        new_ax.set_xticks([])
        new_ax.spines["top"].set_visible(False)
        new_ax.spines["right"].set_visible(False)
        new_ax.spines["bottom"].set_visible(False)
        new_ax.spines["left"].set_visible(False)

    # Switch focus back to the main axis
    plt.sca(ax)


def format_yaxis(ax):
    """
    Move the exponent into the ylabel
    You have to call plt.draw() before this, otherwise the offset will be blank
    """
    plt.draw()
    ax.yaxis.offsetText.set_visible(False)
    units_string = (
        f"{Quantity(1, ax.yaxis.units).units:~P}" if ax.yaxis.units else "Unitless"
    )

    units_string = units_string.lstrip("1")

    # Get the axis scale factor (i.e. 10^19 for the density)
    scale_factor = ax.get_yaxis().get_offset_text().get_text()

    label_string = (
        f"$10^{{{scale_factor.split('e')[-1]}}}${units_string}"
        if scale_factor
        else f"{units_string}"
    )

    ax.set_ylabel(label_string)


def make_colorbar(
    ax, mappable, units, as_title: bool = True, remove_offset: bool = True
):
    """
    Adds a colorbar to the image, with a units label
    """
    cbar = plt.colorbar(mappable, ax=ax)
    cax = cbar.ax

    plt.draw()
    units_string = f"{Quantity(1, units).units:~P}" if units else ""

    if remove_offset:
        # If the units string starts with a '1' (i.e. 1/m^3), remove the 1
        units_string = units_string.lstrip("1")
        cax.yaxis.offsetText.set_visible(False)

        # Get the axis scale factor (i.e. 10^19 for the density)
        scale_factor = cax.get_yaxis().get_offset_text().get_text()

        label_string = (
            f"[$10^{{{scale_factor.split('e')[-1]}}}${units_string}]"
            if scale_factor
            else f"[{units_string}]"
        )
    else:
        label_string = f"[{units_string}]"

    if as_title:
        cax.set_title(label_string, fontsize=plt.rcParams["ytick.labelsize"])
    else:
        cax.set_ylabel(label_string, rotation=270, labelpad=15)

    return cbar


def make_field_direction_string(field_direction: str):
    """Returns the field direction as a nice string"""
    field_direction_rename = {
        "forward_field": "Forward field",
        "reversed_field": "Reversed field",
    }
    return field_direction_rename[field_direction]


def make_diagnostic_string(diagnostic: str):
    """Returns the diagnostic as a nice string"""
    diagnostic_rename = {
        "LFS-LP": "Low-field-side target",
        "LFS-IR": "Low-field-side target",
        "HFS-LP": "High-field-side target",
        "TS": "Divertor Thomson",
        "RDPA": "Divertor volume",
        "FHRP": "Outboard midplane",
    }
    return diagnostic_rename[diagnostic]


def make_observable_string(observable: Observable):
    """Returns the observable as a compact string"""
    observable_string = observable.name

    shorten_title_rename = {
        "Standard deviation": "Std. dev.",
        "of the": "of",
        "Pearson kurtosis": "Kurtosis",
        "Plasma velocity normalised to local sound speed": "Mach number",
        "Ion saturation current": "$J_{sat}$",
        "ion saturation current": "$J_{sat}$",
    }
    for key, value in shorten_title_rename.items():
        observable_string = observable_string.replace(key, value)

    return observable_string


def make_labels(field_direction: str, diagnostic: str, observable: Observable):
    """
    Returns strings which can be used in a title
    """

    return (
        make_field_direction_string(field_direction),
        make_diagnostic_string(diagnostic),
        make_observable_string(observable),
    )


def label_subplots(axs: np.ndarray, prefix: str = None):
    """Add subplot labels to an array of subplots"""
    plt.draw()

    index = -1

    for ax in axs.flatten():

        if not ax.get_visible():
            continue
        index += 1

        # Convert integer to character
        if prefix is None:
            label = f"{chr(index + 65)}"
        else:
            label = f"{prefix}{chr(index + 65)}"

        annotation_kwargs = dict(
            xycoords="axes fraction",
            bbox=dict(boxstyle="round", alpha=0.9, facecolor="white", edgecolor="grey"),
        )

        annotate = ax.annotate(label, xy=(0.05, 0.85), **annotation_kwargs)

        # Try to not overlap with the legend
        legend = ax.get_legend()
        try:
            legend_extent = legend.get_window_extent()
        except AttributeError:
            continue
        annotate_extent = annotate.get_window_extent()

        if (
            (legend_extent.x0 <= annotate_extent.x0 <= legend_extent.x1)
            or (legend_extent.x0 <= annotate_extent.x1 <= legend_extent.x1)
        ) and (
            (legend_extent.y0 <= annotate_extent.y0 <= legend_extent.y1)
            or (legend_extent.y0 <= annotate_extent.y1 <= legend_extent.y1)
        ):
            # Prevent annotation from overlapping with legend

            annotate.remove()
            ax.annotate(label, xy=(0.05, 0.1), **annotation_kwargs)
