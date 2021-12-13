from pathlib import Path
import matplotlib.pyplot as plt
import sys
from tcvx21 import test_session


def savefig(
    fig,
    output_path: Path = None,
    facecolor="w",
    bbox_inches="tight",
    dpi=300,
    show=False,
    close=True,
):
    """
    Save a figure to file
    """

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, facecolor=facecolor, bbox_inches=bbox_inches, dpi=dpi)

    if show and not test_session:
        plt.show()
    elif close:
        plt.close(fig)
