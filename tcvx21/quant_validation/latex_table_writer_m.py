"""
Apologies for hard-coding. This makes a latex table which has conditional formatting applied and
expands the diagnostic, observable into nice compact symbolic forms

Not suitable for picking up directly
"""
import tcvx21
from tcvx21.quant_validation.ricci_metric_m import level_of_agreement_function
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

diagnostics_latex = {
    "FHRP": "Fast\\\\horizontally-\\\\reciprocating\\\\probe (FHRP)\\\\for outboard\\\\midplane",
    "TS": "Thomson scattering\\\\(TS) for divertor\\\\entrance",
    "RDPA": "Reciprocating\\\\divertor probe\\\\array (RDPA)\\\\for divertor\\\\volume",
    "LFS-IR": "Infrared camera (IR)\\\\for low-field-side target",
    "LFS-LP": "Wall Langmuir\\\\probes for\\\\low-field-side\\\\target",
    "HFS-LP": "Wall Langmuir\\\\probes for\\\\high-field-side\\\\target",
}

observable_latex = {
    "density": "$n$",
    "electron_temp": "$T_e$",
    "potential": "$V_{pl}$",
    "jsat": "$J_{sat}$",
    "jsat_std": "$\\sigma\\left(J_{sat}\\right)$",
    "jsat_skew": "$\\mathrm{skew}\\left(J_{sat}\\right)$",
    "jsat_kurtosis": "$\\mathrm{kurt}\\left(J_{sat}\\right)$",
    "current": "$J_\\parallel$",
    "current_std": "$\\sigma\\left(J_\\parallel\\right)$",
    "vfloat": "$V_{fl}$",
    "vfloat_std": "$\\sigma\\left(V_{fl}\\right)$",
    "q_parallel": "$q_\\parallel$",
    "lambda_q": "$\\lambda_q$",
    "mach_number": "$M_\\parallel$",
}

# Chi as a latex symbol
chi_latex = "$\\chi$"


def tuple_to_string(tuple_in: tuple) -> str:
    """Converts an RGB tuple to a latex-xcolor compatible string"""
    return ", ".join([f"{val:6.3f}" for val in tuple_in])


def write_number_w_color(
    value,
    file,
    cell_clip=None,
    cmap="RdYlGn_r",
    text_clip_fraction=False,
    expand=1.4,
    debug=False,
):
    """
    Writes a 'value' to a file, applying conditional cell colouring and text colouring

    It is assumed that all values are >= 0.0, or np.nan
    cell_clip gives the 100% intensity value, above which all values are set to 100%. This also sets the
    normalisation of the range < cell_clip: a linear ramp from 0 to cell_clip is used

    cmap is the name of a matplotlib colormap, default set to the red-blue diverging colormap
    text_clip_fraction switches from black to white text for values above cell_clip * text_clip_fraction
    expand sets the 0% and 100% intensity of the colormap to outside the normalised value space. This is useful
    to make the cell color closer to white, which makes the cells more readable
    """

    if not np.isnan(value):
        if cell_clip is None:
            cellcolor, fontcolor = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
        else:
            assert value >= 0.0

            norm = colors.Normalize(
                vmin=cell_clip * (1 - expand), vmax=cell_clip * expand
            )
            normalised = norm(np.min((value, cell_clip)))
            cellcolor = plt.cm.get_cmap(cmap)(normalised)[:-1]

            if text_clip_fraction:
                # Set value < text_clip to black, and > text_clip to white
                fontcolor = (
                    (0.0, 0.0, 0.0)
                    if value < text_clip_fraction * cell_clip
                    else (1.0, 1.0, 1.0)
                )
            else:
                fontcolor = (0.0, 0.0, 0.0)

        text = f"{value:.3}"
        if "e" in text:
            text = f"{int(float(text))}"

    else:
        cellcolor, fontcolor = (0.4, 0.4, 0.4), (1.0, 1.0, 1.0)
        text = f"{'NaN':6}"

    if not debug:
        file.write(
            f"\\cellcolor[rgb]{{{tuple_to_string(cellcolor)}}}"
            f"\\textcolor[rgb]{{{tuple_to_string(fontcolor)}}}"
            f"{{{text:6}}}"
        )
    else:
        file.write(f"{text:6}")


def process_case_key(case_key):
    """Convert the toroidal field sign to math-type, to get it to print nicely"""
    return case_key.replace("+", "($+$)").replace("-", "($-$)")


def make_colorbar(
    cell_clip, label, output_path=None, cmap="RdYlGn_r", expand=1.4, eps=1e-2
):

    plt.style.use(tcvx21.style_sheet)
    cnorm = colors.Normalize(vmin=cell_clip * (1 - expand), vmax=cell_clip * expand)

    fig, ax = plt.subplots(figsize=(5, 1))
    dummy_array = np.array([[0, cell_clip]])

    im = ax.imshow(dummy_array, cmap=cmap, norm=cnorm)
    ax.set_visible(False)

    # left, bottom, width, height in normalised axis units
    cax = plt.axes([0.1, 0.2, 0.8, 0.1])

    plt.colorbar(
        im,
        orientation="horizontal",
        cax=cax,
        extend="max",
        ticks=np.arange(cell_clip + eps),
        boundaries=np.linspace(0, cell_clip + eps, num=1000),
    )
    cax.set_ylabel(
        label, rotation=0, labelpad=10, y=-0.5, size=plt.rcParams["axes.titlesize"]
    )

    if output_path is not None:
        tcvx21.plotting.savefig(fig, output_path)


def write_cases_to_latex(cases: dict, output_file: Path):
    """
    Write a formatted conditionally colored table to a file
    """

    for case in cases.values():
        case.calculate_metric_terms()

    case_keys = list(cases.keys())
    n_cases = len(cases)

    make_colorbar(
        cell_clip=5.0, label="$d_j$", output_path=output_file.parent / "colorbar.png"
    )

    with open(output_file, "w") as f:

        # Header
        f.write(f"\\begin{{tabular}}{{ll{n_cases * 'cc'}}}\n")
        f.write(f"\\toprule\n")

        case_header = " & ".join(
            [
                f"\\multicolumn{{2}}{{c}}{{{process_case_key(case_key)}}}"
                for case_key in case_keys
            ]
        )
        f.write(f" & & {case_header} \\\\ \n")

        case_header = " & ".join(n_cases * ["$d_j$ & $S$"])
        f.write(f"Diagnostic & observable & {case_header} \\\\ \n")
        f.write(f"\\midrule\n")

        # Write the results in
        for diagnostic, diagnostic_styled in diagnostics_latex.items():

            # Assume that all cases have the same hierarchies, so take
            # the first value
            hierarchies = cases[case_keys[0]].hierarchy[diagnostic]
            n_observables = len(hierarchies)

            f.write(
                f"\\multirow{{{n_observables + 1}}}{{*}}{{\\makecell{{{diagnostic_styled}}}}}\n"
            )

            d = np.nan * np.ones((n_observables, n_cases))
            S = np.nan * np.ones((n_observables, n_cases))
            H = np.nan * np.ones(n_observables)
            i = -1

            for observable, observable_styled in observable_latex.items():
                if observable in hierarchies.keys():
                    i += 1
                    H[i] = 1.0 / hierarchies[observable]
                    f.write(f"& {observable_styled:40} & ")

                    for j, case in enumerate(cases.values()):
                        d[i, j] = case.distance[diagnostic][observable]
                        write_number_w_color(d[i, j], file=f, cell_clip=5.0)
                        f.write(" & ")

                        S[i, j] = case.sensitivity[diagnostic][observable]
                        write_number_w_color(S[i, j], file=f)

                        if not j == n_cases - 1:
                            # Don't write an alignment character for the last entry on a line
                            f.write(" & ")

                    f.write("\\\\\n")

            f.write(f"\\cline{{2-{2 * n_cases + 2}}}\n")

            Q = np.nansum(H[:, np.newaxis] * S, axis=0)
            chi = (
                np.nansum(H[:, np.newaxis] * S * level_of_agreement_function(d), axis=0)
                / Q
            )

            # Write diagnostic chi and Q

            diagnostic_summary = (
                "$\\left(\\chi; Q\\right)$\\textsubscript" + f"{{{diagnostic}}}"
            )
            f.write(f"& {diagnostic_summary:40} & ")

            for j in range(n_cases):
                f.write(
                    "\\multicolumn{{2}}{{c}}{{$ \\textbf{{({chi:.2}; \\ {Q:.3})}} $ }}".format(
                        chi=chi[j], Q=Q[j]
                    )
                )

                if not j == n_cases - 1:
                    # Don't write an alignment character for the last entry on a line
                    f.write(" & ")

            f.write("\\\\\n")
            f.write("\\midrule\n")

        # Write the overall agreement
        f.write(f"Overall\n& {chi_latex + '; $Q$':40} & ")

        for j, case in enumerate(cases.values()):
            chi, Q = case.compute_chi()

            f.write(
                "\\multicolumn{{2}}{{c}}{{$ \\textbf{{({chi:.2}; \\ {Q:.3})}} $ }}".format(
                    chi=chi, Q=Q
                )
            )

            if not j == n_cases - 1:
                # Don't write an alignment character for the last entry on a line
                f.write(" & ")
            else:
                f.write("\\\\\n")

        # Footer
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}")
