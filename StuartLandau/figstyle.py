"""Shared figure style for the glycolytic-oscillator / phase-inference figures.

Extracted verbatim from ``GlycolyticOscillatorGPF.ipynb`` so that other
notebooks produce visually identical output.

The house style builds axes at a *fixed physical size* via
``fig.add_axes`` with explicit margins, rather than letting matplotlib
choose. Figures are therefore saved with no ``bbox_inches='tight'`` --
the PDF bounding box is deterministic and axes frames align across
panels of a multi-panel figure.

Usage
-----
>>> import figstyle as fs
>>> fig, ax = fs.figax(6.5, 4.5)                    # cm
>>> ax.plot(t, y, **fs.DATA, color=fs.C_DATA)
>>> fs.style_axes(ax, xlabel=r'time, $t$', ylabel=r'measurement, $y_k$')
>>> fs.save(fig, 'plots/MyFigure.pdf')
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

__all__ = [
    "CM", "CMAP", "CMAP_PHASE",
    "C_DATA", "C_MODEL", "C_THEORY", "C_GUIDE",
    "FS_LABEL", "FS_TICK", "FS_LEGEND", "FS_CBAR_LABEL",
    "LW_SPINE", "LW_TICK",
    "DATA", "TRACE", "CYCLE", "PHASE_PTS", "PHASE_LINE",
    "MSD_PTS", "THEORY", "MEAN", "SCATTER", "GUIDE",
    "BAND", "BAND_PRED", "BAND_FAINT",
    "MARGINS_LABELLED", "MARGINS_COMPACT",
    "retina", "figax", "figax_cbar", "style_axes", "legend",
    "pi_ticks", "save",
]

# ---------------------------------------------------------------------------
# units and colormaps
# ---------------------------------------------------------------------------

CM: float = 1 / 2.54
"""Centimetres per inch, for expressing physical figure sizes in cm."""

CMAP: Colormap = plt.cm.viridis
"""Sequential map; data colours are drawn from fixed sample points."""

CMAP_PHASE: Colormap = plt.cm.twilight
"""Cyclic map -- phase colouring only, never sequential quantities."""

# Colour roles. Sampling one map at fixed points keeps every figure on the
# same ramp regardless of how many series a given panel happens to have.
C_DATA = CMAP(0.2)      # observed / noisy data
C_MODEL = CMAP(0.6)     # inferred, fitted, posterior-predictive
C_THEORY = CMAP(0.0)    # closed-form prediction
C_GUIDE = "k"           # axvline / axhline reference marks

# ---------------------------------------------------------------------------
# typography and line weights (points)
# ---------------------------------------------------------------------------

FS_LABEL = 8            # axis labels
FS_TICK = 7             # tick labels
FS_LEGEND = 7           # legend entries
FS_CBAR_LABEL = 8       # colorbar label

LW_SPINE = 0.5          # axes frame
LW_TICK = 0.5           # tick marks

# ---------------------------------------------------------------------------
# mark specs -- splat into a plot call, e.g. ax.plot(x, y, **TRACE)
# ---------------------------------------------------------------------------

DATA = dict(marker=".", ls="-", ms=3, mew=0, lw=0.5, alpha=0.8)
TRACE = dict(marker=".", ls="-", ms=3, mew=0, lw=0.25)
CYCLE = dict(ls="-", lw=1.5)
PHASE_PTS = dict(marker=".", ls="none", ms=2, mew=0)
PHASE_LINE = dict(ls="-", lw=0.5)
MSD_PTS = dict(marker="o", ls="none", ms=3, mew=0.5, mfc="None")
THEORY = dict(ls="-", lw=0.5)
MEAN = dict(lw=1)
SCATTER = dict(s=10, alpha=0.8, lw=0)
GUIDE = dict(color=C_GUIDE, lw=0.5, ls="--")

# Uncertainty bands: never stroked, alpha set by what the band means.
BAND = dict(lw=0, alpha=0.2)          # +/- 1 s.d.
BAND_PRED = dict(lw=0, alpha=0.25)    # 95% posterior predictive
BAND_FAINT = dict(lw=0, alpha=0.1)    # MSD envelope, marginal fills

# ---------------------------------------------------------------------------
# margin presets (inches), as used in the notebook
# ---------------------------------------------------------------------------

MARGINS_LABELLED = (0.6, 0.2, 0.6, 0.2)
"""left, right, bottom, top -- panels with both axis labels."""

MARGINS_COMPACT = (0.5, 0.5, 0.5, 0.5)
"""left, right, bottom, top -- small square marginal/contour panels."""


def retina() -> None:
    """Request retina-resolution inline output in IPython.

    No-op outside IPython.
    """
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    except ImportError:
        pass


def figax(
    width_cm: float,
    height_cm: float,
    margins_in: tuple[float, float, float, float] = MARGINS_LABELLED,
) -> tuple[Figure, Axes]:
    """Create a figure whose axes frame has an exact physical size.

    Parameters
    ----------
    width_cm, height_cm : float
        Axes frame dimensions in centimetres (the frame, not the figure).
    margins_in : tuple of float
        ``(left, right, bottom, top)`` margins in inches, reserved for tick
        labels and axis labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    left, right, bottom, top = margins_in
    ax_w = width_cm * CM
    ax_h = height_cm * CM

    fig_w = ax_w + left + right
    fig_h = ax_h + bottom + top

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes(
        [left / fig_w, bottom / fig_h, ax_w / fig_w, ax_h / fig_h]
    )
    return fig, ax


def figax_cbar(
    width_cm: float,
    height_cm: float,
    margins_cm: tuple[float, float, float, float] = (1.5, 1.5, 1.0, 0.5),
    gap_cm: float = 0.0,
    cbar_width_cm: float = 0.2,
) -> tuple[Figure, Axes, Axes]:
    """Create a figure with a fixed-size axes plus an attached colorbar axes.

    Parameters
    ----------
    width_cm, height_cm : float
        Main axes frame dimensions in centimetres.
    margins_cm : tuple of float
        ``(left, right, bottom, top)`` margins in centimetres. The right
        margin is measured from the outer edge of the colorbar.
    gap_cm : float
        Separation between the main axes and the colorbar, in centimetres.
    cbar_width_cm : float
        Colorbar width in centimetres.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        Main axes.
    cax : matplotlib.axes.Axes
        Colorbar axes.
    """
    left, right, bottom, top = (m * CM for m in margins_cm)
    ax_w = width_cm * CM
    ax_h = height_cm * CM
    gap = gap_cm * CM
    cbar_w = cbar_width_cm * CM

    fig_w = left + ax_w + gap + cbar_w + right
    fig_h = bottom + ax_h + top

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h, ax_w / fig_w, ax_h / fig_h])
    cax = fig.add_axes(
        [(left + ax_w + gap) / fig_w, bottom / fig_h, cbar_w / fig_w, ax_h / fig_h]
    )
    return fig, ax, cax


def style_axes(
    ax: Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    label_size: int = FS_LABEL,
) -> Axes:
    """Apply the house tick, spine and label styling in place.

    Ticks point inward at 0.5 pt width with 7 pt labels; all four spines are
    thinned to 0.5 pt.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to restyle.
    xlabel, ylabel : str, optional
        Axis labels. Skipped when None, so existing labels survive.
    label_size : int
        Axis label size in points. The notebook uses 8 pt for full-width
        panels and 9 pt for a few small ones.

    Returns
    -------
    matplotlib.axes.Axes
        The same axes, for chaining.
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_size)

    for which in ("major", "minor"):
        ax.tick_params(
            axis="both",
            which=which,
            labelsize=FS_TICK,
            width=LW_TICK,
            direction="in",
        )

    for spine in ax.spines.values():
        spine.set_linewidth(LW_SPINE)

    return ax


def legend(ax: Axes, boxed: bool = False, **kwargs):
    """Add a legend at the house size.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    boxed : bool
        If True, draw an opaque backing box with no visible edge -- used when
        the legend overlaps data. Otherwise the legend is frameless.
    **kwargs
        Forwarded to ``ax.legend`` (``loc``, ``ncol``, ...).

    Returns
    -------
    matplotlib.legend.Legend
    """
    style = (
        dict(frameon=True, edgecolor="none")
        if boxed
        else dict(frameon=False)
    )
    return ax.legend(fontsize=FS_LEGEND, **style, **kwargs)


def pi_ticks(ax: Axes, axis: str = "y", turns: float = 1.0, quarter: bool = True) -> None:
    """Label a phase axis in fractions of pi.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    axis : {'x', 'y'}
        Which axis to relabel.
    turns : float
        Extent in units of 2*pi. ``1.0`` gives 0..2pi, ``2.0`` gives 0..4pi.
    quarter : bool
        If True, tick every pi/2 with ``\\frac`` labels (the convention for
        single-turn phase axes). If False, tick every whole pi -- what the
        multi-turn waveform panel uses.

    Notes
    -----
    Applies to ``cax`` objects too, since a colorbar axes is an Axes.
    """
    step = 0.5 if quarter else 1.0
    n = int(round(2 * turns / step))
    ticks = [i * step * np.pi for i in range(n + 1)]

    labels = []
    for i in range(n + 1):
        num = i * step  # multiples of pi
        if num == 0:
            labels.append(r"$0$")
        elif num == 1:
            labels.append(r"$\pi$")
        elif num == int(num):
            labels.append(rf"${int(num)}\pi$")
        else:
            labels.append(rf"$\frac{{{int(2 * num)}\pi}}{{2}}$".replace("1\\pi", "\\pi"))

    setter_ticks = ax.set_yticks if axis == "y" else ax.set_xticks
    setter_labels = ax.set_yticklabels if axis == "y" else ax.set_xticklabels
    setter_ticks(ticks)
    setter_labels(labels)


def save(fig: Figure, path: str | Path) -> None:
    """Save as vector PDF without altering the computed bounding box.

    No ``bbox_inches='tight'`` -- that would discard the exact axes geometry
    that ``figax`` was used to establish.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str or pathlib.Path
        Destination. Parent directories are created if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
