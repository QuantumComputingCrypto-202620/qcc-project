"""Shared matplotlib style: one categorical palette in fixed order, recessive axes, and figures
saved as both PNG and PDF."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# Validated categorical palette (light surface). Assign in this order; never cycle past it.
SERIES_COLORS: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#6250d6",  # violet
)
# Secondary encoding so identity never relies on color alone.
SERIES_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P")
TEXT = "#1f1f1e"
MUTED = "#6b6a63"
GRID = "#e4e3dd"
REFERENCE = "#8a8980"  # neutral color for reference lines (thresholds, exact values)
SURFACE = "#fcfcfb"

FIGSIZE_SINGLE = (6.4, 4.0)
FIGSIZE_WIDE = (10.0, 4.0)


def series_style(index: int) -> dict[str, object]:
    """Line style for the ``index``-th series: palette color, marker, 2 px line."""
    if index >= len(SERIES_COLORS):
        raise ValueError(
            f"only {len(SERIES_COLORS)} series colors; facet or group the extra series"
        )
    return {
        "color": SERIES_COLORS[index],
        "marker": SERIES_MARKERS[index],
        "linewidth": 1.5,
        "markersize": 5,
        "markeredgecolor": SURFACE,
        "markeredgewidth": 0.8,
    }


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": TEXT,
            "ytick.labelcolor": TEXT,
            "font.size": 9,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def save_figure(
    fig: Figure, stem: Path | str, formats: Sequence[str] = ("png", "pdf")
) -> list[Path]:
    """Save ``fig`` as ``<stem>.png`` and ``<stem>.pdf`` and close it."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(f".{fmt}") for fmt in formats]
    for path in paths:
        fig.savefig(path, metadata={"CreationDate": None} if path.suffix == ".pdf" else None)
    plt.close(fig)
    return paths
