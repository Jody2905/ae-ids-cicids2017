"""Shared matplotlib styling. Each model keeps the same colour in every figure."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_NAMES = {"ae": "Autoencoder", "if": "Isolation Forest", "rf": "Random Forest",
               "rf_unseen": "Random Forest (family unseen)"}
# Categorical slots 1-4 of a colour-blind-validated palette, fixed per model.
MODEL_COLORS = {"ae": "#2a78d6", "if": "#eb6834", "rf": "#1baf7a", "rf_unseen": "#eda100"}

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
        "savefig.dpi": 160, "savefig.bbox": "tight",
    })
    return plt
