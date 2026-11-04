# plotting.py

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

TOOL_STYLES = {
    "truth": ("truth", "#52514e"),
    "catella": ("catella", "#2a78d6"),
    "hiddenfoot": ("hiddenfoot", "#c8407e"),
    "ft": ("ft", "#eb6834"),
    "fiberhmm_pre": ("fiberhmm (pretrained)", "#1baf7a"),
    "fiberhmm": ("fiberhmm (retrained)", "#eda100"),
}


def _mod_qual_rows(ddir, mol_ids, nbp):
    tsv = pd.read_csv(Path(ddir) / "synth.tsv", sep="\t")
    rows = np.full((len(mol_ids), nbp), np.nan)
    index = {m: i for i, m in enumerate(mol_ids)}
    for mol_id, grp in tsv.groupby("read_id", sort=False):
        if mol_id in index:
            pos = grp["ref_position"].to_numpy()
            rows[index[mol_id], pos] = grp["mod_qual"].to_numpy()
    return rows


# Occupancy page layout (inches)
_OCC_PAGE = (8.27, 11.69)
_OCC_TRACK = 0.28
_OCC_GAP = 0.3
_OCC_TOP = 0.8
_OCC_BOTTOM = 1.1
_OCC_LEFT = 1.5
_OCC_RIGHT = 0.4


def _occupancy_per_page(ntrack):
    usable = _OCC_PAGE[1] - _OCC_TOP - _OCC_BOTTOM
    return max(1, int(usable // (ntrack * _OCC_TRACK + _OCC_GAP)))


def _occupancy_figure(molecules, title, nbp, trim):
    width, height = _OCC_PAGE
    fig = Figure(figsize=_OCC_PAGE)
    x0 = _OCC_LEFT / width
    w = 1 - (_OCC_LEFT + _OCC_RIGHT) / width
    th = _OCC_TRACK / height
    y = 1 - _OCC_TOP / height
    for mol_id, qual, tracks in molecules:
        y -= _OCC_GAP / height
        for k, (label, occ, color) in enumerate(
            [("mod_qual", None, None), *tracks]
        ):
            y -= th
            ax = fig.add_axes((x0, y + 0.25 * th, w, 0.75 * th))
            ax.set_xlim(-0.5, nbp - 0.5)
            ax.tick_params(labelbottom=False, labelsize=7)
            if k == 0:
                image = ax.imshow(
                    qual[None, :],
                    aspect="auto",
                    cmap="Purples",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                    extent=(-0.5, nbp - 0.5, 0, 1),
                )
                ax.set_title(mol_id, fontsize=8, loc="left")
            _draw_track(ax, label, occ, color, nbp, trim)
    ax.tick_params(labelbottom=True)
    ax.set_xlabel("position (bp)", fontsize=8)
    fig.suptitle(title, fontsize=9, y=1 - 0.25 / height, va="top")
    cax = fig.add_axes((0.35, 0.45 / height, 0.3, 0.08 / height))
    bar = fig.colorbar(image, cax=cax, orientation="horizontal")
    bar.set_label(
        "mod_qual = P(methylated | signal); white = no site", fontsize=7
    )
    bar.ax.tick_params(labelsize=6)
    return fig


def _draw_track(ax, label, occ, color, nbp, trim):
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if occ is None:
        return
    ax.axvspan(0, trim, color="#eeeeee", zorder=0)
    ax.axvspan(nbp - trim, nbp, color="#eeeeee", zorder=0)
    x = np.arange(len(occ))
    ax.fill_between(x, occ, step="mid", color=color, alpha=0.85, lw=0)
    ax.set_ylim(0, 1)


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
    ax.set_axisbelow(True)
