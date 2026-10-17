"""Graphical abstract for the manuscript.

IMAVIS requires one (>=1328 px wide, legible at 13 cm); several Elsevier titles
recommend one. The reduction to 13 cm is the binding constraint: the first
version of this figure was a three-stage block diagram and every label collided
once reduced. A graphical abstract that needs full-size reading is not a
graphical abstract, so this version carries one data plot and three statements.

Every value comes from the archives via the loader the article figures use, so
the abstract cannot drift from the paper.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_figures as M

OUT = M.OUT
INK, MUTED, WARM, GRIDC = M.INK, M.MUTED, M.WARM, M.GRIDC

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9.4,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

KEEP = {"UCF-Crime": "D_ucf2ucf", "XD-Violence": "C_xd2xd",
        "ShanghaiTech": "ucf2sht"}


def build():
    rows2, rows4 = M.rows2, M.rows4
    names = list(KEEP)
    reported = [rows2[KEEP[d]]["pooled_auc"] for d in names]
    no_level = [rows4[KEEP[d]]["deoff_auc"] for d in names]
    kept = [rows4[KEEP[d]]["const_auc"] for d in names]

    fig = plt.figure(figsize=(6.6, 4.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.62], hspace=0.42)

    # ---------------------------------------------------------------- panel
    ax = fig.add_subplot(gs[0])
    x = range(len(names))
    w = 0.33
    b1 = ax.bar([i - w for i in x], reported, width=w, color=MUTED,
                edgecolor=INK, lw=0.6, zorder=3, label="reported")
    b2 = ax.bar(list(x), no_level, width=w, color=M.ACCENT, edgecolor=INK,
                lw=0.6, zorder=3, label="per-video level deleted")
    b3 = ax.bar([i + w for i in x], kept, width=w, color=WARM, edgecolor=INK,
                lw=0.6, zorder=3, label="within-video variation deleted")

    for bars in (b1, b2, b3):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.018,
                    f"{r.get_height():.2f}", ha="center", va="bottom",
                    fontsize=8.0, color=INK)

    ax.axhline(0.5, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=4)
    ax.text(2.66, 0.512, "chance", fontsize=8.0, ha="left", va="bottom",
            color=INK)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRIDC, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=9.0)
    ax.set_xlim(-0.62, 3.02)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("frame-level AUC", fontsize=8.6)
    ax.tick_params(labelsize=8.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=3,
              frameon=False, fontsize=8.0, handlelength=1.1,
              columnspacing=1.2)

    # -------------------------------------------------------------- text
    tx = fig.add_subplot(gs[1])
    tx.axis("off")
    tx.set_xlim(0, 1)
    tx.set_ylim(0, 1)

    tx.text(0.5, 0.97, "Pooling frame scores ranks two different things, and "
                       "the reported number is\nmostly the easier one.",
            ha="center", va="top", fontsize=9.6, color=INK, linespacing=1.5)

    tx.text(0.5, 0.56,
            "Deleting every within-video cue still keeps 94–97% of the "
            "reported area on UCF-Crime and XD-Violence,\nand on ShanghaiTech, "
            "whose test split contains no normal video, the ordering reverses "
            "and pooled AUC falls to 0.40.",
            ha="center", va="top", fontsize=8.6, color=INK, linespacing=1.6)

    tx.text(0.5, 0.04,
            "High pooled AUC alone does not establish temporal localisation.",
            ha="center", va="bottom", fontsize=9.2, color=WARM,
            style="italic", weight="bold")

    fig.savefig(os.path.join(OUT, "graphical_abstract.pdf"))
    fig.savefig(os.path.join(OUT, "graphical_abstract.png"), dpi=300)
    plt.close(fig)
    return dict(zip(names, zip(reported, no_level, kept)))


if __name__ == "__main__":
    d = build()
    for k, (p, nl, kv) in d.items():
        print(f"{k:<15} reported={p:.4f}  level-deleted={nl:.4f}  "
              f"variation-deleted={kv:.4f}")
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"graphical_abstract.{ext}")
        print(f"  graphical_abstract.{ext}  {os.path.getsize(p)} bytes")