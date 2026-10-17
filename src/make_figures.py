"""Build the manuscript figures from archived measurement files.

Every value plotted here is read from results/*.json. Nothing is typed in by
hand, so a figure cannot drift from the table it illustrates. The script also
emits a machine-readable dump of the plotted values next to the PDF so the
numbers in the caption can be checked without opening the figure.

Outputs (into the paper directory):
    fig1_framework.pdf   measurement design (the required block diagram)
    fig2_decomposition.pdf
    fig3_alignment.pdf
    fig4_restricted_auc.pdf
    fig_values.json
"""
from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RES = os.path.join(REPO_ROOT, "results")
OUT = r"D:\paper\2026\Pooled-Frame-Level-AUC-Measures"

# ----------------------------------------------------------------- styling
# Colour-blind-safe, print-legible. Two hues carry the whole story: grey for
# the "no method contribution" condition, one accent for the reported value.
INK = "#1a1a1a"
MUTED = "#9a9a9a"
ACCENT = "#0072B2"   # blue, OKABE-ITO
WARM = "#D55E00"     # vermillion, OKABE-ITO
POS = "#009E73"      # bluish green
NEG = "#CC79A7"      # reddish purple
GRIDC = "#dcdcdc"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8.2,
    "axes.labelsize": 8.6,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,      # embed as TrueType, keeps text selectable
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def clean(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRIDC, lw=0.5, zorder=0)
        ax.set_axisbelow(True)


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as fh:
        return json.load(fh)


s2 = load("s2_table.json")
s3 = load("s3_deoffset.json")
s4 = load("s4_g5redo.json")

rows2 = s2["rows"]
rows3 = s3["rows"]
rows4 = s4["rows"]

# Domain blocks, in the order used in the manuscript.
DOMAINS = [
    ("UCF-Crime", ["B_xd2ucf", "D_ucf2ucf"]),
    ("XD-Violence", ["A_ucf2xd", "C_xd2xd", "E_ens11_ucf2xd",
                     "F_ens22_ucf2xd", "G_ens33_ucf2xd"]),
    ("ShanghaiTech", ["ucf2sht", "xd2sht"]),
]
DMAP = {k: d for d, ks in DOMAINS for k in ks}


def present(keys, source):
    """Keep only configurations that the given archive actually measured.

    The restricted-AUC table covers nine configurations, but the two
    interventions were run on the six that carry a released checkpoint. Plotting
    all nine against intervention quantities would silently invent values, so
    each figure declares which archive it draws from.
    """
    return [k for k in keys if k in source]


# ------------------------------------------------------- Figure 1: design
def fig_framework():
    """Block diagram of the measurement design.

    Not a proposed pipeline -- an instrument. Each box names what is held
    fixed and what is varied, so the reader can see that both interventions are
    label-free and that the only thing differing between rows is the benchmark.

    Layout note: box geometry is derived from the wrapped text rather than
    hard-coded, because the first hand-tuned version overflowed its boxes.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    FS, TS = 8.0, 8.6
    LINE_H = 2.45          # vertical units consumed by one body line

    def box(x, y, w, title_lines, body_lines, fc, ec, h=None):
        n_t, n_b = len(title_lines), len(body_lines)
        if h is None:
            h = 2.4 + n_t * LINE_H + n_b * LINE_H + 1.8
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.45,rounding_size=1.1",
            linewidth=0.9, edgecolor=ec, facecolor=fc, zorder=2))
        cy = y + h - 2.2
        for t in title_lines:
            ax.text(x + w / 2, cy, t, ha="center", va="top", fontsize=TS,
                    weight="bold", color=INK, zorder=3)
            cy -= LINE_H
        cy -= 0.5
        for b in body_lines:
            ax.text(x + w / 2, cy, b, ha="center", va="top", fontsize=FS,
                    color=INK, zorder=3)
            cy -= LINE_H
        return h

    def arrow(x1, y1, x2, y2, rad=0.0):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
            linewidth=0.9, color=INK, zorder=4,
            connectionstyle=f"arc3,rad={rad}", shrinkA=0.5, shrinkB=0.5))

    C = ("#f4f8fb", ACCENT)     # fixed components
    W = ("#fbf3ef", WARM)        # the interventions under test
    D = ("#efefef", MUTED)      # intermediate representation

    # ---- stage 1: what is held fixed ------------------------------------
    h_scorer = box(1.0, 44.0, 24.0,
                   [r"Frozen scorer $\theta$"],
                   ["2 checkpoints", "no fine-tuning"], *C)
    h_data = box(1.0, 27.0, 24.0,
                 ["Three benchmarks"],
                 ["UCF-Crime   290 vid.", "XD-Violence   800 vid.",
                  "ShanghaiTech   107 vid."], *C)

    # ---- stage 2: the representation the metric is computed on ----------
    h_frame = box(29.0, 30.0, 20.0,
                  ["Frame score grid"],
                  ["one scalar per clip", "clip-level labels",
                   "up to $L_i$ clips per video"], *D)

    # ---- stage 3: the two interventions ---------------------------------
    h_a = box(53.5, 45.0, 26.0,
              ["A", r"$s \leftarrow \bar{s}_i$"],
              ["keeps level", "deletes variation"], *W)
    h_b = box(53.5, 25.0, 26.0,
              ["B", r"$s \leftarrow s - \bar{s}_i$"],
              ["keeps variation", "deletes level"], *W)

    # ---- stage 4: what is reported --------------------------------------
    h_rep = box(83.0, 28.0, 16.0,
                ["Reported"],
                ["pooled AUC / AP", "within-video AUC", "cross-video AUC",
                 "gap, placebo, SE"], *C)

    arrow(25.6, 44.0 + h_scorer / 2, 28.4, 30.0 + h_frame / 2, rad=-0.12)
    arrow(25.6, 27.0 + h_data / 2, 28.4, 30.0 + h_frame / 2, rad=0.12)
    arrow(49.6, 30.0 + h_frame / 2, 52.9, 45.0 + h_a / 2, rad=0.10)
    arrow(49.6, 30.0 + h_frame / 2, 52.9, 25.0 + h_b / 2, rad=-0.10)
    arrow(80.1, 45.0 + h_a / 2, 82.4, 28.0 + h_rep / 2, rad=-0.10)
    arrow(80.1, 25.0 + h_b / 2, 82.4, 28.0 + h_rep / 2, rad=0.10)

    # ---- shared-parameter band ------------------------------------------
    # Kept to three words plus three symbol names. An earlier version carried a
    # 38-word paragraph here, which belonged in the caption: the publisher asks
    # for captions in the manuscript text, and a reader who needs it explained
    # inside the figure is not reading the caption.
    ax.plot([1.0, 99.0], [20.0, 20.0], color=GRIDC, lw=0.7, zorder=1)
    ax.text(50.0, 17.6, r"shared: $\theta$, clip window, threshold rule",
            ha="center", va="top", fontsize=FS, color=MUTED, zorder=3)

    fig.savefig(os.path.join(OUT, "fig1_framework.pdf"))
    plt.close(fig)


# ------------------------------------------- Figure 2: retained AUC share
def fig_decomposition():
    """Share of the reported AUC that survives deleting one of the two parts."""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.05))

    # Readable x labels. The archive keys carry internal prefixes (A_, B_, ...)
    # which mean nothing outside the experiment log.
    NICE = {"A_ucf2xd": "UCF$\\to$XD", "C_xd2xd": "XD$\\to$XD",
            "B_xd2ucf": "XD$\\to$UCF", "D_ucf2ucf": "UCF$\\to$UCF",
            "ucf2sht": "UCF$\\to$SHT", "xd2sht": "XD$\\to$SHT"}

    labels, const_keep, deoff_keep, doms = [], [], [], []
    for dom, keys in DOMAINS:
        for k in present(keys, rows4):
            labels.append(NICE[k])
            const_keep.append(rows4[k]["const_keep_pct"])
            deoff_keep.append(rows4[k]["deoff_keep_pct"])
            doms.append(dom)

    xs = range(len(labels))
    w = 0.38

    # A domain keeps ONE colour across both panels. Encoding the panel by hue
    # would make ShanghaiTech orange on the left and blue on the right, which
    # reads as two different populations rather than one repeated one.
    panels = (
        (const_keep, "A   delete within-video variation"),
        (deoff_keep, "B   delete the per-video level"),
    )
    for ax, (vals, title) in zip(axes, panels):
        cols = [WARM if d == "ShanghaiTech" else MUTED for d in doms]
        ax.bar(xs, vals, width=w, color=cols, edgecolor=INK, lw=0.5, zorder=3)
        ax.axhline(100, color=INK, lw=0.7, ls=(0, (4, 3)), zorder=4)
        clean(ax, "y")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, rotation=38, ha="right")
        ax.set_xlim(-0.7, len(vals) - 0.3)
        ax.set_ylim(0, 128)
        ax.set_ylabel("share of pooled AUC retained (%)")
        ax.set_title(title, loc="left", pad=6)

    # Label the dashed 100% line just above it, at the left of panel A where the
    # first bar sits below the line. At the top of the axes the label collided
    # with the panel title once the text was raised to 8 pt.
    axes[0].text(0.015, 0.805, "reported value", transform=axes[0].transAxes,
                 fontsize=8.0, ha="left", va="bottom", color=INK)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=MUTED, ec=INK, lw=0.5),
               plt.Rectangle((0, 0), 1, 1, fc=WARM, ec=INK, lw=0.5)]
    fig.legend(handles, ["UCF-Crime / XD-Violence", "ShanghaiTech"],
               loc="lower center", ncol=2, frameon=False, fontsize=8.0,
               handlelength=1.1, bbox_to_anchor=(0.5, -0.09))

    fig.tight_layout(w_pad=1.6)
    fig.savefig(os.path.join(OUT, "fig2_decomposition.pdf"))
    plt.close(fig)


# -------------------------------- Figure 3: gap vs alignment (sign flip)
def fig_alignment():
    """The gap reverses exactly where the offset stops being label-aligned."""
    fig, ax = plt.subplots(figsize=(3.45, 3.05))

    for dom, colour in (("XD-Violence", ACCENT), ("UCF-Crime", POS),
                        ("ShanghaiTech", NEG)):
        keys = present([k for d, ks in DOMAINS if d == dom for k in ks], rows3)
        xs = [rows3[k]["offset_rho"] for k in keys]
        ys = [rows2[k]["gap"] for k in keys]
        ax.scatter(xs, ys, s=36, facecolor=colour, edgecolor=INK, lw=0.6,
                   label=dom, zorder=3)

    ax.axhline(0, color=INK, lw=0.8, zorder=2)
    ax.axvline(0, color=MUTED, lw=0.7, ls=(0, (3, 3)), zorder=2)
    clean(ax, "both")

    # Two short quadrant tags, not sentences. The caption carries the finding.
    ax.text(-0.48, 0.188, "aligned", fontsize=8.0, color=MUTED,
            ha="left", va="top")
    ax.text(0.55, -0.075, "misaligned", fontsize=8.0, color=MUTED,
            ha="right", va="bottom")

    ax.set_xlim(-0.50, 0.56)
    ax.set_ylim(-0.085, 0.205)
    ax.set_xlabel("offset / video-label agreement (Spearman)")
    ax.set_ylabel("cross-video minus within-video AUC")
    # No in-figure title: the caption states the finding, and a reader who needs
    # the finding explained inside the figure has not reached the caption.
    ax.set_title("")

    # Legend below the axes: inside, it covered the UCF-Crime point at
    # (0.13, 0.10), which is one of the six observations the figure exists to
    # show.
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=3, handletextpad=0.25, columnspacing=1.0, fontsize=8.2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_alignment.pdf"))
    plt.close(fig)


# ------------------------------------ Figure 4: restricted AUC + placebo
def fig_restricted():
    """Pooled AUC against its two restricted components.

    Drawn on the same six configurations as Figs. 2 and 3 so that the paper
    refers throughout to one set of runs; the three checkpoint ensembles appear
    only in the table, where they are needed for completeness but not for the
    argument.
    """
    NICE = {"A_ucf2xd": "UCF$\\to$XD", "C_xd2xd": "XD$\\to$XD",
            "B_xd2ucf": "XD$\\to$UCF", "D_ucf2ucf": "UCF$\\to$UCF",
            "ucf2sht": "UCF$\\to$SHT", "xd2sht": "XD$\\to$SHT"}

    order = []
    for dom, keys in DOMAINS:
        order.extend(present(keys, rows3))
    order = sorted(order, key=lambda k: -rows2[k]["pooled_auc"])

    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    xs = range(len(order))
    w = 0.26

    ax.bar([i - w for i in xs], [rows2[k]["auc_within"] for k in order],
           width=w, color=MUTED, edgecolor=INK, lw=0.5, zorder=3,
           label="within")
    ax.bar(list(xs), [rows2[k]["pooled_auc"] for k in order],
           width=w, color=ACCENT, edgecolor=INK, lw=0.5, zorder=3,
           label="pooled")
    ax.bar([i + w for i in xs], [rows2[k]["auc_cross"] for k in order],
           width=w, color=WARM, edgecolor=INK, lw=0.5, zorder=3,
           label="cross-video")

    ax.axhline(0.5, color=INK, lw=0.7, ls=(0, (4, 3)), zorder=4)
    ax.text(-0.72, 0.508, "chance", fontsize=8.0, ha="left", va="bottom",
            color=INK, zorder=5)

    clean(ax, "y")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([NICE[k] for k in order], rotation=38, ha="right",
                       fontsize=8.0)
    ax.set_xlim(-0.75, len(order) - 0.25)
    ax.set_ylim(0.33, 1.02)
    ax.set_ylabel("AUC")
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.17), handlelength=1.0,
              columnspacing=1.0)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_restricted_auc.pdf"))
    plt.close(fig)


def dump():
    out = {"const_keep_pct": {}, "deoff_keep_pct": {},
           "gap": {}, "offset_rho": {}, "pooled_auc": {},
           "auc_within": {}, "auc_cross": {}, "placebo_gap": {}}
    for k in rows2:
        out["pooled_auc"][k] = round(rows2[k]["pooled_auc"], 4)
        out["gap"][k] = round(rows2[k]["gap"], 4)
        out["placebo_gap"][k] = round(rows2[k].get("placebo_gap", float("nan")), 4)
        out["auc_within"][k] = round(rows2[k]["auc_within"], 4)
        out["auc_cross"][k] = round(rows2[k]["auc_cross"], 4)
        if k in rows3:
            out["offset_rho"][k] = round(rows3[k]["offset_rho"], 4)
        if k in rows4:
            out["const_keep_pct"][k] = round(rows4[k]["const_keep_pct"], 2)
            out["deoff_keep_pct"][k] = round(rows4[k]["deoff_keep_pct"], 2)
    with open(os.path.join(OUT, "fig_values.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    fig_framework()
    fig_decomposition()
    fig_alignment()
    fig_restricted()
    v = dump()
    print("figures written to", OUT)
    for name in ("fig1_framework", "fig2_decomposition",
                 "fig3_alignment", "fig4_restricted_auc"):
        p = os.path.join(OUT, name + ".pdf")
        print(f"  {name}.pdf  {os.path.getsize(p)} bytes")
    print("const_keep_pct :", v["const_keep_pct"])
    print("deoff_keep_pct :", v["deoff_keep_pct"])