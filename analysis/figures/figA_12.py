# -*- coding: utf-8 -*-
"""Fig 12 + Fig 11 —— 精修版"""
import os, sys, json
sys.path.insert(0, r"W:\虚拟细胞\analysis")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import *
apply()

A = r"W:\虚拟细胞\analysis"

# ==================== Fig 12 ====================
d = json.load(open(os.path.join(A, "supp_08b_official_specificity.json"), encoding="utf-8"))
rows = d["orig"]; M = np.array(d["matrix_orig"]); CH = d["channels"]
name2i = {c: i for i, c in enumerate(CH)}
order = sorted(range(len(rows)), key=lambda k: -rows[k]["specificity"])
ylab = [rows[k]["channel"] for k in order]
spec = np.array([rows[k]["specificity"] for k in order])
diag = np.array([rows[k]["diag"] for k in order])
bother = np.array([rows[k]["best_other"] for k in order])

fig = plt.figure(figsize=(9.4, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.12], height_ratios=[1.0, 0.92],
                      wspace=0.42, hspace=0.52, left=0.075, right=0.975, top=0.90, bottom=0.075)

# (a) 23x23 矩阵
axA = ax = fig.add_subplot(gs[:, 0])
im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=float(np.nanmax(M)), origin="upper", aspect="equal")
ax.set_xticks(range(23)); ax.set_yticks(range(23))
ax.set_xticklabels(CH, rotation=90, fontsize=4.6)
ax.set_yticklabels(CH, fontsize=4.6)
for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
    if t.get_text() in ("TRITC", "Cy5"):
        t.set_color(C_CTRL); t.set_fontweight("bold")
ax.set_xlabel("Reference mask channel", fontsize=FS_BASE, labelpad=2)
ax.set_ylabel("Virtual channel", fontsize=FS_BASE, labelpad=2)
for i in range(23):
    ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, ec="#3fd8ff", lw=0.6))
cb = fig.colorbar(im, ax=ax, fraction=0.043, pad=0.025)
cb.set_label("Dice", fontsize=FS_BASE); cb.ax.tick_params(labelsize=FS_TICK)
ax.set_title("Every channel against every reference mask", fontsize=FS_BASE, loc="left", pad=5)
# 圈出"互相可互换"的髓系/免疫通道块
_b = ["CD68_1:100", "PD-L1", "CD16", "CD11c", "CD4"]
_ix = [CH.index(c) for c in _b]
_r0, _r1 = min(_ix) - 0.5, max(_ix) + 0.5
ax.add_patch(plt.Rectangle((_r0, _r0), _r1 - _r0, _r1 - _r0, fill=False, ec="#ffd166", lw=1.4, ls="-", zorder=5))
ax.annotate("near-interchangeable\nimmune family", xy=(_r1, _r0 + 1.0), xytext=(16.4, 7.6),
            fontsize=5.4, color="#ffd166", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="#ffd166", lw=0.7, shrinkA=0, shrinkB=1))

# (b) 特异性 lollipop
axB = ax = fig.add_subplot(gs[0, 1])
y = np.arange(len(spec))
cols = [C_FOCAL if v > 0 else C_CTRL for v in spec]
for k in range(len(spec)):
    ax.plot([0, spec[k]], [y[k], y[k]], color=cols[k], lw=1.3, solid_capstyle="butt", zorder=1)
ax.scatter(spec, y, s=24, c=cols, zorder=3, edgecolors="white", linewidths=0.5)
ax.axvline(0, color="#444444", lw=0.7)
ax.set_yticks(y); ax.set_yticklabels(ylab, fontsize=5.4)
for t in ax.get_yticklabels():
    if t.get_text() in ("TRITC", "Cy5"): t.set_color(C_CTRL); t.set_fontweight("bold")
    if t.get_text() == "DAPI": t.set_color(C_FOCAL); t.set_fontweight("bold")
ax.invert_yaxis()
ax.set_xlabel("Specificity (paired $-$ best competing)")
ax.set_xlim(min(spec) - 0.045, max(spec) + 0.13)
ax.set_title("Only DAPI clears its own confusion set", fontsize=FS_BASE, loc="left", pad=5)
_it = ylab.index("TRITC")
ax.annotate("TRITC: declared\nbackground", xy=(spec[_it], y[_it]), xytext=(spec[_it] + 0.055, y[_it] + 3.1),
            fontsize=5.4, color=C_CTRL, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=C_CTRL, lw=0.6, shrinkA=1, shrinkB=2))

# (c) 配对 vs 最强错配
axC = ax = fig.add_subplot(gs[1, 1])
ax.plot([0, 0.75], [0, 0.75], color="#999999", lw=0.7, ls="--", zorder=1)
other = [k for k in range(len(rows)) if rows[k]["channel"] not in ("TRITC", "Cy5") and rows[k]["channel"] != "DAPI"]
ctl = [k for k in range(len(rows)) if rows[k]["channel"] in ("TRITC", "Cy5")]
ax.scatter([rows[k]["best_other"] for k in other], [rows[k]["diag"] for k in other],
           s=22, c=C_GREY, edgecolors="white", linewidths=0.5, zorder=3)
ax.scatter([rows[k]["best_other"] for k in ctl], [rows[k]["diag"] for k in ctl],
           s=38, c=C_CTRL, marker="D", edgecolors="white", linewidths=0.5, zorder=4)
ax.scatter([rows[name2i["DAPI"]]["best_other"]], [rows[name2i["DAPI"]]["diag"]],
           s=48, c=C_FOCAL, marker="s", edgecolors="white", linewidths=0.5, zorder=4)
ax.set_xlabel("Best competing channel")
ax.set_ylabel("Own channel")
ax.set_xlim(0.02, 0.70); ax.set_ylim(0.02, 0.82)
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=4, mfc=C_GREY, mec="white", label="protein channel"),
                   Line2D([], [], marker="D", ls="", ms=4.5, mfc=C_CTRL, mec="white", label="background channel"),
                   Line2D([], [], marker="s", ls="", ms=5, mfc=C_FOCAL, mec="white", label="DAPI")],
          frameon=False, fontsize=5.6, loc="upper left", handletextpad=0.25, borderaxespad=0.5)
ax.set_title("Marker channels sit on the identity line", fontsize=FS_BASE, loc="left", pad=5)

plabel(axA, "a"); plabel(axB, "b"); plabel(axC, "c")
audit(fig, "Fig12")
save(fig, "Fig12_official_specificity")
plt.close(fig)
