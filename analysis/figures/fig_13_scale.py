"""Fig 13：HEMIT 物理尺度标定 + 尺度扫描（重写版：直接标注线端，避免图例冲突）"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"C:\Users\PC\.agents\skills\figure-style")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kernel import apply_figure_style, panel_letter

apply_figure_style(frame="open", sizes=(8, 7, 6))
OUT = r"W:\虚拟细胞\figures"; A = r"W:\虚拟细胞\analysis"
C1, C2, C3 = "#1f6fb4", "#d1622f", "#5b8c5a"

a = json.load(open(os.path.join(A, "supp_09a_hemit_scale.json"), encoding="utf-8"))
b = json.load(open(os.path.join(A, "supp_09b_hemit_scale_sweep.json"), encoding="utf-8"))
GIGA, M0, SC = b["giga"], b["m0"], b["scales"]
fov = np.array([256*M0/s for s in SC])
FOV_TRAIN = 256*GIGA
res = b["result"]
XT = [40, 60, 80, 100, 150]

fig = plt.figure(figsize=(8.8, 5.9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], height_ratios=[1.0, 1.0],
                      wspace=0.32, hspace=0.60)

# ---------------- (a) 核直径标定 ----------------
ax = fig.add_subplot(gs[:, 0])
gq = {float(k): v for k, v in a["gigatime_pct"].items()}
hq = {float(k): v for k, v in a["hemit_pct"].items()}
for row, (q, col) in enumerate([(gq, C1), (hq, C2)]):
    ax.plot([q[10], q[90]], [row, row], color=col, lw=1.0, zorder=1)
    ax.plot([q[25], q[75]], [row, row], color=col, lw=6.0, solid_capstyle="butt", alpha=0.30, zorder=1)
    ax.scatter([q[50]], [row], s=60, c=col, edgecolors="white", linewidths=0.8, zorder=3)
    ax.annotate(f"{q[50]:.1f} px", xy=(q[50], row), xytext=(0, 30), textcoords="offset points",
                ha="center", fontsize=6.8, color=col, fontweight="bold")
ax.set_yticks([0, 1])
ax.set_yticklabels(["GigaTIME official\n(0.2302 µm/px, known)", "HEMIT\n(scale unknown)"], fontsize=6.8)
ax.set_ylim(1.62, -0.62); ax.set_xlim(4, 46)
ax.set_xlabel("Nuclear equivalent diameter (px)\n(identical segmentation on both datasets)", fontsize=7)
ax.set_title("DAPI nuclear size pins the unknown scale", fontsize=7.5, loc="left", pad=6)
_s = a["mpp_sensitivity"]
ax.text(0.03, 0.055,
        f"median 25.6 px = {a['ref_nucleus_um']:.2f} µm  (known)\n"
        f"→ HEMIT 20.7 px  ⇒  $m_0$ = {a['hemit_mpp_estimate']:.3f} µm/px\n"
        f"      = {a['ratio_to_gigatime']:.2f}× the training scale\n"
        f"sensitivity (Q25/Q50/Q75): " + " / ".join(f"{_s[k]:.3f}" for k in ("25","50","75")),
        transform=ax.transAxes, fontsize=6.2, color="#333333", ha="left", va="bottom")
ax.text(0.97, 0.98, "box = IQR, whisker = 10–90th pct", transform=ax.transAxes,
        fontsize=5.8, color="#888888", va="top", ha="right")
panel_letter(ax, "a", dx=-0.24, dy=1.03, fontsize=11)

# ---------------- (b) Dice vs 视场 ----------------
ax = fig.add_subplot(gs[0, 1])
ax.axvspan(FOV_TRAIN*0.94, FOV_TRAIN*1.06, color="#e2e2e2", lw=0, zorder=0)
for g, col in (("DAPI", C1), ("CK", C2), ("CD3", C3)):
    y = np.array(res[g]["dice_default"])
    ax.plot(fov, y, color=col, lw=1.3, marker="o", ms=3.4, mec="white", mew=0.5, zorder=3)
    ax.annotate(g, xy=(fov[-1], y[-1]), xytext=(5, 0), textcoords="offset points",
                fontsize=6.5, color=col, va="center", fontweight="bold")
ax.set_xscale("log"); ax.set_xticks(XT); ax.set_xticklabels([str(v) for v in XT])
ax.set_xlim(fov.min()*0.91, fov.max()*1.22)
ax.set_xlabel("Physical field of view per 256-px window (µm)", fontsize=7)
ax.set_ylabel("Dice vs. mIHC (Otsu)", fontsize=7)
ax.annotate("training FOV", xy=(FOV_TRAIN, 0.0), xytext=(FOV_TRAIN, 0.055),
            fontsize=5.6, color="#777777", ha="center", clip_on=False)
ax.set_title("DAPI peaks at the training scale; CK and CD3 do not", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "b", dx=-0.24, dy=1.085, fontsize=11)

# ---------------- (c) 配对通道排名 ----------------
ax = fig.add_subplot(gs[1, 1])
ax.axvspan(FOV_TRAIN*0.94, FOV_TRAIN*1.06, color="#e2e2e2", lw=0, zorder=0)
for g, col in (("DAPI", C1), ("CK", C2), ("CD3", C3)):
    rk = np.array([res["specificity_vs_scale"][g][str(s)]["rank"] for s in SC], float)
    ax.plot(fov, rk, color=col, lw=1.3, marker="o", ms=3.4, mec="white", mew=0.5, zorder=3)
    ax.annotate(g, xy=(fov[-1], rk[-1]), xytext=(5, 0), textcoords="offset points",
                fontsize=6.5, color=col, va="center", fontweight="bold")
ax.axhline(1, color="#555555", lw=0.8, ls="--", zorder=1)
ax.set_yscale("log"); ax.set_yticks([1, 2, 5, 10, 20]); ax.set_yticklabels(["1", "2", "5", "10", "20"])
ax.set_ylim(0.85, 30); ax.invert_yaxis()
ax.set_xscale("log"); ax.set_xticks(XT); ax.set_xticklabels([str(v) for v in XT])
ax.set_xlim(fov.min()*0.91, fov.max()*1.30)
ax.set_xlabel("Physical field of view per 256-px window (µm)", fontsize=7)
ax.set_ylabel("Rank of paired channel\n(1 = best of 23; higher = worse)", fontsize=7)
ax.text(fov.max()*1.16, 1.0, "rank 1", fontsize=5.6, color="#555555", va="center", ha="right")
ax.set_title("Only DAPI stays rank 1 across a 4-fold range", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "c", dx=-0.24, dy=1.085, fontsize=11)

fig.savefig(os.path.join(OUT, "Fig13_hemit_scale.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Fig13_hemit_scale.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Fig13 done")
