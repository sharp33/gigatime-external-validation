# -*- coding: utf-8 -*-
"""Fig 13 / Fig 2 / Fig 3 —— 精修版"""
import os, sys, json
sys.path.insert(0, r"W:\虚拟细胞\analysis")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import *
apply()
A = r"W:\虚拟细胞\analysis"

# ==================== Fig 13 ====================
a13 = json.load(open(os.path.join(A, "supp_09a_hemit_scale.json"), encoding="utf-8"))
b13 = json.load(open(os.path.join(A, "supp_09b_hemit_scale_sweep.json"), encoding="utf-8"))
GIGA, M0, SC = b13["giga"], b13["m0"], b13["scales"]
fov = np.array([256 * M0 / s for s in SC]); FOV_T = 256 * GIGA
res = b13["result"]; XT = [40, 60, 80, 100, 150]
fig = plt.figure(figsize=(9.4, 5.8))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.18], height_ratios=[1.0, 1.0],
                      wspace=0.34, hspace=0.62, left=0.09, right=0.975, top=0.90, bottom=0.095)
axA = ax = fig.add_subplot(gs[:, 0])
gq = {float(k): v for k, v in a13["gigatime_pct"].items()}
hq = {float(k): v for k, v in a13["hemit_pct"].items()}
for row, (q, col) in enumerate([(gq, C_FOCAL), (hq, C_CTRL)]):
    ax.plot([q[10], q[90]], [row, row], color=col, lw=1.0, zorder=1)
    ax.plot([q[25], q[75]], [row, row], color=col, lw=6.0, solid_capstyle="butt", alpha=0.30, zorder=1)
    ax.scatter([q[50]], [row], s=58, c=col, edgecolors="white", linewidths=0.8, zorder=3)
    ax.annotate("%.1f px" % q[50], xy=(q[50], row), xytext=(0, 26), textcoords="offset points",
                ha="center", fontsize=FS_SEC, color=col, fontweight="bold")
ax.set_yticks([0, 1])
ax.set_yticklabels(["GigaTIME official\n0.2302 $\\mu$m/px (known)", "HEMIT\nscale unknown"], fontsize=FS_TICK)
ax.set_ylim(1.62, -0.62); ax.set_xlim(4, 46)
ax.set_xlabel("Nuclear equivalent diameter (px)\n(identical segmentation, both datasets)")
ax.set_title("DAPI nuclear size pins the unknown scale", fontsize=FS_BASE, loc="left", pad=5)
_s = a13["mpp_sensitivity"]
ax.text(0.03, 0.045, "median 25.6 px = %.2f $\\mu$m  (known)\n"
        "$\\rightarrow$ HEMIT 20.7 px  $\\Rightarrow$  $m_0$ = %.3f $\\mu$m/px\n"
        "     = %.2f$\\times$ the training scale\n"
        "sensitivity (Q25/Q50/Q75): " % (a13["ref_nucleus_um"], a13["hemit_mpp_estimate"], a13["ratio_to_gigatime"])
        + " / ".join("%.3f" % _s[k] for k in ("25", "50", "75")),
        transform=ax.transAxes, fontsize=FS_TICK - 0.2, color="#333333", ha="left", va="bottom")
ax.text(0.97, 0.975, "box = IQR, whisker = 10-90th pct", transform=ax.transAxes,
        fontsize=FS_TICK - 0.6, color="#999999", va="top", ha="right")

axB = ax = fig.add_subplot(gs[0, 1])
ax.axvspan(FOV_T * 0.94, FOV_T * 1.06, color="#e6e6e6", lw=0, zorder=0)
for g, col in (("DAPI", C_FOCAL), ("CK", C_CTRL), ("CD3", "#5b8c5a")):
    y = np.array(res[g]["dice_default"])
    ax.plot(fov, y, color=col, lw=1.3, marker="o", ms=3.2, mec="white", mew=0.5, zorder=3)
    ax.annotate(g, xy=(fov[-1], y[-1]), xytext=(5, 0), textcoords="offset points",
                fontsize=FS_SEC, color=col, va="center", fontweight="bold")
ax.set_xscale("log"); ax.set_xticks(XT); ax.set_xticklabels([str(v) for v in XT], fontsize=FS_TICK)
ax.set_xlim(fov.min() * 0.90, fov.max() * 1.30)
ax.set_xlabel("Physical field of view per 256-px window ($\\mu$m)")
ax.set_ylabel("Dice vs. mIHC (Otsu)")
ax.annotate("training\nfield of view", xy=(FOV_T, 0.015), xytext=(FOV_T, 0.075),
            fontsize=FS_TICK - 0.6, color="#888888", ha="center", va="bottom")
ax.set_title("DAPI peaks at the training scale; CK and CD3 do not", fontsize=FS_BASE, loc="left", pad=5)

axC = ax = fig.add_subplot(gs[1, 1])
ax.axvspan(FOV_T * 0.94, FOV_T * 1.06, color="#e6e6e6", lw=0, zorder=0)
for g, col in (("DAPI", C_FOCAL), ("CK", C_CTRL), ("CD3", "#5b8c5a")):
    rk = np.array([res["specificity_vs_scale"][g][str(s)]["rank"] for s in SC], float)
    ax.plot(fov, rk, color=col, lw=1.3, marker="o", ms=3.2, mec="white", mew=0.5, zorder=3)
    ax.annotate(g, xy=(fov[-1], rk[-1]), xytext=(5, 0), textcoords="offset points",
                fontsize=FS_SEC, color=col, va="center", fontweight="bold")
ax.axhline(1, color="#555555", lw=0.8, ls="--", zorder=1)
ax.set_yscale("log"); ax.set_yticks([1, 2, 5, 10, 20]); ax.set_yticklabels(["1", "2", "5", "10", "20"], fontsize=FS_TICK)
ax.set_ylim(0.85, 30); ax.invert_yaxis()
ax.set_xscale("log"); ax.set_xticks(XT); ax.set_xticklabels([str(v) for v in XT], fontsize=FS_TICK)
ax.set_xlim(fov.min() * 0.90, fov.max() * 1.30)
ax.set_xlabel("Physical field of view per 256-px window ($\\mu$m)")
ax.set_ylabel("Rank of nominal channel\n(1 = best of 23)")

ax.set_title("Only DAPI stays rank 1 across a four-fold range", fontsize=FS_BASE, loc="left", pad=5)
plabel(axA, "a"); plabel(axB, "b"); plabel(axC, "c")
audit(fig, "Fig13"); save(fig, "Fig13_hemit_scale"); plt.close(fig)

# ==================== Fig 2 ====================
mv = json.load(open(os.path.join(A, "supp_03_l1_modelvariant.json"), encoding="utf-8"))
df = json.load(open(os.path.join(A, "supp_03c_density_fairness.json"), encoding="utf-8"))
CONDS = [("flash@1024_reinhard", "Flash\nslide"), ("flash@1024_none", "Flash\nnone"),
         ("orig@1024_reinhard", "orig\nslide"), ("orig@1024_none", "orig\nnone"),
         ("orig@512_reinhard", "orig512\nslide"), ("orig@512_none", "orig512\nnone")]
CONDS2 = [c for c in CONDS if "%s|DAPI" % c[0] in df]
CH3 = ["DAPI", "CK", "CD3"]; COLS3 = [C_FOCAL, C_CTRL, "#5b8c5a"]
fig = plt.figure(figsize=(9.4, 3.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.30, 1.0], wspace=0.34,
                      left=0.055, right=0.985, top=0.86, bottom=0.235)
for pi, ttl in enumerate(("Default threshold (0.5)", "Density-matched threshold")):
    ax = fig.add_subplot(gs[0, pi])
    use = CONDS if pi == 0 else CONDS2
    x = np.arange(len(use)); w = 0.26
    for k, nm in enumerate(CH3):
        vals, base = [], []
        for cond, _ in use:
            if pi == 0:
                vals.append(mv[cond][nm]["dice"]); base.append(mv[cond][nm]["baseline"])
            else:
                vals.append(df["%s|%s" % (cond, nm)]["dice_density_matched"])
                base.append(df["%s|%s" % (cond, nm)]["baseline"])
        ax.bar(x + (k - 1) * w, vals, w, color=COLS3[k], zorder=2)
        ax.hlines(base[0], x[0] - 1.5 * w, x[-1] + 1.5 * w, color=COLS3[k], lw=0.8, ls="--", alpha=0.75, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in use], fontsize=FS_TICK - 0.6)
    ax.set_ylim(0, 0.92)
    if pi == 0: ax.set_ylabel("Dice vs. mIHC (Otsu)")
    hs = [Line2D([], [], marker="s", ls="", ms=5, mfc=c, mec="none", label=n) for n, c in zip(CH3, COLS3)]
    hs.append(Line2D([], [], color="#777777", ls="--", lw=0.9, label="random baseline"))
    ax.legend(handles=hs, frameon=False, fontsize=FS_TICK - 0.8, loc="upper right",
              handletextpad=0.3, borderaxespad=0.3)
    ax.set_title(ttl, fontsize=FS_BASE, loc="left", pad=5)
    if pi == 0: axA = ax
    else: axB = ax
axC = ax = fig.add_subplot(gs[0, 2])
for k, nm in enumerate(CH3):
    for cond, _ in CONDS2:
        dd = df["%s|%s" % (cond, nm)]
        ax.scatter([dd["gt_density"]], [dd["pred_density"]], s=26, c=COLS3[k], zorder=3,
                   edgecolors="white", linewidths=0.4)
ax.plot([0.03, 0.45], [0.03, 0.45], color="#a8a8a8", lw=0.8, ls="--", zorder=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([0.05, 0.1, 0.2, 0.3]); ax.set_xticklabels(["0.05", "0.1", "0.2", "0.3"], fontsize=FS_TICK)
ax.minorticks_off()
ax.set_xlabel("Ground-truth density"); ax.set_ylabel("Predicted density")
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=4, mfc=c, mec="white", label=n) for n, c in zip(CH3, COLS3)],
          frameon=False, fontsize=FS_TICK - 0.8, loc="lower right", handletextpad=0.3, borderaxespad=0.4)
ax.set_title("Density calibration breaks", fontsize=FS_BASE, loc="left", pad=5)
plabel(axA, "a"); plabel(axB, "b"); plabel(axC, "c")
audit(fig, "Fig2"); save(fig, "Fig02_L1_HEMIT"); plt.close(fig)
