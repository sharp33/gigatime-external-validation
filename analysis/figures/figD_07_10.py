# -*- coding: utf-8 -*-
"""Fig 7 / Fig 10 / Fig 13 —— 精修版"""
import os, sys, json
sys.path.insert(0, r"W:\虚拟细胞\analysis")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import *
apply()
A = r"W:\虚拟细胞\analysis"
CH = ["DAPI","TRITC","Cy5","PD-1_1:200","CD14","CD4","T-bet","CD34","CD68_1:100","CD16","CD11c",
      "CD138","CD20","CD3_1:1000","CD8","PD-L1","CK_1:150","Ki67_1:150","Tryptase","Actin-D",
      "Caspase3-D","PHH3-B","Transgelin"]

# ==================== Fig 7 ====================
hem = json.load(open(os.path.join(A, "supp_09b_hemit_scale_sweep.json"), encoding="utf-8"))
sp = json.load(open(os.path.join(A, "l2", "l2_specificity.json"), encoding="utf-8"))
foc = json.load(open(os.path.join(A, "l3", "l3_paired_focus.json"), encoding="utf-8"))
comp = json.load(open(os.path.join(A, "supp_05_composition_axis.json"), encoding="utf-8"))
c11 = json.load(open(os.path.join(A, "supp_11_composition_recovery.json"), encoding="utf-8"))
pp7 = json.load(open(os.path.join(A, "orion", "preproc_ablation_summary.json"), encoding="utf-8"))["per_patient"]
rate41 = {}
for pat in sorted(pp7):
    for dd in pp7[pat]["tile"]["detail"]:
        if dd.get("paired_rho") is not None:
            rate41.setdefault(dd["marker"], []).append(1 if dd["specific"] else 0)
L1_win = sum(1 for g in ("DAPI", "CK", "CD3") if hem["result"]["specificity_vs_scale"][g]["1.0"]["rank"] == 1)
L2_win = sum(1 for x in sp["specificity"] if x["specificity"] > 0)
L3_win = sum(1 for t in ("CD20", "Caspase3", "CD31") if foc[t]["rank"] == 1)
L4_win = sum(1 for m, v in rate41.items() if np.mean(v) > 0.5)
LAY = [("L1\nHEMIT", L1_win, 3, "945\npatches", 945), ("L2\nVisium", L2_win, 11, "15,489\nspots", 15489),
       ("L3\nTCGA", L3_win, 3, "222\npatients", 222), ("L4\nORION", L4_win, 13, "994,729\ncells", 994729)]
wr = [100 * w / t for _, w, t, _, _ in LAY]

fig = plt.figure(figsize=(9.4, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.82, 1.25], wspace=0.34,
                      left=0.065, right=0.985, top=0.86, bottom=0.24)
axA = ax = fig.add_subplot(gs[0, 0])
ax.bar(range(4), wr, 0.60, color=[C_FOCAL if v >= 25 else C_CTRL for v in wr], zorder=2)
for k, (nm, w, t, nl, _) in enumerate(LAY):
    ax.text(k, wr[k] + 1.6, "%d/%d" % (w, t), ha="center", fontsize=FS_TICK - 0.2, color="#333333")
ax.set_xticks(range(4))
ax.set_xticklabels(["%s\n%s" % (n, nl) for n, _, _, nl, _ in LAY], fontsize=FS_TICK - 0.8)
ax.set_ylim(0, 44); ax.set_ylabel("Pairs passing specificity (%)")
ax.set_title("Most pairs fail at every layer", fontsize=FS_BASE, loc="left", pad=5)

axB = ax = fig.add_subplot(gs[0, 1])
cv = [comp["orion"]["rho_pc1_vs_real_axis"], c11["association"]["composition"]["pc1"]["rho"]]
ax.barh([0, 1], cv, 0.50, color=C_FOCAL, zorder=2)
ax.set_yticks([0, 1]); ax.set_yticklabels(["cell level\n994,729 cells", "patient level\n41 patients"],
                                          fontsize=FS_TICK - 0.6)
ax.invert_yaxis(); ax.set_xlim(0, 0.92)
ax.set_xlabel("$\\rho$: virtual vs measured composition\n(TCGA: 56 of 175 proteins track the same axis)")
for k, v in enumerate(cv):
    ax.text(v + 0.02, k, "%.2f" % v, va="center", fontsize=FS_TICK, color="#333333")
ax.set_title("Composition recovers (ORION)", fontsize=FS_BASE, loc="left", pad=5)

axC = ax = fig.add_subplot(gs[0, 2])
for k, (nm, w, t, nl, unit) in enumerate(LAY):
    ax.scatter(unit, wr[k], s=46, c=C_FOCAL if wr[k] >= 25 else C_CTRL, zorder=3,
               edgecolors="white", linewidths=0.6)
    ax.annotate(nm.split("\n")[0], xy=(unit, wr[k]), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=FS_TICK - 0.4, color="#666666")
ax.axhspan(0, 20, color="#f5f5f5", zorder=0, lw=0)
ax.set_xscale("log"); ax.set_xlim(120, 4.5e6); ax.set_ylim(-3, 46)
ax.set_xlabel("Measured units in the layer (log)"); ax.set_ylabel("Pairs passing specificity (%)")
ax.set_title("More measurement does not rescue it", fontsize=FS_BASE, loc="left", pad=5)
plabel(axA, "a"); plabel(axB, "b"); plabel(axC, "c")
audit(fig, "Fig7"); save(fig, "Fig07_summary"); plt.close(fig)

# ==================== Fig 10 ====================
o = json.load(open(os.path.join(A, "supp_03_official_sample.json"), encoding="utf-8"))
p10 = json.load(open(os.path.join(A, "orion", "preproc_ablation_summary.json"), encoding="utf-8"))
VAR = p10["variants"]; agg = p10["aggregate"]; pcm = p10["n_specific_mean"]
t10 = json.load(open(os.path.join(A, "orion", "tile_sensitivity.json"), encoding="utf-8"))
rb10 = json.load(open(os.path.join(A, "supp_04_l3_robustness.json"), encoding="utf-8"))
abl = rb10["aggregation_ablation"]

fig = plt.figure(figsize=(9.4, 6.2))
gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.70, left=0.075, right=0.975, top=0.90, bottom=0.10)
axA = ax = fig.add_subplot(gs[0, 0])
sizes = [256, 512, 556]
fl = [o["flash@%d" % s]["mean"] for s in sizes]; og = [o["orig@%d" % s]["mean"] for s in sizes]
x = np.arange(3); w = 0.36
ax.bar(x - w/2, og, w, color=C_CTRL, zorder=2)
ax.bar(x + w/2, fl, w, color=C_FOCAL, zorder=2)
ax.axhline(o["official_precomputed"], color="#444444", lw=0.9, ls="--", zorder=3)
ax.text(2.45, o["official_precomputed"] + 0.012, "authors' stored value", fontsize=FS_TICK - 0.6,
        color="#444444", ha="right")
ax.set_xticks(x); ax.set_xticklabels([str(s) for s in sizes], fontsize=FS_TICK)
ax.set_xlabel("Input resolution (px)"); ax.set_ylabel("Dice (official protocol)")
ax.set_ylim(0, 0.42)
ax.legend(handles=[Line2D([], [], marker="s", ls="", ms=5, mfc=C_CTRL, mec="none", label="original U-Net++"),
                   Line2D([], [], marker="s", ls="", ms=5, mfc=C_FOCAL, mec="none", label="GigaTIME-Flash")],
          frameon=False, fontsize=FS_TICK - 0.6, loc="upper left", handletextpad=0.3, borderaxespad=0.4)
ax.set_title("Distillation did not change the metric", fontsize=FS_BASE, loc="left", pad=5)

axB = ax = fig.add_subplot(gs[0, 1])
marks = ["Pan-CK", "E-cadherin", "CD20", "CD3e", "CD4", "CD31", "Hoechst", "PD-L1", "Ki67", "CD68", "SMA", "PD-1", "CD8a"]
COLS = {"reinhard": C_FOCAL, "none": C_GREEN if False else "#5b8c5a", "tile": C_CTRL}
LBL = {"reinhard": "slide-level", "none": "none", "tile": "per-tile"}
y = np.arange(len(marks))
for k, m in enumerate(marks):
    vals = [agg[v][m]["mean"] for v in VAR]
    ax.plot([min(vals), max(vals)], [k, k], color="#e2e2e2", lw=4.0, solid_capstyle="round", zorder=1)
    for v in VAR:
        ax.scatter([agg[v][m]["mean"]], [k], s=20, c=COLS[v], zorder=3, edgecolors="white", linewidths=0.4)
ax.axvline(0, color="#666666", lw=0.7)
ax.set_yticks(y); ax.set_yticklabels(marks, fontsize=FS_TICK - 0.4)
ax.invert_yaxis(); ax.set_xlim(-0.16, 0.86)
ax.set_xlabel("Cross-patient mean residualised $\\rho$")
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=4, mfc=COLS[v], mec="white", label=LBL[v]) for v in VAR]
          + [Line2D([], [], color="#e2e2e2", lw=3, label="range")],
          frameon=False, fontsize=FS_TICK - 0.8, loc="lower right", handletextpad=0.3, borderaxespad=0.4)
ax.set_title("Normalisation shifts values, not the ranking", fontsize=FS_BASE, loc="left", pad=5)

axC = ax = fig.add_subplot(gs[1, 0])
TL = [50, 100, 200, 400, 800, 1600]
sel = [("DAPI", C_FOCAL), ("CK_1:150", C_CTRL), ("CD68_1:100", "#5b8c5a"),
       ("CD3_1:1000", C_ALT), ("CD20", C_GOLD)]
for nm, col in sel:
    ci = CH.index(nm); curve = []
    for n_ in TL:
        vs = [float(np.array(t10[pat][str(n_)])[ci]) for pat in t10 if t10[pat].get(str(n_)) is not None]
        curve.append(sum(vs) / len(vs))
    curve = np.array(curve); dev = 100 * np.abs(curve / curve[-1] - 1)
    ax.plot(TL, dev, color=col, lw=1.3, marker="o", ms=3.2, mec="white", mew=0.5, zorder=3)
    ax.annotate(nm.split("_")[0], xy=(TL[0], dev[0]), xytext=(7, 0), textcoords="offset points",
                fontsize=FS_TICK - 0.4, color=col, va="center")
ax.axvline(400, color="#bbbbbb", lw=0.9, ls=":", zorder=1)
ax.text(400, ax.get_ylim()[1] * 0.97, "L3 used 400", fontsize=FS_TICK - 0.6, color="#777777",
        ha="center", va="top")
ax.set_xscale("log"); ax.set_xticks(TL); ax.set_xticklabels([str(v) for v in TL], fontsize=FS_TICK)
ax.set_xlim(42, 2100)
ax.set_xlabel("Sampled tiles per slide"); ax.set_ylabel("Deviation from the\n1,600-tile estimate (%)")
ax.set_title("400 tiles suffice for abundant channels", fontsize=FS_BASE, loc="left", pad=5)

axD = ax = fig.add_subplot(gs[1, 1])
tg = ["CD20", "Caspase-3", "CD31"]; aggk = ["mean", "median", "max"]
x = np.arange(3); w = 0.26
for i, (a_, col) in enumerate(zip(aggk, [C_FOCAL, C_CTRL, "#5b8c5a"])):
    key = {"CD20": "CD20", "Caspase-3": "Caspase3", "CD31": "CD31"}
    vals = [abl["%s|%s|spearman" % (key[t_], a_)] for t_ in tg]
    ax.bar(x + (i - 1) * w, vals, w, color=col, zorder=2)
ax.axhline(0, color="#444444", lw=0.8, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(tg, fontsize=FS_TICK)
ax.set_ylim(-0.21, 0.21); ax.set_ylabel("Paired-channel Spearman $\\rho$")
ax.legend(handles=[Line2D([], [], marker="s", ls="", ms=5, mfc=c, mec="none", label=a_)
                   for a_, c in zip(aggk, [C_FOCAL, C_CTRL, "#5b8c5a"])],
          frameon=False, fontsize=FS_TICK - 0.6, loc="lower right", ncol=3,
          handletextpad=0.3, columnspacing=0.8, borderaxespad=0.4)
ax.set_title("Aggregation choice changes nothing", fontsize=FS_BASE, loc="left", pad=5)
plabel(axA, "a"); plabel(axB, "b"); plabel(axC, "c"); plabel(axD, "d")
audit(fig, "Fig10"); save(fig, "Fig10_robustness"); plt.close(fig)
