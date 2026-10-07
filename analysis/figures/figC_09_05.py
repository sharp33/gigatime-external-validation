# -*- coding: utf-8 -*-
"""Fig 9 / Fig 5+6 / Fig 7 —— 精修版"""
import os, sys, json
sys.path.insert(0, r"W:\虚拟细胞\analysis")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import *
apply()
A = r"W:\虚拟细胞\analysis"

# ==================== Fig 9 ====================
d9 = json.load(open(os.path.join(A, "supp_06_predictive.json"), encoding="utf-8"))
allr = np.array([r for _, r in d9["all_proteins_cv_rho"]])
ptg = d9["paired_targets"]; ax9 = d9["axes"]
DISP = {"CD20": "CD20", "Caspase3": "Caspase-3", "CD31": "CD31"}
fig = plt.figure(figsize=(9.4, 3.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.05], wspace=0.36,
                      left=0.06, right=0.985, top=0.86, bottom=0.18)

axA = ax = fig.add_subplot(gs[0, 0])
ax.hist(allr, bins=26, color="#dcdcdc", edgecolor="#c0c0c0", linewidth=0.4, zorder=1)
med = float(np.median(allr))
ax.axvline(med, color=C_GREY, lw=1.0, ls="--", zorder=2)
ymax = ax.get_ylim()[1]
ax.text(med - 0.008, ymax * 0.97, "median %.2f" % med, fontsize=FS_TICK - 0.4, color="#777777",
        ha="right", va="top")
for (tname, col), hgt in zip(zip(ptg, [C_CTRL, C_FOCAL, C_ALT]), (0.90, 0.72, 0.54)):
    vv = ptg[tname]["cv_rho_all"]
    ax.axvline(vv, color=col, lw=1.1, zorder=3)
    ax.text(vv + 0.007, ymax * hgt, DISP[tname], fontsize=FS_TICK - 0.2, color=col, va="top", ha="left")
ax.set_xlabel("Held-out CV Spearman $\\rho$"); ax.set_ylabel("RPPA proteins")
ax.set_xlim(-0.02, 0.60)
ax.set_title("The cognate protein is not special", fontsize=FS_BASE, loc="left", pad=5)

axB = ax = fig.add_subplot(gs[0, 1])
x = np.arange(3); w = 0.34
sing = [ptg[t]["cv_rho_single"] for t in ptg]; alls = [ptg[t]["cv_rho_all"] for t in ptg]
ax.bar(x - w/2, sing, w, color=C_GREY, zorder=2)
ax.bar(x + w/2, alls, w, color=C_FOCAL, zorder=2)
ax.set_xticks(x); ax.set_xticklabels([DISP[t] for t in ptg], fontsize=FS_TICK)
ax.set_ylim(0, 0.52); ax.set_ylabel("Held-out CV $\\rho$")
ax.legend(handles=[Line2D([], [], marker="s", ls="", ms=5, mfc=C_GREY, mec="none", label="nominal channel only"),
                   Line2D([], [], marker="s", ls="", ms=5, mfc=C_FOCAL, mec="none", label="all 21 channels")],
          frameon=False, fontsize=FS_TICK - 0.6, loc="upper left", handletextpad=0.3, borderaxespad=0.4)
ax.set_title("21 channels jointly", fontsize=FS_BASE, loc="left", pad=5)

axC = ax = fig.add_subplot(gs[0, 2])
labs = ["CD20", "Caspase-3", "CD31", "RPPA\nPC1", "RPPA\nPC2"]
vals = [ptg["CD20"]["cv_rho_all"], ptg["Caspase3"]["cv_rho_all"], ptg["CD31"]["cv_rho_all"],
        ax9["PC1"]["cv_rho_all"], ax9["PC2"]["cv_rho_all"]]
cols = [C_FOCAL]*3 + [C_CTRL]*2
ax.bar(range(5), vals, 0.60, color=cols, zorder=2)
ax.axvline(2.5, color="#e0e0e0", lw=0.9)
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=FS_TICK - 0.6)
ax.set_ylim(0, 0.68); ax.set_ylabel("Held-out CV $\\rho$")
ax.text(1.0, 0.955, "single protein", transform=ax.get_xaxis_transform(), ha="center",
        va="bottom", fontsize=FS_TICK - 0.6, color="#777777")
ax.text(3.5, 0.955, "global composition axis", transform=ax.get_xaxis_transform(), ha="center",
        va="bottom", fontsize=FS_TICK - 0.6, color=C_CTRL)
ax.set_title("Global axes are easier to predict", fontsize=FS_BASE, loc="left", pad=5)
plabel(axA, "a"); plabel(axB, "b"); plabel(axC, "c")
audit(fig, "Fig9"); save(fig, "Fig09_predictive_cv"); plt.close(fig)

# ==================== Fig 5 （a,b,c） ====================
p = json.load(open(os.path.join(A, "orion", "preproc_ablation_summary.json"), encoding="utf-8"))
V = "tile"; PP = p["per_patient"]; pats = sorted(PP); npat = len(pats)
NEG = ["AF1", "Argo550", "CD45", "FOXP3", "CD45RO", "CD163"]
PAIR = ["Pan-CK", "E-cadherin", "CD20", "CD3e", "CD4", "CD31", "Hoechst", "PD-L1", "Ki67", "CD68", "SMA", "PD-1", "CD8a"]
paired_rho = {m: [] for m in PAIR}; spec_pass = {m: [] for m in PAIR}; bother = {m: [] for m in PAIR}
neg_rho = {m: [] for m in NEG}
for pat in pats:
    for dd in PP[pat][V]["detail"]:
        mk = dd["marker"]
        if mk in PAIR and dd.get("paired_rho") is not None:
            paired_rho[mk].append(dd["paired_rho"]); spec_pass[mk].append(1 if dd["specific"] else 0)
            bother[mk].append(dd["best_other"])
        if mk in NEG and dd.get("best_rho") is not None:
            neg_rho[mk].append(abs(dd["best_rho"]))
order = sorted(PAIR, key=lambda m: -np.mean(spec_pass[m]))

fig = plt.figure(figsize=(9.4, 3.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 1.0, 1.0], wspace=0.36,
                      left=0.085, right=0.985, top=0.86, bottom=0.235)
axA = ax = fig.add_subplot(gs[0, 0])
rate = np.array([np.mean(spec_pass[m]) for m in order]) * 100
cols = [C_FOCAL if v >= 25 else (C_CTRL if v > 0 else C_GREY) for v in rate]
ax.barh(range(len(order)), rate, color=cols, height=0.62, zorder=2)
for k, v in enumerate(rate):
    ax.text(v + 1.8, k, "%.0f%%" % v, va="center", fontsize=FS_TICK - 0.6, color="#333333")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=FS_TICK - 0.6)
for t in ax.get_yticklabels():
    if t.get_text() in ("Pan-CK", "E-cadherin"): t.set_fontweight("bold")
ax.invert_yaxis(); ax.set_xlim(0, 108)
ax.set_xlabel("Patients where the nominal channel wins (%)")
ax.text(0.98, 0.02, "$n$ = %d patients" % npat, transform=ax.transAxes, ha="right",
        fontsize=FS_TICK - 0.6, color="#777777")
ax.set_title("Specificity holds only for the epithelial pair", fontsize=FS_BASE, loc="left", pad=5)

axB = ax = fig.add_subplot(gs[0, 1])
rng = np.random.default_rng(3)
for k, m in enumerate(order):
    v = np.array(paired_rho[m])
    ax.scatter(v, np.full(len(v), k) + rng.uniform(-0.17, 0.17, len(v)), s=6, c=C_GREY, alpha=0.6,
               zorder=2, linewidths=0)
    ax.plot([np.median(v)] * 2, [k - 0.31, k + 0.31], color="#2b2b2b", lw=2.0, zorder=3)
ax.axvline(0, color="#666666", lw=0.7)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=FS_TICK - 0.6)
ax.invert_yaxis(); ax.set_xlim(-0.62, 1.02)
ax.set_xlabel("Residualised paired $\\rho$")
ax.text(0.02, 0.03, "bar = median", transform=ax.transAxes, fontsize=FS_TICK - 0.8, color="#777777")
ax.set_title("Immune channels carry a weak shared signal", fontsize=FS_BASE, loc="left", pad=5)

axC = ax = fig.add_subplot(gs[0, 2])
ax.plot([0, 0.85], [0, 0.85], color="#a8a8a8", lw=0.7, ls="--", zorder=1)
for m in PAIR:
    ax.scatter(np.abs(bother[m]), np.abs(paired_rho[m]), s=11, c=C_GREY, alpha=0.55, linewidths=0, zorder=3)
for m, col in (("Pan-CK", C_FOCAL), ("E-cadherin", C_LIGHT), ("CD20", C_CTRL), ("CD3e", C_ALT)):
    ax.scatter(np.abs(bother[m]), np.abs(paired_rho[m]), s=24, c=col, edgecolors="white",
               linewidths=0.4, zorder=4)
ax.set_xlim(0.02, 0.76); ax.set_ylim(0.02, 0.98)
ax.set_xlabel("Best competing channel  $|\\rho|$"); ax.set_ylabel("Nominal channel  $|\\rho|$")
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.6, mfc=C_GREY, mec="white", label="all 13 pairs"),
                   Line2D([], [], marker="o", ls="", ms=4.4, mfc=C_FOCAL, mec="white", label="Pan-CK"),
                   Line2D([], [], marker="o", ls="", ms=4.4, mfc=C_LIGHT, mec="white", label="E-cadherin"),
                   Line2D([], [], marker="o", ls="", ms=4.4, mfc=C_CTRL, mec="white", label="CD20"),
                   Line2D([], [], marker="o", ls="", ms=4.4, mfc=C_ALT, mec="white", label="CD3e")],
          frameon=False, fontsize=FS_TICK - 1.0, loc="upper left", handletextpad=0.25, borderaxespad=0.4)
ax.set_title("Nominal channels do not stand out", fontsize=FS_BASE, loc="left", pad=5)
plabel(axA, "a"); plabel(axB, "b"); plabel(axC, "c")
audit(fig, "Fig5"); save(fig, "Fig05_ORION_patients"); plt.close(fig)

# ==================== Fig 6 （负对照单独成图） ====================
fig = plt.figure(figsize=(6.6, 3.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.34,
                      left=0.14, right=0.98, top=0.85, bottom=0.20)
axA = ax = fig.add_subplot(gs[0, 0])
rng = np.random.default_rng(7)
for k, m in enumerate(NEG):
    v = np.array(neg_rho[m])
    ax.scatter(v, np.full(len(v), k) + rng.uniform(-0.18, 0.18, len(v)), s=7, c=C_CTRL, alpha=0.6,
               linewidths=0, zorder=2)
    ax.plot([np.median(v)] * 2, [k - 0.32, k + 0.32], color=C_CTRL, lw=2.2, zorder=3)
for k, m in enumerate(["Pan-CK", "CD3e", "CD20"]):
    v = np.abs(np.array(paired_rho[m])); yy = len(NEG) + 0.85 + k
    ax.scatter(v, np.full(len(v), yy) + rng.uniform(-0.18, 0.18, len(v)), s=7, c=C_FOCAL, alpha=0.6,
               linewidths=0, zorder=2)
    ax.plot([np.median(v)] * 2, [yy - 0.32, yy + 0.32], color=C_FOCAL, lw=2.2, zorder=3)
ax.set_yticks(list(range(len(NEG))) + [len(NEG) + 0.85 + k for k in range(3)])
ax.set_yticklabels(NEG + ["Pan-CK", "E-cadherin", "CD3e"], fontsize=FS_TICK - 0.4)
for t in ax.get_yticklabels():
    if t.get_text() in ("Pan-CK", "E-cadherin", "CD3e"): t.set_fontweight("bold")
ax.invert_yaxis(); ax.set_xlim(0, 0.86)
ax.axhline(len(NEG) + 0.2, color="#dcdcdc", lw=1.0)
ax.set_xlabel("Best-matching virtual channel  $|\\rho|$")
ax.text(0.985, 0.965, "medians: 0.312 vs 0.163", transform=ax.transAxes, ha="right", va="top",
        fontsize=FS_TICK - 0.4, color="#555555")
ax.set_title("No virtual counterpart needed", fontsize=FS_BASE, loc="left", pad=5)

axB = ax = fig.add_subplot(gs[0, 1])
win = 0
for pat in pats:
    det = PP[pat][V]["detail"]
    nb = [abs(x["best_rho"]) for x in det if x["marker"] in NEG and x.get("best_rho") is not None]
    pb = [abs(x["paired_rho"]) for x in det if x["marker"] in PAIR and x.get("paired_rho") is not None]
    if nb and pb and max(nb) > max(pb): win += 1
frac = 100.0 * win / npat
ax.bar([0], [frac], 0.55, color=C_CTRL, zorder=2)
ax.bar([0], [100 - frac], 0.55, bottom=[frac], color="#e8e8e8", zorder=2)
ax.text(0, frac / 2, "%.0f%%" % frac, ha="center", va="center", fontsize=FS_BASE + 1,
        color="white", fontweight="bold")
ax.text(0, frac + (100 - frac) / 2, "%.0f%%" % (100 - frac), ha="center", va="center",
        fontsize=FS_BASE, color="#666666")
ax.set_xlim(-0.6, 0.6); ax.set_ylim(0, 100); ax.set_xticks([])
ax.set_ylabel("Patients (%)")
ax.text(0, frac + 2.5, "a negative control\noutperformed every\nnominal pair", ha="center", va="bottom",
        fontsize=FS_TICK - 0.4, color="#444444")
ax.set_title("Per patient, in %d of %d" % (win, npat), fontsize=FS_BASE, loc="left", pad=5)
plabel(axA, "a"); plabel(axB, "b")
audit(fig, "Fig6"); save(fig, "Fig06_negative_controls"); plt.close(fig)
