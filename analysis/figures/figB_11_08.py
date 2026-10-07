# -*- coding: utf-8 -*-
"""Fig 11 / Fig 8 / Fig 9 —— 精修版"""
import os, sys, json
sys.path.insert(0, r"W:\虚拟细胞\analysis")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from figstyle import *
apply()
A = r"W:\虚拟细胞\analysis"

# ==================== Fig 11 ====================
d = json.load(open(os.path.join(A, "supp_11_composition_recovery.json"), encoding="utf-8"))
pc1 = np.array(d["virtual_pc1"]); pre = np.array(d["virtual_prespec"])
true = np.array(d["true_comp_score"]); n = d["n_patients"]
r, p = stats.spearmanr(pc1, true)

fig = plt.figure(figsize=(9.4, 6.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.10], height_ratios=[1.0, 0.86],
                      wspace=0.34, hspace=0.60, left=0.075, right=0.975, top=0.90, bottom=0.085)

axA = ax = fig.add_subplot(gs[:, 0])
sl, ic = np.polyfit(pc1, true, 1)
xs = np.linspace(pc1.min() - 0.3, pc1.max() + 0.3, 100)
ax.plot(xs, sl * xs + ic, color=C_FOCAL, lw=1.2, zorder=2)
resid = true - (sl * pc1 + ic); se = resid.std(ddof=2)
xl = np.linspace(pc1.min(), pc1.max(), 100)
band = 1.96 * se * np.sqrt(1/n + (xl - pc1.mean())**2 / ((n - 1) * pc1.std(ddof=1)**2))
ax.fill_between(xl, sl * xl + ic - band, sl * xl + ic + band, color=C_FOCAL, alpha=0.14, lw=0, zorder=1)
ax.scatter(pc1, true, s=30, c=C_FOCAL, edgecolors="white", linewidths=0.5, zorder=3)
ax.axhline(0, color="#c8c8c8", lw=0.6, ls=":", zorder=0)
ax.set_xlabel("Virtual composition score (PC1, per-patient mean)")
ax.set_ylabel("Measured composition\n(epithelial $-$ immune, single cell)")
ax.set_title("Virtual channels recover tumour composition", fontsize=FS_BASE, loc="left", pad=5)
ax.set_xlim(pc1.min() - 1.1, pc1.max() + 1.1)
ax.set_ylim(true.min() - 0.030, true.max() + 0.045)
ax.text(0.975, 0.055, "Spearman $\\rho$ = %.2f\n$p$ = %.1e\n$n$ = %d patients" % (r, p, n),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_SEC)
o = np.argsort(pc1)
for k, tag, off in ((o[-1], "epithelial-rich", (-58, 2)), (o[0], "immune-rich", (58, -3))):
    ax.annotate(tag, xy=(pc1[k], true[k]), xytext=off, textcoords="offset points",
                fontsize=FS_TICK, color="#666666", ha="center", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#999999", shrinkA=1, shrinkB=2))

axB = ax = fig.add_subplot(gs[0, 1])
lab = {"epi": "Epithelial", "tcell": "T cell", "mye": "Myeloid", "endo": "Endothelial", "stroma": "Stromal"}
keys = ["epi", "tcell", "mye", "endo", "stroma"]
r1 = np.array([d["association"][k]["pc1"]["rho"] for k in keys])
r2 = np.array([d["association"][k]["prespec"]["rho"] for k in keys])
y = np.arange(len(keys))
for k in range(len(keys)):
    ax.plot([0, r1[k]], [y[k] - 0.17, y[k] - 0.17], color=C_FOCAL, lw=1.4, zorder=1)
    ax.plot([0, r2[k]], [y[k] + 0.17, y[k] + 0.17], color=C_LIGHT, lw=1.4, zorder=1)
ax.scatter(r1, y - 0.17, s=26, c=C_FOCAL, edgecolors="white", linewidths=0.5, zorder=3)
ax.scatter(r2, y + 0.17, s=26, c=C_LIGHT, edgecolors="white", linewidths=0.5, zorder=3)
ax.axvline(0, color="#444444", lw=0.7)
ax.set_yticks(y); ax.set_yticklabels([lab[k] for k in keys], fontsize=FS_TICK)
for t, k in zip(ax.get_yticklabels(), keys):
    if k in ("epi", "tcell", "mye"): t.set_fontweight("bold")
ax.invert_yaxis(); ax.set_xlim(-0.86, 1.32)
ax.set_xlabel("Spearman $\\rho$ with measured abundance")
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=4, mfc=C_FOCAL, mec="white", label="virtual PC1"),
                   Line2D([], [], marker="o", ls="", ms=4, mfc=C_LIGHT, mec="white", label="pre-specified (no fitting)")],
          frameon=False, fontsize=FS_TICK - 0.3, loc="center right", handletextpad=0.3, borderaxespad=0.2)
ax.set_title("Immune and epithelial compartments drive the axis", fontsize=FS_BASE, loc="left", pad=5)

axC = ax = fig.add_subplot(gs[1, 1])
items = [("Composition\n(41 patients)", [d["association"]["composition"]["pc1"]["rho"],
                                          d["association"]["composition"]["prespec"]["rho"]], [C_FOCAL, C_LIGHT]),
         ("Pan-CK", [d["cell_level_paired"]["Pan-CK"]], [C_GREY]),
         ("CD3e", [d["cell_level_paired"]["CD3e"]], [C_GREY]),
         ("CD20", [d["cell_level_paired"]["CD20"]], [C_GREY]),
         ("CD68", [d["cell_level_paired"]["CD68"]], [C_GREY])]
xs, ys, cs = [], [], []
for i, (nm, vals, cols) in enumerate(items):
    off = (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.19
    for v, c, o2 in zip(vals, cols, off):
        xs.append(i + o2); ys.append(v); cs.append(c)
for x, v, c in zip(xs, ys, cs):
    ax.plot([x, x], [0, v], color=c, lw=1.4, zorder=1)
ax.scatter(xs, ys, s=32, c=cs, edgecolors="white", linewidths=0.5, zorder=3)
ax.axvline(0.5, color="#e0e0e0", lw=0.9)
ax.axhline(0, color="#444444", lw=0.7)
ax.set_xticks(range(len(items)))
ax.set_xticklabels([nm for nm, _, _ in items], fontsize=FS_TICK)
for t, (nm, _, _) in zip(ax.get_xticklabels(), items):
    if nm.startswith("Composition"): t.set_fontweight("bold"); t.set_color(C_FOCAL)
ax.set_ylim(-0.12, 0.94); ax.set_xlim(-0.55, 4.55)
ax.set_ylabel("Spearman $\\rho$")
ax.text(3.0, 0.845, "single marker, cell level", ha="center", va="center",
        fontsize=FS_TICK - 0.4, color="#777777")
for x, v in zip(xs, ys):
    if x < 0.5 or (x > 3.5 and v < 0.12):
        ax.text(x, v + 0.035, "%.2f" % v, ha="center", va="bottom", fontsize=FS_TICK - 0.4, color="#444444")
ax.set_title("Composition is recoverable; single-marker identity is not", fontsize=FS_BASE, loc="left", pad=5)

plabel(axA, "a"); plabel(axB, "b"); plabel(axC, "c")
audit(fig, "Fig11"); save(fig, "Fig11_composition_recovery"); plt.close(fig)

# ==================== Fig 8 ====================
d8 = json.load(open(os.path.join(A, "supp_04b_crossreact_matrix.json"), encoding="utf-8"))
Rm = np.array(d8["matrix"]); CH8 = d8["channels"]; PR = d8["proteins"]; n8 = d8["n_patients"]
cr = json.load(open(os.path.join(A, "supp_04_l3_robustness.json"), encoding="utf-8"))["cross_reactivity"]
order_ch = sorted(range(23), key=lambda i: -(np.abs(Rm[i]) > 0.3).sum())
order_pr = np.argsort(-np.abs(Rm).mean(0))
R2 = Rm[np.ix_(order_ch, order_pr)]
lab8 = [CH8[i] for i in order_ch]
PAIR = {"CD20": "CD20|CD20", "Caspase-3": "CASP3|Caspase-3_active", "CD31": "PECAM1|CD31"}
PAIRCH = {"CD20": "CD20", "Caspase-3": "Caspase3-D", "CD31": "CD34"}

fig = plt.figure(figsize=(9.4, 5.8))
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0],
                      wspace=0.34, hspace=0.78, left=0.085, right=0.975, top=0.90, bottom=0.175)
axA = ax = fig.add_subplot(gs[:, 0])
im = ax.imshow(R2, cmap=DIV, vmin=-0.55, vmax=0.55, aspect="auto", origin="upper")
ax.set_yticks(range(23)); ax.set_yticklabels(lab8, fontsize=4.8)
for t in ax.get_yticklabels():
    if t.get_text() in ("TRITC", "Cy5"): t.set_color(C_CTRL); t.set_fontweight("bold")
ax.set_xticks([]); ax.set_ylabel("Virtual channel")
ax.set_xlabel("RPPA proteins (175), ordered by mean cross-reactivity\n(outlined = its own cognate protein)", labelpad=44)
cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
cb.set_label("Spearman $\\rho$", fontsize=FS_TICK + 0.5); cb.ax.tick_params(labelsize=FS_TICK)
for tname, pname in PAIR.items():
    if pname not in PR: continue
    j = int(np.where(np.array(PR) == pname)[0][0]); jj = int(np.where(order_pr == j)[0][0])
    i = int(np.where(np.array(lab8) == PAIRCH[tname])[0][0])
    ax.add_patch(plt.Rectangle((jj - 0.5, i - 0.5), 1, 1, fill=False, ec="#111111", lw=1.0))
_dx = {"CD31": -4.0, "CD20": 0.0, "Caspase-3": 4.5}
for tname, pname in PAIR.items():
    j = int(np.where(np.array(PR) == pname)[0][0]); jj = int(np.where(order_pr == j)[0][0])
    ax.annotate(tname, xy=(jj, 22.6), xytext=(jj + _dx[tname], 25.4), fontsize=5.2, ha="center",
                color="#111111", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#666666"))
ax.set_title("Every virtual channel correlates with dozens of unrelated proteins", fontsize=FS_BASE, loc="left", pad=5)

axB = ax = fig.add_subplot(gs[0, 1])
y = np.arange(23)
for k, i in enumerate(order_ch):
    a = np.abs(Rm[i])
    ax.plot([np.percentile(a, 10), np.percentile(a, 90)], [k, k], color="#dcdcdc", lw=3.2,
            solid_capstyle="round", zorder=1)
ax.scatter([np.median(np.abs(Rm[i])) for i in order_ch], y, s=15, c=C_GREY, zorder=2)
pt, pc = [], []
for tname, chn in PAIRCH.items():
    i = order_ch.index(CH8.index(chn)); j = PR.index(PAIR[tname])
    pt.append(float(Rm[CH8.index(chn), j])); pc.append(i)
ax.scatter(np.abs(pt), pc, s=40, c=C_CTRL, marker="D", edgecolors="white", linewidths=0.6, zorder=4)
ax.set_yticks([]); ax.invert_yaxis(); ax.set_xlim(0.0, 0.72)
ax.set_xlabel("$|\\rho|$ across 175 proteins")
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.6, mfc=C_GREY, mec="white", label="median"),
                   Line2D([], [], marker="D", ls="", ms=4.2, mfc=C_CTRL, mec="white", label="its own cognate protein")],
          frameon=False, fontsize=FS_TICK - 0.4, loc="lower right", handletextpad=0.3, borderaxespad=0.4)
ax.set_ylabel("Virtual channel\n(rows as in a)", fontsize=FS_TICK)
ax.set_title("The cognate protein is an average correlate", fontsize=FS_BASE, loc="left", pad=5)

axC = ax = fig.add_subplot(gs[1, 1])
names = list(PAIR.keys())
pct = [cr[PAIRCH[t]]["paired_percentile"] for t in names]
ax.barh(range(3), pct, color=[C_CTRL if q < 50 else C_LIGHT for q in pct], height=0.52, zorder=2)
for k, q in enumerate(pct):
    ax.text(q + 2.5, k, "%.0fth" % q, va="center", fontsize=FS_TICK, color="#333333")
ax.axvline(50, color="#666666", lw=0.8, ls="--")
ax.set_yticks(range(3)); ax.set_yticklabels(names, fontsize=FS_TICK)
ax.invert_yaxis(); ax.set_xlim(0, 118); ax.set_ylim(2.75, -0.75)
ax.set_xlabel("Percentile of the cognate protein\nwithin its channel's own 175 correlations")
ax.text(52, -0.62, "median", fontsize=FS_TICK - 0.6, color="#666666", ha="left", va="center")
ax.set_title("No better than an arbitrary protein", fontsize=FS_BASE, loc="left", pad=5)
plabel(axA, "a"); plabel(axB, "b"); plabel(axC, "c")
audit(fig, "Fig8"); save(fig, "Fig08_crossreactivity"); plt.close(fig)
