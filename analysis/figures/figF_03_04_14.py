# -*- coding: utf-8 -*-
"""Fig 3 / Fig 4 / Fig 14 —— 精修版"""
import os, sys, json, glob
sys.path.insert(0, r"W:\虚拟细胞\analysis")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from figstyle import *
apply()
A = r"W:\虚拟细胞\analysis"
CH = ["DAPI","TRITC","Cy5","PD-1_1:200","CD14","CD4","T-bet","CD34","CD68_1:100","CD16","CD11c",
      "CD138","CD20","CD3_1:1000","CD8","PD-L1","CK_1:150","Ki67_1:150","Tryptase","Actin-D",
      "Caspase3-D","PHH3-B","Transgelin"]

# ==================== Fig 3 ====================
sp = json.load(open(os.path.join(A, "l2", "l2_specificity.json"), encoding="utf-8"))
Mx = np.array(sp["matrix"]); chs = sp["channels"]; mods = sp["modules"]; spec = sp["specificity"]
PAIRED = {d["module"]: d["channel"] for d in spec}
fig = plt.figure(figsize=(9.4, 3.7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.32,
                      left=0.095, right=0.975, top=0.86, bottom=0.235)
axA = ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(Mx, cmap=DIV, vmin=-0.30, vmax=0.30, aspect="auto")
ax.set_xticks(range(len(mods))); ax.set_xticklabels(mods, rotation=45, ha="right", fontsize=FS_TICK - 0.4)
ax.set_yticks(range(len(chs))); ax.set_yticklabels(chs, fontsize=FS_TICK - 0.4)
for j, m in enumerate(mods):
    if m in PAIRED and PAIRED[m] in chs:
        i = chs.index(PAIRED[m])
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#3fd8ff", lw=1.0))
ax.set_xlabel("Spatial-transcriptomics module"); ax.set_ylabel("Virtual channel")
cb = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
cb.set_label("Spearman $\\rho$", fontsize=FS_TICK + 0.5); cb.ax.tick_params(labelsize=FS_TICK)
ax.set_title("Virtual channels correlate with every module", fontsize=FS_BASE, loc="left", pad=5)

axB = ax = fig.add_subplot(gs[0, 1])
sp2 = sorted(spec, key=lambda d: -d["specificity"])
y = np.arange(len(sp2))
ax.barh(y, [d["specificity"] for d in sp2], color=[C_FOCAL if d["specificity"] > 0 else C_CTRL for d in sp2],
        height=0.60, zorder=2)
ax.axvline(0, color="#444444", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([d["channel"] for d in sp2], fontsize=FS_TICK - 0.6)
ax.invert_yaxis(); ax.set_xlim(-0.22, 0.15)
ax.set_xlabel("Specificity  $|\\rho_{nominal}|-\\max|\\rho_{other}|$")
ax.set_title("Only CK clears its own module", fontsize=FS_BASE, loc="left", pad=5)
plabel(axA, "a"); plabel(axB, "b")
audit(fig, "Fig3"); save(fig, "Fig03_L2_gse230424"); plt.close(fig)

# ==================== Fig 4 ====================
foc = json.load(open(os.path.join(A, "l3", "l3_paired_focus.json"), encoding="utf-8"))
rb = json.load(open(os.path.join(A, "supp_04_l3_robustness.json"), encoding="utf-8"))
L3 = os.path.join(A, "l3")
lines = open(os.path.join(L3, "thca_rppa.txt"), encoding="utf-8", errors="replace").read().strip().split("\n")
prots = [l.split("\t")[0] for l in lines[1:]]
M = np.array([l.split("\t")[1:] for l in lines[1:]], dtype=float)
p2c = {s[:12]: i for i, s in enumerate(lines[0].split("\t")[1:])}
vecs = {}
for fp in glob.glob(os.path.join(L3, "*_vec.npy")):
    vecs[os.path.basename(fp).replace("_vec.npy", "")] = np.load(fp)
p2v = {}
for n2, v in vecs.items(): p2v.setdefault(n2[:12], []).append(v)
p2v = {p: np.mean(vs, 0) for p, vs in p2v.items()}
common = sorted(set(p2v) & set(p2c)); cols = [p2c[p] for p in common]
V = np.array([p2v[p] for p in common]); n = len(common)
TGT = [("CD20", "CD20", "CD20|CD20", "CD20"),
       ("Caspase-3", "Caspase3-D", "CASP3|Caspase-3_active", "Caspase3"),
       ("CD31", "CD34", "PECAM1|CD31", "CD31")]
fig = plt.figure(figsize=(9.4, 6.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.34], height_ratios=[1.0, 0.80],
                      wspace=0.34, hspace=0.62, left=0.095, right=0.955, top=0.90, bottom=0.095)
axA = ax = fig.add_subplot(gs[0, 0])
yv = M[prots.index("CD20|CD20")][cols]; xv = V[:, CH.index("CD20")]
r, p = stats.spearmanr(xv, yv)
ax.scatter(xv, yv, s=15, c=C_GREY, alpha=0.75, linewidths=0, zorder=3)
sl, ic = np.polyfit(xv, yv, 1); xx = np.linspace(xv.min(), xv.max(), 50)
ax.plot(xx, sl * xx + ic, color=C_CTRL, lw=1.1, zorder=2)
ax.set_xlabel("Virtual CD20 density"); ax.set_ylabel("RPPA CD20 (bulk)")
ax.set_xlim(xv.min() - 0.004, xv.max() + 0.008); ax.set_ylim(yv.min() - 0.2, yv.max() + 0.25)
ax.text(0.965, 0.94, "$\\rho$ = %+.3f\n$p$ = %.3f\n$n$ = %d patients" % (r, p, n),
        transform=ax.transAxes, ha="right", va="top", fontsize=FS_SEC)
ax.set_title("The nominal channel is not a readout", fontsize=FS_BASE, loc="left", pad=5)

axB = ax = fig.add_subplot(gs[:, 1])
matr = np.array([[abs(foc[t[3]]["all_rhos"][c]) for c in CH] for t in TGT])
im = ax.imshow(matr, cmap=SEQ, aspect="auto", vmin=0, vmax=0.45)
ax.set_yticks(range(3))
ax.set_yticklabels(["%s\nrank %d/23" % (t[0], foc[t[3]]["rank"]) for t in TGT], fontsize=FS_TICK)
ax.set_xticks(range(23)); ax.set_xticklabels(CH, rotation=90, fontsize=4.6)
for t in ax.get_xticklabels():
    if t.get_text() in ("TRITC", "Cy5"): t.set_color("#3fd8ff"); t.set_fontweight("bold")
for i, t in enumerate(TGT):
    j = CH.index(t[1])
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#3fd8ff", lw=1.1))
cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
cb.set_label("$|\\rho|$", fontsize=FS_TICK + 0.5); cb.ax.tick_params(labelsize=FS_TICK)
ax.set_xlabel("Virtual channel")
ax.set_title("The nominal channel never ranks first", fontsize=FS_BASE, loc="left", pad=5)

axC = ax = fig.add_subplot(gs[1, 0])
labs = ["CD20", "Caspase-3", "CD31"]
sh = rb["split_half"]
sc = [sh[k]["sign_consistency"] * 100 for k in ("CD20", "Caspase3", "CD31")]
ax.barh(range(3), sc, color=[C_FOCAL if v > 85 else (C_CTRL if v < 70 else C_GREY) for v in sc],
        height=0.55, zorder=2)
ax.axvline(50, color="#666666", lw=0.8, ls="--", zorder=3)
ax.set_yticks(range(3)); ax.set_yticklabels(labs, fontsize=FS_TICK); ax.invert_yaxis()
ax.set_xlim(0, 112); ax.set_xlabel("Split-half sign consistency (%)")
for k, v in enumerate(sc):
    ax.text(v + 2, k, "%.0f%%" % v, va="center", fontsize=FS_TICK, color="#333333")
ax.text(51, -0.40, "chance", fontsize=FS_TICK - 0.6, color="#666666", ha="left", va="center")
ax.set_ylim(2.7, -0.85)
ax.set_title("Only the CD20 sign reproduces", fontsize=FS_BASE, loc="left", pad=5)
plabel(axA, "a"); plabel(axB, "b"); plabel(axC, "c")
audit(fig, "Fig4"); save(fig, "Fig04_L3_tcga"); plt.close(fig)

# ==================== Fig 14 ====================
cl = json.load(open(os.path.join(A, "supp_10_clinical_assoc.json"), encoding="utf-8"))
fig = plt.figure(figsize=(9.4, 3.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.92, 1.35], wspace=0.34,
                      left=0.065, right=0.985, top=0.86, bottom=0.225)
axA = ax = fig.add_subplot(gs[0, 0])
km = cl["km"]; tt = np.array(km["t"]); ee = np.array(km["e"]); gg = np.array(km["g"])
for k, (lbl, col) in enumerate((("PC1 low/mid", C_GREY), ("PC1 top tertile", C_FOCAL))):
    m = (gg == k); ts = np.sort(np.unique(tt[m]))
    surv = [1 - ee[(tt <= tv) & m].sum() / m.sum() for tv in ts]
    ax.step(ts, surv, where="post", color=col, lw=1.3)
ax.set_ylim(0.62, 1.02); ax.set_xlim(0, 160)
ax.set_xlabel("Overall survival (months)"); ax.set_ylabel("Survival probability")
ax.legend(handles=[Line2D([], [], color=C_GREY, lw=1.3, label="PC1 low / mid"),
                   Line2D([], [], color=C_FOCAL, lw=1.3, label="PC1 top tertile")],
          frameon=False, fontsize=FS_TICK - 0.8, loc="lower left", handletextpad=0.4, borderaxespad=0.4)
ax.set_title("No survival separation (log-rank $p$ = %.2f)" % km["p"], fontsize=FS_BASE, loc="left", pad=5)
ax.text(0.97, 0.05, "%d events / %d patients" % (km["events"], km["n"]), transform=ax.transAxes,
        ha="right", fontsize=FS_TICK - 0.6, color="#777777")

axB = ax = fig.add_subplot(gs[0, 1])
ass = cl["association"]; items = [("Stage", ass["AJCC_PATHOLOGIC_TUMOR_STAGE"]), ("Age", ass["AGE"])]
for k, (nm, dd) in enumerate(items):
    se = 1 / np.sqrt(dd["n"] - 3); z = stats.norm.ppf(0.975)
    ax.plot([dd["stat"] - z * se * 0.55, dd["stat"] + z * se * 0.55], [k, k], color=C_FOCAL, lw=1.4)
    ax.scatter([dd["stat"]], [k], s=32, c=C_FOCAL, zorder=3, edgecolors="white", linewidths=0.5)
    ax.text(0.60, k, "$p$ = %.2f" % dd["p"], fontsize=FS_TICK, va="center", ha="right", color="#333333")
ax.axvline(0, color="#444444", lw=0.8)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Stage", "Age"], fontsize=FS_TICK); ax.invert_yaxis()
ax.set_xlim(-0.42, 0.62); ax.set_ylim(1.7, -0.7)
ax.set_xlabel("Spearman $\\rho$ with composition axis")
ax.set_title("No clinical association", fontsize=FS_BASE, loc="left", pad=5)

axC = ax = fig.add_subplot(gs[0, 2])
ld = [cl["pc1_loadings"][c] for c in CH]
ax.bar(range(23), ld, 0.74, color=[C_CTRL if v < 0 else C_FOCAL for v in ld], zorder=2)
ax.axhline(0, color="#444444", lw=0.8, zorder=3)
ax.set_xticks(range(23)); ax.set_xticklabels(CH, rotation=90, fontsize=4.6)
ax.set_ylabel("PC1 loading")
ax.set_ylim(min(ld) * 1.18, max(ld) * 3.2)
ax.set_title("PC1 is an immune-low axis (%.0f%% of variance)" % (cl["pc1_explained"] * 100),
             fontsize=FS_BASE, loc="left", pad=5)
plabel(axA, "a"); plabel(axB, "b"); plabel(axC, "c")
audit(fig, "Fig14"); save(fig, "Fig14_clinical"); plt.close(fig)
