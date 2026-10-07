"""Fig 8（交叉反应谱，L3 222 病人）与 Fig 9（有监督预测性检验）"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"C:\Users\PC\.agents\skills\figure-style")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from kernel import apply_figure_style, panel_letter

apply_figure_style(frame="open", sizes=(8, 7, 6))
OUT = r"W:\虚拟细胞\figures"; A = r"W:\虚拟细胞\analysis"
os.makedirs(OUT, exist_ok=True)
C_FOCAL, C_LIGHT, C_GREY, C_CTRL = "#1f6fb4", "#8fbcd9", "#9a9a9a", "#d1622f"
DIV = LinearSegmentedColormap.from_list("bo", ["#2b6ca3", "#7fb2d4", "#f2f2f2", "#e8a97e", "#c15a1f"])

# ================= Fig 8 =================
d = json.load(open(os.path.join(A, "supp_04b_crossreact_matrix.json"), encoding="utf-8"))
Rm = np.array(d["matrix"]); CH = d["channels"]; PR = d["proteins"]; n = d["n_patients"]
cr = json.load(open(os.path.join(A, "supp_04_l3_robustness.json"), encoding="utf-8"))["cross_reactivity"]
order_ch = sorted(range(23), key=lambda i: -(np.abs(Rm[i]) > 0.3).sum())
order_pr = np.argsort(-np.abs(Rm).mean(0))
R2 = Rm[np.ix_(order_ch, order_pr)]
lab = [CH[i] for i in order_ch]
PAIR = {"CD20": "CD20|CD20", "Caspase-3": "CASP3|Caspase-3_active", "CD31": "PECAM1|CD31"}
PAIRCH = {"CD20": "CD20", "Caspase-3": "Caspase3-D", "CD31": "CD34"}

fig = plt.figure(figsize=(9.0, 5.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0],
                      wspace=0.30, hspace=0.85)

# (a) 23x175 矩阵
ax = fig.add_subplot(gs[:, 0])
v = float(np.percentile(np.abs(Rm), 99))
im = ax.imshow(R2, cmap=DIV, vmin=-0.55, vmax=0.55, aspect="auto", origin="upper")
ax.set_yticks(range(23)); ax.set_yticklabels(lab, fontsize=5.4)
ax.set_xlabel("RPPA proteins (175), ordered by mean cross-reactivity", fontsize=7, labelpad=20)
ax.set_ylabel("Virtual channel", fontsize=7)
ax.set_xticks([])
for t in ax.get_yticklabels():
    if t.get_text() in ("TRITC", "Cy5"):
        t.set_color(C_CTRL); t.set_fontweight("bold")
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("Spearman $\\rho$", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
# 配对靶蛋白在列上的位置
for tname, pname in PAIR.items():
    if pname not in PR: continue
    j = int(np.where(np.array(PR) == pname)[0][0]); jj = int(np.where(order_pr == j)[0][0])
    i = int(np.where(np.array(lab) == PAIRCH[tname])[0][0])
    ax.add_patch(plt.Rectangle((jj-0.5, i-0.5), 1, 1, fill=False, ec="black", lw=0.9))
    _dx = {"CD31": -3.0, "CD20": -1.6, "Caspase-3": 3.0}[tname]
    ax.annotate(tname, xy=(jj, 22.55), xytext=(jj + _dx, 26.2), fontsize=5.6, ha="center", color="#222222",
                annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=0.5, color="#666666"))
ax.set_title("Every virtual channel correlates with dozens of unrelated proteins", fontsize=7.5, loc="left", pad=4)
ax.text(1.0, -0.20, f"n = {n} patients", transform=ax.transAxes, fontsize=5.8, color="#666666", ha="right")
panel_letter(ax, "a", dx=-0.13, dy=1.02, fontsize=11)

# (b) |rho| 分布 + 配对靶位置
ax = fig.add_subplot(gs[0, 1])
y = np.arange(23)
for k, i in enumerate(order_ch):
    a = np.abs(Rm[i])
    ax.plot([np.percentile(a, 10), np.percentile(a, 90)], [k, k], color="#d8d8d8", lw=3.0, solid_capstyle="round", zorder=1)
ax.scatter([np.median(np.abs(Rm[i])) for i in order_ch], y, s=16, c=C_GREY, zorder=2, label="median over 175 proteins")
pt, pc = [], []
for tname, chn in PAIRCH.items():
    i = order_ch.index(CH.index(chn)); j = PR.index(PAIR[tname])
    pt.append(float(Rm[CH.index(chn), j])); pc.append(i)
ax.scatter(np.abs(pt), pc, s=42, c=C_CTRL, marker="D", edgecolors="white", linewidths=0.6, zorder=4,
           label="its own cognate protein")
ax.set_yticks([]); ax.invert_yaxis()
ax.set_xlim(0.0, 0.60)
ax.set_xlabel("$|\\rho|$ across the 175 RPPA proteins", fontsize=7)
ax.legend(frameon=False, fontsize=5.8, loc="upper right", handletextpad=0.3, borderaxespad=0.2)
ax.set_title("The cognate protein is an average correlate", fontsize=7.5, loc="left", pad=4)
ax.text(0.0, 1.005, "rows ordered as in (a)", transform=ax.transAxes, fontsize=5.6, color="#777777", va="bottom")
panel_letter(ax, "b", dx=-0.12, dy=1.075, fontsize=11)

# (c) 配对靶百分位
ax = fig.add_subplot(gs[1, 1])
names = list(PAIR.keys())
pct = [cr[PAIRCH[t]]["paired_percentile"] for t in names]
ax.barh(range(3), pct, color=[C_CTRL if p < 50 else C_LIGHT for p in pct], height=0.5, zorder=2)
for k, p in enumerate(pct):
    ax.text(p+2, k, f"{p:.0f}th", va="center", fontsize=6.5, color="#333333")
ax.axvline(50, color="#666666", lw=0.8, ls="--")
ax.text(52, 1.35, "median of its own 175\ncorrelations", fontsize=5.6, color="#666666", ha="left", va="center")
ax.set_yticks(range(3)); ax.set_yticklabels(names, fontsize=6.5)
ax.invert_yaxis(); ax.set_xlim(0, 118); ax.set_ylim(2.9, -0.6)
ax.set_xlabel("Percentile of the cognate protein within\nthe channel's own 175 correlations", fontsize=7)
ax.set_title("At the 34th–54th percentile: no better than chance", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "c", dx=-0.30, dy=1.16, fontsize=11)
fig.savefig(os.path.join(OUT, "Fig08_crossreactivity.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Fig08_crossreactivity.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Fig8 done")

# ================= Fig 9 =================
d9 = json.load(open(os.path.join(A, "supp_06_predictive.json"), encoding="utf-8"))
allr = np.array([r for _, r in d9["all_proteins_cv_rho"]])
ptg = d9["paired_targets"]; axes_ = d9["axes"]
fig = plt.figure(figsize=(8.6, 3.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.36)

ax = fig.add_subplot(gs[0, 0])
ax.hist(allr, bins=26, color="#d8d8d8", edgecolor="#bbbbbb", linewidth=0.4, zorder=1)
ax.axvline(np.median(allr), color=C_GREY, lw=1.0, ls="--", zorder=2)
ax.text(np.median(allr)+0.01, ax.get_ylim()[1]*0.94, f"median {np.median(allr):.2f}", fontsize=5.8, color="#666666")
_disp = {"CD20": "CD20", "Caspase3": "Caspase-3", "CD31": "CD31"}
for (tname, col), hgt in zip(zip(ptg, [C_CTRL, C_FOCAL, "#7d5ba6"]), (0.94, 0.78, 0.62)):
    vv = ptg[tname]["cv_rho_all"]
    ax.axvline(vv, color=col, lw=1.1, zorder=3)
    ax.text(vv+0.006, ax.get_ylim()[1]*hgt, _disp[tname], fontsize=5.8, color=col, va="top", ha="left")
ax.set_xlabel("Held-out CV Spearman $\\rho$", fontsize=7)
ax.set_ylabel("RPPA proteins", fontsize=7)
ax.set_title("The cognate protein is not special", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "a", dx=-0.26, dy=1.10, fontsize=11)

ax = fig.add_subplot(gs[0, 1])
x = np.arange(3); w = 0.34
sing = [ptg[t]["cv_rho_single"] for t in ptg]
alls = [ptg[t]["cv_rho_all"] for t in ptg]
ax.bar(x-w/2, sing, w, color=C_GREY, zorder=2, label="paired channel only")
ax.bar(x+w/2, alls, w, color=C_FOCAL, zorder=2, label="all 21 virtual channels")
ax.set_xticks(x); ax.set_xticklabels([{"Caspase3": "Caspase-3"}.get(t, t) for t in ptg], fontsize=6.5)
ax.set_ylim(0, 0.50); ax.set_ylabel("Held-out CV $\\rho$", fontsize=7)
ax.legend(frameon=False, fontsize=5.8, loc="upper left", handletextpad=0.35)
ax.set_title("21 channels jointly", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "b", dx=-0.28, dy=1.10, fontsize=11)

ax = fig.add_subplot(gs[0, 2])
labs = ["CD20\nprotein", "Caspase-3\nprotein", "CD31\nprotein", "RPPA\nPC1", "RPPA\nPC2"]
vals = [ptg["CD20"]["cv_rho_all"], ptg["Caspase3"]["cv_rho_all"], ptg["CD31"]["cv_rho_all"],
        axes_["PC1"]["cv_rho_all"], axes_["PC2"]["cv_rho_all"]]
cols = [C_FOCAL]*3 + [C_CTRL]*2
ax.bar(range(5), vals, 0.62, color=cols, zorder=2)
ax.axvline(2.5, color="#dddddd", lw=0.8)
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=5.8)
ax.set_ylim(0, 0.60); ax.set_ylabel("Held-out CV $\\rho$", fontsize=7)
ax.text(1.0, 0.955, "single protein", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6, color="#666666")
ax.text(3.5, 0.955, "composition axis", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6, color=C_CTRL)
ax.set_title("Global axes are easier", fontsize=7.5, loc="left", pad=16)
panel_letter(ax, "c", dx=-0.28, dy=1.19, fontsize=11)
fig.savefig(os.path.join(OUT, "Fig09_predictive_cv.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Fig09_predictive_cv.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Fig9 done")
