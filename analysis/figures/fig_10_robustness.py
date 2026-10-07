"""Fig 10：稳健性四联 —— 模型变体 / 染色归一化 / 采样收敛 / 聚合口径"""
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
C1, C2, C3, CG = "#1f6fb4", "#d1622f", "#5b8c5a", "#9a9a9a"
CH = ["DAPI","TRITC","Cy5","PD-1_1:200","CD14","CD4","T-bet","CD34","CD68_1:100","CD16","CD11c",
      "CD138","CD20","CD3_1:1000","CD8","PD-L1","CK_1:150","Ki67_1:150","Tryptase","Actin-D",
      "Caspase3-D","PHH3-B","Transgelin"]

fig = plt.figure(figsize=(9.0, 6.0))
gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.66)

# --- (a) 模型变体 ---
o = json.load(open(os.path.join(A, "supp_03_official_sample.json"), encoding="utf-8"))
sizes = [256, 512, 556]
fl = [o[f"flash@{s}"]["mean"] for s in sizes]; og = [o[f"orig@{s}"]["mean"] for s in sizes]
ax = fig.add_subplot(gs[0, 0])
x = np.arange(3); w = 0.36
ax.bar(x-w/2, og, w, color=C2, zorder=2, label="original GigaTIME (U-Net++)")
ax.bar(x+w/2, fl, w, color=C1, zorder=2, label="GigaTIME-Flash")
ax.axhline(o["official_precomputed"], color="#444444", lw=0.9, ls="--", zorder=3)
ax.text(2.42, o["official_precomputed"]+0.012, "authors' stored value", fontsize=5.6, color="#444444", ha="right")
ax.set_xticks(x); ax.set_xticklabels([f"{s}" for s in sizes], fontsize=6.5)
ax.set_xlabel("Input resolution (px)", fontsize=7); ax.set_ylabel("Dice (official cell-hull protocol)", fontsize=7)
ax.set_ylim(0, 0.42)
ax.legend(frameon=False, fontsize=5.8, loc="upper left", handletextpad=0.4, borderaxespad=0.3)
ax.set_title("Distillation did not change the metric", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "a", dx=-0.22, dy=1.09, fontsize=11)

# --- (b) 染色归一化消融 ---
p = json.load(open(os.path.join(A, "orion", "preproc_ablation_summary.json"), encoding="utf-8"))
VAR = p["variants"]; agg = p["aggregate"]
marks = ["Pan-CK","E-cadherin","CD20","CD3e","CD4","CD31","Hoechst","PD-L1","Ki67","CD68","SMA","PD-1","CD8a"]
COLS = {"reinhard": C1, "none": C3, "tile": C2}
LBL = {"reinhard": "slide-level", "none": "none", "tile": "per-tile (= original L4)"}
ax = fig.add_subplot(gs[0, 1])
y = np.arange(len(marks))
for k, m in enumerate(marks):
    vals = [agg[v][m]["mean"] for v in VAR]
    ax.plot([min(vals), max(vals)], [k, k], color="#dddddd", lw=4.0, solid_capstyle="round", zorder=1)
    for v in VAR:
        ax.scatter([agg[v][m]["mean"]], [k], s=22, c=COLS[v], zorder=3,
                   label=LBL[v] if k == 0 else None, edgecolors="white", linewidths=0.4)
ax.axvline(0, color="#666666", lw=0.7)
ax.set_yticks(y); ax.set_yticklabels(marks, fontsize=6)
ax.invert_yaxis()
ax.set_xlabel("Cross-patient mean residualised $\\rho$", fontsize=7)
ax.legend(frameon=False, fontsize=5.8, loc="upper left", handletextpad=0.3, borderaxespad=0.4)
ax.set_title("Normalisation shifts all values, not the ranking", fontsize=7.5, loc="left", pad=4)
pcm = p["n_specific_mean"]
ax.text(0.02, 0.03, "specificity passes: " + " · ".join(f"{LBL[v].split(' ')[0]} {pcm[v]:.1f}/13" for v in VAR),
        transform=ax.transAxes, fontsize=5.6, color="#666666")
panel_letter(ax, "b", dx=-0.30, dy=1.09, fontsize=11)

# --- (c) 采样收敛 ---
t = json.load(open(os.path.join(A, "orion", "tile_sensitivity.json"), encoding="utf-8"))
TL = [50, 100, 200, 400, 800, 1600]
sel = [("DAPI", C1), ("CK_1:150", C2), ("CD68_1:100", C3), ("CD3_1:1000", "#7d5ba6"), ("CD20", "#c9a227")]
ax = fig.add_subplot(gs[1, 0])
for nm, col in sel:
    ci = CH.index(nm)
    curve = []
    for n_ in TL:
        vs = [np.array(t[pat][str(n_)])[ci] for pat in t if t[pat].get(str(n_)) is not None]
        curve.append(float(np.mean(vs)))
    curve = np.array(curve)
    dev = 100*np.abs(curve/curve[-1]-1)
    ax.plot(TL, dev, color=col, lw=1.3, marker="o", ms=3.2, mec="white", mew=0.5, zorder=3)
    ax.annotate(nm.replace("_1:150","").replace("_1:100",""), xy=(TL[0], dev[0]),
                xytext=(7, 0), textcoords="offset points", fontsize=6, color=col, va="center")
ax.axvline(400, color="#bbbbbb", lw=0.9, ls=":", zorder=1)
ax.text(400, ax.get_ylim()[1]*0.95, "L3 used 400", fontsize=5.6, color="#777777", ha="center", va="top")
ax.set_xscale("log"); ax.set_xticks(TL); ax.set_xticklabels([str(v) for v in TL], fontsize=6)
ax.set_xlim(45, 2400)
ax.set_xlabel("Sampled tiles per slide", fontsize=7)
ax.set_ylabel("Deviation from the 1,600-tile\nestimate (%)", fontsize=7)
ax.set_title("400 tiles suffice for abundant channels", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "c", dx=-0.22, dy=1.09, fontsize=11)

# --- (d) 聚合口径 ---
rb = json.load(open(os.path.join(A, "supp_04_l3_robustness.json"), encoding="utf-8"))
abl = rb["aggregation_ablation"]
ax = fig.add_subplot(gs[1, 1])
tg = ["CD20", "Caspase3", "CD31"]; aggk = ["mean", "median", "max"]
x = np.arange(3); w = 0.26
for i, (a_, col) in enumerate(zip(aggk, [C1, C2, C3])):
    vals = [abl[f"{t_}|{a_}|spearman"] for t_ in tg]
    ax.bar(x+(i-1)*w, vals, w, color=col, zorder=2, label=a_)
ax.axhline(0, color="#444444", lw=0.8, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(tg, fontsize=6.5)
ax.set_ylim(-0.20, 0.20); ax.set_ylabel("Paired-channel Spearman $\\rho$", fontsize=7)
ax.legend(frameon=False, fontsize=5.8, loc="lower right", ncol=3, handletextpad=0.3, columnspacing=0.8)
ax.set_title("Aggregation choice changes nothing", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "d", dx=-0.22, dy=1.09, fontsize=11)

fig.savefig(os.path.join(OUT, "Fig10_robustness.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Fig10_robustness.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Fig10 done")
