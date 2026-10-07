"""Fig 2（L1 HEMIT 独立机构）、Fig 3（L2 Visium）、Fig 4（L3 222 病人）"""
import os, sys, json, glob
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"C:\Users\PC\.agents\skills\figure-style")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from kernel import apply_figure_style, panel_letter

apply_figure_style(frame="open", sizes=(8, 7, 6))
OUT = r"W:\虚拟细胞\figures"; A = r"W:\虚拟细胞\analysis"
os.makedirs(OUT, exist_ok=True)
C1, C2, C3, CG = "#1f6fb4", "#d1622f", "#5b8c5a", "#9a9a9a"
DIV = LinearSegmentedColormap.from_list("bo", ["#2b6ca3", "#7fb2d4", "#f2f2f2", "#e8a97e", "#c15a1f"])
CH = ["DAPI","TRITC","Cy5","PD-1_1:200","CD14","CD4","T-bet","CD34","CD68_1:100","CD16","CD11c",
      "CD138","CD20","CD3_1:1000","CD8","PD-L1","CK_1:150","Ki67_1:150","Tryptase","Actin-D",
      "Caspase3-D","PHH3-B","Transgelin"]

# ============================================================ Fig 2
mv = json.load(open(os.path.join(A, "supp_03_l1_modelvariant.json"), encoding="utf-8"))
df = json.load(open(os.path.join(A, "supp_03c_density_fairness.json"), encoding="utf-8"))
CONDS = [("flash@1024_reinhard","Flash\nslide"), ("flash@1024_none","Flash\nnone"),
         ("orig@1024_reinhard","orig\nslide"), ("orig@1024_none","orig\nnone"),
         ("orig@512_reinhard","orig512\nslide"), ("orig@512_none","orig512\nnone")]
CH3 = ["DAPI", "CK", "CD3"]
CMAP_KEY = {"DAPI": "DAPI", "CK": "CK", "CD3": "CD3"}
KEY3 = {"DN": ("orig@1024_none","DAPI"), "CN": ("orig@512_none","CD3"), "CC": ("orig@512_none","CK"),
        "DR": ("flash@1024_reinhard","DAPI")}

fig = plt.figure(figsize=(9.0, 3.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.35, 1.0], wspace=0.34)
CONDS2 = [c for c in CONDS if f"{c[0]}|DAPI" in df]
for pi, tag in enumerate(("default", "density-matched")):
    ax = fig.add_subplot(gs[0, pi])
    use = CONDS if pi == 0 else CONDS2
    x = np.arange(len(use)); w = 0.26
    for k, (nm, col) in enumerate(zip(CH3, [C1, C2, C3])):
        vals, base = [], []
        for cond, _ in use:
            if pi == 0:
                vals.append(mv[cond][nm]["dice"]); base.append(mv[cond][nm]["baseline"])
            else:
                key = f"{cond}|{nm}"
                vals.append(df[key]["dice_density_matched"]); base.append(df[key]["baseline"])
        ax.bar(x+(k-1)*w, vals, w, color=col, zorder=2, label=nm if pi == 0 else None)
        ax.hlines(base[0], x[0]-1.5*w, x[-1]+1.5*w, color=col, lw=0.8, ls="--", zorder=3, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in use], fontsize=5.4, rotation=32, ha="right")
    ax.set_ylim(0, 0.88); ax.set_ylabel("Dice vs. mIHC (Otsu)" if pi == 0 else "", fontsize=7)
    ax.set_title(f"{'Default' if pi==0 else 'Density-matched'} threshold", fontsize=7.5, loc="left", pad=4)
    if pi == 0:
        from matplotlib.lines import Line2D
        h, l = ax.get_legend_handles_labels()
        h.append(Line2D([0], [0], color="#777777", ls="--", lw=0.8)); l.append("random baseline")
        ax.legend(h, l, frameon=False, fontsize=5.8, loc="upper right", ncol=1,
                  handletextpad=0.35, borderaxespad=0.2)
    if pi == 1:
        ax.plot([], [], color="#777777", ls="--", lw=0.8, label="random baseline")
        ax.legend(frameon=False, fontsize=5.8, loc="upper right", handletextpad=0.35, borderaxespad=0.2)
    panel_letter(ax, "ab"[pi], dx=-0.20, dy=1.10, fontsize=10)

ax = fig.add_subplot(gs[0, 2])
for k, nm in enumerate(CH3):
    for cond, _ in CONDS2:
        d = df[f"{cond}|{nm}"]
        ax.scatter([d["gt_density"]], [d["pred_density"]], s=26, c=[C1, C2, C3][k], zorder=3,
                   edgecolors="white", linewidths=0.4, label=nm if cond == CONDS[0][0] else None)
ax.plot([0.02, 0.7], [0.02, 0.7], color="#999999", lw=0.8, ls="--", zorder=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([0.05, 0.1, 0.2, 0.3]); ax.set_xticklabels(["0.05", "0.1", "0.2", "0.3"], fontsize=6)
ax.minorticks_off()
ax.set_xlabel("Ground-truth density", fontsize=7); ax.set_ylabel("Predicted density", fontsize=7)
ax.legend(frameon=False, fontsize=5.8, loc="lower right", handletextpad=0.3)
ax.set_title("Density calibration breaks across conditions", fontsize=7.5, loc="left", pad=4)
ax.text(0.03, 0.40, "4 conditions\nwith density-\nmatched control", transform=ax.transAxes,
        fontsize=5.2, color="#888888", va="top")
panel_letter(ax, "c", dx=-0.26, dy=1.10, fontsize=10)
fig.savefig(os.path.join(OUT, "Fig02_L1_HEMIT.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Fig02_L1_HEMIT.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Fig2 done")

# ============================================================ Fig 3
sp = json.load(open(os.path.join(A, "l2", "l2_specificity.json"), encoding="utf-8"))
Mx = np.array(sp["matrix"]); chs = sp["channels"]; mods = sp["modules"]; spec = sp["specificity"]
PAIRED = {d["module"]: d["channel"] for d in spec}
fig = plt.figure(figsize=(8.6, 3.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.30)
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(Mx, cmap=DIV, vmin=-0.30, vmax=0.30, aspect="auto")
ax.set_xticks(range(len(mods))); ax.set_xticklabels(mods, rotation=45, ha="right", fontsize=5.6)
ax.set_yticks(range(len(chs))); ax.set_yticklabels(chs, fontsize=5.6)
for j, m in enumerate(mods):
    if m in PAIRED and PAIRED[m] in chs:
        i = chs.index(PAIRED[m]); ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, ec="#00d5ff", lw=1.0))
ax.set_xlabel("Spatial-transcriptomics module", fontsize=7); ax.set_ylabel("Virtual channel", fontsize=7)
cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02); cb.set_label("Spearman $\\rho$", fontsize=6.5); cb.ax.tick_params(labelsize=6)
ax.set_title("Virtual channels correlate with every module", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "a", dx=-0.16, dy=1.06, fontsize=10)
ax = fig.add_subplot(gs[0, 1])
sp2 = sorted(spec, key=lambda d: -d["specificity"])
y = np.arange(len(sp2))
cols = [C1 if d["specificity"] > 0 else C2 for d in sp2]
ax.barh(y, [d["specificity"] for d in sp2], color=cols, height=0.6, zorder=2)
ax.axvline(0, color="#444444", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([d["channel"] for d in sp2], fontsize=5.8)
ax.invert_yaxis()
ax.set_xlabel("Specificity  $|\\rho_{paired}|-\\max|\\rho_{other}|$", fontsize=7)
ax.set_title("Only CK clears its own module", fontsize=7.5, loc="left", pad=4)
ax.set_xlim(-0.22, 0.14)
panel_letter(ax, "b", dx=-0.30, dy=1.06, fontsize=10)
fig.savefig(os.path.join(OUT, "Fig03_L2_gse230424.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Fig03_L2_gse230424.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Fig3 done  n_positive_spec=%.0f/%d" % (sum(1 for d in spec if d["specificity"] > 0), len(spec)))
