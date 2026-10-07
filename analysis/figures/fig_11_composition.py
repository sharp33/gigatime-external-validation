"""论文主图 Fig 11：虚拟组成轴复原肿瘤真实细胞组成（ORION-CRC，41 病人）"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"C:\Users\PC\.agents\skills\figure-style")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from kernel import apply_figure_style, panel_letter

apply_figure_style(frame="open", sizes=(8, 7, 6))
OUT = r"W:\虚拟细胞\figures"; os.makedirs(OUT, exist_ok=True)
d = json.load(open(r"W:\虚拟细胞\analysis\supp_11_composition_recovery.json", encoding="utf-8"))

C_FOCAL = "#1f6fb4"   # 虚拟组成轴（主角）
C_LIGHT = "#8fbcd9"   # 预先指定分数（同色系浅色，§4.3）
C_GREY  = "#9a9a9a"   # 细胞级单标记读数（对照）

pc1 = np.array(d["virtual_pc1"]); pre = np.array(d["virtual_prespec"])
true = np.array(d["true_comp_score"]); L = d["true_composition"]
n = d["n_patients"]

fig = plt.figure(figsize=(8.8, 5.9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.10], height_ratios=[1.0, 1.0],
                      wspace=0.36, hspace=0.68)

# ---------------- (a) 散点：虚拟组成 vs 真实组成 ----------------
ax = fig.add_subplot(gs[:, 0])
r, p = stats.spearmanr(pc1, true)
sl, ic = np.polyfit(pc1, true, 1)
xx = np.linspace(pc1.min()-0.3, pc1.max()+0.3, 100)
ax.plot(xx, sl*xx+ic, color=C_FOCAL, lw=1.2, zorder=2)
resid = true - (sl*pc1+ic); se = resid.std(ddof=2)
xs = np.linspace(pc1.min(), pc1.max(), 100)
sx = pc1.std(ddof=1); n_ = len(pc1)
band = 1.96*se*np.sqrt(1/n_ + (xs-pc1.mean())**2/((n_-1)*sx**2))
ax.fill_between(xs, sl*xs+ic-band, sl*xs+ic+band, color=C_FOCAL, alpha=0.14, lw=0, zorder=1)
ax.scatter(pc1, true, s=30, c=C_FOCAL, edgecolors="white", linewidths=0.5, zorder=3)
ax.axhline(0, color="#bbbbbb", lw=0.6, ls=":", zorder=0)
ax.set_xlabel("Virtual composition score\n(PC1 of 23 virtual channels, per-patient mean)", fontsize=7)
ax.set_ylabel("Measured composition\n(single-cell positivity: epithelial − immune)", fontsize=7)
ax.set_title("Virtual channels recover tumour composition", fontsize=7.5, loc="left", pad=4)
ax.text(0.97, 0.05, f"Spearman $\\rho$ = {r:.2f}\n$p$ = {p:.1e}\nn = {n} patients",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#333333")
# §6.9 直接标注极值
o = np.argsort(pc1)
for k, tag, off in ((o[-1], "epithelial-rich", (-52, -6)), (o[0], "immune-rich", (46, 4))):
    ax.annotate(tag, xy=(pc1[k], true[k]), xytext=off, textcoords="offset points",
                fontsize=6, color="#555555", ha="center", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#888888", shrinkA=1, shrinkB=2))
ax.margins(0.10)
panel_letter(ax, "a", dx=-0.30, dy=1.035, fontsize=11)

# ---------------- (b) 逐谱系 rho ----------------
ax = fig.add_subplot(gs[0, 1])
lab = {"epi": "Epithelial", "tcell": "T cell", "mye": "Myeloid", "endo": "Endothelial", "stroma": "Stromal"}
keys = ["epi", "tcell", "mye", "endo", "stroma"]
y = np.arange(len(keys))
r1 = np.array([d["association"][k]["pc1"]["rho"] for k in keys])
r2 = np.array([d["association"][k]["prespec"]["rho"] for k in keys])
for k in range(len(keys)):
    ax.plot([0, r1[k]], [y[k]-0.16, y[k]-0.16], color=C_FOCAL, lw=1.4, zorder=1)
    ax.plot([0, r2[k]], [y[k]+0.16, y[k]+0.16], color=C_LIGHT, lw=1.4, zorder=1)
ax.scatter(r1, y-0.16, s=28, c=C_FOCAL, edgecolors="white", linewidths=0.5, zorder=3,
           label="virtual PC1")
ax.scatter(r2, y+0.16, s=28, c=C_LIGHT, edgecolors="white", linewidths=0.5, zorder=3,
           label="pre-specified score")
ax.axvline(0, color="#444444", lw=0.7)
ax.set_yticks(y); ax.set_yticklabels([lab[k] for k in keys], fontsize=6.5)
for t_, k in zip(ax.get_yticklabels(), keys):
    if k in ("epi", "tcell", "mye"): t_.set_fontweight("bold")
ax.invert_yaxis()
ax.set_xlim(-0.85, 0.85)
ax.set_xlabel("Spearman $\\rho$ with measured abundance", fontsize=7)
ax.legend(frameon=False, fontsize=6, loc="lower left", handletextpad=0.35, borderaxespad=0.1)
ax.set_title("Immune and epithelial compartments drive the axis", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "b", dx=-0.24, dy=1.085, fontsize=11)

# ---------------- (c) 组成 vs 身份 的对比（本文核心对照） ----------------
ax = fig.add_subplot(gs[1, 1])
items = [("Composition\n(41 patients)", [d["association"]["composition"]["pc1"]["rho"],
                                          d["association"]["composition"]["prespec"]["rho"]],
          [C_FOCAL, C_LIGHT]),
         ("Pan-CK", [d["cell_level_paired"]["Pan-CK"]], [C_GREY]),
         ("CD3e",   [d["cell_level_paired"]["CD3e"]],   [C_GREY]),
         ("CD20",   [d["cell_level_paired"]["CD20"]],   [C_GREY]),
         ("CD68",   [d["cell_level_paired"]["CD68"]],   [C_GREY])]
xs, ys, cs = [], [], []
for i, (nm, vals, cols) in enumerate(items):
    off = (np.arange(len(vals))-(len(vals)-1)/2)*0.18
    for v, c, o in zip(vals, cols, off):
        xs.append(i+o); ys.append(v); cs.append(c)
for x, v, c in zip(xs, ys, cs):
    ax.plot([x, x], [0, v], color=c, lw=1.4, zorder=1)
ax.scatter(xs, ys, s=34, c=cs, edgecolors="white", linewidths=0.5, zorder=3)
ax.axvline(0.5, color="#dddddd", lw=0.8)
ax.axhline(0, color="#444444", lw=0.7)
ax.set_xticks(range(len(items)))
ax.set_xticklabels([nm for nm, _, _ in items], fontsize=6.5)
for t_, nm in zip(ax.get_xticklabels(), [nm for nm, _, _ in items]):
    if nm.startswith("Composition"): t_.set_fontweight("bold"); t_.set_color(C_FOCAL)
ax.set_ylim(-0.10, 0.88)
ax.set_ylabel("Spearman $\\rho$", fontsize=7)
ax.text(0.0, 0.962, "patient-level composition", transform=ax.get_xaxis_transform(),
        ha="center", va="bottom", fontsize=6, color=C_FOCAL)
ax.text(3.0, 0.962, "cell-level single marker (within-patient)", transform=ax.get_xaxis_transform(),
        ha="center", va="bottom", fontsize=6, color="#666666")
for x, v in zip(xs, ys):
    if x < 0.5 or x > 3.5 and v < 0.11:
        ax.text(x, v+0.035, f"{v:.2f}", ha="center", va="bottom", fontsize=5.8, color="#444444")
ax.set_title("Composition is recoverable; single-marker identity is not", fontsize=7.5, loc="left", pad=16)
panel_letter(ax, "c", dx=-0.12, dy=1.115, fontsize=11)

fig.savefig(os.path.join(OUT, "Fig11_composition_recovery.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Fig11_composition_recovery.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Fig11 done  rho=%.3f p=%.2e" % (r, p))
