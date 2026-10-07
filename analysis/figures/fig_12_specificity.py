"""论文主图 Fig 11（组成复原）与 Fig 12（官方口径交叉通道特异性）
遵循 figure-style：role-mapped 字号、外置刻度、无框图例、300 dpi、CVD-safe 配色。
"""
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
OUT = r"W:\虚拟细胞\figures"; os.makedirs(OUT, exist_ok=True)
A = r"W:\虚拟细胞\analysis"

# ---- 统一色板（两个图共用；§4.1 threading）----
C_FOCAL = "#1f6fb4"      # 焦点（本文主张的对象）
C_CTRL  = "#d1622f"      # 负对照 / 背景通道
C_MUTED = "#9a9a9a"      # 其他
C_POS   = "#1f6fb4"      # 特异性 > 0
C_NEG   = "#d1622f"      # 特异性 < 0

# ==================================================================
#  Fig 12 —— 官方口径下的交叉通道特异性
# ==================================================================
d = json.load(open(os.path.join(A, "supp_08b_official_specificity.json"), encoding="utf-8"))
rows = d["orig"]; M = np.array(d["matrix_orig"]); CH = d["channels"]
name2i = {c: i for i, c in enumerate(CH)}
order = sorted(range(len(rows)), key=lambda k: -rows[k]["specificity"])
ylab = [rows[k]["channel"] for k in order]
spec = np.array([rows[k]["specificity"] for k in order])
diag = np.array([rows[k]["diag"] for k in order])
bother = np.array([rows[k]["best_other"] for k in order])

fig = plt.figure(figsize=(8.8, 5.7))
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.0],
                      wspace=0.30, hspace=0.55)

# --- (a) 23x23 Dice 矩阵 ---
ax = fig.add_subplot(gs[:, 0])
im = ax.imshow(M, cmap="magma", vmin=0, vmax=np.nanmax(M), origin="upper")
ax.set_xticks(range(23)); ax.set_yticks(range(23))
ax.set_xticklabels(CH, rotation=90, fontsize=5.0)
ax.set_yticklabels(CH, fontsize=5.0)
for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
    if t.get_text() in ("TRITC", "Cy5"):
        t.set_color(C_CTRL); t.set_fontweight("bold")
ax.set_xlabel("Reference mask channel (ground-truth mIF)", fontsize=7)
ax.set_ylabel("Virtual channel (prediction)", fontsize=7)
for i in range(23):
    ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, ec="#42e2ff", lw=0.55))
cb = fig.colorbar(im, ax=ax, fraction=0.044, pad=0.03)
cb.set_label("Dice", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
panel_letter(ax, "a", dx=-0.24, dy=1.035, fontsize=11)
ax.text(1.0, -0.235, "n = 50 released sample tiles", transform=ax.transAxes,
        fontsize=5.8, color="#666666", ha="right", va="top")

# --- (b) 特异性 lollipop ---
ax = fig.add_subplot(gs[0, 1])
y = np.arange(len(spec))
cols = [C_POS if v > 0 else C_NEG for v in spec]
for k in range(len(spec)):
    ax.plot([0, spec[k]], [y[k], y[k]], color=cols[k], lw=1.4, solid_capstyle="butt", zorder=1)
ax.scatter(spec, y, s=26, c=cols, zorder=3, edgecolors="white", linewidths=0.5)
ax.axvline(0, color="#444444", lw=0.7)
ax.set_yticks(y); ax.set_yticklabels(ylab, fontsize=6)
for t in ax.get_yticklabels():
    if t.get_text() in ("TRITC", "Cy5"):
        t.set_color(C_CTRL); t.set_fontweight("bold")
    if t.get_text() == "DAPI":
        t.set_color(C_FOCAL); t.set_fontweight("bold")
ax.invert_yaxis()
ax.set_xlabel("Specificity  $D_{i,i}-\\max_{j\\neq i} D_{i,j}$", fontsize=7)
ax.set_xlim(min(spec)-0.035, max(spec)+0.055)
_it = ylab.index("TRITC")
ax.annotate("background channel\n(declared by the authors)", xy=(spec[_it]*0.62, y[_it]+0.45),
            xytext=(spec.max()*0.30, y[_it]+4.2), fontsize=6, color=C_CTRL, ha="left",
            arrowprops=dict(arrowstyle="-", color=C_CTRL, lw=0.6, shrinkA=0, shrinkB=2))
ax.set_title("Only DAPI separates from its own confusion set", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "b", dx=-0.30, dy=1.075, fontsize=11)

# --- (c) 配对 Dice vs 最强错配 ---
ax = fig.add_subplot(gs[1, 1])
ax.plot([0, 0.75], [0, 0.75], color="#999999", lw=0.7, ls="--", zorder=1)
is_ctrl = [k for k in range(len(rows)) if rows[k]["channel"] in ("TRITC", "Cy5")]
is_foc = [name2i["DAPI"]]
other = [k for k in range(len(rows)) if k not in is_ctrl and k != name2i["DAPI"]]
ax.scatter([rows[k]["best_other"] for k in other], [rows[k]["diag"] for k in other],
           s=22, c=C_MUTED, edgecolors="white", linewidths=0.5, zorder=3, label="marker channel")
ax.scatter([rows[k]["best_other"] for k in is_ctrl], [rows[k]["diag"] for k in is_ctrl],
           s=34, c=C_CTRL, marker="D", edgecolors="white", linewidths=0.5, zorder=4, label="background channel")
ax.scatter([rows[k]["best_other"] for k in is_foc], [rows[k]["diag"] for k in is_foc],
           s=44, c=C_FOCAL, marker="s", edgecolors="white", linewidths=0.5, zorder=4, label="DAPI")
ax.set_xlabel("Best other channel  $\\max_{j\\neq i} D_{i,j}$", fontsize=7)
ax.set_ylabel("Its own channel  $D_{i,i}$", fontsize=7)
ax.set_xlim(0.0, 0.60); ax.set_ylim(0.0, 0.75)
ax.legend(frameon=False, loc="upper left", fontsize=6, handletextpad=0.4)
ax.set_title("Marker channels stay within 0.1 of their best confusable channel", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "c", dx=-0.30, dy=1.085, fontsize=11)

fig.savefig(os.path.join(OUT, "Fig12_official_specificity.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Fig12_official_specificity.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Fig12 done")
