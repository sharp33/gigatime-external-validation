"""Fig 5（ORION 41 病人：跨病人通过率与配对 rho 分布）与 Fig 6（负对照）"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"C:\Users\PC\.agents\skills\figure-style")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kernel import apply_figure_style, panel_letter

apply_figure_style(frame="open", sizes=(8, 7, 6))
OUT = r"W:\虚拟细胞\figures"; A = r"W:\虚拟细胞\analysis\orion"
C1, C2, CG = "#1f6fb4", "#d1622f", "#9a9a9a"
p = json.load(open(os.path.join(A, "preproc_ablation_summary.json"), encoding="utf-8"))
V = "tile"                       # 原 L4 口径（逐 tile 归一化）
PP = p["per_patient"]; pats = sorted(PP)
NEG = ["AF1", "Argo550", "CD45", "FOXP3", "CD45RO", "CD163"]
PAIR = ["Pan-CK", "E-cadherin", "CD20", "CD3e", "CD4", "CD31", "Hoechst", "PD-L1", "Ki67", "CD68", "SMA", "PD-1", "CD8a"]
npat = len(pats)

def col(mk, field):
    out = {}
    for pat in pats:
        for d in PP[pat][V]["detail"]:
            if d["marker"] == mk and d.get(field) is not None:
                out.setdefault(mk, []).append(d[field])
    return out

paired_rho = {m: [] for m in PAIR}; spec_pass = {m: [] for m in PAIR}; bother = {m: [] for m in PAIR}
for pat in pats:
    for d in PP[pat][V]["detail"]:
        if d["marker"] in PAIR and d.get("paired_rho") is not None:
            paired_rho[d["marker"]].append(d["paired_rho"])
            spec_pass[d["marker"]].append(1 if d["specific"] else 0)
            bother[d["marker"]].append(d["best_other"])
neg_rho = {m: [] for m in NEG}
for pat in pats:
    for d in PP[pat][V]["detail"]:
        if d["marker"] in NEG and d.get("best_rho") is not None:
            neg_rho[d["marker"]].append(abs(d["best_rho"]))
order = sorted(PAIR, key=lambda m: -np.mean(spec_pass[m]))
print("通过率:", {m: f"{np.mean(spec_pass[m])*100:.0f}%" for m in order})

fig = plt.figure(figsize=(9.0, 6.0))
gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.68)

# --- (a) 通过率 ---
ax = fig.add_subplot(gs[0, 0])
rate = np.array([np.mean(spec_pass[m]) for m in order])*100
cols = [C1 if r >= 25 else (C2 if r > 0 else CG) for r in rate]
ax.barh(range(len(order)), rate, color=cols, height=0.62, zorder=2)
for k, r in enumerate(rate):
    ax.text(r+1.5, k, f"{r:.0f}%", va="center", fontsize=6, color="#333333")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6)
for t in ax.get_yticklabels():
    if t.get_text() in ("Pan-CK", "E-cadherin"): t.set_fontweight("bold")
ax.invert_yaxis(); ax.set_xlim(0, 100)
ax.set_xlabel(f"Patients where the paired channel beats all others (%)", fontsize=7)
ax.set_title("Specificity holds for the epithelial pair, not for immune markers", fontsize=7.5, loc="left", pad=4)
ax.text(0.98, 0.02, f"n = {npat} patients", transform=ax.transAxes, ha="right", fontsize=5.8, color="#666666")
panel_letter(ax, "a", dx=-0.24, dy=1.075, fontsize=11)

# --- (b) 配对 rho 分布 ---
ax = fig.add_subplot(gs[0, 1])
rng = np.random.default_rng(3)
for k, m in enumerate(order):
    v = np.array(paired_rho[m])
    ax.scatter(v, np.full(len(v), k)+rng.uniform(-0.16, 0.16, len(v)), s=7, c=CG, alpha=0.65,
               zorder=2, linewidths=0)
    ax.plot([np.median(v)]*2, [k-0.30, k+0.30], color="#2b2b2b", lw=2.0, zorder=3)
ax.axvline(0, color="#666666", lw=0.7)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6)
ax.invert_yaxis(); ax.set_xlim(-0.6, 0.95)
ax.set_xlabel("Paired-channel residualised $\\rho$ (one point per patient)", fontsize=7)
ax.set_title("Immune channels carry a real but weak shared signal", fontsize=7.5, loc="left", pad=4)
ax.text(0.02, 0.02, "bar = median", transform=ax.transAxes, fontsize=5.6, color="#666666")
panel_letter(ax, "b", dx=-0.24, dy=1.075, fontsize=11)

# --- (c) 配对 vs 最强错配 ---
ax = fig.add_subplot(gs[1, 0])
ax.plot([0, 0.85], [0, 0.85], color="#999999", lw=0.7, ls="--", zorder=1)
for m in PAIR:
    ax.scatter(np.abs(bother[m]), np.abs(paired_rho[m]), s=12, c=CG, alpha=0.6, linewidths=0, zorder=3)
for m, col in (("Pan-CK", C1), ("E-cadherin", C1), ("CD20", C2), ("CD3e", C2)):
    ax.scatter(np.abs(bother[m]), np.abs(paired_rho[m]), s=26, c=col, edgecolors="white",
               linewidths=0.4, zorder=4, label=m)
ax.set_xlim(0, 0.72); ax.set_ylim(0, 0.95)
ax.set_xlabel("Best other channel  $|\\rho|$", fontsize=7)
ax.set_ylabel("Its own paired channel  $|\\rho|$", fontsize=7)
ax.legend(frameon=False, fontsize=5.8, loc="upper left", handletextpad=0.3, borderaxespad=0.3)
ax.set_title("Paired channels do not stand out", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "c", dx=-0.22, dy=1.075, fontsize=11)

# --- (d) 负对照 ---
ax = fig.add_subplot(gs[1, 1])
y = np.arange(len(NEG))
for k, m in enumerate(NEG):
    v = np.array(neg_rho[m])
    ax.scatter(v, np.full(len(v), k)+rng.uniform(-0.16, 0.16, len(v)), s=7, c=C2, alpha=0.6, linewidths=0, zorder=2)
    ax.plot([np.median(v)]*2, [k-0.30, k+0.30], color=C2, lw=2.0, zorder=3)
# 参照：三个最强的配对通道
for k, m in enumerate(["Pan-CK", "CD3e", "CD20"]):
    v = np.abs(np.array(paired_rho[m]))
    ax.scatter(v, np.full(len(v), len(NEG)+0.7+k)+rng.uniform(-0.16, 0.16, len(v)), s=7, c=C1, alpha=0.6,
               linewidths=0, zorder=2)
    ax.plot([np.median(v)]*2, [len(NEG)+0.7+k-0.30, len(NEG)+0.7+k+0.30], color=C1, lw=2.0, zorder=3)
ax.set_yticks(list(y)+[len(NEG)+0.7+k for k in range(3)])
ax.set_yticklabels(NEG+["Pan-CK","CD3e","CD20"], fontsize=6)
for t in ax.get_yticklabels():
    if t.get_text() in ("Pan-CK","CD3e","CD20"): t.set_fontweight("bold")
ax.invert_yaxis(); ax.set_xlim(0, 0.95)
ax.axhline(len(NEG)+0.1, color="#dddddd", lw=0.8)
ax.set_xlabel("Best-matching virtual channel  $|\\rho|$", fontsize=7)
ax.set_title("Markers with no virtual counterpart match equally well", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "d", dx=-0.24, dy=1.075, fontsize=11)

fig.savefig(os.path.join(OUT, "Fig05_ORION_patients.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Fig05_ORION_patients.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Fig5/6 done")

# ---------------- Fig 6：负对照 vs 配对 的直接对比 ----------------
win = []
for pat in pats:
    det = PP[pat][V]["detail"]
    nb = [abs(d["best_rho"]) for d in det if d["marker"] in NEG and d.get("best_rho") is not None]
    pb = [abs(d["paired_rho"]) for d in det if d["marker"] in PAIR and d.get("paired_rho") is not None]
    if nb and pb: win.append(max(nb) > max(pb))
print("负对照胜出比例: %.0f%%" % (100*np.mean(win)))
