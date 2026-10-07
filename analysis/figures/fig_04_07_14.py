"""Fig 4（L3 222 病人）、Fig 7（四层综合）、Fig 14（临床关联）"""
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
C1, C2, C3, CG = "#1f6fb4", "#d1622f", "#5b8c5a", "#9a9a9a"
DIV = LinearSegmentedColormap.from_list("bo", ["#2b6ca3", "#7fb2d4", "#f2f2f2", "#e8a97e", "#c15a1f"])
CH = ["DAPI","TRITC","Cy5","PD-1_1:200","CD14","CD4","T-bet","CD34","CD68_1:100","CD16","CD11c",
      "CD138","CD20","CD3_1:1000","CD8","PD-L1","CK_1:150","Ki67_1:150","Tryptase","Actin-D",
      "Caspase3-D","PHH3-B","Transgelin"]

# ================= Fig 4 =================
foc = json.load(open(os.path.join(A, "l3", "l3_paired_focus.json"), encoding="utf-8"))
rb = json.load(open(os.path.join(A, "supp_04_l3_robustness.json"), encoding="utf-8"))
# 重算 222 病人的向量与 RPPA
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

fig = plt.figure(figsize=(9.0, 5.8))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.30], height_ratios=[1.0, 1.0], wspace=0.34, hspace=0.66)

# (a) 判决性散点：虚拟 CD20 vs RPPA CD20
ax = fig.add_subplot(gs[0, 0])
y = M[prots.index("CD20|CD20")][cols]
x = V[:, CH.index("CD20")]
r, p = stats.spearmanr(x, y)
ax.scatter(x, y, s=16, c=CG, alpha=0.75, linewidths=0, zorder=3)
sl, ic = np.polyfit(x, y, 1); xx = np.linspace(x.min(), x.max(), 50)
ax.plot(xx, sl*xx+ic, color=C2, lw=1.1, zorder=2)
ax.set_xlabel("Virtual CD20 density", fontsize=7); ax.set_ylabel("RPPA CD20 (bulk)", fontsize=7)
ax.set_title("The paired channel is not a readout", fontsize=7.5, loc="left", pad=4)
ax.text(0.97, 0.92, f"$\\rho$ = {r:+.3f}   $p$ = {p:.3f}\nn = {n} patients", transform=ax.transAxes,
        ha="right", va="top", fontsize=6.2, color="#333333")
panel_letter(ax, "a", dx=-0.26, dy=1.07, fontsize=10)

# (b) 3 靶 x 23 通道 |rho| 热图
ax = fig.add_subplot(gs[:, 1])
matr = np.array([[abs(foc[t[3]]["all_rhos"][c]) for c in CH] for t in TGT])
im = ax.imshow(matr, cmap="magma", aspect="auto", vmin=0, vmax=0.45)
ax.set_yticks(range(3))
ax.set_yticklabels([f"{t[0]}\nrank {foc[t[3]]['rank']}/23" for t in TGT], fontsize=6.2)
ax.set_xticks(range(23)); ax.set_xticklabels(CH, rotation=90, fontsize=5.0)
for t in ax.get_xticklabels():
    if t.get_text() in ("TRITC", "Cy5"): t.set_color("#00d5ff"); t.set_fontweight("bold")
for i, t in enumerate(TGT):
    j = CH.index(t[1]); ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, ec="#00d5ff", lw=1.1))
cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02); cb.set_label("$|\\rho|$", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
ax.set_xlabel("Virtual channel", fontsize=7)
ax.set_title("The paired channel never ranks first", fontsize=7.5, loc="left", pad=4)
ax.text(1.0, -0.30, "cyan = background channels", transform=ax.transAxes, ha="right", fontsize=5.6, color="#666666")
panel_letter(ax, "b", dx=-0.08, dy=1.03, fontsize=10)

# (c) 稳健性小结
ax = fig.add_subplot(gs[1, 0])
sh = rb["split_half"]
labs = ["CD20", "Caspase-3", "CD31"]
sc = [sh[k]["sign_consistency"]*100 for k in ("CD20", "Caspase3", "CD31")]
ax.barh(range(3), sc, color=[C1 if v > 85 else (C2 if v < 70 else CG) for v in sc], height=0.55, zorder=2)
ax.axvline(50, color="#666666", lw=0.8, ls="--")
ax.set_yticks(range(3)); ax.set_yticklabels(labs, fontsize=6.5); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_xlabel("Split-half sign consistency (%)", fontsize=7)
for k, v in enumerate(sc):
    ax.text(v+1.5, k, f"{v:.0f}%", va="center", fontsize=6, color="#333333")
ax.set_title("Only the CD20 sign reproduces", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "c", dx=-0.26, dy=1.11, fontsize=10)
fig.savefig(os.path.join(OUT, "Fig04_L3_tcga.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Fig04_L3_tcga.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Fig4 done  CD20 rho=%.3f" % r)

# ================= Fig 7：四层综合 =================
hem = json.load(open(os.path.join(A, "supp_09b_hemit_scale_sweep.json"), encoding="utf-8"))
sp = json.load(open(os.path.join(A, "l2", "l2_specificity.json"), encoding="utf-8"))
hemrank = {g: hem["result"]["specificity_vs_scale"][g]["1.0"]["rank"] for g in ("DAPI", "CK", "CD3")}
L1_win = sum(1 for g in hemrank if hemrank[g] == 1)
L2_win = sum(1 for d in sp["specificity"] if d["specificity"] > 0)
L3_win = sum(1 for t in TGT if foc[t[3]]["rank"] == 1)
pp = json.load(open(os.path.join(A, "orion", "preproc_ablation_summary.json"), encoding="utf-8"))["per_patient"]
pats41 = sorted(pp)
rate41 = {}
for pat in pats41:
    for d in pp[pat]["tile"]["detail"]:
        if d.get("paired_rho") is not None:
            rate41.setdefault(d["marker"], []).append(1 if d["specific"] else 0)
L4_win = sum(1 for m, v in rate41.items() if np.mean(v) > 0.5)
LAYERS = [("L1\nHEMIT", L1_win, 3), ("L2\nVisium", L2_win, 11),
          ("L3\nTCGA", L3_win, 3), ("L4\nORION", L4_win, 13)]
NLAB = ["945\npatches", "15,489\nspots", "222\npatients", "994,729\ncells"]
NUNITS = [945, 15489, 222, 994729]

fig = plt.figure(figsize=(9.0, 3.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.85, 1.25], wspace=0.34)
ax = fig.add_subplot(gs[0, 0])
wr = [100*w/t for _, w, t in LAYERS]
ax.bar(range(4), wr, 0.6, color=[C1 if v > 20 else C2 for v in wr], zorder=2)
for k, (nm, w, t) in enumerate(LAYERS):
    ax.text(k, wr[k]+2.5, f"{w}/{t}", ha="center", fontsize=6.2, color="#333333")
ax.set_xticks(range(4))
ax.set_xticklabels([f"{n}\n{NLAB[k]}" for k, (n, _, _) in enumerate(LAYERS)], fontsize=5.6)
ax.set_ylim(0, 45); ax.set_ylabel("Marker pairs passing specificity (%)", fontsize=7)
ax.set_title("Most pairs fail specificity at every layer", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "a", dx=-0.24, dy=1.12, fontsize=10)

ax = fig.add_subplot(gs[0, 1])
comp = json.load(open(os.path.join(A, "supp_05_composition_axis.json"), encoding="utf-8"))
c11 = json.load(open(os.path.join(A, "supp_11_composition_recovery.json"), encoding="utf-8"))
cv = [comp["orion"]["rho_pc1_vs_real_axis"], c11["association"]["composition"]["pc1"]["rho"]]
ax.barh([0, 1], cv, 0.5, color=C1, zorder=2)
ax.set_yticks([0, 1]); ax.set_yticklabels(["cell level\n(994,729 cells)", "patient level\n(41 patients)"], fontsize=6.2)
ax.invert_yaxis(); ax.set_xlim(0, 0.90)
ax.set_xlabel("$\\rho$: virtual axis vs. measured composition", fontsize=6.8)
for k, v in enumerate(cv):
    ax.text(v+0.015, k, f"{v:.2f}", va="center", fontsize=6.4, color="#333333")
ax.set_title("Composition recovers (ORION)", fontsize=7.5, loc="left", pad=4)
ax.text(0.0, -0.62, "TCGA (independent panel): 56 / 175 RPPA proteins\nsignificantly track the same axis",
        fontsize=5.6, color="#666666", va="top")
panel_letter(ax, "b", dx=-0.30, dy=1.12, fontsize=10)

ax = fig.add_subplot(gs[0, 2])
ax.scatter(NUNITS, wr, s=48, c=[C1 if v > 20 else C2 for v in wr], zorder=3, edgecolors="white", linewidths=0.6)
for k, (nm, w, t) in enumerate(LAYERS):
    ax.annotate(nm.split("\n")[0], xy=(NUNITS[k], wr[k]), xytext=(0, 11), textcoords="offset points",
                ha="center", fontsize=5.8, color="#555555")
ax.axhspan(0, 20, color="#f4f4f4", zorder=0, lw=0)
ax.set_xscale("log"); ax.set_xlim(120, 4e6); ax.set_ylim(-3, 45)
ax.set_xlabel("Number of measured units in the layer (log)", fontsize=7)
ax.set_ylabel("Pairs passing specificity (%)", fontsize=7)
ax.set_title("More measurement does not rescue it", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "c", dx=-0.20, dy=1.12, fontsize=10)
fig.savefig(os.path.join(OUT, "Fig07_summary.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Fig07_summary.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Fig7 done  L1=%d/3 L2=%d/11 L3=%d/3 L4=%d/13" % (L1_win, L2_win, L3_win, L4_win))

# ================= Fig 14：临床关联（阴性） =================
cl = json.load(open(os.path.join(A, "supp_10_clinical_assoc.json"), encoding="utf-8"))
fig = plt.figure(figsize=(9.0, 3.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25], wspace=0.34)
# (a) KM
ax = fig.add_subplot(gs[0, 0])
km = cl["km"]
t = np.array(km["t"]); e = np.array(km["e"]); g = np.array(km["g"])
for k, (lbl, col) in enumerate((("PC1 low/mid", CG), ("PC1 top tertile", C1))):
    m = (g == k)
    tt = np.sort(np.unique(t[m])); surv = []
    for tv in tt:
        surv.append(1 - e[(t <= tv) & m].sum()/m.sum())
    ax.step(tt, surv, where="post", color=col, lw=1.3, label=lbl)
ax.set_ylim(0.6, 1.02); ax.set_xlim(0, 160)
ax.set_xlabel("Overall survival (months)", fontsize=7); ax.set_ylabel("Survival probability", fontsize=7)
ax.legend(frameon=False, fontsize=5.8, loc="lower left")
ax.set_title(f"No survival separation (log-rank $p$ = {km['p']:.2f})", fontsize=7.5, loc="left", pad=4)
ax.text(0.97, 0.06, f"{km['events']} events / {km['n']} patients", transform=ax.transAxes, ha="right",
        va="bottom", fontsize=5.8, color="#666666")
panel_letter(ax, "a", dx=-0.24, dy=1.12, fontsize=10)
# (b) forest
ax = fig.add_subplot(gs[0, 1])
ass = cl["association"]
items = [("Stage", ass["AJCC_PATHOLOGIC_TUMOR_STAGE"], C1), ("Age", ass["AGE"], C1)]
ypos = [0, 1]
for k, (nm, d, col) in enumerate(items):
    n_ = d["n"]; r_ = d["stat"]
    se = 1/np.sqrt(n_-3)
    z = stats.norm.ppf(0.975)
    ax.plot([r_-z*se*0.55, r_+z*se*0.55], [k, k], color=col, lw=1.4)
    ax.scatter([r_], [k], s=34, c=col, zorder=3, edgecolors="white", linewidths=0.5)
    ax.text(0.42, k, f"$p$ = {d['p']:.2f}", fontsize=6, va="center", color="#333333")
ax.axvline(0, color="#444444", lw=0.8)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Stage", "Age"], fontsize=6.5); ax.invert_yaxis()
ax.set_xlim(-0.45, 0.62); ax.set_ylim(1.7, -0.7)
ax.text(0.0, -0.62, "Stage, age and sex are all null;\nsurvival is underpowered (14 events)",
        fontsize=5.6, color="#666666", va="top")
ax.set_xlabel("Spearman $\\rho$ with the virtual composition axis", fontsize=6.8)
ax.set_title("No clinical association", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "b", dx=-0.28, dy=1.12, fontsize=10)
# (c) loadings
ax = fig.add_subplot(gs[0, 2])
ld = [cl["pc1_loadings"][c] for c in CH]
ax.bar(range(23), ld, 0.72, color=[C2 if v < 0 else C1 for v in ld], zorder=2)
ax.axhline(0, color="#444444", lw=0.8)
ax.set_xticks(range(23)); ax.set_xticklabels(CH, rotation=90, fontsize=5.0)
ax.set_ylabel("PC1 loading", fontsize=7)
ax.set_title(f"PC1 = immune-low axis ({cl['pc1_explained']*100:.0f}% variance)", fontsize=7.5, loc="left", pad=4)
panel_letter(ax, "c", dx=-0.20, dy=1.12, fontsize=10)
fig.savefig(os.path.join(OUT, "Fig14_clinical.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Fig14_clinical.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Fig14 done")
