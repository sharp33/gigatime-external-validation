"""补充实验 S5：正面结论 —— 虚拟通道的共享轴是否对应真实的"组织组成轴"？

思路：前面四层都说明"单通道不特异"。那么反面问题是：**它们的共享成分有没有生物学意义？**
  (A) ORION-CRC 细胞级（n≈20 万细胞）：定义真实"上皮 ↔ 免疫"组成轴，
      看虚拟通道 PC1 与它的相关，并与"任一单通道 vs 自身配对标记"对比。
  (B) TCGA-THCA 病人级（n=222）：虚拟通道 PC1 与 175 个 RPPA 蛋白的相关谱，
      看是否出现自洽的"上皮高 / 免疫低"生物学轴。
零 GPU（复用已保存的细胞级/切片级向量）。
"""
import os, sys, glob, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
from scipy import stats

OUT = r"W:\虚拟细胞\analysis"
ORION = os.path.join(OUT, "orion")
CH = ["DAPI","TRITC","Cy5","PD-1_1:200","CD14","CD4","T-bet","CD34","CD68_1:100","CD16","CD11c",
      "CD138","CD20","CD3_1:1000","CD8","PD-L1","CK_1:150","Ki67_1:150","Tryptase","Actin-D",
      "Caspase3-D","PHH3-B","Transgelin"]

EPI = ["Pan-CK", "E-cadherin"]
IMM = ["CD45", "CD3e", "CD4", "CD8a", "CD20", "CD68", "CD163", "FOXP3", "CD45RO"]

report = {}

# ---------------- (A) ORION 细胞级 ----------------
print("="*96)
print("  (A) ORION-CRC：虚拟通道的共享轴 vs 真实'上皮<->免疫'组成轴（细胞级）")
print("="*96)
files = sorted(glob.glob(os.path.join(ORION, "*_reinhard_cells.npz")))
if not files:
    files = sorted([f for f in glob.glob(os.path.join(ORION, "*_cells.npz"))
                    if not any(f.endswith(f"_{v}_cells.npz") for v in ("reinhard","none","tile"))])
print(f"  使用 {len(files)} 个病人文件: {[os.path.basename(f) for f in files]}")

Vs, Xs, pats, MARKS = [], [], [], None
for f in files:
    z = np.load(f)
    marks = [k[2:] for k in z.files if k.startswith("r_")]
    if not all(m in marks for m in EPI + IMM):
        print(f"  {os.path.basename(f)}: 缺少组成轴所需标记，跳过"); continue
    if MARKS is None: MARKS = list(marks)
    V = z["V"].astype(np.float64)
    Xfull = np.stack([z[f"r_{m}"] for m in marks], 1)
    Vs.append(V); Xs.append(Xfull); pats.append(os.path.basename(f).split("_")[0])
V = np.vstack(Vs); Xf = np.vstack(Xs)
e = Xf[:, MARKS.index("Pan-CK")] if "Pan-CK" in MARKS else Xf[:, MARKS.index("E-cadherin")]
imm_idx = [MARKS.index(m) for m in IMM if m in MARKS]
i_axis = Xf[:, imm_idx].mean(1)
X = np.column_stack([e, i_axis])
print(f"  合并 {len(pats)} 病人 / {len(V)} 细胞；组成轴 = mean(Pan-CK, E-cadherin) - mean({'/'.join(m for m in IMM if m in MARKS)})")

def zs(a): return (a - a.mean(0, keepdims=True)) / (a.std(0, keepdims=True) + 1e-9)
Vz = zs(V)
# 虚拟通道 PC1（共享/组成轴）
U, S, Vt = np.linalg.svd(Vz - Vz.mean(0), full_matrices=False)
ev = S**2/(S**2).sum()
pc1 = (Vz - Vz.mean(0)) @ Vt[0]
# 真实组成轴：上皮 - 免疫（各自 z 化后相减）
real_axis = zs(X[:, [0]]).ravel() - zs(X[:, [1]]).ravel()

rho_pc1_real = stats.spearmanr(pc1, real_axis).statistic
print(f"\n  虚拟通道 PC1 解释方差 {ev[0]*100:.1f}%（PC2 {ev[1]*100:.1f}%, PC3 {ev[2]*100:.1f}%）")
print(f"  虚拟 PC1 vs 真实(上皮-免疫)轴   Spearman rho = {rho_pc1_real:+.3f}")
print(f"\n  各虚拟通道与真实组成轴的 rho（绝对值前 8）:")
rr = np.array([stats.spearmanr(V[:, j], real_axis).statistic for j in range(len(CH))])
for j in np.argsort(-np.abs(rr))[:8]:
    print(f"    {CH[j]:14s} {rr[j]:+8.3f}")
ref = {"Pan-CK":"CK_1:150","E-cadherin":"CK_1:150","CD3e":"CD3_1:1000","CD20":"CD20",
       "CD68":"CD68_1:100","CD4":"CD4","CD8a":"CD8","Hoechst":"DAPI"}
print("  配对通道自相关（同批细胞）:")
pair_rhos = {}
for mk, chn in ref.items():
    if mk not in MARKS: continue
    r = stats.spearmanr(V[:, CH.index(chn)], Xf[:, MARKS.index(mk)]).statistic
    pair_rhos[mk] = float(r)
    print(f"    {mk:12s} -> {chn:14s} {r:+8.3f}")
print(f"\n  => 虚拟 PC1 vs 真实组成轴 rho={rho_pc1_real:+.3f}；最佳单配对 rho={max(pair_rhos.values()):+.3f}")

print("\n  ★ 核心对照：每个通道 与'自己的标记' vs 与'真实组成轴'的相关（同批细胞，未残差化）")
print(f"  {'ORION 标记':12s} {'虚拟通道':14s} {'|vs 自己标记|':>14s} {'|vs 组成轴|':>12s} {'比值':>8s}")
own_vs_comp = {}
for mk, chn in ref.items():
    if mk not in MARKS: continue
    a = abs(stats.spearmanr(V[:, CH.index(chn)], Xf[:, MARKS.index(mk)]).statistic)
    b = abs(stats.spearmanr(V[:, CH.index(chn)], real_axis).statistic)
    own_vs_comp[mk] = dict(channel=chn, abs_rho_own=float(a), abs_rho_composition=float(b),
                           ratio=float(b/(a+1e-9)))
    print(f"  {mk:12s} {chn:14s} {a:14.3f} {b:12.3f} {b/(a+1e-9):8.2f}")
mm = np.mean([v["ratio"] for v in own_vs_comp.values()])
print(f"\n  平均比值 = {mm:.2f}  ->  虚拟通道与'全局组织组成轴'的相关平均是与'自己标记'相关的 {mm:.1f} 倍")


report["orion"] = dict(n_patients=len(pats), n_cells=int(len(V)), patients=pats,
                       pc_explained=[float(x) for x in ev[:5]], rho_pc1_vs_real_axis=float(rho_pc1_real),
                       channel_vs_real_axis={CH[j]: float(rr[j]) for j in range(len(CH))},
                       paired_rhos=pair_rhos,
                       own_vs_composition=own_vs_comp,
                       mean_ratio_comp_over_own=float(mm))

# ---------------- (B) TCGA-THCA 病人级 ----------------
print("\n" + "="*96)
print("  (B) TCGA-THCA：虚拟通道 PC1 与 175 个 RPPA 蛋白的相关谱")
print("="*96)
L3 = os.path.join(OUT, "l3")
lines = open(os.path.join(L3,"thca_rppa.txt"), encoding="utf-8", errors="replace").read().strip().split("\n")
prots = [l.split("\t")[0] for l in lines[1:]]
M = np.array([l.split("\t")[1:] for l in lines[1:]], dtype=float)
p2c = {s[:12]: i for i, s in enumerate(lines[0].split("\t")[1:])}
vecs = {}
for f in glob.glob(os.path.join(L3, "*_vec.npy")):
    vecs[os.path.basename(f).replace("_vec.npy","")] = np.load(f)
pat2vec = {}
for n2, v in vecs.items(): pat2vec.setdefault(n2[:12], []).append(v)
pat2vec = {p: np.mean(vs, 0) for p, vs in pat2vec.items()}
common = sorted(set(pat2vec) & set(p2c)); cols = [p2c[p] for p in common]
Vt2 = np.array([pat2vec[p] for p in common]); n = len(common)
print(f"  n = {n} 病人")
Vz2 = zs(Vt2)
U2, S2, Vt2b = np.linalg.svd(Vz2 - Vz2.mean(0), full_matrices=False)
ev2 = S2**2/(S2**2).sum()
pc1t = (Vz2 - Vz2.mean(0)) @ Vt2b[0]
print(f"  虚拟通道 PC1 解释方差 {ev2[0]*100:.1f}%（PC2 {ev2[1]*100:.1f}%）")
print(f"  PC1 载荷最高的 6 个通道: " + ", ".join(
    f"{CH[j]}({Vt2b[0][j]:+.2f})" for j in np.argsort(-np.abs(Vt2b[0]))[:6]))
ok = [i for i in range(len(prots)) if np.isfinite(M[i][cols]).all()]
rp = np.array([stats.spearmanr(pc1t, M[i][cols]).statistic for i in ok])
pv = np.array([stats.spearmanr(pc1t, M[i][cols]).pvalue for i in ok])
order = np.argsort(-np.abs(rp))
thr = stats.t.ppf(1-0.025/len(ok), n-2)/np.sqrt(n-2+stats.t.ppf(1-0.025/len(ok), n-2)**2)
print(f"  可评估蛋白 {len(ok)} 个；Bonferroni 阈值 |rho| > {thr:.3f}")
print(f"  通过 Bonferroni 的蛋白数: {(np.abs(rp) > thr).sum()}")
print("\n  |rho| 最高的 15 个蛋白（+ 方向为'上皮/增殖高'侧）:")
for k in order[:15]:
    print(f"    {prots[ok[k]]:34s} rho={rp[k]:+7.3f}  p={pv[k]:.2e}")
print("\n  最低的 10 个蛋白:")
for k in order[-10:]:
    print(f"    {prots[ok[k]]:34s} rho={rp[k]:+7.3f}  p={pv[k]:.2e}")

# 关心蛋白是否出现且方向自洽
watch = {"CDH1|E-Cadherin": +1, "CDH3|P-Cadherin": +1, "CLDN7|Claudin-7": +1,
         "COL6A1|Collagen_VI": -1, "FN1|Fibronectin": -1, "CDH2|N-Cadherin": None,
         "PECAM1|CD31": -1, "CD20|CD20": -1, "PCNA|PCNA": None,
         "CASP3|Caspase-3_active": None, "CTNNB1|beta-Catenin": None}
print("\n  关键蛋白落点:")
wout = {}
for a, sgn in watch.items():
    if a in prots:
        i = prots.index(a)
        if not np.isfinite(M[i][cols]).all(): print(f"    {a:34s} 缺失值"); continue
        r = stats.spearmanr(pc1t, M[i][cols]).statistic
        pct = float((np.abs(rp) < abs(r)).mean()*100)
        wout[a] = dict(rho=float(r), percentile=pct)
        print(f"    {a:34s} rho={r:+7.3f}  位于全部蛋白的 {pct:.0f} 百分位")
    else:
        print(f"    {a:34s} 不在 RPPA 面板中")

report["tcga"] = dict(n_patients=n, pc_explained=[float(x) for x in ev2[:5]],
                      pc1_top_channels={CH[j]: float(Vt2b[0][j]) for j in np.argsort(-np.abs(Vt2b[0]))[:8]},
                      bonferroni_threshold=float(thr), n_pass=int((np.abs(rp) > thr).sum()),
                      top_proteins=[[prots[ok[k]], float(rp[k]), float(pv[k])] for k in order[:20]],
                      watch=wout)
json.dump(report, open(os.path.join(OUT,"supp_05_composition_axis.json"),"w"), indent=1)
print("\n-> supp_05_composition_axis.json")
