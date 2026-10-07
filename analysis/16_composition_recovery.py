"""B 级正面结果：虚拟组成轴能否在**病人层面**复原肿瘤的真实细胞组成？

为什么这个设计干净：
  - 参照（ground truth）来自 ORION 的真实单细胞 19 标记强度，与虚拟预测完全独立
  - 病人层面（n=41）避免了 S5 的细胞级分析把"病人间差异"和"病人内差异"混在一起
  - 虚拟组成轴用**病人均值向量**做主成分，与细胞级参照不共享噪声

同时报告一个预先指定的、不含数据拟合的虚拟组成分数（CK - 免疫通道均值），
以防 PC1 被批评为"用数据自己定义自己"。
"""
import os, sys, glob, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
from scipy import stats

O = r"W:\虚拟细胞\analysis\orion"; OUT = r"W:\虚拟细胞\analysis"
CH = ["DAPI","TRITC","Cy5","PD-1_1:200","CD14","CD4","T-bet","CD34","CD68_1:100","CD16","CD11c",
      "CD138","CD20","CD3_1:1000","CD8","PD-L1","CK_1:150","Ki67_1:150","Tryptase","Actin-D",
      "Caspase3-D","PHH3-B","Transgelin"]

LINE = {"epi": ["Pan-CK", "E-cadherin"], "tcell": ["CD3e", "CD4", "CD8a"],
        "mye":  ["CD45", "CD68", "CD163"], "endo": ["CD31"], "stroma": ["SMA"]}
V_IMM = ["CD3_1:1000", "CD4", "CD14", "CD11c", "CD68_1:100", "CD8", "PD-1_1:200"]

files = sorted(glob.glob(os.path.join(O, "*_reinhard_cells.npz")))
pats, Vm, F = [], [], {k: [] for k in LINE}
for f in files:
    z = np.load(f)
    marks = [k[2:] for k in z.files if k.startswith("r_")]
    if not all(m in marks for m in ["Pan-CK", "CD45", "CD3e"]): continue
    V = z["V"].astype(np.float64)
    R = {m: z[f"r_{m}"].astype(np.float64) for m in marks}
    # 每个标记在该病人内 z 化 -> 阳性 = z>1（约 16%）
    Z = {m: (v - v.mean())/(v.std()+1e-9) for m, v in R.items()}
    rec = {}
    for lin, ms in LINE.items():
        ms = [m for m in ms if m in Z]
        pos = np.zeros(len(V), bool)
        for m in ms: pos |= (Z[m] > 1)
        rec[lin] = float(pos.mean())
    F["epi"].append(rec["epi"]); F["tcell"].append(rec["tcell"]); F["mye"].append(rec["mye"])
    F["endo"].append(rec["endo"]); F["stroma"].append(rec["stroma"])
    Vm.append(V.mean(0)); pats.append(os.path.basename(f).split("_")[0])

Vm = np.array(Vm); n = len(pats)
F = {k: np.array(v) for k, v in F.items()}
true_comp = F["epi"] - (F["tcell"] + F["mye"] + F["endo"]) / 3
print(f"病人 {n} 例")
print(f"真实组成分数 = 上皮阳性率 − (T/髓系/内皮阳性率均值)")
print(f"  上皮率 median={np.median(F['epi']):.3f}  T median={np.median(F['tcell']):.3f}  "
      f"髓系 median={np.median(F['mye']):.3f}  内皮 median={np.median(F['endo']):.3f}  "
      f"间质 median={np.median(F['stroma']):.3f}")

def zs(a): return (a - a.mean(0, keepdims=True))/(a.std(0, keepdims=True)+1e-9)
Vz = zs(Vm)
U, S, Vt = np.linalg.svd(Vz - Vz.mean(0), full_matrices=False)
ev = S**2/(S**2).sum()
pc1 = (Vz - Vz.mean(0)) @ Vt[0]
if np.corrcoef(pc1, true_comp)[0, 1] < 0: pc1 = -pc1; sgn = -1
else: sgn = 1
print(f"\n病人层面虚拟 PC1 解释方差 {ev[0]*100:.1f}%（PC2 {ev[1]*100:.1f}%）")
print("PC1 载荷前 6（正 = 上皮侧）: " +
      ", ".join(f"{CH[j]}({sgn*Vt[0][j]:+.2f})" for j in np.argsort(-np.abs(Vt[0]))[:6]))

# 预先指定的虚拟组成分数（不做任何拟合）
pre_spec = Vm[:, CH.index("CK_1:150")] - Vm[:, [CH.index(c) for c in V_IMM]].mean(1)

print("\n" + "="*94)
print("  ★ 病人层面：虚拟组成 vs 真实细胞组成（n=%d 病人，Spearman）" % n)
print("="*94)
print(f"{'真实组分':16s} {'虚拟 PC1':>22s} {'预先指定分(CK−免疫)':>24s}")
res = {}
for k, name in (("epi","上皮阳性率"), ("tcell","T 细胞阳性率"), ("mye","髓系阳性率"),
                ("endo","内皮阳性率"), ("stroma","间质阳性率")):
    r1, p1 = stats.spearmanr(pc1, F[k]); r2, p2 = stats.spearmanr(pre_spec, F[k])
    res[k] = dict(pc1=dict(rho=float(r1), p=float(p1)), prespec=dict(rho=float(r2), p=float(p2)))
    print(f"{name:16s} {r1:+9.3f} (p={p1:.3f})   {r2:+9.3f} (p={p2:.3f})")
r1, p1 = stats.spearmanr(pc1, true_comp); r2, p2 = stats.spearmanr(pre_spec, true_comp)
res["composition"] = dict(pc1=dict(rho=float(r1), p=float(p1)), prespec=dict(rho=float(r2), p=float(p2)))
print(f"{'★ 综合组成分数':16s} {r1:+9.3f} (p={p1:.2e})   {r2:+9.3f} (p={p2:.2e})")

print("\n  对照：细胞级配对通道 vs 自身标记（同一批数据，病人内）")
print(f"  {'标记':14s} {'虚拟通道':14s} {'病人内 rho(中位)':>18s}")
ctr = {}
for mk, chn in (("Pan-CK","CK_1:150"), ("CD3e","CD3_1:1000"), ("CD20","CD20"), ("CD68","CD68_1:100")):
    rs = []
    for f in files:
        z = np.load(f); marks = [k[2:] for k in z.files if k.startswith("r_")]
        if mk not in marks: continue
        rs.append(stats.spearmanr(z["V"][:, CH.index(chn)], z[f"r_{mk}"]).statistic)
    ctr[mk] = float(np.median(rs))
    print(f"  {mk:14s} {chn:14s} {np.median(rs):+18.3f}")

json.dump(dict(n_patients=n, patients=pats, pc1_explained=float(ev[0]),
               pc1_loadings={CH[j]: float(sgn*Vt[0][j]) for j in range(23)},
               true_composition={k: [float(x) for x in v] for k, v in F.items()},
               true_comp_score=[float(x) for x in true_comp],
               virtual_pc1=[float(x) for x in pc1], virtual_prespec=[float(x) for x in pre_spec],
               association=res, cell_level_paired=ctr),
          open(os.path.join(OUT, "supp_11_composition_recovery.json"), "w"), indent=1)
print("\n-> supp_11_composition_recovery.json")
