"""B 级正面结果：虚拟"组织组成轴"在 TCGA-THCA 上有没有临床/生物学价值？

我们已经知道（S5）：
  - 虚拟通道 PC1 解释 ~30% 方差，是一个"上皮 <-> 免疫/内皮"的组织组成轴
  - 单通道读不出自己的标记（特异性全负）
本脚本问反面问题：**这个组成轴本身有没有用？**
  1) 组织学亚型：PTC / FTC / PDTC / ATC（去分化程度递增）—— ATC 是免疫冷、间质富集的
  2) 分期 / 年龄 / 性别
  3) 总生存（OS）
并对"单个最佳虚拟通道"做同样的比较，看组成轴是否比任何单通道更强。
"""
import os, sys, json, glob
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
from scipy import stats

L3 = r"W:\虚拟细胞\analysis\l3"; OUT = r"W:\虚拟细胞\analysis"
CH = ["DAPI","TRITC","Cy5","PD-1_1:200","CD14","CD4","T-bet","CD34","CD68_1:100","CD16","CD11c",
      "CD138","CD20","CD3_1:1000","CD8","PD-L1","CK_1:150","Ki67_1:150","Tryptase","Actin-D",
      "Caspase3-D","PHH3-B","Transgelin"]

raw = json.load(open(os.path.join(L3, "thca_clinical_raw.json"), encoding="utf-8"))
clin = {}
for x in raw:
    clin.setdefault(x["patientId"], {})[x["clinicalAttributeId"]] = x["value"]
print(f"临床数据: {len(clin)} 病人, {len(set(k for v in clin.values() for k in v))} 个字段")

vecs = {}
for f in glob.glob(os.path.join(L3, "*_vec.npy")):
    vecs[os.path.basename(f).replace("_vec.npy","")] = np.load(f)
pat2vec = {}
for n, v in vecs.items(): pat2vec.setdefault(n[:12], []).append(v)
pat2vec = {p: np.mean(vs, 0) for p, vs in pat2vec.items()}
pats = sorted(set(pat2vec) & set(clin))
V = np.array([pat2vec[p] for p in pats])
n = len(pats)
print(f"虚拟通道 + 临床可配对: {n} 病人\n")

def zs(a): return (a - a.mean(0, keepdims=True)) / (a.std(0, keepdims=True) + 1e-9)
Vz = zs(V)
U, S, Vt = np.linalg.svd(Vz - Vz.mean(0), full_matrices=False)
ev = S**2/(S**2).sum()
pc1 = (Vz - Vz.mean(0)) @ Vt[0]
print(f"虚拟 PC1 解释方差 {ev[0]*100:.1f}%（PC2 {ev[1]*100:.1f}%）")
print(f"PC1 载荷前 6: " + ", ".join(f"{CH[j]}({Vt[0][j]:+.2f})" for j in np.argsort(-np.abs(Vt[0]))[:6]))
print()

def get(p, k): return clin.get(p, {}).get(k)

# ---------- 1) 组织学亚型 ----------
print("="*92)
print("  1) 组织学亚型 / 去分化程度")
print("="*92)
for key in ("HISTOLOGICAL_DIAGNOSIS", "SUBTYPE", "CANCER_TYPE_DETAILED"):
    vals = [get(p, key) for p in pats]
    from collections import Counter
    c = Counter(v for v in vals if v is not None)
    if len(c) > 1:
        print(f"  {key}: {dict(c.most_common(10))}")
print()
key = "HISTOLOGICAL_DIAGNOSIS"
groups = {}
for i, p in enumerate(pats):
    v = get(p, key)
    if v: groups.setdefault(v, []).append(pc1[i])
if len(groups) >= 2:
    ks = stats.kruskal(*[np.array(g) for g in groups.values() if len(g) >= 3])
    print(f"  PC1 跨亚型 Kruskal-Wallis: H={ks.statistic:.2f}, p={ks.pvalue:.3e}")
    for g, vals in sorted(groups.items(), key=lambda t: -np.median(t[1])):
        if len(vals) < 2: continue
        print(f"    {g:34s} n={len(vals):3d}  中位 PC1 = {np.median(vals):+7.3f}  "
              f"(IQR {np.percentile(vals,25):+.2f} ~ {np.percentile(vals,75):+.2f})")

# ---------- 2) 分期 / 年龄 / 性别 ----------
print("\n" + "="*92)
print("  2) 分期 / 年龄 / 性别")
print("="*92)
def stage_num(sv):
    if not sv: return None
    s2 = sv.upper().replace("STAGE ", "").strip()
    for a, b in (("IV","4"),("III","3"),("II","2"),("I","1")):
        if s2.startswith(a): return int(b)
    return None
for key in ("TUMOR_STAGE", "AJCC_PATHOLOGIC_TUMOR_STAGE"):
    sv = [stage_num(get(p, key)) for p in pats]
    ok = [(x, y) for x, y in zip(pc1, sv) if y is not None]
    if len(ok) > 20:
        r, pv = stats.spearmanr([a for a, _ in ok], [b for _, b in ok])
        print(f"  PC1 vs {key:32s} n={len(ok):3d}  rho={r:+.3f}  p={pv:.3e}")
age = [(x, float(y)) for x, y in ((pc1[i], get(p, "AGE")) for i, p in enumerate(pats)) if y]
if age:
    r, pv = stats.spearmanr([a for a, _ in age], [b for _, b in age])
    print(f"  PC1 vs AGE{'':26s} n={len(age):3d}  rho={r:+.3f}  p={pv:.3e}")
sexm = [pc1[i] for i, p in enumerate(pats) if get(p, "SEX") == "Male"]
sexf = [pc1[i] for i, p in enumerate(pats) if get(p, "SEX") == "Female"]
if sexm and sexf:
    mw = stats.mannwhitneyu(sexm, sexf)
    print(f"  PC1 vs SEX  Male n={len(sexm)} median={np.median(sexm):+.3f} | "
          f"Female n={len(sexf)} median={np.median(sexf):+.3f}  MWU p={mw.pvalue:.3f}")

# ---------- 3) 总生存 ----------
print("\n" + "="*92)
print("  3) 总生存（按 PC1 三分位分组的 log-rank）")
print("="*92)
def logrank(t, e, g):
    g = np.asarray(g); t = np.asarray(t, float); e = np.asarray(e, int)
    O1 = E1 = V1 = 0.0
    for tt in np.unique(t[e == 1]):
        at = t >= tt; n_at = at.sum()
        n1 = ((g == 1) & at).sum(); n0 = ((g == 0) & at).sum()
        d = ((t == tt) & (e == 1)).sum()
        d1 = ((t == tt) & (e == 1) & (g == 1)).sum()
        if n_at <= 1: continue
        E1 += d * n1 / n_at; O1 += d1
        V1 += d * (n1/n_at) * (1 - n1/n_at) * (n_at - d) / (n_at - 1)
    chi2 = (O1 - E1)**2 / V1 if V1 > 0 else 0.0
    return chi2, float(stats.chi2.sf(chi2, 1))
os_m = [get(p, "OS_MONTHS") for p in pats]; os_s = [get(p, "OS_STATUS") for p in pats]
ok = [(i, float(m), 1 if (s and "DECEASED" in s.upper()) else 0)
      for i, (m, s) in enumerate(zip(os_m, os_s)) if m and s]
if len(ok) > 30:
    t = np.array([x[1] for x in ok]); e = np.array([x[2] for x in ok])
    q = np.quantile(pc1[[x[0] for x in ok]], [1/3, 2/3])
    g = (pc1[[x[0] for x in ok]] > q[1]).astype(int)
    chi2, pv = logrank(t, e, g)
    print(f"  n={len(ok)}  事件={int(e.sum())}  PC1 高三分位 vs 其余: chi2={chi2:.3f} p={pv:.3f}")
    c, pv2 = stats.spearmanr(pc1[[x[0] for x in ok]], t)
    print(f"  PC1 vs OS_MONTHS Spearman: rho={c:+.3f} p={pv2:.3f}")

# ---------- 4) 单通道 vs 组成轴 ----------
print("\n" + "="*92)
print("  4) 组成轴 vs 单通道：对最强关联的对比")
print("="*92)
key = "HISTOLOGICAL_DIAGNOSIS"
gl = [get(p, key) for p in pats]
gg = {}
for i, g in enumerate(gl):
    if g: gg.setdefault(g, []).append(i)
big = {g: idx for g, idx in gg.items() if len(idx) >= 3}
if len(big) >= 2:
    yb = np.concatenate([np.full(len(ix), k) for k, ix in enumerate(big.values())])
    idxb = np.concatenate(list(big.values()))
    # 用 eta^2 度量每个通道对亚型的区分能力
    def eta2(x, y):
        gm = [x[y == k].mean() for k in np.unique(y)]
        ssb = sum(((x[y == k].mean() - x.mean())**2 * (y == k).sum() for k in np.unique(y)))
        return float(ssb / (((x - x.mean())**2).sum() + 1e-12))
    e_pc1 = eta2(pc1[idxb], yb)
    rows = sorted(((eta2(V[idxb, j], yb), CH[j]) for j in range(23)), reverse=True)
    print(f"  亚型分组: {dict((k, len(v)) for k, v in big.items())}")
    print(f"  **虚拟 PC1 的 eta^2 = {e_pc1:.4f}**")
    print("  单通道 eta^2 前 6: " + ", ".join(f"{c}({v:.4f})" for v, c in rows[:6]))
    print(f"  -> PC1 {'优于' if e_pc1 > rows[0][0] else '不优于'} 最佳单通道（{rows[0][1]}）")

stats_out = {}
for key2 in ("TUMOR_STAGE", "AJCC_PATHOLOGIC_TUMOR_STAGE"):
    sv2 = [stage_num(get(p, key2)) for p in pats]
    okk = [(x, y) for x, y in zip(pc1, sv2) if y is not None]
    if len(okk) > 20:
        rr, pp = stats.spearmanr([a for a, _ in okk], [b for _, b in okk])
        stats_out[key2] = dict(n=len(okk), kind="spearman", stat=float(rr), p=float(pp))
if age:
    rr, pp = stats.spearmanr([a for a, _ in age], [b for _, b in age])
    stats_out["AGE"] = dict(n=len(age), kind="spearman", stat=float(rr), p=float(pp))
if sexm and sexf:
    mw2 = stats.mannwhitneyu(sexm, sexf)
    stats_out["SEX"] = dict(n=len(sexm)+len(sexf), kind="mwu", stat=float(mw2.statistic), p=float(mw2.pvalue),
                            median_male=float(np.median(sexm)), median_female=float(np.median(sexf)))
km = None
if len(ok) > 30:
    t2 = np.array([x[1] for x in ok]); e2 = np.array([x[2] for x in ok])
    g2 = (pc1[[x[0] for x in ok]] > np.quantile(pc1[[x[0] for x in ok]], 2/3)).astype(int)
    chi2b, pvb = logrank(t2, e2, g2)
    km = dict(n=len(ok), events=int(e2.sum()), chi2=float(chi2b), p=float(pvb),
              t=[float(v) for v in t2], e=[int(v) for v in e2], g=[int(v) for v in g2],
              groups=[[float(x) for x in pc1[[x[0] for x in ok]][g2 == k]] for k in (0, 1)])
    stats_out["OS_LOGRANK"] = dict(n=len(ok), kind="logrank", stat=float(chi2b), p=float(pvb))
json.dump(dict(n_patients=n, pc1_explained=float(ev[0]), pc1_loadings={CH[j]: float(Vt[0][j]) for j in range(23)},
               histology={g: [float(x) for x in v] for g, v in groups.items()} if groups else {},
               association=stats_out, km=km, patients=pats, pc1=[float(x) for x in pc1]),
          open(os.path.join(OUT, "supp_10_clinical_assoc.json"), "w"), indent=1)
print("\n-> supp_10_clinical_assoc.json")
