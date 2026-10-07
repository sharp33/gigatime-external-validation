"""L3 聚焦分析：配对通道值 + 置换零分布 + 多重检验"""
import os, sys, json, glob
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from scipy import stats

L3 = r"W:\虚拟细胞\analysis\l3"
CH = ["DAPI","TRITC","Cy5","PD-1_1:200","CD14","CD4","T-bet","CD34","CD68_1:100","CD16","CD11c",
      "CD138","CD20","CD3_1:1000","CD8","PD-L1","CK_1:150","Ki67_1:150","Tryptase","Actin-D",
      "Caspase3-D","PHH3-B","Transgelin"]
PAIR = {"CD20":"CD20", "Caspase3":"Caspase3-D", "CD31":"CD34"}   # CD31 概念对应 CD34(内皮)

vecs = {}
for f in glob.glob(os.path.join(L3, "*_vec.npy")):
    vecs[os.path.basename(f).replace("_vec.npy","")] = np.load(f)
pat2vec = {}
for n,v in vecs.items(): pat2vec.setdefault(n[:12], []).append(v)
pat2vec = {p: np.mean(vs,0) for p,vs in pat2vec.items()}

lines = open(os.path.join(L3,"thca_rppa.txt"), encoding="utf-8", errors="replace").read().strip().split("\n")
hdr = lines[0].split("\t"); samples = hdr[1:]
prots = [l.split("\t")[0] for l in lines[1:]]
M = np.array([l.split("\t")[1:] for l in lines[1:]], dtype=float)
p2c = {s[:12]:i for i,s in enumerate(samples)}
common = sorted(set(pat2vec) & set(p2c))
V = np.array([pat2vec[p] for p in common]); cols=[p2c[p] for p in common]
n = len(common)
print(f"n = {n} 病人 (病人级平均)")
print(f"|rho| > {stats.t.ppf(0.975,n-2)/np.sqrt(n-2+stats.t.ppf(0.975,n-2)**2):.3f} 为 p<0.05;  Bonferroni(23 检验) 需 |rho| > {stats.t.ppf(1-0.025/23,n-2)/np.sqrt(n-2+stats.t.ppf(1-0.025/23,n-2)**2):.3f}")

rng = np.random.default_rng(0)
print("\n" + "="*92)
print("  L3：配对通道 vs RPPA（病人级 Spearman, n=%d）" % n)
print("="*92)
print(f"{'RPPA 目标':12s} {'配对虚拟通道':14s} {'rho':>8s} {'p':>9s} {'Bonf':>7s} {'该通道排名':>10s} {'随机零|rho|':>11s}")
out={}
for tname, chname in PAIR.items():
    ag = {"CD20":"CD20|CD20","Caspase3":"CASP3|Caspase-3_active","CD31":"PECAM1|CD31"}[tname]
    y = M[prots.index(ag)][cols]
    rhos = []
    for ch in CH:
        rhos.append(stats.spearmanr(V[:, CH.index(ch)], y).statistic)
    rhos = np.array(rhos)
    ci = CH.index(chname)
    rho = rhos[ci]; p = stats.spearmanr(V[:,ci], y).pvalue
    rank = int((np.abs(rhos) > abs(rho)).sum()) + 1
    # 随机零：打乱 y
    nulls = [abs(stats.spearmanr(V[:,ci], rng.permutation(y)).statistic) for _ in range(2000)]
    p_null = float(np.mean(np.array(nulls) >= abs(rho)))
    bonf = "通过" if p < 0.05/23 else "不通过"
    print(f"{tname:12s} {chname:14s} {rho:+8.3f} {p:9.4f} {bonf:>7s} {rank:>6d}/23 {np.mean(nulls):11.3f}")
    out[tname] = dict(paired_channel=chname, rho=float(rho), p=float(p), rank=rank,
                      bonferroni_pass=bool(p<0.05/23), null_mean=float(np.mean(nulls)),
                      p_perm=p_null, all_rhos={c: float(r) for c,r in zip(CH, rhos)})

print("\n=== 背景通道对照（TRITC/Cy5 本应无意义，若它们排名靠前说明相关是伪影）===")
for tname in PAIR:
    ag = {"CD20":"CD20|CD20","Caspase3":"CASP3|Caspase-3_active","CD31":"PECAM1|CD31"}[tname]
    y = M[prots.index(ag)][cols]
    r = {c: abs(stats.spearmanr(V[:, CH.index(c)], y).statistic) for c in CH}
    order = sorted(r, key=lambda c: -r[c])
    print(f"  {tname:10s} 前三: " + ", ".join(f"{c}({r[c]:.3f})" for c in order[:3]))
json.dump(out, open(os.path.join(L3,"l3_paired_focus.json"),"w"), indent=1)
print("\n-> l3_paired_focus.json")
