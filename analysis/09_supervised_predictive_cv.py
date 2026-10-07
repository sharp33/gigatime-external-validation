"""补充实验 S6：预测性检验（交叉验证）—— 虚拟通道到底能不能"读出"某个蛋白？

L3 用的是相关性（无监督），这里换成**有监督 + 留出集**：
  用法特征 X = 22 个虚拟通道密度；目标 y = RPPA 某个蛋白。
  重复 5 折交叉验证，报**留出集 Spearman**，与三种对照比：
    (1) 只用一个通道（配对通道）
    (2) 用全部 22 个通道（线性岭回归）
    (3) 置换对照（打乱 y）
  再对"全局组成轴"（RPPA PC1）做同样的事 —— 如果虚拟通道能预测整体组成，
  却预测不了具体蛋白，就把"编码组成、不编码身份"量化到 R² 上。
零 GPU。
"""
import os, sys, glob, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
from scipy import stats

L3 = r"W:\虚拟细胞\analysis\l3"
OUT = r"W:\虚拟细胞\analysis"
CH = ["DAPI","TRITC","Cy5","PD-1_1:200","CD14","CD4","T-bet","CD34","CD68_1:100","CD16","CD11c",
      "CD138","CD20","CD3_1:1000","CD8","PD-L1","CK_1:150","Ki67_1:150","Tryptase","Actin-D",
      "Caspase3-D","PHH3-B","Transgelin"]
FEAT = [c for c in CH if c not in ("TRITC","Cy5")]     # 21 个有生物学含义的虚拟通道
PAIR = {"CD20": ("CD20","CD20|CD20"), "Caspase3": ("Caspase3-D","CASP3|Caspase-3_active"),
        "CD31": ("CD34","PECAM1|CD31")}

lines = open(os.path.join(L3,"thca_rppa.txt"), encoding="utf-8", errors="replace").read().strip().split("\n")
prots = [l.split("\t")[0] for l in lines[1:]]
M = np.array([l.split("\t")[1:] for l in lines[1:]], dtype=float)
p2c = {s2[:12]: i for i, s2 in enumerate(lines[0].split("\t")[1:])}
vecs = {}
for f in glob.glob(os.path.join(L3, "*_vec.npy")):
    vecs[os.path.basename(f).replace("_vec.npy","")] = np.load(f)
pat2vec = {}
for n2, v in vecs.items(): pat2vec.setdefault(n2[:12], []).append(v)
pat2vec = {p: np.mean(vs,0) for p, vs in pat2vec.items()}
common = sorted(set(pat2vec) & set(p2c)); cols = [p2c[p] for p in common]
X = np.array([[pat2vec[p][CH.index(c)] for c in FEAT] for p in common])
n = len(common)
print(f"n = {n} 病人, 特征 = {len(FEAT)} 个虚拟通道")
Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)

def cv_rho(Xf, y, folds=5, reps=20, seed=0, ridge=1.0):
    """重复 K 折 CV，返回留出集 Spearman 与 R^2 的均值"""
    rng = np.random.default_rng(seed); rhos = []; r2s = []
    for _ in range(reps):
        idx = rng.permutation(n)
        for k in range(folds):
            te = idx[k::folds]; tr = np.setdiff1d(idx, te)
            A = np.column_stack([np.ones(len(tr)), Xf[tr]])
            B = np.linalg.solve(A.T@A + ridge*np.eye(A.shape[1]), A.T@y[tr])
            pred = np.column_stack([np.ones(len(te)), Xf[te]]) @ B
            r = stats.spearmanr(pred, y[te]).statistic
            ss = 1 - ((y[te]-pred)**2).sum()/(((y[te]-y[tr].mean())**2).sum()+1e-12)
            if np.isfinite(r): rhos.append(r)
            r2s.append(ss)
    return float(np.mean(rhos)), float(np.mean(r2s))

print("\n" + "="*100)
print("  S6.1 三个配对靶蛋白的留出集预测（5 折 x 20 次重复，岭回归）")
print("="*100)
print(f"{'靶蛋白':10s} {'配对通道':12s} {'单通道 CV rho':>14s} {'全部21通道 CV rho':>18s} {'全部21通道 CV R2':>16s} {'置换零|rho|均值':>16s}")
res61 = {}
rng0 = np.random.default_rng(7)
for tname,(chn, ag) in PAIR.items():
    y = M[prots.index(ag)][cols]
    j = FEAT.index(chn)
    r1,_ = cv_rho(Xz[:, [j]], y)
    r2, rr2 = cv_rho(Xz, y)
    # 置换零：200 次打乱 y 后重做整个 CV（每次用不同折划分），取 |rho| 均值
    nulls = []
    for it in range(200):
        r0, _ = cv_rho(Xz, rng0.permutation(y), reps=1, seed=1000+it)
        nulls.append(r0)
    r0 = float(np.mean(nulls))
    res61[tname] = dict(paired_channel=chn, cv_rho_single=r1, cv_rho_all=r2, cv_r2_all=rr2, cv_rho_permuted=r0)
    print(f"{tname:10s} {chn:12s} {r1:+14.3f} {r2:+18.3f} {rr2:+16.3f} {abs(r0):16.3f}")

# ---- 全局组成轴 ----
print("\n" + "="*100)
print("  S6.2 同一个虚拟通道集合，能不能预测 RPPA 的全局组成轴（PC1/PC2）？")
print("="*100)
ok = [i for i in range(len(prots)) if np.isfinite(M[i][cols]).all()]
Xp = M[ok][:, cols].T
Z = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-9)
U, S, Vt = np.linalg.svd(Z, full_matrices=False)
ev = S**2/(S**2).sum()
print(f"  RPPA 可用蛋白 {len(ok)}；PC1 解释 {ev[0]*100:.1f}%，PC2 {ev[1]*100:.1f}%")
res62 = {}
for k in range(3):
    y = U[:, k]*S[k]
    r_all, r2_all = cv_rho(Xz, y)
    # 只用一个最佳通道
    jbest = int(np.argmax([abs(stats.spearmanr(Xz[:, j], y).statistic) for j in range(len(FEAT))]))
    r_1, _ = cv_rho(Xz[:, [jbest]], y)
    res62[f"PC{k+1}"] = dict(explained=float(ev[k]), cv_rho_all=r_all, cv_r2_all=r2_all,
                             best_channel=FEAT[jbest], cv_rho_best_single=r_1)
    print(f"  PC{k+1}（解释 {ev[k]*100:4.1f}% 方差）: 21通道 CV rho={r_all:+.3f} (R2={r2_all:+.3f}) "
          f"| 最佳单通道 {FEAT[jbest]} CV rho={r_1:+.3f}")

# ---- 全部 175 蛋白的零分布 ----
print("\n" + "="*100)
print("  S6.3 对全部可用 RPPA 蛋白做同样检验：CV rho 的分布（虚拟通道能预测多少蛋白？）")
print("="*100)
allr = []
for i in ok:
    y = M[i][cols]
    if np.std(y) == 0: continue
    r, _ = cv_rho(Xz, y, folds=5, reps=3)
    allr.append((prots[i], r))
allr.sort(key=lambda t: -t[1])
arr = np.array([r for _, r in allr])
print(f"  可评估蛋白 {len(allr)} 个；CV rho 中位数 {np.median(arr):+.3f}，最大 {arr.max():+.3f}，"
      f"|rho|>0.3 的 {int((np.abs(arr)>0.3).sum())} 个")
print("  前 10 名: " + ", ".join(f"{a.split('|')[-1]}({r:+.2f})" for a, r in allr[:10]))
print("  三个配对靶的位置: " + ", ".join(
    f"{t}={dict(allr).get(prots[prots.index(PAIR[t][1])], float('nan')):+.3f}" for t in PAIR))
pct = {t: float((np.abs(arr) < abs(dict(allr)[prots[prots.index(PAIR[t][1])]])).mean()*100) for t in PAIR}
print("  对应百分位: " + ", ".join(f"{t}={v:.0f}%" for t, v in pct.items()))

json.dump(dict(n_patients=n, n_features=len(FEAT), paired_targets=res61, axes=res62,
               all_proteins_cv_rho=[[a, float(r)] for a, r in allr],
               median_cv_rho=float(np.median(arr)), max_cv_rho=float(arr.max()),
               n_abs_gt_03=int((np.abs(arr)>0.3).sum()),
               paired_percentile=pct, pca_explained=[float(x) for x in ev[:5]]),
          open(os.path.join(OUT,"supp_06_predictive.json"),"w"), indent=1)
print("\n-> supp_06_predictive.json")
