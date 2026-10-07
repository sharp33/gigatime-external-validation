"""补充实验 S4：L3（TCGA-THCA RPPA）结论的稳健性 / 聚合口径消融（零 GPU，纯统计）

测试 L3 的阴性结论是否依赖于：
  1) 聚合口径：per-slide -> per-patient 的 mean / median / max
  2) 相关系数：Spearman / Pearson / 秩变换后 Pearson
  3) 半样本可复现性（split-half sign consistency）
  4) 全局"组织组成轴"假说：虚拟通道是否主要编码 RPPA 的第一主成分（组成轴），
     去掉该轴后配对通道与靶蛋白的相关是否进一步消失 —— 与 L4 残差化结论互相印证
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
PAIR = {"CD20": ("CD20", "CD20|CD20"),
        "Caspase3": ("Caspase3-D", "CASP3|Caspase-3_active"),
        "CD31": ("CD34", "PECAM1|CD31")}

lines = open(os.path.join(L3,"thca_rppa.txt"), encoding="utf-8", errors="replace").read().strip().split("\n")
hdr = lines[0].split("\t"); samples = hdr[1:]
prots = [l.split("\t")[0] for l in lines[1:]]
M = np.array([l.split("\t")[1:] for l in lines[1:]], dtype=float)
p2c = {s[:12]: i for i, s in enumerate(samples)}
print(f"RPPA 矩阵 {M.shape} (蛋白 x 样本), 蛋白示例: {prots[:6]} ...")

vecs = {}
for f in glob.glob(os.path.join(L3, "*_vec.npy")):
    vecs[os.path.basename(f).replace("_vec.npy","")] = np.load(f)
pat2vec = {}
for n, v in vecs.items(): pat2vec.setdefault(n[:12], []).append(v)
pat2vec = {p: np.array(vs) for p, vs in pat2vec.items()}
common = sorted(set(pat2vec) & set(p2c))
cols = [p2c[p] for p in common]
n = len(common)
print(f"匹配病人 n = {n}（slide {sum(len(v) for p,v in pat2vec.items() if p in p2c)}）")

AGG = {"mean": lambda a: a.mean(0), "median": lambda a: np.median(a,0), "max": lambda a: a.max(0)}
Vagg = {k: np.array([AGG[k](pat2vec[p]) for p in common]) for k in AGG}

# ---------- 1) 聚合 x 相关系数 消融 ----------
def rho_of(x, y, how):
    if how == "spearman": return float(stats.spearmanr(x, y).statistic)
    if how == "pearson":  return float(stats.pearsonr(x, y).statistic)
    if how == "rank_pearson": return float(stats.pearsonr(stats.rankdata(x), stats.rankdata(y)).statistic)
    raise ValueError(how)

rng = np.random.default_rng(0)
print("\n" + "="*100)
print("  1) L3 配对通道 rho：聚合口径 x 相关系数（n=%d 病人）" % n)
print("="*100)
print(f"{'靶蛋白':10s} {'虚拟通道':12s} " + " ".join(f"{a+'/'+h:>18s}" for a in AGG for h in ("spearman","pearson")))
abl = {}
for tname,(chn, ag) in PAIR.items():
    y = M[prots.index(ag)][cols]
    if np.std(y) == 0:
        print(f"{tname:10s} 该 RPPA 靶在队列中无变异，跳过"); continue
    row = []
    for a in AGG:
        for h in ("spearman","pearson"):
            r = rho_of(Vagg[a][:, CH.index(chn)], y, h); row.append(r)
            abl[f"{tname}|{a}|{h}"] = r
    print(f"{tname:10s} {chn:12s} " + " ".join(f"{v:+18.3f}" for v in row))

# ---------- 2) 半样本可复现性 ----------
print("\n" + "="*100)
print("  2) 半样本可复现性（随机对半 500 次；配对通道 rho 的符号一致率与 |rho| 分布）")
print("="*100)
half = {}
for tname,(chn, ag) in PAIR.items():
    y = M[prots.index(ag)][cols]; x = Vagg["mean"][:, CH.index(chn)]
    r1s, r2s = [], []
    for _ in range(500):
        idx = rng.permutation(n); h1, h2 = idx[:n//2], idx[n//2:]
        r1s.append(stats.spearmanr(x[h1], y[h1]).statistic)
        r2s.append(stats.spearmanr(x[h2], y[h2]).statistic)
    r1s = np.array(r1s); r2s = np.array(r2s)
    same = float(np.mean(np.sign(r1s) == np.sign(r2s)))
    rr = float(stats.pearsonr(r1s, r2s).statistic)
    half[tname] = dict(sign_consistency=same, split_half_r=rr,
                       mean_abs_rho=float(np.mean(np.abs(np.concatenate([r1s,r2s])))))
    print(f"  {tname:10s} 半样本符号一致率 {same*100:5.1f}%   两半 rho 相关 r={rr:+.3f}   平均|rho|={np.mean(np.abs(np.concatenate([r1s,r2s]))):.3f}")

# ---------- 3) 组成轴假说 ----------
print("\n" + "="*100)
print("  3) 组成轴假说：RPPA 主成分 vs 虚拟通道")
print("="*100)
Xp = M[:, cols].T                      # 病人 x 蛋白
mask = np.isfinite(Xp).all(0) & (Xp.std(0) > 0)
Xp = Xp[:, mask]; pn = [prots[i] for i in np.where(mask)[0]]
Z = (Xp - Xp.mean(0)) / Xp.std(0)
U, S, Vt = np.linalg.svd(Z, full_matrices=False)
ev = S**2 / (S**2).sum()
pc = U[:, :5] * S[:5]
print(f"  可用蛋白 {len(pn)}，前 5 个主成分解释方差: {np.round(ev[:5]*100,1)}%")
pc1_load = sorted(zip(pn, Vt[0]), key=lambda t: -abs(t[1]))
print("  PC1 载荷最高 8 个蛋白: " + ", ".join(f"{a}({b:+.2f})" for a,b in pc1_load[:8]))

Vmean = Vagg["mean"]
print(f"\n  {'虚拟通道':14s} {'rho(PC1)':>10s} {'rho(PC2)':>10s} {'rho(PC3)':>10s}  配对靶: 原始rho -> 控PC1后rho(偏相关)")
comp = {}
for c in CH:
    rs = [float(stats.spearmanr(Vmean[:, CH.index(c)], pc[:, k]).statistic) for k in range(3)]
    comp[c] = rs
order = sorted(CH, key=lambda c: -abs(comp[c][0]))
print("  与 PC1 相关性最强的 6 个虚拟通道: " + ", ".join(f"{c}({comp[c][0]:+.3f})" for c in order[:6]))

def partial_spearman(x, y, Zc):
    """控制 Zc 后的偏 Spearman（对秩做线性回归取残差）"""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    C = np.column_stack([np.ones(len(Zc)), Zc])
    ex = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
    ey = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    return float(stats.pearsonr(ex, ey).statistic)

print(f"\n  {'靶蛋白':10s} {'配对通道':12s} {'原始 rho':>9s} {'控 PC1 偏 rho':>13s} {'控 PC1-3 偏 rho':>15s}")
pa = {}
for tname,(chn, ag) in PAIR.items():
    y = M[prots.index(ag)][cols]; x = Vmean[:, CH.index(chn)]
    r0 = float(stats.spearmanr(x, y).statistic)
    r1 = partial_spearman(x, y, pc[:, [0]])
    r13 = partial_spearman(x, y, pc[:, :3])
    pa[tname] = dict(paired_channel=chn, raw_rho=r0, partial_pc1=r1, partial_pc13=r13)
    print(f"  {tname:10s} {chn:12s} {r0:+9.3f} {r1:+13.3f} {r13:+15.3f}")

# ---------- 4) 全通道扫描：最好通道能否通过 Bonferroni ----------
print("\n" + "="*100)
print("  4) 全通道最佳 |rho| 与多重检验（每个靶 23 次检验）")
print("="*100)
best = {}
for tname,(chn, ag) in PAIR.items():
    y = M[prots.index(ag)][cols]
    rr = np.array([stats.spearmanr(Vmean[:, CH.index(c)], y).statistic for c in CH])
    pv = np.array([stats.spearmanr(Vmean[:, CH.index(c)], y).pvalue for c in CH])
    j = int(np.argmax(np.abs(rr)))
    thr = stats.t.ppf(1-0.025/23, n-2)/np.sqrt(n-2+stats.t.ppf(1-0.025/23, n-2)**2)
    best[tname] = dict(best_channel=CH[j], best_rho=float(rr[j]), best_p=float(pv[j]),
                       bonferroni_threshold=float(thr), passes=bool(pv[j] < 0.05/23),
                       paired_channel=chn, paired_rho=float(rr[CH.index(chn)]),
                       paired_rank=int((np.abs(rr) > abs(rr[CH.index(chn)])).sum())+1,
                       n_tests=int(len(CH)))
    print(f"  {tname:10s} 最佳通道 {CH[j]:14s} rho={rr[j]:+.3f} p={pv[j]:.4f} "
          f"| Bonferroni 阈值 |rho|>{thr:.3f} -> {'通过' if pv[j]<0.05/23 else '不通过'} "
          f"| 配对通道 {chn} rho={rr[CH.index(chn)]:+.3f} 排名 {best[tname]['paired_rank']}/23")

# ---------- 5) 交叉反应谱：每个虚拟通道 vs 全部 175 个 RPPA 蛋白 ----------
print("\n" + "="*100)
print("  5) 交叉反应谱（虚拟通道 vs 全部 %d 个 RPPA 蛋白）：配对靶落在自身相关分布的什么位置" % len(pn))
print("="*100)
Rp = np.array([stats.spearmanr(Vmean[:, CH.index(c)], M[prots.index(a)][cols], nan_policy="omit").statistic
               for c in CH for a in pn if np.isfinite(M[prots.index(a)][cols]).all()])
nprot = int(np.isfinite(np.array([M[prots.index(a)][cols] for a in pn])).all(1).sum())
rho_mat = np.zeros((len(CH), nprot)); pn_ok = []
for j, a in enumerate([a for a in pn if np.isfinite(M[prots.index(a)][cols]).all()]):
    yv = M[prots.index(a)][cols]
    pn_ok.append(a)
    for i, c in enumerate(CH):
        rho_mat[i, j] = stats.spearmanr(Vmean[:, CH.index(c)], yv).statistic
print(f"  蛋白 {nprot} 个；|rho| 全局 95 分位 = {np.percentile(np.abs(rho_mat), 95):.3f}, 最大 = {np.abs(rho_mat).max():.3f}")
print(f"\n  {'虚拟通道':14s} {'中位|rho|':>10s} {'95分位|rho|':>12s} {'|rho|>0.3 蛋白数':>16s}   配对靶 |rho| / 百分位")
cross = {}
for c in CH:
    i = CH.index(c); a = np.abs(rho_mat[i])
    cross[c] = dict(median_abs_rho=float(np.median(a)), p95_abs_rho=float(np.percentile(a,95)),
                    n_gt_03=int((a>0.3).sum()), n_proteins=nprot)
mapped = {v[0]: (k, v[1]) for k, v in PAIR.items()}
for c in sorted(CH, key=lambda x: -cross[x]["n_gt_03"]):
    extra = ""
    if c in mapped:
        tname, ag = mapped[c]; yv = M[prots.index(ag)][cols]
        rr = np.array([stats.spearmanr(Vmean[:, CH.index(cc)], yv).statistic for cc in CH])
        pv = np.array([stats.spearmanr(Vmean[:, CH.index(cc)], yv).pvalue for cc in CH])
        aa = np.abs(rho_mat[CH.index(c)])
        pr = float((aa < abs(rr[CH.index(c)])).mean()*100)
        extra = f"   {tname}: |rho|={abs(rr[CH.index(c)]):.3f} (在自身 {nprot} 蛋白分布中位于 {pr:.0f} 百分位)"
        cross[c]["paired_target"] = tname; cross[c]["paired_abs_rho"] = float(abs(rr[CH.index(c)])); cross[c]["paired_percentile"] = pr
    print(f"  {c:14s} {cross[c]['median_abs_rho']:10.3f} {cross[c]['p95_abs_rho']:12.3f} {cross[c]['n_gt_03']:16d}{extra}")

# ---------- 6) L4 式残差化：回归掉"整体 RPPA 强度 + 整体虚拟激活" ----------
print("\n" + "="*100)
print("  6) L4 式残差化（回归掉 整体RPPA均值 + 整体虚拟激活）后的配对 rho 与排名")
print("="*100)
Rall = M[:, cols].T.astype(np.float64)
def zs(a): return (a-a.mean(0,keepdims=True))/(a.std(0,keepdims=True)+1e-9)
def residualize(Aa, C):
    C1 = np.column_stack([np.ones(len(C)), C]); B,*_ = np.linalg.lstsq(C1, Aa, rcond=None); return Aa - C1@B
shared = np.column_stack([np.nanmean(Rall,1), Vmean.mean(1)])
Vr = residualize(zs(Vmean), shared); Rr = residualize(zs(Rall), shared)
resid = {}
print(f"  {'靶蛋白':10s} {'配对通道':12s} {'残差化 rho':>11s} {'排名':>7s} {'其他最大':>9s} {'特异性':>8s}")
for tname,(chn, ag) in PAIR.items():
    i = prots.index(ag)
    if not np.isfinite(Rall[:, i]).all():
        print(f"  {tname:10s} 含缺失值，跳过"); continue
    yv = Rr[:, i]
    rr = np.array([stats.spearmanr(Vr[:, CH.index(c)], yv).statistic for c in CH])
    ci = CH.index(chn)
    others = max([(abs(rr[k]), k) for k in range(len(CH)) if k != ci])
    resid[tname] = dict(paired_channel=chn, residual_rho=float(rr[ci]),
                        rank=int((np.abs(rr) > abs(rr[ci])).sum())+1,
                        best_other_channel=CH[others[1]], best_other_abs=float(others[0]),
                        specificity=float(abs(rr[ci])-others[0]))
    print(f"  {tname:10s} {chn:12s} {rr[ci]:+11.3f} {resid[tname]['rank']:5d}/23 {others[0]:9.3f} {abs(rr[ci])-others[0]:+8.3f}")

json.dump(dict(cross_reactivity=cross, residualized=resid,
               global_95pct_abs_rho=float(np.percentile(np.abs(rho_mat),95)),
               n_patients=n, n_slides=int(sum(len(v) for p,v in pat2vec.items() if p in p2c)),
               aggregation_ablation=abl, split_half=half, composition_axis=comp,
               partial=pa, best_channel=best,
               pca_explained=[float(x) for x in ev[:5]], pc1_top_proteins=[[a,float(b)] for a,b in pc1_load[:8]],
               rppa_coverage=dict(n_proteins=int(len(pn)), proteins_used=pn)),
          open(os.path.join(OUT,"supp_04_l3_robustness.json"),"w"), indent=1)
print("\n-> supp_04_l3_robustness.json")
