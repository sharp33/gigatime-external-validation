"""L2 特异性对照：完整 通道×模块 相关矩阵（跨样本平均）+ 随机基因集零分布"""
import os, sys, gzip, glob, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np, pandas as pd
from scipy import stats
from scipy.sparse import coo_matrix

D = r"W:\虚拟细胞\data\GSE230424"; OUT = r"W:\虚拟细胞\analysis\l2"
gsms = {"P1":"GSM7221915","P2":"GSM7221916","P3":"GSM7221917","P4":"GSM7221918"}
CH = {n:i for i,n in enumerate(["DAPI","TRITC","Cy5","PD-1_1:200","CD14","CD4","T-bet","CD34",
      "CD68_1:100","CD16","CD11c","CD138","CD20","CD3_1:1000","CD8","PD-L1","CK_1:150",
      "Ki67_1:150","Tryptase","Actin-D","Caspase3-D","PHH3-B","Transgelin"])}
MODULES = {
 "Tcell":["CD3D","CD3E","CD2","TRAC","IL7R"], "Cytotoxic":["CD8A","GZMB","PRF1","NKG7","GZMA"],
 "Bcell":["MS4A1","CD79A","CD79B","CD19"], "Plasma":["MZB1","JCHAIN","SDC1","IGHG1"],
 "Macrophage":["CD68","CD163","CSF1R","MSR1","C1QA"], "Monocyte":["CD14","LYZ","S100A8","FCN1"],
 "Endothelial":["PECAM1","VWF","CD34","CLDN5"], "Epithelium":["EPCAM","KRT8","KRT18","KRT19","KRT7"],
 "Prolif":["MKI67","TOP2A","PCNA","CCNB1"], "Stroma":["FAP","COL1A1","COL1A2","ACTA2","POSTN"],
 "Mast":["TPSAB1","TPSB2","CPA3"],
}
PAIR = {"CD3_1:1000":"Tcell","CD8":"Cytotoxic","CD20":"Bcell","CD138":"Plasma","CD68_1:100":"Macrophage",
        "CD14":"Monocyte","CD34":"Endothelial","CK_1:150":"Epithelium","Ki67_1:150":"Prolif",
        "Tryptase":"Mast","Transgelin":"Stroma"}
CHS = ["DAPI"] + list(PAIR.keys())
mnames = list(MODULES.keys())

R = np.full((len(CHS), len(mnames), 4), np.nan)
NULL = np.full((len(CHS), 4), np.nan)     # 随机基因集零分布（配对通道用）
rng = np.random.default_rng(7)

for si,(s,g) in enumerate(gsms.items()):
    V = np.load(os.path.join(OUT, f"{s}_spotX.npy"))
    keep_bc = open(os.path.join(OUT, f"{s}_barcodes.txt")).read().split("\n")
    lines = gzip.open(os.path.join(D, f"{g}_{s}_matrix.mtx.gz")).read().split(b"\n")
    ng, ns, _ = map(int, lines[2].split())
    arr = np.loadtxt(lines[3:], dtype=np.float32)
    M = coo_matrix((arr[:,2],(arr[:,0].astype(int)-1, arr[:,1].astype(int)-1)), shape=(ng,ns)).toarray().astype(np.float32)
    feat = [l.split("\t")[1].upper() for l in gzip.open(os.path.join(D, f"{g}_{s}_features.tsv.gz")).read().decode().split("\n") if l.strip() and len(l.split("\t"))>1]
    bcs = [l.strip() for l in gzip.open(os.path.join(D, f"{g}_{s}_barcodes.tsv.gz")).read().decode().split("\n") if l.strip()]
    bidx = {b:i for i,b in enumerate(bcs)}
    sel = [bidx[b] for b in keep_bc if b in bidx]
    Ms = M[:, sel]; gidx = {gg:i for i,gg in enumerate(feat)}
    tot = Ms.sum(0, keepdims=True); tot[tot==0]=1
    X = np.log1p(Ms/tot*1e4)
    scores = {}
    for mn, gl in MODULES.items():
        ii = [gidx[gg] for gg in gl if gg in gidx]
        z = X[ii,:]; z = (z - z.mean(1,keepdims=True))/(z.std(1,keepdims=True)+1e-9)
        scores[mn] = z.mean(0)
    for ci, ch in enumerate(CHS):
        v = V[:, CH[ch]]
        for mi, mn in enumerate(mnames):
            R[ci, mi, si] = stats.spearmanr(v, scores[mn]).statistic
        # 零分布：同规模随机基因集（只跑一次）
        n_sz = len(MODULES[PAIR.get(ch, "Tcell")]) if ch in PAIR else 5
        nulls = []
        for _ in range(200):
            ri = rng.choice(len(feat), size=n_sz, replace=False)
            z = X[ri,:]; z = (z - z.mean(1,keepdims=True))/(z.std(1,keepdims=True)+1e-9)
            nulls.append(abs(stats.spearmanr(v, z.mean(0)).statistic))
        NULL[ci, si] = np.mean(nulls)
    print(f"  {s} done")

Rm = np.nanmean(R, axis=2)
print("\n" + "="*118)
print("  L2 特异性矩阵：跨 4 样本平均 |Spearman rho|  (行=虚拟通道, 列=ST 模块)")
print("="*118)
print(f"{'':14s}" + "".join(f"{m[:10]:>11s}" for m in mnames))
for ci, ch in enumerate(CHS):
    row = f"{ch[:13]:14s}"
    for mi, mn in enumerate(mnames):
        mark = "*" if PAIR.get(ch)==mn else " "
        row += f"{Rm[ci,mi]:>10.3f}{mark}"
    print(row)
print("\n  * = 预期配对；|rho| 已跨样本平均（若某样本符号相反会抵消）")

print("\n=== 特异性得分：配对模块 |rho| − 其他模块最大 |rho| ===")
print(f"{'通道':14s} {'配对':11s} {'配对|rho|':>10s} {'其他最大':>10s} {'特异性':>9s} {'随机零':>9s}")
spec_rows=[]
for ci, ch in enumerate(CHS):
    if ch not in PAIR: continue
    mi = mnames.index(PAIR[ch])
    pv = abs(Rm[ci,mi])
    others = [abs(Rm[ci,mj]) for mj in range(len(mnames)) if mj!=mi]
    om = max(others)
    print(f"{ch:14s} {PAIR[ch]:11s} {pv:10.3f} {om:10.3f} {pv-om:+9.3f} {np.nanmean(NULL[ci]):9.3f}")
    spec_rows.append(dict(channel=ch, module=PAIR[ch], paired=pv, best_other=om, specificity=pv-om,
                          null=float(np.nanmean(NULL[ci]))))
json.dump(dict(matrix=Rm.tolist(), channels=CHS, modules=mnames, specificity=spec_rows),
          open(os.path.join(OUT,"l2_specificity.json"),"w"), indent=1)
print("\n-> l2_specificity.json")
