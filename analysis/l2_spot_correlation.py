"""L2 步骤2：ST 模块评分 + 与虚拟通道的 spot 级相关

用法: python l2_correlate.py P1
"""
import os, sys, glob, gzip, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
from scipy import stats
from scipy.sparse import coo_matrix

D = r"W:\虚拟细胞\data\GSE230424"; OUT = r"W:\虚拟细胞\analysis\l2"
sample = sys.argv[1] if len(sys.argv)>1 else "P1"
gsm = {"P1":"GSM7221915","P2":"GSM7221916","P3":"GSM7221917","P4":"GSM7221918"}[sample]

def gz(p):
    with gzip.open(p,"rb") as f: return f.read()

# 模型通道索引
CH = {n:i for i,n in enumerate(["DAPI","TRITC","Cy5","PD-1_1:200","CD14","CD4","T-bet","CD34",
      "CD68_1:100","CD16","CD11c","CD138","CD20","CD3_1:1000","CD8","PD-L1","CK_1:150",
      "Ki67_1:150","Tryptase","Actin-D","Caspase3-D","PHH3-B","Transgelin"])}

MODULES = {
 "Tcell":       ["CD3D","CD3E","CD2","TRAC","IL7R"],
 "Cytotoxic":   ["CD8A","GZMB","PRF1","NKG7","GZMA"],
 "Bcell":       ["MS4A1","CD79A","CD79B","CD19"],
 "Plasma":      ["MZB1","JCHAIN","SDC1","IGHG1"],
 "Macrophage":  ["CD68","CD163","CSF1R","MSR1","C1QA"],
 "Monocyte":    ["CD14","LYZ","S100A8","FCN1"],
 "Endothelial": ["PECAM1","VWF","CD34","CLDN5"],
 "Epithelium":  ["EPCAM","KRT8","KRT18","KRT19","KRT7"],
 "Prolif":      ["MKI67","TOP2A","PCNA","CCNB1"],
 "Stroma":      ["FAP","COL1A1","COL1A2","ACTA2","POSTN"],
 "Mast":        ["TPSAB1","TPSB2","CPA3"],
}
CH_PAIR = {"DAPI":"", "CD3_1:1000":"Tcell", "CD8":"Cytotoxic", "CD20":"Bcell", "CD138":"Plasma",
           "CD68_1:100":"Macrophage", "CD14":"Monocyte", "CD34":"Endothelial", "CK_1:150":"Epithelium",
           "Ki67_1:150":"Prolif", "Transgelin":"Stroma", "Tryptase":"Mast"}

# ---- ST 矩阵 ----
print("loading ST matrix...")
lines = gz(os.path.join(D, f"{gsm}_{sample}_matrix.mtx.gz")).split(b"\n")
ng, ns, nnz = map(int, lines[2].split())
arr = np.loadtxt(lines[3:], dtype=np.float32)
M = coo_matrix((arr[:,2], (arr[:,0].astype(int)-1, arr[:,1].astype(int)-1)), shape=(ng, ns)).toarray().astype(np.float32)
feat = [l.split("\t")[1].upper() for l in gz(os.path.join(D, f"{gsm}_{sample}_features.tsv.gz")).decode().split("\n") if l.strip() and len(l.split("\t"))>1]
bcs = [l.strip() for l in gz(os.path.join(D, f"{gsm}_{sample}_barcodes.tsv.gz")).decode().split("\n") if l.strip()]
print(f"  matrix: {M.shape}  genes={len(feat)} barcodes={len(bcs)}")

# ---- 对齐到我们保留的 spot ----
keep_bc = open(os.path.join(OUT, f"{sample}_barcodes.txt")).read().split("\n")
bidx = {b:i for i,b in enumerate(bcs)}
sel = [bidx[b] for b in keep_bc if b in bidx]
print(f"  kept spots matched: {len(sel)} / {len(keep_bc)}")
Ms = M[:, sel]
gidx = {g:i for i,g in enumerate(feat)}
tot = Ms.sum(0, keepdims=True); tot[tot==0] = 1
X = np.log1p(Ms / tot * 1e4)

# ---- 模块评分（z-scored mean）----
scores = {}
for nm, gl in MODULES.items():
    ii = [gidx[g] for g in gl if g in gidx]
    if not ii: 
        print(f"  module {nm}: no genes found"); continue
    z = X[ii, :]
    z = (z - z.mean(1, keepdims=True)) / (z.std(1, keepdims=True) + 1e-9)
    scores[nm] = z.mean(0)
    print(f"  module {nm:12s} genes={len(ii)}/{len(gl)}")

# ---- 虚拟通道 ----
V = np.load(os.path.join(OUT, f"{sample}_spotX.npy"))
print(f"  virtual channels: {V.shape}")

rows=[]
print(f"\n=== {sample}: 虚拟通道 vs ST 模块 的相关（Spearman, n={len(sel)} spots）===")
print(f"{'虚拟通道':14s} {'配对模块':12s} {'rho':>8s} {'p':>10s}   |  最佳模块   rho")
for chname, mod in CH_PAIR.items():
    if chname not in CH: continue
    v = V[:, CH[chname]]
    best=("",0)
    for mn, sc in scores.items():
        rho,_ = stats.spearmanr(v, sc)
        if abs(rho) > abs(best[1]): best=(mn, rho)
    line = f"{chname:14s} "
    if mod and mod in scores:
        rho,p = stats.spearmanr(v, scores[mod])
        line += f"{mod:12s} {rho:8.3f} {p:10.2e}"
        rows.append(dict(channel=chname, module=mod, rho=float(rho), p=float(p)))
    else:
        line += f"{'-':12s} {'-':>8s} {'-':>10s}"
    line += f"   |  {best[0]:12s} {best[1]:+.3f}"
    print(line)
json.dump(dict(sample=sample, n_spots=len(sel), pairs=rows,
               modules={k: list(v) for k,v in MODULES.items()}),
          open(os.path.join(OUT, f"{sample}_corr.json"),"w"), indent=1)
print("-> saved", sample)
