"""L3：虚拟通道密度 vs RPPA 真实蛋白"""
import os, sys, glob, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from scipy import stats

L3 = r"W:\虚拟细胞\analysis\l3"
CH = ["DAPI","TRITC","Cy5","PD-1_1:200","CD14","CD4","T-bet","CD34","CD68_1:100","CD16","CD11c",
      "CD138","CD20","CD3_1:1000","CD8","PD-L1","CK_1:150","Ki67_1:150","Tryptase","Actin-D",
      "Caspase3-D","PHH3-B","Transgelin"]

# 载入每张切片的向量
vecs = {}
for f in glob.glob(os.path.join(L3, "*_vec.npy")):
    name = os.path.basename(f).replace("_vec.npy","")
    vecs[name] = np.load(f)
print("slide vectors:", len(vecs))
if len(vecs) == 0:
    print("no vectors yet"); sys.exit(0)

# patient = 文件名前 12 字符
pat2vec = {}
for name, v in vecs.items():
    p = name[:12]
    pat2vec.setdefault(p, []).append(v)
pat2vec = {p: np.mean(vs, axis=0) for p, vs in pat2vec.items()}
print("unique patients with slides:", len(pat2vec))

# RPPA
lines = open(os.path.join(L3, "thca_rppa.txt"), encoding="utf-8", errors="replace").read().strip().split("\n")
hdr = lines[0].split("\t"); samples = hdr[1:]
prots = [l.split("\t")[0] for l in lines[1:]]
M = np.array([l.split("\t")[1:] for l in lines[1:]], dtype=float)   # antigen x sample
pat2col = {s[:12]: i for i, s in enumerate(samples)}
print("RPPA samples:", len(samples))

common = sorted(set(pat2vec) & set(pat2col))
print("共同病人:", len(common))
if len(common) < 8:
    print("样本太少，等更多切片下完再跑"); sys.exit(0)

V = np.array([pat2vec[p] for p in common])            # n x 23
cols = [pat2col[p] for p in common]

TARGETS = {"CD20": "CD20|CD20", "Caspase3": "CASP3|Caspase-3_active", "CD31": "PECAM1|CD31"}
print("\n" + "="*88)
print("  L3：虚拟通道密度 vs RPPA 真实蛋白（病人级 Spearman）")
print("="*88)
res = {}
for tname, ag in TARGETS.items():
    if ag not in prots:
        print(f"{tname}: 抗原 {ag} 不在 RPPA 中"); continue
    y = M[prots.index(ag)][cols]
    print(f"\n--- RPPA {tname} ({ag}) ---  n={len(common)}")
    rows = []
    for ci, ch in enumerate(CH):
        rho, p = stats.spearmanr(V[:, ci], y)
        rows.append((ch, rho, p))
    rows.sort(key=lambda x: -abs(x[1]))
    for ch, rho, p in rows[:6]:
        star = "  <== 配对" if (tname=="CD20" and ch=="CD20") or (tname=="Caspase3" and ch=="Caspase3-D") or (tname=="CD31" and ch=="CD34") else ""
        print(f"   {ch:14s} rho={rho:+.3f}  p={p:.4f}{star}")
    res[tname] = {ch: dict(rho=float(r), p=float(p)) for ch, r, p in rows}
json.dump(dict(n=len(common), patients=common, results=res),
          open(os.path.join(L3, "l3_rppa_corr.json"), "w"), indent=1)
print("\n-> l3_rppa_corr.json")
