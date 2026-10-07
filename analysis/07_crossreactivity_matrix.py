"""补算 L3 的完整交叉反应矩阵 (23 虚拟通道 x 175 RPPA 蛋白)，供 Fig 8 使用"""
import os, sys, glob, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
from scipy import stats

L3 = r"W:\虚拟细胞\analysis\l3"; OUT = r"W:\虚拟细胞\analysis"
CH = ["DAPI","TRITC","Cy5","PD-1_1:200","CD14","CD4","T-bet","CD34","CD68_1:100","CD16","CD11c",
      "CD138","CD20","CD3_1:1000","CD8","PD-L1","CK_1:150","Ki67_1:150","Tryptase","Actin-D",
      "Caspase3-D","PHH3-B","Transgelin"]
lines = open(os.path.join(L3,"thca_rppa.txt"), encoding="utf-8", errors="replace").read().strip().split("\n")
prots = [l.split("\t")[0] for l in lines[1:]]
M = np.array([l.split("\t")[1:] for l in lines[1:]], dtype=float)
p2c = {s2[:12]: i for i, s2 in enumerate(lines[0].split("\t")[1:])}
vecs = {}
for f in glob.glob(os.path.join(L3, "*_vec.npy")):
    vecs[os.path.basename(f).replace("_vec.npy","")] = np.load(f)
pat2vec = {}
for n2, v in vecs.items(): pat2vec.setdefault(n2[:12], []).append(v)
pat2vec = {p: np.mean(vs, 0) for p, vs in pat2vec.items()}
common = sorted(set(pat2vec) & set(p2c)); cols = [p2c[p] for p in common]
V = np.array([pat2vec[p] for p in common]); n = len(common)
keep = [i for i in range(len(prots)) if np.isfinite(M[i][cols]).all()]
R = np.zeros((23, len(keep)))
for j, i in enumerate(keep):
    y = M[i][cols]
    for ci in range(23):
        R[ci, j] = stats.spearmanr(V[:, ci], y).statistic
print(f"n={n} 病人, 矩阵 {R.shape}")
json.dump(dict(matrix=R.tolist(), channels=CH, proteins=[prots[i] for i in keep], n_patients=n,
               n_proteins=len(keep)), open(os.path.join(OUT, "supp_04b_crossreact_matrix.json"), "w"))
print("-> supp_04b_crossreact_matrix.json")
