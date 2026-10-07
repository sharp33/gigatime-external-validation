"""ORION 残差化特异性分析 v2"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
from scipy import stats
sys.path.insert(0, r"W:\虚拟细胞\repo")
from gigatime_flash import CHANNEL_NAMES

OUT = r"W:\虚拟细胞\analysis\orion"
pat = sys.argv[1] if len(sys.argv)>1 else "CRC01"
z = np.load(os.path.join(OUT, f"{pat}_cells.npz"))
V = z["V"].astype(np.float64)
IDX = {c: i for i, c in enumerate(CHANNEL_NAMES)}
CH = [c for c in CHANNEL_NAMES if c not in ("TRITC", "Cy5")]
MARKERS = [k[2:] for k in z.files if k.startswith("r_")]
X = np.stack([z[f"r_{m}"] for m in MARKERS], 1).astype(np.float64)
print(f"{pat}: V={V.shape} X={X.shape}")

def zs(a): return (a - a.mean(0, keepdims=True)) / (a.std(0, keepdims=True) + 1e-9)
def residualize(A, C):
    C1 = np.column_stack([np.ones(len(C)), C])
    B, *_ = np.linalg.lstsq(C1, A, rcond=None)
    return A - C1 @ B

shared = np.column_stack([X.mean(1), V.mean(1)])
Xr = residualize(zs(X), shared)
Vr = residualize(zs(V), shared)

R = np.zeros((len(MARKERS), len(CH)))
for i in range(len(MARKERS)):
    for j, c in enumerate(CH):
        R[i, j] = stats.spearmanr(Vr[:, IDX[c]], Xr[:, i]).statistic

PAIR = {"Hoechst":"DAPI","CD31":"CD34","CD68":"CD68_1:100","CD4":"CD4","CD8a":"CD8",
        "CD20":"CD20","PD-L1":"PD-L1","CD3e":"CD3_1:1000","PD-1":"PD-1_1:200",
        "Ki67":"Ki67_1:150","Pan-CK":"CK_1:150","E-cadherin":"CK_1:150","SMA":"Actin-D"}

print("\n" + "="*94)
print("  残差化特异性（已回归掉'整体荧光量 + 整体虚拟激活'共享轴）")
print("="*94)
print(f"{'ORION 标记':12s} {'配对通道':14s} {'配对 rho':>9s} {'其他最大':>9s} {'特异性':>9s}  判定")
res = []; n_ok = 0
for mk in MARKERS:
    i = MARKERS.index(mk); ch = PAIR.get(mk)
    if ch is None:
        jm = int(np.argmax(np.abs(R[i])))
        print(f"{mk:12s} {'(负对照)':14s} {'-':>9s} {R[i,jm]:+9.3f} {'-':>9s}  最佳={CH[jm]}")
        res.append(dict(marker=mk, paired=None, best_channel=CH[jm], best_rho=float(R[i,jm])))
        continue
    j = CH.index(ch)
    pv = R[i, j]
    others = [(abs(R[i, k]), k) for k in range(len(CH)) if k != j]
    om, om_j = max(others)
    ok = abs(pv) > om
    n_ok += int(ok)
    print(f"{mk:12s} {ch:14s} {pv:+9.3f} {om:9.3f} {abs(pv)-om:+9.3f}  {'OK 特异性' if ok else 'FAIL 不特异'}")
    res.append(dict(marker=mk, paired=ch, paired_rho=float(pv), best_other=float(om),
                    best_other_channel=CH[om_j], specificity=float(abs(pv)-om), specific=bool(ok)))
print(f"\n通过特异性: {n_ok}/{len(PAIR)}")
json.dump(dict(pat=pat, n_cells=int(V.shape[0]), residual_matrix=R.tolist(), channels=CH,
               markers=MARKERS, detail=res, n_specific=int(n_ok), n_paired=len(PAIR)),
          open(os.path.join(OUT, f"{pat}_residual_specificity.json"), "w"), indent=1)
print("-> " + pat + "_residual_specificity.json")
