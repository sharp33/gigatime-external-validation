"""ORION 特异性矩阵 v2（修正索引 bug：CH 是显示顺序，索引用 23 列的字典）"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
from scipy import stats
sys.path.insert(0, r"W:\虚拟细胞\repo")
from gigatime_flash import CHANNEL_NAMES

OUT = r"W:\虚拟细胞\analysis\orion"
pat = sys.argv[1] if len(sys.argv)>1 else "CRC01"
z = np.load(os.path.join(OUT, f"{pat}_cells.npz"))
V = z["V"]
MARKERS = [k[2:] for k in z.files if k.startswith("r_")]
IDX = {c:i for i,c in enumerate(CHANNEL_NAMES)}          # 关键修正
CH = [c for c in CHANNEL_NAMES if c not in ("TRITC","Cy5")]
PAIR = {"Hoechst":"DAPI","CD31":"CD34","CD68":"CD68_1:100","CD4":"CD4","CD8a":"CD8",
        "CD20":"CD20","PD-L1":"PD-L1","CD3e":"CD3_1:1000","PD-1":"PD-1_1:200",
        "Ki67":"Ki67_1:150","Pan-CK":"CK_1:150","E-cadherin":"CK_1:150","SMA":"Actin-D"}
ALL = [m for m in MARKERS]
print(f"{pat}: {V.shape[0]} cells | markers={len(ALL)} | V cols={V.shape[1]}")

R = np.zeros((len(ALL), len(CH)))
for i, mk in enumerate(ALL):
    y = z[f"r_{mk}"]
    for j, c in enumerate(CH):
        R[i, j] = stats.spearmanr(V[:, IDX[c]], y).statistic

print("\n" + "="*112)
print("  特异性矩阵 v2（行=ORION 实测标记, 列=GigaTIME 虚拟通道）  * = 预期配对")
print("="*112)
print(f"{'ORION':12s}" + "".join(f"{c[:10]:>11s}" for c in CH))
for i, mk in enumerate(ALL):
    row = f"{mk:12s}"
    for j, c in enumerate(CH):
        row += f"{R[i,j]:>10.3f}{'*' if PAIR.get(mk)==c else ' '}"
    print(row)

print("\n" + "="*86)
print("  特异性 = 配对 |rho| − 其他最大 |rho|")
print("="*86)
print(f"{'ORION 标记':12s} {'配对通道':14s} {'配对|rho|':>10s} {'其他最大':>10s} {'特异性':>9s}  判定")
res=[]; n_ok=0
for mk in ALL:
    ch = PAIR.get(mk); i = ALL.index(mk)
    if ch is None:
        jm = int(np.argmax(np.abs(R[i])))
        print(f"{mk:12s} {'(无对应)':14s} {'-':>10s} {abs(R[i,jm]):10.3f} {'-':>9s}  最佳={CH[jm]}")
        res.append(dict(marker=mk,paired=None,best_other=float(abs(R[i,jm])),best_channel=CH[jm]))
        continue
    j = CH.index(ch); pv = abs(R[i,j])
    om = max(abs(R[i,k]) for k in range(len(CH)) if k!=j)
    ok = pv > om
    n_ok += ok
    print(f"{mk:12s} {ch:14s} {pv:10.3f} {om:10.3f} {pv-om:+9.3f}  {'OK 特异性' if ok else 'FAIL 不特异'}")
    res.append(dict(marker=mk,paired=ch,paired_rho=float(R[i,j]),best_other=float(om),
                    specificity=float(pv-om),specific=bool(ok)))
print(f"\n通过特异性: {n_ok}/{len(PAIR)}")
json.dump(dict(pat=pat,n_cells=int(V.shape[0]),matrix=R.tolist(),channels=CH,markers=ALL,
               detail=res, n_specific=int(n_ok), n_paired=len(PAIR)),
          open(os.path.join(OUT,f"{pat}_specificity.json"),"w"), indent=1)
print("-> "+pat+"_specificity.json")
