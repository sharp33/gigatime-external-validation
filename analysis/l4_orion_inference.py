"""ORION-CRC 细胞级验证：虚拟通道 vs 单细胞实测标记强度

设计：
  - 抽样 tile（而非逐细胞），每个 tile 内的细胞批量查表 -> 高效
  - 尺度：ORION mpp=0.325 -> 裁 181 原生px 上采样到 256 (eff≈0.2302)
  - 参照：每个细胞的 19 个标记强度（CSV）
用法: python orion_eval.py CRC01 [n_tiles]
"""
import os, sys, glob, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"W:\虚拟细胞\repo")
import numpy as np, pandas as pd, torch, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from gigatime_flash import load_model, CHANNEL_NAMES, mean as IMNET_MEAN, std as IMNET_STD

D = r"W:\虚拟细胞\data\ORION_CRC"; OUT = r"W:\虚拟细胞\analysis\orion"
os.makedirs(OUT, exist_ok=True)
pat = sys.argv[1] if len(sys.argv)>1 else "CRC01"
ntile = int(sys.argv[2]) if len(sys.argv)>2 else 800

# ORION 标记 -> GigaTIME 模型通道名（None = 无对应，作负对照）
PAIR = {"Hoechst":"DAPI", "CD31":"CD34", "CD68":"CD68_1:100", "CD4":"CD4", "CD8a":"CD8",
        "CD20":"CD20", "PD-L1":"PD-L1", "CD3e":"CD3_1:1000", "PD-1":"PD-1_1:200",
        "Ki67":"Ki67_1:150", "Pan-CK":"CK_1:150", "E-cadherin":"CK_1:150", "SMA":"Actin-D"}
NEG = ["AF1","Argo550","CD45","FOXP3","CD45RO","CD163"]   # 无对应虚拟通道 -> 负对照

def to_lab(x): return cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
def from_lab(l):
    l=l.copy(); l[:,:,0]=np.clip(l[:,:,0],0,100); l[:,:,1]=np.clip(l[:,:,1],-127,127); l[:,:,2]=np.clip(l[:,:,2],-127,127)
    return np.clip(cv2.cvtColor(l.astype(np.float32), cv2.COLOR_LAB2RGB),0,1)

# --- 读 CSV ---
csvp = os.path.join(D, f"{pat}_cells.csv")
print(f"loading {csvp} ...")
df = pd.read_csv(csvp)
print(f"  cells: {len(df)}  cols: {len(df.columns)}")
print(f"  X range [{df.X_centroid.min():.0f},{df.X_centroid.max():.0f}]  Y range [{df.Y_centroid.min():.0f},{df.Y_centroid.max():.0f}]")

# --- H&E ---
hep = os.path.join(D, f"{pat}_HE.ome.tif")
import tiffslide
ts = tiffslide.TiffSlide(hep)
W, H = ts.dimensions
print(f"  HE: {W} x {H}")
# 尺度
mp = None
for k in ("tiffslide.mpp-x","openslide.mpp-x"):
    if k in ts.properties:
        try: mp = float(ts.properties[k]); break
        except Exception: pass
if mp is None:
    mk = tiffslide.TiffSlide(os.path.join(D, f"{pat}_mask.ome.tif"))
    mp = float(mk.properties.get("tiffslide.mpp-x", 0.325))
print(f"  mpp = {mp:.4f}")
TARGET = 0.2302
UP = 256; N_IN = int(round(UP*TARGET/mp)); scale = UP/N_IN; eff = mp/scale
print(f"  裁 {N_IN} 原生px -> 上采样 {UP}px | eff={eff:.4f} (偏差 {abs(eff/TARGET-1)*100:.1f}%)")

# --- 染色目标统计 ---
off=sorted(glob.glob(r"W:\虚拟细胞\data\sample_test_data\**\*_he.png",recursive=True))
A=np.concatenate([to_lab(np.asarray(Image.open(p).convert("RGB").resize((256,256),Image.BILINEAR),dtype=np.float32)/255.).reshape(-1,3) for p in off[:40]],0)
TM,TS=A.mean(0),A.std(0)

# --- 源统计（全图抽样）---
rng = np.random.default_rng(1); srcs=[]
for _ in range(40):
    x0=int(rng.integers(0,max(1,W-512))); y0=int(rng.integers(0,max(1,H-512)))
    t=np.asarray(ts.read_region((x0,y0),0,(512,512)).convert("RGB"),dtype=np.float32)/255.
    srcs.append(to_lab(t).reshape(-1,3))
S=np.concatenate(srcs,0); SM,SS=S.mean(0),S.std(0)+1e-6
print(f"  源 LAB {SM.round(1)} / 目标 {TM.round(1)}")

model, device, _ = load_model(weights_path=r"W:\虚拟细胞\models\gigatime-flash\model.pth")

# --- 抽样 tile（只取有细胞的区域）---
cx = df.X_centroid.to_numpy(); cy = df.Y_centroid.to_numpy()
rng2 = np.random.default_rng(7)
# 用细胞位置做加权：随机取细胞，以其为中心取 tile
picks = rng2.choice(len(df), size=min(ntile*3, len(df)), replace=False)
centers = {}
for i in picks:
    tx = int(cx[i]//N_IN)*N_IN; ty = int(cy[i]//N_IN)*N_IN
    centers[(tx,ty)] = True
centers = list(centers.keys())[:ntile]
print(f"  抽样 tile: {len(centers)}")

cells_v = []; cells_r = {k: [] for k in list(PAIR)+NEG}
CH = {n:i for i,n in enumerate(CHANNEL_NAMES)}
yy, xx = np.mgrid[0:UP,0:UP]
t0=time.time(); used=0
for (tx, ty) in centers:
    x1, y1 = tx+N_IN, ty+N_IN
    if x1 > W or y1 > H: continue
    a = np.asarray(ts.read_region((tx,ty),0,(N_IN,N_IN)).convert("RGB"), dtype=np.float32)/255.
    lab = to_lab(a); sm, ss = lab.reshape(-1,3).mean(0), lab.reshape(-1,3).std(0)+1e-6
    a = from_lab((lab-sm)/ss*TS+TM)
    a = np.asarray(Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8)).resize((UP,UP),Image.BILINEAR),dtype=np.float32)/255.
    ten = torch.from_numpy(((a-IMNET_MEAN)/IMNET_STD).transpose(2,0,1)).unsqueeze(0)
    with torch.no_grad():
        pr = torch.sigmoid(model(ten.to(device))).squeeze(0).cpu().numpy()
    pr = np.moveaxis(pr,0,-1) if pr.shape[0]==23 else pr     # (256,256,23)
    m = (cx>=tx)&(cx<x1)&(cy>=ty)&(cy<y1)
    idxs = np.where(m)[0]
    for i in idxs:
        u = int((cx[i]-tx)*scale); v = int((cy[i]-ty)*scale)
        if u<0 or v<0 or u>=UP or v>=UP: continue
        cells_v.append(pr[v,u,:])
        for k in cells_r: cells_r[k].append(df[k].to_numpy()[i])
    used += 1
    if used % 200 == 0: print(f"    tiles={used} cells={len(cells_v)} {time.time()-t0:.0f}s", flush=True)

V = np.array(cells_v); print(f"\n完成: {used} tiles, {len(V)} 细胞, {time.time()-t0:.0f}s")
if len(V) < 100:
    print("细胞太少"); sys.exit(1)
R = {k: np.array(v) for k,v in cells_r.items()}
np.savez(os.path.join(OUT, f"{pat}_cells.npz"), V=V, **{f"r_{k}": R[k] for k in R})

from scipy import stats
print("\n" + "="*86)
print(f"  ORION {pat}: 虚拟通道 vs 单细胞实测标记（细胞级 Spearman, n={len(V)}）")
print("="*86)
print(f"{'ORION 标记':12s} {'GigaTIME 通道':14s} {'rho':>8s} {'p':>10s}   {'随机零':>8s}")
res = {}
rngn = np.random.default_rng(3)
for k, chname in PAIR.items():
    y = R[k]
    if np.std(y) == 0: continue
    col = V[:, CH[chname]]
    rho, p = stats.spearmanr(col, y)
    null = np.mean([abs(stats.spearmanr(col, rngn.permutation(y)).statistic) for _ in range(200)])
    print(f"{k:12s} {chname:14s} {rho:+8.3f} {p:10.2e}   {null:8.4f}")
    res[k] = dict(channel=chname, rho=float(rho), p=float(p), null=float(null))
print("\n--- 负对照（ORION 有、GigaTIME 无对应通道）---")
neg = {}
ALLCH = [c for c in CHANNEL_NAMES if c not in ("TRITC","Cy5")]
for k in NEG:
    y = R[k]
    if np.std(y)==0: continue
    best = max(((stats.spearmanr(V[:,CH[c]], y).statistic, c) for c in ALLCH), key=lambda t: abs(t[0]))
    print(f"{k:12s} {'(无对应)':14s} 最佳匹配的是 {best[1]:14s} rho={best[0]:+.3f}")
    neg[k] = dict(best_channel=best[1], rho=float(best[0]))
json.dump(dict(pat=pat, n_cells=int(len(V)), pairs=res, negatives=neg,
               mpp=mp, eff_umpp=eff, n_tiles=used), open(os.path.join(OUT,f"{pat}_result.json"),"w"), indent=1)
print("\n-> orion/"+pat+"_result.json")
