"""L2 步骤1 v3：GSE230424 -> spot 级虚拟通道

关键修正（v3）：
  v1 bug: 按 tile 做 Reinhard -> 抹掉空间变异                -> 改全局统计
  v2 bug: 重叠 tile 用"均值的均值" -> 加权错误               -> 改不重叠 + sum/npx
  v3 bug: 原生 256px 铺砖 -> 模型视野 118µm，训练域是 59µm  -> 裁 128 原生px 上采样到 256
用法: python l2_run_gigatime_v3.py P1
"""
import os, sys, glob, json, gzip, time, shutil
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"W:\虚拟细胞\repo")
import numpy as np, pandas as pd, torch, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from gigatime_flash import load_model, CHANNEL_NAMES, mean as IMNET_MEAN, std as IMNET_STD

D = r"W:\虚拟细胞\data\GSE230424"; OUT = r"W:\虚拟细胞\analysis\l2"
os.makedirs(OUT, exist_ok=True)
sample = sys.argv[1] if len(sys.argv) > 1 else "P1"
gsm = {"P1":"GSM7221915","P2":"GSM7221916","P3":"GSM7221917","P4":"GSM7221918"}[sample]

def to_lab(x): return cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
def from_lab(l):
    l=l.copy(); l[:,:,0]=np.clip(l[:,:,0],0,100); l[:,:,1]=np.clip(l[:,:,1],-127,127); l[:,:,2]=np.clip(l[:,:,2],-127,127)
    return np.clip(cv2.cvtColor(l.astype(np.float32), cv2.COLOR_LAB2RGB),0,1)

pos = pd.read_csv(os.path.join(D, f"{gsm}_{sample}_tissue_positions_list.csv.gz"), header=None,
                  names=["barcode","in_tissue","array_row","array_col","px_row","px_col"])
barcode = pos["barcode"].to_numpy(dtype=str)
in_tissue = pos["in_tissue"].to_numpy(dtype=int)
px_row = pos["px_row"].to_numpy(dtype=float); px_col = pos["px_col"].to_numpy(dtype=float)
sel = in_tissue == 1
print(f"{sample}: spots={len(barcode)} in_tissue={int(sel.sum())}")

P = np.stack([px_row[sel], px_col[sel]],1)
idx = np.random.default_rng(0).choice(len(P), size=min(400,len(P)), replace=False)
sub = P[idx]; dm = np.sqrt(((sub[:,None,:]-sub[None,:,:])**2).sum(-1)); np.fill_diagonal(dm,1e9)
spacing_px = float(np.median(dm.min(1))); umpp = 100.0/spacing_px
print(f"  尺度 {umpp:.4f} um/px (GigaTIME 0.2302, 比 {umpp/0.2302:.3f}x)")

he_jpg = os.path.join(OUT, f"{sample}_HE.jpg")
if not os.path.exists(he_jpg):
    with gzip.open(os.path.join(D, f"{gsm}_{sample}_HE.jpg.gz"),"rb") as fi, open(he_jpg,"wb") as fo:
        shutil.copyfileobj(fi, fo, 1<<24)
HE = Image.open(he_jpg); W,H = HE.size
print(f"  HE {W} x {H}")

rng = np.random.default_rng(1); srcs=[]
for _ in range(60):
    x0=int(rng.integers(0,max(1,W-256))); y0=int(rng.integers(0,max(1,H-256)))
    t=np.asarray(HE.crop((x0,y0,x0+256,y0+256)).convert("RGB"),dtype=np.float32)/255.
    srcs.append(to_lab(t).reshape(-1,3))
S=np.concatenate(srcs,0); SM,SS=S.mean(0),S.std(0)+1e-6
off=sorted(glob.glob(r"W:\虚拟细胞\data\sample_test_data\**\*_he.png",recursive=True))
A=np.concatenate([to_lab(np.asarray(Image.open(p).convert("RGB").resize((256,256),Image.BILINEAR),dtype=np.float32)/255.).reshape(-1,3) for p in off[:40]],0)
TM,TS=A.mean(0),A.std(0)
print(f"  源 LAB {SM.round(2)} / 目标 LAB {TM.round(2)}")
def norm_tile(a01): return from_lab((to_lab(a01)-SM)/SS*TS+TM)

model, device, _ = load_model(weights_path=r"W:\虚拟细胞\models\gigatime-flash\model.pth")

UP=256
TARGET=0.2302                      # GigaTIME 训练域
N_IN = int(round(256*TARGET/umpp))  # 按样本自适应：使 eff≈TARGET
N_IN = max(64, min(N_IN, 256))
scale=UP/N_IN; eff=umpp/scale
print(f"  自适应 N_IN={N_IN} | 裁 {N_IN} 原生px -> 上采样 {UP}px | 有效尺度 {eff:.4f} um/px (偏差 {abs(eff/TARGET-1)*100:.1f}%)")

spot_xy = np.stack([px_col[sel], px_row[sel]],1)
spot_r_eff = 27.5/eff
print(f"  spot 半径 {spot_r_eff:.1f} px (上采样空间)")
r_nat = 27.5/umpp

acc = np.zeros((int(sel.sum()),23)); npx = np.zeros(int(sel.sum()), dtype=np.int64)
yy,xx = np.mgrid[0:UP,0:UP]
t0=time.time(); n_tile=0
for y0 in range(0,H,N_IN):
    for x0 in range(0,W,N_IN):
        x1,y1 = min(x0+N_IN,W), min(y0+N_IN,H)
        if x1-x0<N_IN or y1-y0<N_IN: continue
        m = ((spot_xy[:,0]>=x0-r_nat)&(spot_xy[:,0]<x1+r_nat)&
             (spot_xy[:,1]>=y0-r_nat)&(spot_xy[:,1]<y1+r_nat))
        if not m.any(): continue
        a = np.asarray(HE.crop((x0,y0,x1,y1)).convert("RGB"),dtype=np.float32)/255.
        a = norm_tile(a)
        a = np.asarray(Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8)).resize((UP,UP),Image.BILINEAR),dtype=np.float32)/255.
        ten = torch.from_numpy(((a-IMNET_MEAN)/IMNET_STD).transpose(2,0,1)).unsqueeze(0)
        with torch.no_grad():
            pr = torch.sigmoid(model(ten.to(device))).squeeze(0).cpu().numpy()
        pr = np.moveaxis(pr,0,-1) if pr.shape[0]==23 else pr
        for i in np.where(m)[0]:
            cx=(spot_xy[i,0]-x0)*scale; cy=(spot_xy[i,1]-y0)*scale
            if cx<-spot_r_eff or cy<-spot_r_eff or cx>UP+spot_r_eff or cy>UP+spot_r_eff: continue
            k = ((xx-cx)**2+(yy-cy)**2)<=spot_r_eff**2
            if k.sum()<4: continue
            acc[i]+=pr[k].sum(0); npx[i]+=int(k.sum())
        n_tile+=1
        if n_tile%800==0: print(f"    tiles={n_tile} elapsed={time.time()-t0:.0f}s")
print(f"  完成 {n_tile} tiles, {time.time()-t0:.0f}s")
vals = acc/np.maximum(npx,1)[:,None]; keep = npx>0
print(f"  有效 spot {int(keep.sum())}/{int(sel.sum())} | 中位覆盖像素 {int(np.median(npx[keep]))}")
np.save(os.path.join(OUT,f"{sample}_spotX.npy"), vals[keep].astype(np.float32))
np.save(os.path.join(OUT,f"{sample}_spotXY.npy"), spot_xy[keep])
open(os.path.join(OUT,f"{sample}_barcodes.txt"),"w").write("\n".join([b for b,k in zip(barcode[sel],keep) if k]))
json.dump(dict(sample=sample,umpp=umpp,spacing_px=spacing_px,eff_umpp=eff,N_IN=N_IN,UP=UP,
               spot_r_eff_px=spot_r_eff,n_spots=int(keep.sum()),n_tiles=n_tile,channels=CHANNEL_NAMES),
          open(os.path.join(OUT,f"{sample}_meta.json"),"w"), indent=1)
print("-> saved", sample)
