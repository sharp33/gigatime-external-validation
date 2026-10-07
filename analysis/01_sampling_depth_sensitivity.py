"""补充实验：抽样 tile 数敏感性（验证 400 tile 是否足以估计通道密度）"""
import os, sys, glob, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"W:\虚拟细胞\repo")
import numpy as np, pandas as pd, torch, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from scipy import stats
from gigatime_flash import load_model, CHANNEL_NAMES, mean as IMNET_MEAN, std as IMNET_STD

D = r"W:\虚拟细胞\data\ORION_CRC"; OUT = r"W:\虚拟细胞\analysis\orion"
def to_lab(x): return cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
def from_lab(l):
    l=l.copy(); l[:,:,0]=np.clip(l[:,:,0],0,255*0+100); l[:,:,1]=np.clip(l[:,:,1],-127,127); l[:,:,2]=np.clip(l[:,:,2],-127,127)
    return np.clip(cv2.cvtColor(l.astype(np.float32), cv2.COLOR_LAB2RGB),0,1)

model, device, _ = load_model(weights_path=r"W:\虚拟细胞\models\gigatime-flash\model.pth")
off = sorted(glob.glob(r"W:\虚拟细胞\data\sample_test_data\**\*_he.png", recursive=True))
A = np.concatenate([to_lab(np.asarray(Image.open(p).convert("RGB").resize((256,256),Image.BILINEAR),dtype=np.float32)/255.).reshape(-1,3) for p in off[:40]],0)
TM, TS = A.mean(0), A.std(0)

pats = ["CRC01","CRC02","CRC03"]
TILES = [50, 100, 200, 400, 800, 1600]
results = {p: {t: None for t in TILES} for p in pats}

for pat in pats:
    print(f"\n===== {pat} =====", flush=True)
    df = pd.read_csv(os.path.join(D, f"{pat}_cells.csv"))
    cx = df.X_centroid.to_numpy(); cy = df.Y_centroid.to_numpy()
    import tiffslide
    ts = tiffslide.TiffSlide(os.path.join(D, f"{pat}_HE.ome.tif"))
    W, H = ts.dimensions
    mp = None
    for k in ("tiffslide.mpp-x", "openslide.mpp-x"):
        v = ts.properties.get(k)
        if v is not None:
            try: mp = float(v); break
            except Exception: pass
    if mp is None or mp <= 0:
        mk = tiffslide.TiffSlide(os.path.join(D, f"{pat}_mask.ome.tif"))
        v = mk.properties.get("tiffslide.mpp-x")
        mp = float(v) if v is not None else 0.325
    print(f"   HE {W}x{H}  mpp={mp:.4f}", flush=True)
    TARGET = 0.2302; UP = 256; N_IN = int(round(UP*TARGET/mp)); scale = UP/N_IN
    rng = np.random.default_rng(7)
    picks = rng.choice(len(df), size=min(3000, len(df)), replace=False)
    centers = []
    seen = set()
    for i in picks:
        key = (int(cx[i]//N_IN), int(cy[i]//N_IN))
        if key in seen: continue
        seen.add(key); centers.append((key[0]*N_IN, key[1]*N_IN))
    rng.shuffle(centers)
    centers = centers[:max(TILES)]
    # 先跑最多的，再前缀截断
    CH = {n:i for i,n in enumerate(CHANNEL_NAMES)}
    vecs = {}
    for t in sorted(TILES):
        acc = np.zeros(23); npx = np.zeros(len(df))
        sel = centers[:t]
        for (tx,ty) in sel:
            if tx+N_IN>W or ty+N_IN>H: continue
            a = np.asarray(ts.read_region((tx,ty),0,(N_IN,N_IN)).convert("RGB"), dtype=np.float32)/255.
            lab = to_lab(a); sm,ss = lab.reshape(-1,3).mean(0), lab.reshape(-1,3).std(0)+1e-6
            a = from_lab((lab-sm)/ss*TS+TM)
            a = np.asarray(Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8)).resize((UP,UP),Image.BILINEAR),dtype=np.float32)/255.
            ten = torch.from_numpy(((a-IMNET_MEAN)/IMNET_STD).transpose(2,0,1)).unsqueeze(0)
            with torch.no_grad():
                pr = torch.sigmoid(model(ten.to(device))).squeeze(0).cpu().numpy()
            pr = np.moveaxis(pr,0,-1) if pr.shape[0]==23 else pr
            m = (cx>=tx)&(cx<tx+N_IN)&(cy>=ty)&(cy<ty+N_IN)
            for i in np.where(m)[0]:
                u=int((cx[i]-tx)*scale); v=int((cy[i]-ty)*scale)
                if 0<=u<UP and 0<=v<UP:
                    acc += pr[v,u,:]; npx[i]+=1
        ck = npx>0
        results[pat][t] = acc/np.array([len(sel)]) if False else (acc/max(1,ck.sum()))
        print(f"   tiles={t:5d} -> 覆盖细胞 {int(ck.sum())}, DAPI密度={results[pat][t][CH['DAPI']]:.4f}, CK密度={results[pat][t][CH['CK_1:150']]:.4f}", flush=True)

print("\n=== 敏感性汇总：各 tile 数下的通道密度（3 病人均值）===")
print(f"{'tiles':>7s} {'DAPI':>9s} {'CK':>9s} {'CD3':>9s} {'CD20':>9s} {'CD68':>9s}")
base = None
for t in TILES:
    vals = [results[p][t] for p in pats if results[p][t] is not None]
    if not vals: continue
    m = np.mean(vals, axis=0)
    if base is None: base = m
    print(f"{t:>7d} " + " ".join(f"{m[CH[c]]:9.4f}" for c in ["DAPI","CK_1:150","CD3_1:1000","CD20","CD68_1:100"]))
    if t > 50:
        dev = float(np.abs(m/base - 1).mean())
        print(f"{'':>7s}  相对 50 tile 的平均偏差: {dev*100:.2f}%")
json.dump({p: {str(t): (results[p][t].tolist() if results[p][t] is not None else None) for t in TILES} for p in pats},
          open(os.path.join(OUT,"tile_sensitivity.json"),"w"))
print("\n-> tile_sensitivity.json")
