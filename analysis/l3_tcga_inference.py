"""L3 步骤1：TCGA-THCA 切片 -> 虚拟通道密度（每张切片一个 23 维向量）

设计：不跑整张切片（太慢），而是从组织区**随机抽样** ~400 个 tile 估计通道密度。
      密度是比例量，随机抽样是它的无偏估计。
尺度：直接读 SVS 的 mpp 属性（不猜），按 GigaTIME 训练域 0.2302 µm/px 重采样。
用法: python l3_run_gigatime.py <slide.svs> [n_tiles]
"""
import os, sys, glob, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"W:\虚拟细胞\repo")
import numpy as np, torch, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from gigatime_flash import load_model, CHANNEL_NAMES, mean as IMNET_MEAN, std as IMNET_STD

OUT = r"W:\虚拟细胞\analysis\l3"
os.makedirs(OUT, exist_ok=True)
TARGET = 0.2302

def to_lab(x): return cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
def from_lab(l):
    l=l.copy(); l[:,:,0]=np.clip(l[:,:,0],0,100); l[:,:,1]=np.clip(l[:,:,1],-127,127); l[:,:,2]=np.clip(l[:,:,2],-127,127)
    return np.clip(cv2.cvtColor(l.astype(np.float32), cv2.COLOR_LAB2RGB),0,1)

off=sorted(glob.glob(r"W:\虚拟细胞\data\sample_test_data\**\*_he.png",recursive=True))
A=np.concatenate([to_lab(np.asarray(Image.open(p).convert("RGB").resize((256,256),Image.BILINEAR),dtype=np.float32)/255.).reshape(-1,3) for p in off[:40]],0)
TM,TS=A.mean(0),A.std(0)

def slide_vector(slide_path, n_tiles=400, seed=0):
    import tiffslide
    ts = tiffslide.TiffSlide(slide_path)
    props = ts.properties
    mpp = None
    for k in ("tiffslide.mpp-x","openslide.mpp-x","tiffslide.mpp","openslide.mpp"):
        if k in props:
            try: mpp = float(props[k]); break
            except Exception: pass
    if mpp is None or mpp <= 0:
        # 退化：由 level 0 尺寸与 objective 估计
        mpp = 0.25
        print(f"    [warn] 无 mpp 属性，假定 {mpp}")
    W0, H0 = ts.dimensions
    N_native = int(round(256 * TARGET / mpp))
    N_native = max(32, min(N_native, 512))
    # 组织掩膜（用 level 0 的 1/32 缩略图）
    lvl = ts.get_best_level_for_downsample(32)
    tw, th = ts.level_dimensions[lvl]
    thumb = ts.read_region((0,0), lvl, (tw,th)).convert("RGB")
    ta = np.asarray(thumb, dtype=np.float32)
    gray = ta.mean(2)
    tissue = gray < 220            # 非白背景
    frac_tissue = float(tissue.mean())
    ys, xs = np.where(tissue)
    if len(ys) < 100:
        return None, dict(mpp=mpp, error="no_tissue", W=W0, H=H0, tissue_frac=frac_tissue)
    rng = np.random.default_rng(seed)
    acc = np.zeros(len(CHANNEL_NAMES)); used = 0
    tries = 0
    while used < n_tiles and tries < n_tiles*6:
        tries += 1
        k = rng.integers(0, len(ys))
        ty, tx = ys[k], xs[k]
        # 缩略图坐标 -> level0 坐标（缩略图中心）
        x0 = int(tx * W0 / tw) - N_native//2
        y0 = int(ty * H0 / th) - N_native//2
        x0 = max(0, min(x0, W0-N_native)); y0 = max(0, min(y0, H0-N_native))
        if x0 < 0 or y0 < 0: continue
        tile = ts.read_region((x0,y0), 0, (N_native,N_native)).convert("RGB")
        a = np.asarray(tile, dtype=np.float32)/255.
        if a.mean() > 0.92:   # 近白，跳过
            continue
        lab = to_lab(a); SM, SS = lab.reshape(-1,3).mean(0), lab.reshape(-1,3).std(0)+1e-6
        a = from_lab((lab - SM)/SS*TS + TM)
        a = np.asarray(Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8)).resize((256,256), Image.BILINEAR), dtype=np.float32)/255.
        yield_tile = a
        ten = torch.from_numpy(((yield_tile-IMNET_MEAN)/IMNET_STD).transpose(2,0,1)).unsqueeze(0)
        with torch.no_grad():
            pr = torch.sigmoid(model(ten.to(device))).squeeze(0).cpu().numpy()
        pr = np.moveaxis(pr,0,-1) if pr.shape[0]==23 else pr
        acc += (pr > 0.5).reshape(-1, 23).mean(0)     # 该 tile 的密度
        used += 1
    return (acc/max(1,used)), dict(mpp=mpp, N_native=N_native, used=used, tries=tries,
                                   W=W0, H=H0, tissue_frac=frac_tissue,
                                   eff_umpp=mpp*N_native/256)

model, device, _ = load_model(weights_path=r"W:\虚拟细胞\models\gigatime-flash\model.pth")

sl = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv)>2 else 400
t0=time.time()
vec, info = slide_vector(sl, n_tiles=n)
name = os.path.basename(sl).split(".")[0]
if vec is None:
    print("FAILED:", name, info)
else:
    np.save(os.path.join(OUT, f"{name}_vec.npy"), vec.astype(np.float32))
    json.dump(dict(file=os.path.basename(sl), **info, channels=CHANNEL_NAMES, sec=round(time.time()-t0,1)),
              open(os.path.join(OUT, f"{name}_info.json"),"w"), indent=1)
    print(f"{name}: mpp={info['mpp']:.4f} N_native={info['N_native']} used={info['used']} "
          f"eff={info['eff_umpp']:.4f} tissue={info['tissue_frac']:.2f} | {time.time()-t0:.0f}s")
    print("  top5 通道密度:", {CHANNEL_NAMES[i]: round(float(vec[i]),4) for i in np.argsort(-vec)[:5]})
