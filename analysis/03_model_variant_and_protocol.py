"""补充实验 S3：模型变体消融 —— 原版 GigaTIME(CNN, 9.16M) vs GigaTIME-Flash(DINOv2-LoRA, 23.8M)

A) 官方 50 张样例 patch：按官方 Dice 口径复算两个模型，并与作者预存的 dice_metric.json 逐通道对照
   -> 既验证原版实现是否正确，也量化蒸馏造成的损失
B) L1 HEMIT 外部验证：2x2 消融 (模型 x 染色归一化)，同一输入张量、同一 Golden Truth(Otsu)
"""
import os, sys, glob, json, gzip, pickle, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"W:\虚拟细胞\repo")
import numpy as np, torch, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from scipy import stats
import gigatime_flash as GF
import gigatime_orig as GO

OUT = r"W:\虚拟细胞\analysis"
HEMIT = r"W:\虚拟细胞\data\HEMIT"
IMNET_M = np.asarray(GF.mean, dtype=np.float32); IMNET_S = np.asarray(GF.std, dtype=np.float32)

flash, device, _ = GF.load_model(weights_path=r"W:\虚拟细胞\models\gigatime-flash\model.pth")
orig, _ = GO.load_gigatime(device=device)

def to_lab(x): return cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
def from_lab(l):
    l = l.copy(); l[:,:,0] = np.clip(l[:,:,0],0,100)
    l[:,:,1] = np.clip(l[:,:,1],-127,127); l[:,:,2] = np.clip(l[:,:,2],-127,127)
    return np.clip(cv2.cvtColor(l.astype(np.float32), cv2.COLOR_LAB2RGB),0,1)

_off = sorted(glob.glob(r"W:\虚拟细胞\data\sample_test_data\**\*_he.png", recursive=True))
assert _off, "sample test data not found"
_A = np.concatenate([to_lab(np.asarray(Image.open(p).convert("RGB").resize((256,256),Image.BILINEAR),dtype=np.float32)/255.).reshape(-1,3) for p in _off[:40]],0)
TM, TS = _A.mean(0), _A.std(0)
def reinhard(x):
    l = to_lab(x); m = l.reshape(-1,3).mean(0); s = l.reshape(-1,3).std(0)+1e-6
    return from_lab((l-m)/s*TS+TM)

def dice_blocks(pb, gb, block=8):
    H, W = gb.shape; hb, wb = H//block, W//block
    if hb==0 or wb==0: return np.nan
    p = pb[:hb*block,:wb*block].reshape(hb,block,wb,block).transpose(0,2,1,3).reshape(-1,block*block)
    g = gb[:hb*block,:wb*block].reshape(hb,block,wb,block).transpose(0,2,1,3).reshape(-1,block*block)
    k = g.any(1)
    if k.sum()==0: return np.nan
    return float((2*(p[k]&g[k]).sum()+1e-6)/(p[k].sum()+g[k].sum()+1e-6))

def to_hwc(a): return np.moveaxis(a,0,-1) if a.shape[0]==23 else a

def run(model, ten, win=256):
    """通用推理：pad 到 window 的整数倍（Flash 的 DINOv2 编码器强制 256x256 输入），跑完裁回。"""
    b,c,h,w = ten.shape
    ph = (-h) % win; pw = (-w) % win
    t = torch.nn.functional.pad(ten, (0,pw,0,ph), mode="replicate") if (ph or pw) else ten
    with torch.no_grad():
        out = GO.predict_patch_orig(model, t, device, win)
    return torch.sigmoid(out[:,:,:h,:w]).squeeze(0).cpu().numpy()

# ================= A) 官方样例：官方 Dice 口径 =================
print("\n" + "="*78, flush=True)
print("  A) 官方 50 张样例 patch —— 官方 Dice 口径（8x8 block Dice, GT 按作者阈值二值化）")
print("="*78, flush=True)
bank = []
for p in _off[:50]:
    stem = os.path.basename(p).replace("_he.png",""); dn = os.path.dirname(p)
    official = float(json.load(open(os.path.join(dn, stem+"_dice_metric.json"), encoding="utf-8"))["dice"])
    gt = np.load(os.path.join(dn, stem+"_comet.npy"))[:,:,0:23]
    with gzip.open(os.path.join(dn, stem+"_comet_binary_thres_labels.pkl.gz"),"rb") as f: lab = pickle.load(f)
    thr = np.array(lab["thres_list"], float)
    binz = np.array(json.load(open(os.path.join(dn, stem+"_comet_binarized.json"), encoding="utf-8")))
    bank.append((dn, stem, official, gt, thr, binz))
official_mean = float(np.mean([b[2] for b in bank]))
print(f"  载入 {len(bank)} 张官方样例；作者预存 Dice 均值 {official_mean:.4f}", flush=True)

resA = {"official_precomputed": official_mean}
for isz in (256, 512, 556):
    for tag, model in (("flash", flash), ("orig", orig)):
        per_ch = {c: [] for c in range(23)}
        for dn, stem, official, gt, thr, binz in bank:
            img = np.asarray(Image.open(os.path.join(dn, stem+"_he.png")).convert("RGB"), dtype=np.float32)/255.
            if isz != img.shape[0]:
                img = np.asarray(Image.fromarray((img*255).astype(np.uint8)).resize((isz,isz), Image.BILINEAR), dtype=np.float32)/255.
            ten = torch.from_numpy(((img-IMNET_M)/IMNET_S).transpose(2,0,1)).unsqueeze(0)
            pr = to_hwc(run(model, ten))
            if pr.shape[0] != gt.shape[0]:
                pr = np.stack([np.asarray(Image.fromarray((pr[:,:,c]*255).astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), Image.BILINEAR))/255. for c in range(23)], -1)
            pb = pr > 0.5; gtb = gt > thr.reshape(1,1,-1)
            for c in range(23):
                if binz[c] != 1: continue
                d = dice_blocks(pb[:,:,c], gtb[:,:,c])
                if not np.isnan(d): per_ch[c].append(d)
        allv = [v for c in per_ch for v in per_ch[c]]
        pc = {GF.CHANNEL_NAMES[c]: float(np.mean(v)) for c,v in per_ch.items() if v}
        resA[f"{tag}@{isz}"] = dict(mean=float(np.mean(allv)), n=len(allv),
                                    n_channels=len(pc), per_channel=pc)
        print(f"  {tag:5s} input={isz:4d} -> Dice {np.mean(allv):.4f} (n={len(allv)}, ch={len(pc)})", flush=True)
json.dump(resA, open(os.path.join(OUT,"supp_03_official_sample.json"),"w"), indent=1)
print("  -> supp_03_official_sample.json", flush=True)

# ================= B) L1 HEMIT 2x2 消融 =================
print("\n" + "="*78, flush=True)
print("  B) L1 HEMIT 外部验证：模型 x 染色归一化 2x2 消融（输入 1024 原生）")
print("="*78, flush=True)
MAP = {"DAPI": (0,2), "CD3": (13,1), "CK": (16,0)}
allins = sorted(glob.glob(os.path.join(HEMIT,"test","input","*.tif")))
N = int(sys.argv[1]) if len(sys.argv)>1 else len(allins)
ins = allins[:N]
print(f"  HEMIT 可用 {len(allins)}，本次用 {len(ins)}", flush=True)

# (模型, 输入尺度, 归一化)。原版 GigaTIME 官方输入尺度为 512；Flash 为 256 滑窗，两者都测。
conds = [("flash",1024,"reinhard"), ("flash",1024,"none"),
         ("orig", 1024,"reinhard"), ("orig", 1024,"none"),
         ("orig",  512,"reinhard"), ("orig",  512,"none")]
MODEL = {"flash": flash, "orig": orig}
key_of = lambda t,s,n: f"{t}@{s}_{n}"
resB = {key_of(*c): {k: dict(dice=[], base=[]) for k in MAP} for c in conds}
predden = {key_of(*c): {k: [] for k in MAP} for c in conds}
rng = np.random.default_rng(0); t0 = time.time()
for i, p in enumerate(ins):
    lb_p = os.path.join(HEMIT,"test","label", os.path.basename(p))
    raw = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/255.
    lbr = np.asarray(Image.open(lb_p).convert("RGB"), dtype=np.uint8)
    variants = {"reinhard": reinhard(raw), "none": raw}
    gts = {nm: (cv2.threshold(lbr[:,:,lc],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] > 0) for nm,(mc,lc) in MAP.items()}
    for tag, scl, nrm in conds:
        if scl == raw.shape[0]:
            imv = variants[nrm]
        else:
            imv = np.asarray(Image.fromarray((variants[nrm]*255).astype(np.uint8)).resize((scl,scl), Image.BILINEAR), dtype=np.float32)/255.
        ten = torch.from_numpy(((imv-IMNET_M)/IMNET_S).transpose(2,0,1)).unsqueeze(0)
        pr = to_hwc(run(MODEL[tag], ten))
        if pr.shape[0] != raw.shape[0]:
            pr = np.stack([np.asarray(Image.fromarray((pr[:,:,c]*255).astype(np.uint8)).resize(
                (raw.shape[0], raw.shape[1]), Image.BILINEAR))/255. for c in range(23)], -1)
        for nm,(mc,lc) in MAP.items():
            pb = pr[:,:,mc] > 0.5; gb = gts[nm]
            d = dice_blocks(pb, gb)
            if not np.isnan(d): resB[key_of(tag,scl,nrm)][nm]["dice"].append(d)
            predden[key_of(tag,scl,nrm)][nm].append(float(pb.mean()))
            rb = rng.random(gb.shape) < float(gb.mean())
            db = dice_blocks(rb, gb)
            if not np.isnan(db): resB[key_of(tag,scl,nrm)][nm]["base"].append(db)
    if (i+1) % 25 == 0:
        print(f"  {i+1}/{len(ins)}  {time.time()-t0:.0f}s", flush=True)

print("\n" + "="*78, flush=True)
print("  L1 结果表  Dice / 随机基线")
print("="*78, flush=True)
print(f"{'条件':20s} {'DAPI':>18s} {'CK':>18s} {'CD3':>18s}", flush=True)
summary = {}
for tag, scl, nrm in conds:
    key = key_of(tag, scl, nrm); row = []; d = {}
    for nm in ("DAPI","CK","CD3"):
        dd = float(np.mean(resB[key][nm]["dice"])); bb = float(np.mean(resB[key][nm]["base"]))
        d[nm] = dict(dice=dd, baseline=bb, gain=dd-bb, n=len(resB[key][nm]["dice"]),
                     pred_density=float(np.mean(predden[key][nm])))
        row.append(f"{dd:.3f}/{bb:.3f}")
    print(f"{key:20s} " + " ".join(f"{v:>18s}" for v in row), flush=True)
    summary[key] = d
summary["_config"] = dict(n_patches=len(ins),
    models=dict(orig="prov-gigatime/GigaTIME (U-Net++ CNN, 9.16M)", flash="prov-gigatime/gigatime-flash (DINOv2-S+LoRA, 23.8M)"),
    norm="reinhard->GigaTIME-official-LAB-stats | none", input_size=1024, threshold=0.5,
    gt="Otsu on HEMIT label channel")
json.dump(summary, open(os.path.join(OUT,"supp_03_l1_modelvariant.json"),"w"), indent=1)
print("\n-> supp_03_official_sample.json / supp_03_l1_modelvariant.json", flush=True)
