"""补充实验 S3c：CD3 通道在 L1 上的"密度公平性"复核（低内存版）

S3 发现 orig@512_none 的 CD3 Dice 达 0.482（随机基线 0.090），但预测密度 0.2635（GT 约 0.05）。
本脚本在同一批 945 patch 上，对每个 patch 额外做**密度匹配阈值**：
  阈值取到使该 patch 的预测阳性率 = 该 patch 的 GT 阳性率，再算 8x8 block Dice。
若密度匹配后 Dice 掉回基线，说明"CD3 成功"是密度失配伪影，而非定位能力。
单遍计算、逐 patch 释放内存。
"""
import os, sys, glob, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"W:\虚拟细胞\repo")
import numpy as np, torch, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import gigatime_flash as GF, gigatime_orig as GO

OUT = r"W:\虚拟细胞\analysis"; HEMIT = r"W:\虚拟细胞\data\HEMIT"
IMNET_M = np.asarray(GF.mean, dtype=np.float32); IMNET_S = np.asarray(GF.std, dtype=np.float32)
orig, _ = GO.load_gigatime(device=None)
device = next(orig.parameters()).device
flash, _, _ = GF.load_model(weights_path=r"W:\虚拟细胞\models\gigatime-flash\model.pth")
MODEL = {"orig": orig, "flash": flash}

def to_lab(x): return cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
def from_lab(l):
    l = l.copy(); l[:,:,0] = np.clip(l[:,:,0],0,100)
    l[:,:,1] = np.clip(l[:,:,1],-127,127); l[:,:,2] = np.clip(l[:,:,2],-127,127)
    return np.clip(cv2.cvtColor(l.astype(np.float32), cv2.COLOR_LAB2RGB),0,1)
off = sorted(glob.glob(r"W:\虚拟细胞\data\sample_test_data\**\*_he.png", recursive=True))
A = np.concatenate([to_lab(np.asarray(Image.open(p).convert("RGB").resize((256,256),Image.BILINEAR),dtype=np.float32)/255.).reshape(-1,3) for p in off[:40]],0)
TM, TS = A.mean(0), A.std(0)
def reinhard(x):
    l = to_lab(x); m = l.reshape(-1,3).mean(0); s = l.reshape(-1,3).std(0)+1e-6
    return from_lab((l-m)/s*TS+TM)

def run(model, ten, win=256):
    b,c,h,w = ten.shape
    ph = (-h) % win; pw = (-w) % win
    t = torch.nn.functional.pad(ten, (0,pw,0,ph), mode="replicate") if (ph or pw) else ten
    with torch.no_grad():
        out = GO.predict_patch_orig(model, t, device, win)
    return torch.sigmoid(out[:,:,:h,:w]).squeeze(0).cpu().numpy()

def dice_blocks(pb, gb, block=8):
    H, W = gb.shape; hb, wb = H//block, W//block
    p = pb[:hb*block,:wb*block].reshape(hb,block,wb,block).transpose(0,2,1,3).reshape(-1,block*block)
    g = gb[:hb*block,:wb*block].reshape(hb,block,wb,block).transpose(0,2,1,3).reshape(-1,block*block)
    k = g.any(1)
    if k.sum()==0: return np.nan
    return float((2*(p[k]&g[k]).sum()+1e-6)/(p[k].sum()+g[k].sum()+1e-6))

CH3 = [("DAPI",0,2), ("CD3",13,1), ("CK",16,0)]
ins = sorted(glob.glob(os.path.join(HEMIT,"test","input","*.tif")))
CONDS = [("orig",1024,"none"), ("orig",512,"none"), ("orig",1024,"reinhard"), ("flash",1024,"reinhard")]
print(f"HEMIT {len(ins)} patch；条件 {CONDS}\n", flush=True)
acc = {c: {nm: dict(d0=[], d1=[], base=[], pd0=[], gd=[], thr=[]) for nm,_,_ in CH3} for c in CONDS}
t0 = time.time()
for i, p in enumerate(ins):
    lb_p = os.path.join(HEMIT,"test","label", os.path.basename(p))
    raw = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/255.
    lbr = np.asarray(Image.open(lb_p).convert("RGB"), dtype=np.uint8)
    gts = {nm: (cv2.threshold(lbr[:,:,lc],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] > 0) for nm,_,lc in CH3}
    for (tag, scl, nrm) in CONDS:
        imv = raw if nrm == "none" else reinhard(raw)
        if scl != raw.shape[0]:
            imv = np.asarray(Image.fromarray((imv*255).astype(np.uint8)).resize((scl,scl), Image.BILINEAR), dtype=np.float32)/255.
        ten = torch.from_numpy(((imv-IMNET_M)/IMNET_S).transpose(2,0,1)).unsqueeze(0)
        pr = run(MODEL[tag], ten)
        for nm, mc, lc in CH3:
            P = pr[mc].astype(np.float32)
            if P.shape[0] != raw.shape[0]:
                P = np.asarray(Image.fromarray((P*255).astype(np.uint8)).resize((raw.shape[1], raw.shape[0]), Image.BILINEAR), dtype=np.float32)/255.
            G = gts[nm]; gd = float(G.mean())
            d0 = dice_blocks(P > 0.5, G)
            # 密度匹配：阈值取到该 patch 预测阳性率 = GT 阳性率
            thr = float(np.quantile(P, 1-gd))
            d1 = dice_blocks(P > thr, G)
            rb = np.random.default_rng(i*7+mc).random(G.shape) < gd
            bs = dice_blocks(rb, G)
            a = acc[(tag,scl,nrm)][nm]
            if not np.isnan(d0): a["d0"].append(d0)
            if not np.isnan(d1): a["d1"].append(d1)
            if not np.isnan(bs): a["base"].append(bs)
            a["pd0"].append(float((P>0.5).mean())); a["gd"].append(gd); a["thr"].append(thr)
            del P, G
        del pr, ten
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(ins)} {time.time()-t0:.0f}s", flush=True)

print(f"\n{'条件':20s} {'通道':6s} {'GT密度':>8s} {'预测密度':>9s} {'Dice@0.5':>9s} {'Dice密度匹配':>13s} {'随机基线':>9s}")
res = {}
for (tag,scl,nrm) in CONDS:
    for nm,_,_ in CH3:
        a = acc[(tag,scl,nrm)][nm]
        row = dict(gt_density=float(np.mean(a["gd"])), pred_density=float(np.mean(a["pd0"])),
                   mean_thr=float(np.mean(a["thr"])),
                   dice_default=float(np.mean(a["d0"])), dice_density_matched=float(np.mean(a["d1"])),
                   baseline=float(np.mean(a["base"])), n=len(a["d0"]))
        res[f"{tag}@{scl}_{nrm}|{nm}"] = row
        print(f"{tag+'@'+str(scl)+'_'+nrm:20s} {nm:6s} {row['gt_density']:8.4f} {row['pred_density']:9.4f} "
              f"{row['dice_default']:9.3f} {row['dice_density_matched']:13.3f} {row['baseline']:9.3f}", flush=True)
json.dump(res, open(os.path.join(OUT,"supp_03c_density_fairness.json"),"w"), indent=1)
print("\n-> supp_03c_density_fairness.json")
