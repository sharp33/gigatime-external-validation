"""补充实验 S3b：官方样例 Dice 口径对齐 —— 我们复算的 Dice 与作者预存 dice_metric.json 的差距来自哪里？

作者预存 50 张样例的逐 tile 平均 Dice = 0.3357。
我们用 GigaTIME 官方 Dice 口径复算只得 ~0.18。差异的候选来源：
  (a) 是否跳过 GT 全零的 8x8 块（我们跳过 -> 应当更高，不是原因）
  (b) 是否把"真阴性"计入（全图计算，含背景）
  (c) 分辨率（原生 556 vs resize 512）
本脚本对每个候选口径都算一遍，看哪个能落到 0.3357 附近 —— 用来证明"口径对齐"而非"复现失败"。
"""
import os, sys, glob, json, gzip, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"W:\虚拟细胞\repo")
import numpy as np, torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import gigatime_flash as GF, gigatime_orig as GO

OUT = r"W:\虚拟细胞\analysis"
IMNET_M = np.asarray(GF.mean, dtype=np.float32); IMNET_S = np.asarray(GF.std, dtype=np.float32)
flash, device, _ = GF.load_model(weights_path=r"W:\虚拟细胞\models\gigatime-flash\model.pth")
orig, _ = GO.load_gigatime(device=device)

def run(model, ten, win=256):
    b,c,h,w = ten.shape
    ph = (-h) % win; pw = (-w) % win
    t = torch.nn.functional.pad(ten, (0,pw,0,ph), mode="replicate") if (ph or pw) else ten
    with torch.no_grad():
        out = GO.predict_patch_orig(model, t, device, win)
    return torch.sigmoid(out[:,:,:h,:w]).squeeze(0).cpu().numpy()

def to_hwc(a): return np.moveaxis(a,0,-1) if a.shape[0]==23 else a

def dice_blocks(pb, gb, block=8, skip_empty=True):
    H, W = gb.shape; hb, wb = H//block, W//block
    p = pb[:hb*block,:wb*block].reshape(hb,block,wb,block).transpose(0,2,1,3).reshape(-1,block*block)
    g = gb[:hb*block,:wb*block].reshape(hb,block,wb,block).transpose(0,2,1,3).reshape(-1,block*block)
    k = g.any(1) if skip_empty else np.ones(len(g), bool)
    if k.sum()==0: return np.nan
    return float((2*(p[k]&g[k]).sum()+1e-6)/(p[k].sum()+g[k].sum()+1e-6))

def dice_pixel(pb, gb):
    return float((2*(pb&gb).sum()+1e-6)/(pb.sum()+gb.sum()+1e-6))

bank = []
for p in sorted(glob.glob(r"W:\虚拟细胞\data\sample_test_data\**\*_he.png", recursive=True))[:50]:
    stem = os.path.basename(p).replace("_he.png",""); dn = os.path.dirname(p)
    off = float(json.load(open(os.path.join(dn, stem+"_dice_metric.json"), encoding="utf-8"))["dice"])
    gt = np.load(os.path.join(dn, stem+"_comet.npy"))[:,:,0:23]
    with gzip.open(os.path.join(dn, stem+"_comet_binary_thres_labels.pkl.gz"),"rb") as f: lab = pickle.load(f)
    thr = np.array(lab["thres_list"], float)
    binz = np.array(json.load(open(os.path.join(dn, stem+"_comet_binarized.json"), encoding="utf-8")))
    bank.append((dn, stem, off, gt, thr, binz))
print(f"作者预存 Dice 均值 = {np.mean([b[2] for b in bank]):.4f} (n={len(bank)})\n")

protos = {
    "8x8块_跳过空块":  lambda pb, gb: dice_blocks(pb, gb, 8, True),
    "8x8块_含空块":    lambda pb, gb: dice_blocks(pb, gb, 8, False),
    "像素级_全图":      dice_pixel,
}
for isz in (512, 556):
    for tag, model in (("flash", flash), ("orig", orig)):
        acc = {k: [] for k in protos}
        for dn, stem, off, gt, thr, binz in bank:
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
                for k, fn in protos.items():
                    v = fn(pb[:,:,c], gtb[:,:,c])
                    if not np.isnan(v): acc[k].append(v)
        print(f"  {tag:5s} @{isz}: " + " | ".join(f"{k}={np.mean(v):.4f}" for k, v in acc.items()), flush=True)
json.dump(dict(official=float(np.mean([b[2] for b in bank]))),
          open(os.path.join(OUT,"supp_03b_dice_protocol.json"),"w"), indent=1)
print("\n-> supp_03b_dice_protocol.json")
