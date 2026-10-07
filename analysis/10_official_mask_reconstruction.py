"""补充实验 S8：复现作者预存的官方 Dice（dice_metric.json 均值 0.3357）

关键发现 —— 读官方 scripts/prov_data.py 得知：
  1) pkl 里的 comet_array_binary 是 **bit-packed** 的：
       (556,556,3) uint8 --np.unpackbits--> (556,556,24) --[..., :23]--> (556,556,23)
     我们之前直接读原始 3 通道数组，得到的是无意义的打包位。
  2) 官方 dataset 用的参照掩膜**不是**原始阈值化 comet，而是**细胞级凸包掩膜**：
       a. mask[cell_masks==0] = 0                       # 非细胞区置零
       b. 对 labels_dapi / labels_dapi_expanded 的每个连通域取 convex_image(凸包)；
          nuclei 模式只判 DAPI/TRITC/Cy5/Ki67，cell 模式只判其余 19 通道；
          凸包内阳性像素占比 > 0.5（Ki67 > 0.2）则把该通道的凸包位置填 1
  3) 官方输入：556x556 H&E --Resize(512)--> ImageNet Normalize --> 模型（256 滑窗）

本脚本对 50 张样例、多种参照口径各算一遍 Dice，看哪种能落到 0.3357。
"""
import os, sys, glob, gzip, pickle, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"W:\虚拟细胞\repo")
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage.measure import label, regionprops
import gigatime_flash as GF, gigatime_orig as GO

OUT = r"W:\虚拟细胞\analysis"
D = r"W:\虚拟细胞\data\sample_test_data"
IMNET_M = np.asarray(GF.mean, dtype=np.float32); IMNET_S = np.asarray(GF.std, dtype=np.float32)
from gigatime_flash import CHANNEL_NAMES
NUCLEAR = ["DAPI", "TRITIC", "Cy5", "Ki67_1:150 - TRITC"]

orig, device = GO.load_gigatime()
flash, _, _ = GF.load_model(weights_path=r"W:\虚拟细胞\models\gigatime-flash\model.pth")

def unpack_and_load(path):
    with gzip.open(path, "rb") as f: data = pickle.load(f)
    packed = data["comet_array_binary"]
    orig_shape = data["original_shape"]; last = data["original_last_dim"]
    arr = np.unpackbits(packed, axis=-1)[..., :last].reshape(orig_shape)
    data["comet_array_binary"] = arr
    return data

def build_official_mask(pkl, mask_noncell=True, cell_mask_label=True, ratio=0.5):
    mask = pkl["comet_array_binary"].copy()
    if mask_noncell:
        cm = np.stack([pkl["labels_dapi"] if c in NUCLEAR else pkl["labels_dapi_expanded"]
                       for c in CHANNEL_NAMES], -1)
        mask[cm == 0] = 0
    if not cell_mask_label:
        return mask
    mask_new = np.zeros_like(mask)
    for mode, props in (("nuclei", regionprops(label(pkl["labels_dapi"]))),
                        ("cell",   regionprops(label(pkl["labels_dapi_expanded"])))):
        for region in props:
            rmask = getattr(region, "image_convex", None)
            if rmask is None: rmask = region.convex_image
            r0, c0, r1, c1 = region.bbox
            sub = mask[r0:r1, c0:c1, :]
            ratios = sub[rmask].sum(axis=0) / max(1, rmask.sum())
            sel = []
            for ci, cname in enumerate(CHANNEL_NAMES):
                if mode == "nuclei":
                    ok = cname in NUCLEAR
                else:
                    ok = cname not in NUCLEAR
                if ok and ratios[ci] > (0.2 if cname == "Ki67_1:150 - TRITC" else ratio):
                    sel.append(ci)
            for ch in sel:
                view = mask_new[r0:r1, c0:c1, ch]
                view[rmask] = 1
    return mask_new

def dice_pixel(pb, gb):
    s = pb.sum() + gb.sum()
    return float((2*(pb & gb).sum() + 1e-6) / (s + 1e-6)) if s > 0 else np.nan

def run_model(model, ten, win=256):
    b, c, h, w = ten.shape
    ph = (-h) % win; pw = (-w) % win
    t = torch.nn.functional.pad(ten, (0,pw,0,ph), mode="replicate") if (ph or pw) else ten
    with torch.no_grad():
        out = GO.predict_patch_orig(model, t, device, win)
    return torch.sigmoid(out[:,:,:h,:w]).squeeze(0).cpu().numpy()

import torch
files = sorted(glob.glob(os.path.join(D, "**", "*_comet_binary_thres_labels.pkl.gz"), recursive=True))
print(f"样例 {len(files)} 张\n", flush=True)

VARIANTS = ["A_raw_thres", "B_official556", "C_official512", "D_nocellmask"]
acc = {m: {v: [] for v in VARIANTS} for m in ("orig", "flash")}
per_ch = {m: {v: {i: [] for i in range(23)} for v in VARIANTS} for m in ("orig", "flash")}
official_per_tile = []
t0 = time.time()
chk_done = False
for k, p in enumerate(files):
    stem = os.path.basename(p).replace("_comet_binary_thres_labels.pkl.gz", "")
    dn = os.path.dirname(p)
    official_per_tile.append(float(json.load(open(os.path.join(dn, stem + "_dice_metric.json"), encoding="utf-8"))["dice"]))
    binz = np.array(json.load(open(os.path.join(dn, stem + "_comet_binarized.json"), encoding="utf-8")))
    pkl = unpack_and_load(p)
    mask_official = build_official_mask(pkl)
    mask_nocell = build_official_mask(pkl, cell_mask_label=False)
    raw = np.load(os.path.join(dn, stem + "_comet.npy"))[:, :, 0:23]
    thr = np.array(pkl["thres_list"], float)

    if not chk_done:
        ok = np.mean([(pkl["comet_array_binary"][:,:,i] == (raw[:,:,i] > thr[i])).mean() for i in range(23)])
        print(f"  [校验] unpackbits 得到的二值图 vs comet>thres 的一致性 = {ok:.4f}", flush=True)
        chk_done = True

    he = np.asarray(Image.open(os.path.join(dn, stem + "_he.png")).convert("RGB"), dtype=np.float32)/255.
    he512 = np.asarray(Image.fromarray((he*255).astype(np.uint8)).resize((512,512), Image.BILINEAR), dtype=np.float32)/255.
    ten = torch.from_numpy(((he512-IMNET_M)/IMNET_S).transpose(2,0,1)).unsqueeze(0)

    for tag, model in (("orig", orig), ("flash", flash)):
        pr = run_model(model, ten)
        pr = np.moveaxis(pr, 0, -1) if pr.shape[0] == 23 else pr
        pred = pr > 0.5
        H = raw.shape[0]
        pred_native = np.stack([np.asarray(Image.fromarray((pred[:,:,i]*255).astype(np.uint8)).resize((H, H), Image.NEAREST)) > 127
                                for i in range(23)], -1)
        mask512 = np.stack([np.asarray(Image.fromarray((mask_official[:,:,i]*255).astype(np.uint8)).resize((512,512), Image.BILINEAR))/255.
                            for i in range(23)], -1) > 0.5
        for i in range(23):
            if binz[i] != 1: continue
            gts = {"A_raw_thres":   (raw[:,:,i] > thr[i], "native"),
                   "B_official556": (mask_official[:,:,i] > 0, "native"),
                   "C_official512": (mask512[:,:,i], "model"),
                   "D_nocellmask":  (mask_nocell[:,:,i] > 0, "native")}
            for v in VARIANTS:
                g, res = gts[v]
                p = pred_native[:,:,i] if res == "native" else pred[:,:,i]
                d = dice_pixel(p, g)
                if not np.isnan(d):
                    acc[tag][v].append(d); per_ch[tag][v][i].append(d)
    if (k+1) % 10 == 0: print(f"  {k+1}/{len(files)} {time.time()-t0:.0f}s", flush=True)

print(f"\n作者预存 dice_metric.json 均值 = {np.mean(official_per_tile):.4f}\n")
print(f"{'参照口径':18s} {'orig Dice':>12s} {'flash Dice':>12s}")
res = {"official_precomputed": float(np.mean(official_per_tile))}
for v in VARIANTS:
    a = float(np.mean(acc["orig"][v])); b = float(np.mean(acc["flash"][v]))
    res[v] = dict(orig=a, flash=b, n=len(acc["orig"][v]))
    print(f"{v:18s} {a:12.4f} {b:12.4f}   (n={len(acc['orig'][v])})")
print("\n" + "="*84)
print("  ★ 官方口径（B_official556）下的逐通道 Dice —— 关键问题：CD3 是不是也变好了？")
print("="*84)
print(f"{'通道':16s} {'orig':>8s} {'flash':>8s} {'n':>5s}   {'通道':16s} {'orig':>8s} {'flash':>8s} {'n':>5s}")
for i in range(23):
    o = float(np.mean(per_ch["orig"]["B_official556"][i])) if per_ch["orig"]["B_official556"][i] else float("nan")
    fl = float(np.mean(per_ch["flash"]["B_official556"][i])) if per_ch["flash"]["B_official556"][i] else float("nan")
    res.setdefault("per_channel_official", {})[CHANNEL_NAMES[i]] = dict(orig=o, flash=fl,
                                                                       n=len(per_ch["orig"]["B_official556"][i]))
print()
ordr = sorted(range(23), key=lambda i: -(np.nan_to_num(float(np.mean(per_ch['orig']['B_official556'][i]))) if per_ch['orig']['B_official556'][i] else -1))
for i in ordr:
    o = float(np.mean(per_ch["orig"]["B_official556"][i])) if per_ch["orig"]["B_official556"][i] else float("nan")
    fl = float(np.mean(per_ch["flash"]["B_official556"][i])) if per_ch["flash"]["B_official556"][i] else float("nan")
    print(f"  {CHANNEL_NAMES[i]:16s} {o:8.4f} {fl:8.4f} {len(per_ch['orig']['B_official556'][i]):5d}")
for v in VARIANTS:
    res.setdefault("per_channel", {})[v] = {
        CHANNEL_NAMES[i]: dict(orig=float(np.mean(per_ch["orig"][v][i])) if per_ch["orig"][v][i] else None,
                               flash=float(np.mean(per_ch["flash"][v][i])) if per_ch["flash"][v][i] else None,
                               n=len(per_ch["orig"][v][i])) for i in range(23)}
json.dump(res, open(os.path.join(OUT, "supp_08_official_mask.json"), "w"), indent=1)
print("\n-> supp_08_official_mask.json")
