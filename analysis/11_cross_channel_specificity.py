"""补充实验 S8b：官方口径下的**交叉通道特异性**检验（L1 的判决性实验）

S8 发现：用作者的 cell-hull 掩膜，我们能复现官方 Dice（0.364 vs 作者 0.336），
但 **TRITC（作者声明为"背景通道，不参与分析"）拿到 0.446**，比 CD3(0.367)、
CD68(0.430)、CD16(0.397)、CD14(0.315)、CD34(0.302) 都高。

本脚本做 L2/L3/L4 同款的"特异性"检验：
  对每个虚拟通道 i，算它对着**所有**官方掩膜通道 j 的 Dice —— D[i,j]。
  配对值 = D[i,i]；特异性 = D[i,i] − max_{j≠i} D[i,j]。
  若 CD3 的预测对 CD68 的掩膜也打一样的分，那这个 Dice 就不是"CD3 的定位能力"。
另外算随机基线（与 GT 密度匹配的随机掩膜）。
"""
import os, sys, glob, gzip, pickle, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"W:\虚拟细胞\repo")
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage.measure import label, regionprops
import torch
import gigatime_flash as GF, gigatime_orig as GO
from gigatime_flash import CHANNEL_NAMES

OUT = r"W:\虚拟细胞\analysis"
D = r"W:\虚拟细胞\data\sample_test_data"
IMNET_M = np.asarray(GF.mean, dtype=np.float32); IMNET_S = np.asarray(GF.std, dtype=np.float32)
NUCLEAR = ["DAPI", "TRITIC", "Cy5", "Ki67_1:150 - TRITC"]
orig, device = GO.load_gigatime()
flash, _, _ = GF.load_model(weights_path=r"W:\虚拟细胞\models\gigatime-flash\model.pth")

with open(os.path.join(r"W:\虚拟细胞\analysis", "l3_ok.txt"), "w") as _f: pass

def unpack_and_load(path):
    with gzip.open(path, "rb") as f: data = pickle.load(f)
    packed = data["comet_array_binary"]; shape = data["original_shape"]; last = data["original_last_dim"]
    data["comet_array_binary"] = np.unpackbits(packed, axis=-1)[..., :last].reshape(shape)
    return data

def build_official_mask(pkl):
    mask = pkl["comet_array_binary"].copy()
    cm = np.stack([pkl["labels_dapi"] if c in NUCLEAR else pkl["labels_dapi_expanded"] for c in CHANNEL_NAMES], -1)
    mask[cm == 0] = 0
    mask_new = np.zeros_like(mask)
    for mode, props in (("nuclei", regionprops(label(pkl["labels_dapi"]))),
                        ("cell",   regionprops(label(pkl["labels_dapi_expanded"])))):
        for region in props:
            rmask = getattr(region, "image_convex", None)
            if rmask is None: rmask = region.convex_image
            r0, c0, r1, c1 = region.bbox
            sub = mask[r0:r1, c0:c1, :]
            ratios = sub[rmask].sum(axis=0) / max(1, rmask.sum())
            for ci, cname in enumerate(CHANNEL_NAMES):
                ok = (cname in NUCLEAR) if mode == "nuclei" else (cname not in NUCLEAR)
                if ok and ratios[ci] > (0.2 if cname == "Ki67_1:150 - TRITC" else 0.5):
                    mask_new[r0:r1, c0:c1, ci][rmask] = 1
    return mask_new

def run_model(model, ten, win=256):
    b, c, h, w = ten.shape
    ph = (-h) % win; pw = (-w) % win
    t = torch.nn.functional.pad(ten, (0,pw,0,ph), mode="replicate") if (ph or pw) else ten
    with torch.no_grad():
        out = GO.predict_patch_orig(model, t, device, win)
    return torch.sigmoid(out[:,:,:h,:w]).squeeze(0).cpu().numpy()

def dice(a, b):
    s = a.sum() + b.sum()
    return float((2*(a & b).sum() + 1e-6)/(s + 1e-6)) if s > 0 else np.nan

files = sorted(glob.glob(os.path.join(D, "**", "*_comet_binary_thres_labels.pkl.gz"), recursive=True))
print(f"样例 {len(files)} 张\n", flush=True)
Dmat = {m: np.zeros((50, 23, 23)) for m in ("orig", "flash")}
GTpos = np.zeros((50, 23), dtype=bool)
rand_base = np.zeros((50, 23))
rng = np.random.default_rng(0)

for k, p in enumerate(files):
    stem = os.path.basename(p).replace("_comet_binary_thres_labels.pkl.gz", ""); dn = os.path.dirname(p)
    pkl = unpack_and_load(p)
    M = build_official_mask(pkl)                      # (556,556,23)
    GTpos[k] = [M[:,:,i].sum() > 50 for i in range(23)]
    H = M.shape[0]
    he = np.asarray(Image.open(os.path.join(dn, stem+"_he.png")).convert("RGB"), dtype=np.float32)/255.
    he512 = np.asarray(Image.fromarray((he*255).astype(np.uint8)).resize((512,512), Image.BILINEAR), dtype=np.float32)/255.
    ten = torch.from_numpy(((he512-IMNET_M)/IMNET_S).transpose(2,0,1)).unsqueeze(0)
    for tag, model in (("orig", orig), ("flash", flash)):
        pr = run_model(model, ten)
        pr = np.moveaxis(pr, 0, -1) if pr.shape[0] == 23 else pr
        P = np.stack([np.asarray(Image.fromarray(((pr[:,:,i] > 0.5)*255).astype(np.uint8)).resize((H, H), Image.NEAREST)) > 127
                      for i in range(23)], -1)
        for i in range(23):
            if not GTpos[k, i]: continue
            for j in range(23):
                if not GTpos[k, j]: continue
                Dmat[tag][k, i, j] = dice(P[:,:,i], M[:,:,j])
    for j in range(23):
        if not GTpos[k, j]: continue
        rand_base[k, j] = dice(rng.random((H, H)) < float(M[:,:,j].mean()), M[:,:,j] > 0)
    if (k+1) % 10 == 0: print(f"  {k+1}/{len(files)}", flush=True)

print("\n" + "="*104)
print("  ★ L1 官方口径下的交叉通道特异性（对角线 = 配对；特异性 = 对角线 − 其他通道最大值）")
print("="*104)
res = {}
for tag in ("orig", "flash"):
    print(f"\n--- {tag} ---")
    print(f"{'虚拟通道 i':16s} {'配对 D[i,i]':>11s} {'其他最大':>9s} {'最强错配 j':>16s} {'特异性':>8s} {'随机基线':>9s} {'n':>4s}")
    rows = []
    for i in range(23):
        vals = Dmat[tag][:, i, i]; msk = ~np.isnan(vals)
        if msk.sum() < 5: continue
        diag = float(np.nanmean(vals))
        off = []
        for j in range(23):
            if j == i: continue
            v = Dmat[tag][:, i, j]; m = ~np.isnan(v)
            if m.sum() >= 5: off.append((float(np.nanmean(v)), CHANNEL_NAMES[j]))
        if not off: continue
        om, omj = max(off)
        rb = float(np.nanmean(rand_base[:, i][GTpos[:, i]])) if GTpos[:, i].sum() else float("nan")
        rows.append(dict(channel=CHANNEL_NAMES[i], diag=diag, best_other=om, best_other_channel=omj,
                         specificity=diag-om, baseline=rb, n=int(msk.sum())))
        print(f"{CHANNEL_NAMES[i]:16s} {diag:11.4f} {om:9.4f} {omj:>16s} {diag-om:+8.4f} {rb:9.4f} {int(msk.sum()):4d}")
    res[tag] = rows
def meanmat(Dm):
    """对 50 个 tile 取均值，得到 23x23 的 D[i,j]（i=虚拟通道, j=官方掩膜通道）"""
    out = np.full((23, 23), np.nan)
    for i in range(23):
        for j in range(23):
            v = Dm[:, i, j]; m = ~np.isnan(v)
            if m.sum() > 0: out[i, j] = float(v[m].mean())
    return out
json.dump(dict(orig=res["orig"], flash=res["flash"],
               matrix_orig=meanmat(Dmat["orig"]).tolist(),
               matrix_flash=meanmat(Dmat["flash"]).tolist(),
               channels=CHANNEL_NAMES,
               note="D[i,j] = Dice(pred_i, official_mask_j); 特异性 = D[i,i] - max_j D[i,j]"),
          open(os.path.join(OUT, "supp_08b_official_specificity.json"), "w"), indent=1)
print("\n-> supp_08b_official_specificity.json")
