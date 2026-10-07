"""补充实验 S9-B：HEMIT 尺度扫描 —— 结论是否为尺度错配的产物？

S9-A 用 DAPI 核直径把 HEMIT 标定为 ~0.284 µm/px（相对 GigaTIME 训练域 0.2302 为 1.23x），
但核直径假设本身有 ±25% 不确定度。所以这里直接扫：
  把 1024 原生 patch 缩放到 1024*s 后送进模型（s=0.5~2.0），
  等价于假设 HEMIT 尺度为 0.284/s µm/px；预测再缩回 1024 与 GT 比。
同时算"密度匹配 Dice"以消除不同尺度下预测密度漂移的影响，
并算 23x3 交叉通道矩阵看"特异性"随尺度的变化。
"""
import os, sys, glob, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"W:\虚拟细胞\repo")
import numpy as np, torch, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import gigatime_flash as GF, gigatime_orig as GO
from gigatime_flash import CHANNEL_NAMES

OUT = r"W:\虚拟细胞\analysis"; HEMIT = r"W:\虚拟细胞\data\HEMIT"
IMNET_M = np.asarray(GF.mean, dtype=np.float32); IMNET_S = np.asarray(GF.std, dtype=np.float32)
flash, device, _ = GF.load_model(weights_path=r"W:\虚拟细胞\models\gigatime-flash\model.pth")
M0 = 0.2840     # S9-A 标定值
GIGA = 0.2302

def to_lab(x): return cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
def from_lab(l):
    l = l.copy(); l[:,:,0] = np.clip(l[:,:,0],0,100)
    l[:,:,1] = np.clip(l[:,:,1],-127,127); l[:,:,2] = np.clip(l[:,:,2],-127,127)
    return np.clip(cv2.cvtColor(l.astype(np.float32), cv2.COLOR_LAB2RGB),0,1)
off = sorted(glob.glob(r"W:\虚拟细胞\data\sample_test_data\**\*_he.png", recursive=True))
A = np.concatenate([to_lab(np.asarray(Image.open(p).convert("RGB").resize((256,256),Image.BILINEAR),dtype=np.float32)/255.).reshape(-1,3) for p in off[:40]],0)
TM, TS = A.mean(0), A.std(0)
def reinhard(x):
    l = to_lab(x); m = l.reshape(-1,3).mean(0); s_ = l.reshape(-1,3).std(0)+1e-6
    return from_lab((l-m)/s_*TS+TM)

def run(model, ten, win=256):
    b,c,h,w = ten.shape
    ph = (-h) % win; pw = (-w) % win
    t = torch.nn.functional.pad(ten, (0,pw,0,ph), mode="replicate") if (ph or pw) else ten
    with torch.no_grad():
        out = GO.predict_patch_orig(model, t, device, win)
    return torch.sigmoid(out[:,:,:h,:w]).squeeze(0).cpu().numpy()

def dice(pb, gb):
    s_ = pb.sum() + gb.sum()
    return float((2*(pb & gb).sum() + 1e-6)/(s_ + 1e-6)) if s_ > 0 else np.nan

GTCH = {"DAPI": 2, "CD3": 1, "CK": 0}
SCALES = [0.5, 0.707, 1.0, 1.414, 2.0]
ins = sorted(glob.glob(os.path.join(HEMIT, "test", "input", "*.tif")))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng = np.random.default_rng(11)
ins = [ins[i] for i in sorted(rng.choice(len(ins), size=min(N, len(ins)), replace=False))]
print(f"HEMIT patch {len(ins)} 张；尺度 {SCALES}")
print(f"标定 m0={M0} µm/px -> 各 s 对应的等效窗口视场 " +
      ", ".join(f"s={s_}:{256*M0/s_:.0f}µm" for s_ in SCALES) + f"  (训练域 {256*GIGA:.1f}µm)")
print(f"最优 s（若标定正确）= {M0/GIGA:.2f}\n", flush=True)

Z = 1024
M = {s_: np.zeros((23, 3)) for s_ in SCALES}      # Dice(pred_ch, GT_g)
MD = {s_: np.zeros((23, 3)) for s_ in SCALES}     # 密度匹配 Dice
cnt = {s_: np.zeros((23, 3)) for s_ in SCALES}
t0 = time.time()
for k, p in enumerate(ins):
    lb = os.path.join(HEMIT, "test", "label", os.path.basename(p))
    raw = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/255.
    lbr = np.asarray(Image.open(lb).convert("RGB"), dtype=np.uint8)
    he = reinhard(raw)
    GT = {g: (cv2.threshold(lbr[:,:,lc],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] > 0) for g, lc in GTCH.items()}
    for s_ in SCALES:
        sz = int(round(Z*s_))
        im = np.asarray(Image.fromarray((np.clip(he,0,1)*255).astype(np.uint8)).resize((sz,sz), Image.BILINEAR), dtype=np.float32)/255.
        ten = torch.from_numpy(((im-IMNET_M)/IMNET_S).transpose(2,0,1)).unsqueeze(0)
        pr = run(flash, ten)
        pr = np.moveaxis(pr, 0, -1) if pr.shape[0] == 23 else pr
        if pr.shape[0] != Z:
            pr = np.stack([np.asarray(Image.fromarray((pr[:,:,i]*255).astype(np.uint8)).resize((Z, Z), Image.BILINEAR))/255.
                           for i in range(23)], -1)
        for gi, (g, gl) in enumerate(GT.items()):
            gd = float(gl.mean())
            for ci in range(23):
                P = pr[:,:,ci]; pb = P > 0.5
                d = dice(pb, gl)
                if not np.isnan(d): M[s_][ci, gi] += d; cnt[s_][ci, gi] += 1
                thr = float(np.quantile(P, 1-gd))
                dm = dice(P > thr, gl)
                if not np.isnan(dm): MD[s_][ci, gi] += dm
    if (k+1) % 50 == 0: print(f"  {k+1}/{len(ins)} {time.time()-t0:.0f}s", flush=True)

for s_ in SCALES:
    M[s_] /= np.maximum(cnt[s_], 1); MD[s_] /= np.maximum(cnt[s_], 1)
    assert len(cnt[s_]) == 23

print("\n" + "="*100)
print("  配对通道的 Dice 随尺度变化")
print("="*100)
print(f"{'GT 通道':8s} {'虚拟通道':14s} " + " ".join(f"{'s='+str(s_):>10s}" for s_ in SCALES))
res = {}
for g, chn in (("DAPI","DAPI"), ("CK","CK_1:150"), ("CD3","CD3_1:1000")):
    gi = list(GTCH).index(g); ci = CHANNEL_NAMES.index(chn)
    row0 = [M[s_][ci, gi] for s_ in SCALES]; row1 = [MD[s_][ci, gi] for s_ in SCALES]
    print(f"{g:8s} {chn:14s} " + " ".join(f"{v:10.4f}" for v in row0) + "   (默认阈值)")
    print(f"{'':8s} {'':14s} " + " ".join(f"{v:10.4f}" for v in row1) + "   (密度匹配)")
    res[g] = dict(channel=chn, dice_default=row0, dice_density_matched=row1)

print("\n" + "="*100)
print("  交叉通道特异性随尺度变化（Dice(pred_ch, GT_g) 中，配对通道的排名）")
print("="*100)
for g in GTCH:
    gi = list(GTCH).index(g); chn = {"DAPI":"DAPI","CK":"CK_1:150","CD3":"CD3_1:1000"}[g]
    ci = CHANNEL_NAMES.index(chn)
    print(f"\n  --- GT {g} ---")
    for s_ in SCALES:
        col = M[s_][:, gi]
        order = np.argsort(-col)
        rank = int(np.where(order == ci)[0][0]) + 1
        top = [(CHANNEL_NAMES[j], col[j]) for j in order[:3]]
        spec = col[ci] - max(col[j] for j in range(23) if j != ci)
        print(f"    s={s_:<6} 配对 Dice={col[ci]:.4f} 排名={rank:2d}/23 特异性={spec:+.4f} | "
              f"前三: " + ", ".join(f"{a}({b:.3f})" for a, b in top))
        res.setdefault("specificity_vs_scale", {}).setdefault(g, {})[str(s_)] = dict(
            paired_dice=float(col[ci]), rank=rank, specificity=float(spec),
            top3=[[a, float(b)] for a, b in top])
json.dump(dict(m0=M0, giga=GIGA, scales=SCALES, n_patches=len(ins), result=res,
               matrix={str(s_): M[s_].tolist() for s_ in SCALES},
               matrix_density_matched={str(s_): MD[s_].tolist() for s_ in SCALES},
               channels=CHANNEL_NAMES, gt_channels=list(GTCH)),
          open(os.path.join(OUT, "supp_09b_hemit_scale_sweep.json"), "w"), indent=1)
print("\n-> supp_09b_hemit_scale_sweep.json")
