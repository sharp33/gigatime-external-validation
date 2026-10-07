"""补充实验 S9-A：HEMIT 物理尺度（µm/px）的数据驱动标定

问题：HEMIT 的 patch 是 1024x1024 TIFF，dpi=(1,1)，**没有任何分辨率元数据**；
文件名里的坐标 [x,y] 不规则（15 个位点 x 7x9 patch），也推不出尺度。
即我们此前是把未知 µm/px 的图像喂进按 0.2302 µm/px 训练的模型。

标定思路：**两个数据集都有 DAPI**。
  - GigaTIME 官方样例：comet.npy 通道 0 = DAPI，556 px 对应 128 µm -> 0.2302 µm/px（已知）
  - HEMIT：label 通道 2 = DAPI（t13 确认的映射）
用**完全相同的分割流程**测细胞核等效直径的分布，用已知尺度定标未知尺度。
再做一条独立交叉校验：H&E 灰度图上的 LoG blob 尺度（与染色无关的纹理尺度）。
"""
import os, sys, glob, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2
from scipy import ndimage as ndi

OUT = r"W:\虚拟细胞\analysis"
GIGA_UMPP = 128.0/556.0

def nuclei_diameters(dapi, lo=20, hi=3000, open_r=1):
    """统一流程：百分位归一化 -> Otsu -> 形态学开 -> 去小连通域 -> 等效直径"""
    x = dapi.astype(np.float32)
    p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
    x = np.clip((x - p1) / (p99 - p1 + 1e-6), 0, 1)
    u8 = (x*255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = u8 > t
    if open_r > 0:
        k = np.ones((2*open_r+1, 2*open_r+1), np.uint8)
        bw = cv2.morphologyEx(bw.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    bw = ndi.binary_fill_holes(bw)
    lab, n = ndi.label(bw)
    if n == 0: return np.array([])
    areas = ndi.sum(bw, lab, index=np.arange(1, n+1))
    areas = areas[(areas >= lo) & (areas <= hi)]
    return 2*np.sqrt(areas/np.pi)

print("="*84)
print("  A) DAPI 细胞核等效直径标定")
print("="*84)
# --- GigaTIME 官方（已知 0.2302 µm/px）---
D = r"W:\虚拟细胞\data\sample_test_data"
fs = sorted(glob.glob(os.path.join(D, "**", "*_comet.npy"), recursive=True))
gd = []
for p in fs:
    a = np.load(p)[:, :, 0]
    gd.append(nuclei_diameters(a))
gd = np.concatenate([d for d in gd if len(d)])
print(f"  GigaTIME 官方样例 DAPI: n={len(gd)} 核  median={np.median(gd):.2f} px  "
      f"IQR=[{np.percentile(gd,25):.2f}, {np.percentile(gd,75):.2f}]")
ref_um = float(np.median(gd)) * GIGA_UMPP
print(f"  -> 该数据集中位核直径 = {ref_um:.2f} µm（@ 0.2302 µm/px）")

# --- HEMIT ---
H = r"W:\虚拟细胞\data\HEMIT"
ins = sorted(glob.glob(os.path.join(H, "test", "label", "*.tif")))
rng = np.random.default_rng(0)
sel = rng.choice(len(ins), size=min(120, len(ins)), replace=False)
hd = []
for i in sel:
    lb = np.asarray(Image.open(ins[i]).convert("RGB"), dtype=np.uint8)
    hd.append(nuclei_diameters(lb[:, :, 2]))
hd = np.concatenate([d for d in hd if len(d)])
med_px = float(np.median(hd))
print(f"\n  HEMIT DAPI: n={len(hd)} 核（120 张 label）  median={med_px:.2f} px  "
      f"IQR=[{np.percentile(hd,25):.2f}, {np.percentile(hd,75):.2f}]")
mpp = ref_um / med_px
print(f"  -> 推定 HEMIT 尺度 = {ref_um:.2f} µm / {med_px:.2f} px = **{mpp:.4f} µm/px**")
print(f"  -> 相对 GigaTIME 训练域 0.2302 的比值 = {mpp/GIGA_UMPP:.2f}x")
print(f"  -> 1024 px patch 对应 {1024*mpp:.0f} µm；模型 256 窗口对应 {256*mpp:.0f} µm "
      f"(GigaTIME 训练域为 {256*GIGA_UMPP:.1f} µm)")

# --- 用比值反推的等效裁剪尺寸 ---
N_IN = int(round(256 * GIGA_UMPP / mpp))
print(f"  -> 若要复现训练域视场，应从原生图裁 {N_IN} px 上采样到 256")

# --- 敏感性：核直径取不同分位数 ---
print("\n  敏感性（核直径取不同统计量时的 mpp）：")
for q in (25, 50, 75):
    v = float(np.percentile(hd, q)); rv = float(np.percentile(gd, q))
    print(f"    Q{q}: HEMIT {v:6.2f} px, 官方 {rv:6.2f} px -> mpp={rv*GIGA_UMPP/v:.4f} µm/px")

PCTS = [5, 10, 25, 50, 75, 90, 95]
json.dump(dict(gigatime_median_nucleus_px=float(np.median(gd)), ref_nucleus_um=ref_um,
               hemit_median_nucleus_px=med_px, hemit_mpp_estimate=float(mpp),
               ratio_to_gigatime=float(mpp/GIGA_UMPP), N_IN_for_training_fov=int(N_IN),
               n_gigatime_nuclei=int(len(gd)), n_hemit_nuclei=int(len(hd)),
               gigatime_pct={str(p): float(np.percentile(gd, p)) for p in PCTS},
               hemit_pct={str(p): float(np.percentile(hd, p)) for p in PCTS},
               mpp_sensitivity={str(q): float(np.percentile(gd, q)*GIGA_UMPP/np.percentile(hd, q))
                                for q in (25, 50, 75)}),
          open(os.path.join(OUT, "supp_09a_hemit_scale.json"), "w"), indent=1)
print("\n-> supp_09a_hemit_scale.json")
