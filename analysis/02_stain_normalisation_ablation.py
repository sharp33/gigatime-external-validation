"""补充实验 S2：预处理（染色归一化）消融 —— ORION-CRC 单细胞验证

三个变体（同一批 tile、同一随机种子、同一批细胞）：
  reinhard : 全片 LAB 统计 -> GigaTIME 官方 H&E LAB 统计（主分析口径）
  none     : 不做任何归一化，直接 ImageNet 标准化
  tile     : 逐 tile 自归一化 -> GigaTIME 官方统计（已知会抹平片内异质性，作为负对照）

指标：细胞级 Spearman（原始 + 残差化特异性）
用法: python supp_02_preproc.py [npat] [ntile]
"""
import os, sys, glob, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"W:\虚拟细胞\repo")
import numpy as np, pandas as pd, torch, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from scipy import stats
from gigatime_flash import load_model, CHANNEL_NAMES, mean as IMNET_MEAN, std as IMNET_STD

D = r"W:\虚拟细胞\data\ORION_CRC"
OUT = r"W:\虚拟细胞\analysis\orion"
os.makedirs(OUT, exist_ok=True)

PAIR = {"Hoechst":"DAPI", "CD31":"CD34", "CD68":"CD68_1:100", "CD4":"CD4", "CD8a":"CD8",
        "CD20":"CD20", "PD-L1":"PD-L1", "CD3e":"CD3_1:1000", "PD-1":"PD-1_1:200",
        "Ki67":"Ki67_1:150", "Pan-CK":"CK_1:150", "E-cadherin":"CK_1:150", "SMA":"Actin-D"}
NEG = ["AF1","Argo550","CD45","FOXP3","CD45RO","CD163"]
VARIANTS = ["reinhard", "none", "tile"]

def to_lab(x): return cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
def from_lab(l):
    l = l.copy(); l[:,:,0] = np.clip(l[:,:,0],0,100)
    l[:,:,1] = np.clip(l[:,:,1],-127,127); l[:,:,2] = np.clip(l[:,:,2],-127,127)
    return np.clip(cv2.cvtColor(l.astype(np.float32), cv2.COLOR_LAB2RGB),0,1)

off = sorted(glob.glob(r"W:\虚拟细胞\data\sample_test_data\**\*_he.png", recursive=True))
A = np.concatenate([to_lab(np.asarray(Image.open(p).convert("RGB").resize((256,256),Image.BILINEAR),dtype=np.float32)/255.).reshape(-1,3) for p in off[:40]],0)
TM, TS = A.mean(0), A.std(0)
print(f"GigaTIME 官方 H&E LAB 目标: mean={TM.round(2)} std={TS.round(2)}")

import tiffslide
model, device, _ = load_model(weights_path=r"W:\虚拟细胞\models\gigatime-flash\model.pth")
CH = {n:i for i,n in enumerate(CHANNEL_NAMES)}

# ---- 自动发现可用病人 ----
pats = []
for f in sorted(glob.glob(os.path.join(D, "*_HE.ome.tif"))):
    pat = os.path.basename(f).replace("_HE.ome.tif","")
    if all(os.path.exists(os.path.join(D, f"{pat}_{s}")) for s in ("mask.ome.tif","cells.csv")):
        pats.append(pat)
NP = int(sys.argv[1]) if len(sys.argv)>1 else len(pats)
NTILE = int(sys.argv[2]) if len(sys.argv)>2 else 600
pats = pats[:NP]
print(f"可用病人 {len(pats)}: {pats}\n")

TARGET = 0.2302; UP = 256
summary = {}
CACHE = os.environ.get("SUPP02_CACHE", "1") == "1"
for pat in pats:
    pj = os.path.join(OUT, f"{pat}_preproc_ablation.json")
    have_npz = all(os.path.exists(os.path.join(OUT, f"{pat}_{v}_cells.npz")) for v in VARIANTS)
    if CACHE and os.path.exists(pj) and have_npz:
        d = json.load(open(pj, encoding="utf-8"))
        if set(d.get("variants", {}).keys()) == set(VARIANTS):
            summary[pat] = d["variants"]
            print(f"  [cache] {pat}: 已有结果，跳过推理 (n_cells="
                  f"{d['variants'][VARIANTS[0]]['n_cells']})", flush=True)
            continue
    print("="*84, flush=True)
    print(f"  {pat}", flush=True)
    print("="*84, flush=True)
    df = pd.read_csv(os.path.join(D, f"{pat}_cells.csv"))
    ts = tiffslide.TiffSlide(os.path.join(D, f"{pat}_HE.ome.tif"))
    W, H = ts.dimensions
    mk = None
    mp = None
    for k in ("tiffslide.mpp-x","openslide.mpp-x"):
        v = ts.properties.get(k)
        if v is not None:
            try: mp = float(v); break
            except Exception: pass
    if mp is None:
        mk = tiffslide.TiffSlide(os.path.join(D, f"{pat}_mask.ome.tif"))
        v = mk.properties.get("tiffslide.mpp-x"); mp = float(v) if v is not None else 0.325
    N_IN = int(round(UP*TARGET/mp)); scale = UP/N_IN
    cx = df.X_centroid.to_numpy(); cy = df.Y_centroid.to_numpy()
    rng2 = np.random.default_rng(7)
    picks = rng2.choice(len(df), size=min(NTILE*3, len(df)), replace=False)
    centers = {}
    for i in picks:
        centers[(int(cx[i]//N_IN)*N_IN, int(cy[i]//N_IN)*N_IN)] = True
    centers = list(centers.keys())[:NTILE]
    # 源统计（全片抽样）
    rng = np.random.default_rng(1); srcs = []
    for _ in range(40):
        x0 = int(rng.integers(0, max(1, W-512))); y0 = int(rng.integers(0, max(1, H-512)))
        t = np.asarray(ts.read_region((x0,y0),0,(512,512)).convert("RGB"), dtype=np.float32)/255.
        srcs.append(to_lab(t).reshape(-1,3))
    S = np.concatenate(srcs,0); SM, SS = S.mean(0), S.std(0)+1e-6
    print(f"  HE {W}x{H} mpp={mp:.4f} N_IN={N_IN} tiles={len(centers)} | 源 LAB {SM.round(1)}", flush=True)

    MARKERS = [k for k in PAIR] + [k for k in NEG if k in df.columns]
    R = {k: df[k].to_numpy().astype(np.float64) for k in MARKERS}
    V = {v: [] for v in VARIANTS}
    cellidx = []
    t0 = time.time(); used = 0
    for (tx, ty) in centers:
        if tx+N_IN > W or ty+N_IN > H: continue
        a = np.asarray(ts.read_region((tx,ty),0,(N_IN,N_IN)).convert("RGB"), dtype=np.float32)/255.
        lab = to_lab(a)
        sm, ss = lab.reshape(-1,3).mean(0), lab.reshape(-1,3).std(0)+1e-6
        imgs = {"reinhard": from_lab((lab-SM)/SS*TS+TM),
                "none": a,
                "tile": from_lab((lab-sm)/ss*TS+TM)}
        prs = {}
        for v in VARIANTS:
            im = np.asarray(Image.fromarray((np.clip(imgs[v],0,1)*255).astype(np.uint8)).resize((UP,UP),Image.BILINEAR), dtype=np.float32)/255.
            ten = torch.from_numpy(((im-IMNET_MEAN)/IMNET_STD).transpose(2,0,1)).unsqueeze(0)
            with torch.no_grad():
                p = torch.sigmoid(model(ten.to(device))).squeeze(0).cpu().numpy()
            prs[v] = np.moveaxis(p,0,-1) if p.shape[0]==23 else p
        m = (cx>=tx)&(cx<tx+N_IN)&(cy>=ty)&(cy<ty+N_IN)
        for i in np.where(m)[0]:
            u = int((cx[i]-tx)*scale); vv = int((cy[i]-ty)*scale)
            if 0<=u<UP and 0<=vv<UP:
                for v in VARIANTS: V[v].append(prs[v][vv,u,:])
                cellidx.append(int(i))
        used += 1
    print(f"  推理完成: {used} tiles, {len(V['reinhard'])} 细胞, {time.time()-t0:.0f}s", flush=True)
    if len(V["reinhard"]) < 200:
        print("  细胞太少，跳过"); continue
    cellidx = np.asarray(cellidx)
    assert len(cellidx) == len(V["reinhard"]), (len(cellidx), len(V["reinhard"]))
    X = np.stack([R[k][cellidx] for k in MARKERS], 1)   # 只取被抽样到的细胞，与 V 严格对齐
    # 保存
    for v in VARIANTS:
        np.savez(os.path.join(OUT, f"{pat}_{v}_cells.npz"), V=np.array(V[v]),
                 **{f"r_{k}": R[k][cellidx] for k in MARKERS})

    def zs(a): return (a-a.mean(0,keepdims=True))/(a.std(0,keepdims=True)+1e-9)
    def residualize(Aa, C):
        C1 = np.column_stack([np.ones(len(C)), C])
        B, *_ = np.linalg.lstsq(C1, Aa, rcond=None)
        return Aa - C1 @ B
    CHL = [c for c in CHANNEL_NAMES if c not in ("TRITC","Cy5")]

    patres = {}
    print(f"\n  {'变体':10s} {'标记':12s} {'配对通道':14s} {'配对rho':>9s} {'原始rho':>9s} {'其他最大':>9s} {'特异性':>9s}")
    for v in VARIANTS:
        Vv = np.array(V[v], dtype=np.float64)
        shared = np.column_stack([X.mean(1), Vv.mean(1)])
        Xr = residualize(zs(X), shared); Vr = residualize(zs(Vv), shared)
        Vraw = zs(Vv)
        Rm = np.zeros((len(MARKERS), len(CHL)))
        for i in range(len(MARKERS)):
            for j, c in enumerate(CHL):
                Rm[i,j] = stats.spearmanr(Vr[:, CH[ c]], Xr[:, i]).statistic
        det = []
        for mk in MARKERS:
            i = MARKERS.index(mk); chn = PAIR.get(mk)
            if chn is None:
                jm = int(np.argmax(np.abs(Rm[i])))
                det.append(dict(marker=mk, paired=None, best_channel=CHL[jm], best_rho=float(Rm[i,jm])))
                print(f"  {v:10s} {mk:12s} {'(负对照)':14s} {'-':>9s} {'-':>9s} {Rm[i,jm]:+9.3f} {'-':>9s}")
                continue
            j = CHL.index(chn); pv = float(Rm[i,j])
            rho_raw = float(stats.spearmanr(Vraw[:, CH[chn]], X[:, i]).statistic)
            om, om_j = max([(abs(Rm[i,k]), k) for k in range(len(CHL)) if k != j])
            det.append(dict(marker=mk, paired=chn, paired_rho=pv, raw_rho=rho_raw,
                            best_other=float(om), best_other_channel=CHL[om_j],
                            specificity=float(abs(pv)-om), specific=bool(abs(pv)>om)))
            print(f"  {v:10s} {mk:12s} {chn:14s} {pv:+9.3f} {rho_raw:+9.3f} {om:9.3f} {abs(pv)-om:+9.3f}"
                  f"  {'OK' if abs(pv)>om else 'FAIL'}")
        patres[v] = dict(n_cells=int(len(Vv)), residual_matrix=Rm.tolist(), channels=CHL,
                         markers=MARKERS, detail=det,
                         n_specific=int(sum(1 for d in det if d.get("specific"))),
                         n_paired=len(PAIR))
        print()
    summary[pat] = patres
    json.dump(dict(pat=pat, mpp=mp, N_IN=N_IN, n_tiles=used, variants=patres),
              open(os.path.join(OUT, f"{pat}_preproc_ablation.json"),"w"), indent=1)

# ---- 汇总 ----
print("\n" + "="*92, flush=True)
print("  预处理消融汇总：各变体下 13 个配对标记的残差化 rho（跨病人均值）", flush=True)
print("="*92, flush=True)
keys = list(PAIR.items())
hdr = f"{'标记':12s} {'配对通道':14s} " + " ".join(f"{v:>10s}" for v in VARIANTS)
print(hdr)
agg = {v: {} for v in VARIANTS}
for mk, ch in keys:
    line = f"{mk:12s} {ch:14s} "
    for v in VARIANTS:
        vals = []
        for pat, pr in summary.items():
            for d in pr[v]["detail"]:
                if d["marker"] == mk and d.get("paired_rho") is not None:
                    vals.append(d["paired_rho"])
        agg[v][mk] = dict(mean=float(np.mean(vals)) if vals else None, n=len(vals),
                          pos=int(sum(1 for x in vals if x > 0)))
        line += f"{np.mean(vals):+10.3f}" if vals else f"{'-':>10s}"
    print(line)
print()
print(f"{'特异性通过数':12s} {'':14s} " + " ".join(
    f"{np.mean([pr[v]['n_specific'] for pr in summary.values()]):>9.1f}/13" for v in VARIANTS))
json.dump(dict(patients=list(summary), variants=VARIANTS, aggregate=agg,
               n_specific_mean={v: float(np.mean([pr[v]['n_specific'] for pr in summary.values()])) for v in VARIANTS},
               per_patient=summary), open(os.path.join(OUT,"preproc_ablation_summary.json"),"w"), indent=1)
print("\n-> orion/preproc_ablation_summary.json")
