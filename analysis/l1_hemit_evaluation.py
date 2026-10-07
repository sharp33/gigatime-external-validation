"""T1 定稿：正确通道映射 DAPI->ch2, CD3->ch1, CK->ch0"""
import os, sys, glob, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"W:\虚拟细胞\repo")
import numpy as np, torch, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
from scipy import stats
from gigatime_flash import load_model, predict_patch, CHANNEL_NAMES, mean, std

R=r"W:\虚拟细胞\data\HEMIT"; OUT=r"W:\虚拟细胞\analysis"
model, device, _ = load_model(weights_path=r"W:\虚拟细胞\models\gigatime-flash\model.pth")
off=sorted(glob.glob(r"W:\虚拟细胞\data\sample_test_data\**\*_he.png",recursive=True))
def to_lab(x): return cv2.cvtColor(x.astype(np.float32),cv2.COLOR_RGB2LAB)
def from_lab(l):
    l=l.copy(); l[:,:,0]=np.clip(l[:,:,0],0,100); l[:,:,1]=np.clip(l[:,:,1],-127,127); l[:,:,2]=np.clip(l[:,:,2],-127,127)
    return np.clip(cv2.cvtColor(l.astype(np.float32),cv2.COLOR_LAB2RGB),0,1)
A=np.concatenate([to_lab(np.asarray(Image.open(p).convert("RGB").resize((256,256),Image.BILINEAR),dtype=np.float32)/255.).reshape(-1,3) for p in off[:40]],0)
TM,TS=A.mean(0),A.std(0)
def reinhard(x):
    l=to_lab(x); m=l.reshape(-1,3).mean(0); s=l.reshape(-1,3).std(0)+1e-6
    return from_lab((l-m)/s*TS+TM)
def otsu8(ch):
    t,_=cv2.threshold(ch,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU); return ch>t
def dice_b(pb,gb,block=8):
    H,W=gb.shape; hb,wb=H//block,W//block
    if hb==0 or wb==0: return np.nan
    p=pb[:hb*block,:wb*block].reshape(hb,block,wb,block).transpose(0,2,1,3).reshape(-1,block*block)
    g=gb[:hb*block,:wb*block].reshape(hb,block,wb,block).transpose(0,2,1,3).reshape(-1,block*block)
    k=g.any(1)
    if k.sum()==0: return np.nan
    return float((2*(p[k]&g[k]).sum()+1e-6)/(p[k].sum()+g[k].sum()+1e-6))

MAP={"DAPI":(0,2),"CD3":(13,1),"CK":(16,0)}   # 模型通道, label通道
N=int(sys.argv[1]) if len(sys.argv)>1 else 300
ins=sorted(glob.glob(os.path.join(R,"test","input","*.tif")))[:N]
print(f"test patches: {len(ins)}  (原生1024 + Reinhard)")
rng=np.random.default_rng(0)
res={k:dict(dice=[],base=[],pd=[],gd=[]) for k in MAP}
for i,p in enumerate(ins):
    lb_p=os.path.join(R,"test","label",os.path.basename(p))
    he=np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.
    lbr=np.asarray(Image.open(lb_p).convert("RGB"),dtype=np.uint8)
    he=reinhard(he)
    ten=torch.from_numpy(((he-mean)/std).transpose(2,0,1)).unsqueeze(0)
    pr=torch.sigmoid(predict_patch(model,ten,device)).squeeze(0).cpu().numpy()
    pr=np.moveaxis(pr,0,-1) if pr.shape[0]==23 else pr
    for nm,(mc,lc) in MAP.items():
        pb=pr[:,:,mc]>0.5
        gb=otsu8(lbr[:,:,lc])
        d=dice_b(pb,gb)
        if not np.isnan(d): res[nm]["dice"].append(d)
        rb=rng.random(gb.shape)<float(gb.mean())
        db=dice_b(rb,gb)
        if not np.isnan(db): res[nm]["base"].append(db)
        res[nm]["pd"].append(float(pb.mean())); res[nm]["gd"].append(float(gb.mean()))
    if (i+1)%100==0: print(f"  {i+1}/{len(ins)}")

print("\n" + "="*74)
print("  L1 外部验证：GigaTIME-Flash 在 HEMIT（独立机构，结肠癌，mIHC）")
print("="*74)
print(f"{'通道':6s} {'Dice':>8s} {'随机基线':>10s} {'提升':>8s} {'预测密度':>10s} {'真值密度':>10s}")
out={}
for nm in ["DAPI","CK","CD3"]:
    d=float(np.mean(res[nm]["dice"])); b=float(np.mean(res[nm]["base"]))
    pd=float(np.mean(res[nm]["pd"])); gd=float(np.mean(res[nm]["gd"]))
    print(f"{nm:6s} {d:8.3f} {b:10.3f} {d-b:+8.3f} {pd:10.4f} {gd:10.4f}")
    out[nm]=dict(dice=d, baseline=b, gain=d-b, pred_density=pd, gt_density=gd, n=len(res[nm]["dice"]))
out["_config"]=dict(size=1024, normalize="reinhard->GigaTIME-official-stats", split="test", n=N,
                    model="gigatime-flash", weights="prov-gigatime/gigatime-flash")
out["_paper_indomain_dice"]={"DAPI":0.72,"CK":0.35,"CD3":0.20,"note":"论文 Figure 2A 目测"}
json.dump(out, open(os.path.join(OUT,"t13_hemit_eval.json"),"w"), indent=1)
print("\n论文在域内参考: DAPI≈0.72, CK≈0.35, CD3≈0.20")
print("-> t13_hemit_eval.json")
