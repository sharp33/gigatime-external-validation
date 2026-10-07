"""T1.6 公平性检验：CD3 密度校准（把预测阈值调到与真值密度一致）"""
import os, sys, glob, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"W:\虚拟细胞\repo")
import numpy as np, torch, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
from scipy import stats
from gigatime_flash import load_model, predict_patch, mean, std

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

MAP={"DAPI":(0,2),"CD3":(13,1),"CK":(16,0)}
ins=sorted(glob.glob(os.path.join(R,"test","input","*.tif")))[:150]
acc={k:{"fix05":[], "cal":[]} for k in MAP}
for p in ins:
    lbr=np.asarray(Image.open(os.path.join(R,"test","label",os.path.basename(p))).convert("RGB"),dtype=np.uint8)
    he=reinhard(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.)
    ten=torch.from_numpy(((he-mean)/std).transpose(2,0,1)).unsqueeze(0)
    pr=torch.sigmoid(predict_patch(model,ten,device)).squeeze(0).cpu().numpy()
    pr=np.moveaxis(pr,0,-1) if pr.shape[0]==23 else pr
    for nm,(mc,lc) in MAP.items():
        prob=pr[:,:,mc]
        gb=otsu8(lbr[:,:,lc])
        d1=dice_b(prob>0.5, gb)
        if not np.isnan(d1): acc[nm]["fix05"].append(d1)
        # 密度校准：阈值取概率分位数 = 1 - GT 密度
        q=np.quantile(prob, 1.0-float(gb.mean()))
        d2=dice_b(prob>q, gb)
        if not np.isnan(d2): acc[nm]["cal"].append(d2)

print("="*66)
print("  T1.6 阈值公平性检验（n=150 test patch）")
print("="*66)
print(f"{'通道':6s} {'固定0.5':>10s} {'密度校准':>10s} {'变化':>9s}")
for nm in ["DAPI","CK","CD3"]:
    a=float(np.mean(acc[nm]["fix05"])); b=float(np.mean(acc[nm]["cal"]))
    print(f"{nm:6s} {a:10.3f} {b:10.3f} {b-a:+9.3f}")
json.dump({nm:{"fix05":float(np.mean(acc[nm]["fix05"])),"calibrated":float(np.mean(acc[nm]["cal"]))} for nm in MAP},
          open(os.path.join(OUT,"t16_threshold_fairness.json"),"w"), indent=1)
print("\n-> t16_threshold_fairness.json")
