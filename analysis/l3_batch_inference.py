"""L3 批量推理：对已下载的所有切片跑 GigaTIME-Flash"""
import os, sys, glob, subprocess, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
D = r"W:\虚拟细胞\data\TCGA_THCA"
OUT = r"W:\虚拟细胞\analysis\l3"
VPY = r"W:\虚拟细胞\.venv\Scripts\python.exe"
SCRIPT = r"W:\虚拟细胞\analysis\l3_run_gigatime.py"
N = sys.argv[1] if len(sys.argv) > 1 else "400"
slides = sorted(glob.glob(os.path.join(D, "*.svs")))
print(f"found {len(slides)} slides | tiles/slide = {N}")
t0 = time.time(); ok = 0
for i, s in enumerate(slides):
    name = os.path.basename(s).split(".")[0]
    if os.path.exists(os.path.join(OUT, f"{name}_vec.npy")):
        print(f"[{i+1}/{len(slides)}] skip (done)"); continue
    r = subprocess.run([VPY, SCRIPT, s, N], capture_output=True, text=True, encoding="utf-8", errors="replace")
    line = [l for l in (r.stdout or "").split("\n") if "mpp=" in l or "FAILED" in l]
    if line: print(f"[{i+1}/{len(slides)}] {line[-1][:110]}")
    else: print(f"[{i+1}/{len(slides)}] {name[:30]} -> NO OUTPUT; stderr={ (r.stderr or '')[-200:] }")
    if os.path.exists(os.path.join(OUT, f"{name}_vec.npy")): ok += 1
print(f"DONE {ok}/{len(slides)} in {(time.time()-t0)/60:.1f} min")
