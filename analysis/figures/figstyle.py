# -*- coding: utf-8 -*-
"""统一绘图风格 + §9.1 几何审计器"""
import os, sys
sys.path.insert(0, r"C:\Users\PC\.agents\skills\figure-style")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from kernel import apply_figure_style

OUT = r"W:\虚拟细胞\figures"
os.makedirs(OUT, exist_ok=True)

# ---------------- 统一色板（全 12 图共用） ----------------
C_FOCAL = "#1f6fb4"      # 焦点：虚拟通道 / 正面结论 / 有效
C_LIGHT = "#8fbcd9"      # 同色系浅色：辅助系列
C_CTRL  = "#d1622f"      # 负对照 / 背景通道 / 失败
C_GREY  = "#9a9a9a"      # 中性对照
C_DARK  = "#2f2f2f"      # 深色文字
C_ALT   = "#7d5ba6"      # 第三系列（紫）
C_GOLD  = "#c9a227"      # 第四系列（金）

DIV = LinearSegmentedColormap.from_list("div_bo", ["#2b6ca3", "#7fb2d4", "#f4f4f4", "#e8a97e", "#c15a1f"])
SEQ = "magma"

FS_BASE, FS_SEC, FS_TICK = 8.0, 7.0, 6.0
def apply():
    apply_figure_style(frame="open", sizes=(FS_BASE, FS_SEC, FS_TICK))
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": C_DARK, "axes.labelcolor": C_DARK,
        "xtick.color": C_DARK, "ytick.color": C_DARK,
        "font.size": FS_BASE,
    })

def plabel(ax, letter, dx=None, dy=None):
    """稳健的面板字母：放在轴框外左上，按需自动偏移"""
    fig = ax.figure
    fig.canvas.draw()
    ab = ax.get_window_extent(fig.canvas.get_renderer())
    fb = fig.bbox
    x = ab.x0 - 0.02 * fb.width if dx is None else ab.x0 + dx * ab.width
    y = ab.y1 + 0.010 * fb.height if dy is None else ab.y1 + dy * ab.height
    x = max(x, fb.x0 + 0.004 * fb.width)
    y = min(y, fb.y1 - 0.012 * fb.height)
    fig.text(x / fb.width, y / fb.height, letter, fontweight="bold",
             fontsize=FS_BASE + 3, va="bottom", ha="left")

def audit(fig, name="", raise_on=("text-text", "out-of-fig", "text-axes")):
    """§9.1 几何审计：文字-文字重叠、文字出图、文字压轴框（排除刻度标签）"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = []
    for t in fig.findobj(mpl.text.Text):
        try:
            if not t.get_visible() or not t.get_text().strip(): continue
            bb = t.get_window_extent(r)
            if bb.width <= 0 or bb.height <= 0: continue
            texts.append((t, bb))
        except Exception:
            pass
    tick, skip = set(), set()
    for ax in fig.axes:
        xlo, xhi = sorted(ax.get_xlim()); ylo, yhi = sorted(ax.get_ylim())
        for t in list(ax.get_xticklabels(which="both")):
            tick.add(id(t))
            try:
                v = float(t.get_position()[0])
                if not (xlo - 1e-9 <= v <= xhi + 1e-9): skip.add(id(t))
            except Exception: pass
        for t in list(ax.get_yticklabels(which="both")):
            tick.add(id(t))
            try:
                v = float(t.get_position()[1])
                if not (ylo - 1e-9 <= v <= yhi + 1e-9): skip.add(id(t))
            except Exception: pass
    texts = [(t, b) for t, b in texts if id(t) not in skip]
    issues = []
    for i, (a, ba) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            if a is b: continue
            if id(a) in tick and id(b) in tick: continue   # 刻度间重叠另属布局问题
            if ba.overlaps(bb):
                # 允许：完全包含（同一多行文本的分解）与相同文字
                inter = (min(ba.x1, bb.x1) - max(ba.x0, bb.x0)) * (min(ba.y1, bb.y1) - max(ba.y0, bb.y0))
                small = min(ba.width * ba.height, bb.width * bb.height)
                if small > 0 and inter / small > 0.92: continue
                issues.append(("text-text", a.get_text()[:40].replace("\n", "|"),
                               b.get_text()[:40].replace("\n", "|")))
    fb = fig.bbox
    for t, bt in texts:
        if id(t) in tick: continue          # 刻度标签由 bbox_inches='tight' 收纳，不算缺陷
        tolx, toly = 0.035 * fb.width, 0.035 * fb.height   # bbox_inches='tight' 会收纳，超过 3.5% 才算缺陷
        if bt.x0 < fb.x0 - tolx or bt.x1 > fb.x1 + tolx or bt.y0 < fb.y0 - toly or bt.y1 > fb.y1 + toly:
            issues.append(("out-of-fig", t.get_text()[:40].replace("\n", "|"), ""))
    for ax in fig.axes:
        ab = ax.get_window_extent(r)
        for t, bt in texts:
            if id(t) in tick or t.axes is not ax: continue
            if t in (ax.title, ax.xaxis.label, ax.yaxis.label): continue
            inside = ab.x0 <= bt.x0 and bt.x1 <= ab.x1 and ab.y0 <= bt.y0 and bt.y1 <= ab.y1
            if inside or not bt.overlaps(ab): continue
            issues.append(("text-axes", t.get_text()[:40].replace("\n", "|"), ax.get_title()[:24]))
    if issues:
        print(f"  [AUDIT {name}] {len(issues)} 处:")
        seen = set()
        for k, a, b in issues:
            key = (k, a, b)
            if key in seen: continue
            seen.add(key)
            print(f"     {k:10s} | {a!r} <> {b!r}")
    else:
        print(f"  [AUDIT {name}] clean")
    return issues

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    print("  ->", name)
