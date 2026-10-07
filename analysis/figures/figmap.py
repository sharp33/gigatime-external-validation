# -*- coding: utf-8 -*-
"""图件映射的唯一来源：绘图脚本输出名 -> 投稿文件名"""

# 绘图脚本产物 -> 投稿命名
PUBLISH_MAP = {
    "Fig02_L1_HEMIT": "Figure2",
    "Fig03_L2_gse230424": "Figure3",
    "Fig04_L3_tcga": "Figure4",
    "Fig05_ORION_patients": "Figure5",
    "Fig06_negative_controls": "Figure6",
    "Fig07_summary": "Figure7",
    "Fig08_crossreactivity": "Figure8",
    "Fig09_predictive_cv": "Figure9",
    "Fig10_robustness": "Figure10",
    "Fig11_composition_recovery": "Figure11",
    "Fig12_official_specificity": "Figure12",
    "Fig13_hemit_scale": "Figure13",
    "Fig14_clinical": "Figure14",
}

# 稿件内嵌图：图号 -> submit/Figures 下的文件名
MANUSCRIPT_FIGFILE = {1: "Figure1.png", 5: "Figure5.png", 6: "Figure6.png"}
for _k in range(2, 15):
    MANUSCRIPT_FIGFILE.setdefault(_k, "Figure%d.png" % _k)
