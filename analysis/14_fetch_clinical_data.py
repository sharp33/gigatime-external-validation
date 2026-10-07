"""B 级：拉取 TCGA-THCA 临床数据（cBioPortal API），供"虚拟组成轴的临床价值"分析使用"""
import json, os, sys, urllib.request
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"W:\虚拟细胞\analysis\l3"
P = os.path.join(OUT, "thca_rppa.txt")
lines = open(P, encoding="utf-8", errors="replace").read().strip().split("\n")
samples = lines[0].split("\t")[1:]
pats = sorted({s[:12] for s in samples})
print(f"RPPA 样本 {len(samples)}，病人 {len(pats)}")

ATTRS = ["SUBTYPE", "HISTOLOGICAL_DIAGNOSIS", "AJCC_PATHOLOGIC_TUMOR_STAGE", "TUMOR_STAGE",
         "AGE", "SEX", "OS_MONTHS", "OS_STATUS", "DSS_MONTHS", "DSS_STATUS",
         "LYMPH_NODES_EXAMINED_POSITIVE", "RACE", "ETHNICITY", "VITAL_STATUS",
         "PFS_MONTHS", "PFS_STATUS", "GRADE", "TUMOR_TISSUE_SITE", "CANCER_TYPE_DETAILED"]
studies = ["thca_tcga_pan_can_atlas_2018", "thca_tcga"]

def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())

best = None
for st in studies:
    try:
        d = post("https://www.cbioportal.org/api/clinical-data/fetch",
                 {"clinicalDataType": "PATIENT", "attributeIds": ATTRS, "ids": pats})
    except Exception as e:
        print(f"  {st}: FAILED {e}"); continue
    got = {x["patientId"] for x in d} if d else set()
    print(f"  {st}: 返回 {len(d)} 条记录，覆盖 {len(got & set(pats))}/{len(pats)} 病人")
    if best is None or len(got & set(pats)) > best[1]:
        best = (st, len(got & set(pats)), d)

if best:
    st, n, d = best
    print(f"\n采用 {st}（覆盖 {n} 病人）")
    by = {}
    for x in d:
        by.setdefault(x["patientId"], {})[x["clinicalAttributeId"]] = x["value"]
    json.dump(dict(study=st, n_patients=len(by), data=by), open(os.path.join(OUT, "thca_clinical.json"), "w"), indent=1)
    from collections import Counter
    for k in ("SUBTYPE", "HISTOLOGICAL_DIAGNOSIS", "TUMOR_STAGE", "AJCC_PATHOLOGIC_TUMOR_STAGE", "OS_STATUS", "VITAL_STATUS"):
        c = Counter(v.get(k, "(缺)") for v in by.values())
        print(f"  {k:32s} {dict(list(c.most_common(8)))}")
    print("\n-> thca_clinical.json")
