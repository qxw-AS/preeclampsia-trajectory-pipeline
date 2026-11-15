"""核对 GSE204835 / GSE290578 的样本构成与分组信息"""
import gzip
import os
import re
import pandas as pd

D = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01\data\geo"


def read_series_matrix(path):
    """从 series_matrix 提取样本标题与 characteristics 行"""
    meta = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_title"):
                meta["title"] = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
                key = vals[0].split(":")[0].strip() if vals and ":" in vals[0] else "char"
                meta.setdefault("chars", []).append((key, vals))
            elif line.startswith("!Sample_geo_accession"):
                meta["gsm"] = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
    return meta


print("=" * 70)
print("GSE204835")
print("=" * 70)
m = read_series_matrix(os.path.join(D, "GSE204835_series_matrix.txt.gz"))
print("samples:", len(m.get("gsm", [])))
print("first titles:", m.get("title", [])[:4])
for key, vals in m.get("chars", []):
    uniq = pd.Series(vals).value_counts()
    print(f"  [{key}] -> {dict(uniq)}")

cnt_path = os.path.join(D, "GSE204835_counts.csv.gz")
df = pd.read_csv(cnt_path, index_col=0, nrows=5)
print("counts shape (first rows):", df.shape)
print("columns (first 6):", list(df.columns[:6]))
print("index (first 5):", list(df.index[:5]))
full_cols = pd.read_csv(cnt_path, index_col=0, usecols=[0])
with gzip.open(cnt_path, "rt") as f:
    n_rows = sum(1 for _ in f) - 1
print(f"counts matrix: {n_rows} genes x {len(df.columns)} samples")

print()
print("=" * 70)
print("GSE290578")
print("=" * 70)
fl = os.path.join(D, "GSE290578_filelist.txt")
rows = []
with open(fl, encoding="utf-8", errors="replace") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 5 and parts[0] == "File":
            rows.append({"file": parts[1], "size": int(parts[3])})
names = [r["file"] for r in rows]
samples = sorted({re.sub(r"_(barcodes|features|matrix)\.tsv\.gz$", "", n)
                  for n in names if n.endswith((".tsv.gz", ".mtx.gz"))})
print("files:", len(names), "| sample prefixes:", len(samples))
print("samples:", samples)
print("total RAW size: %.1f MB" % (sum(r["size"] for r in rows) / 1024 / 1024))

m2 = read_series_matrix(os.path.join(D, "GSE290578_series_matrix.txt.gz"))
print("series matrix samples:", len(m2.get("gsm", [])))
print("titles:", m2.get("title", []))
for key, vals in m2.get("chars", []):
    print(f"  [{key}] -> {dict(pd.Series(vals).value_counts())}")
