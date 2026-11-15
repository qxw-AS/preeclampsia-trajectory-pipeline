"""GEO 断点续传下载器：小文件优先，失败重试"""
import os
import sys
import time
import requests

OUT = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01\data\geo"
os.makedirs(OUT, exist_ok=True)

FILES = [
    ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE204nnn/GSE204835/suppl/GSE204835_Human_paraffin-embedded_placental_tissue_counts.csv.gz",
     "GSE204835_counts.csv.gz"),
    ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE204nnn/GSE204835/matrix/GSE204835_series_matrix.txt.gz",
     "GSE204835_series_matrix.txt.gz"),
    ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE290nnn/GSE290578/suppl/filelist.txt",
     "GSE290578_filelist.txt"),
    ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE290nnn/GSE290578/matrix/GSE290578_series_matrix.txt.gz",
     "GSE290578_series_matrix.txt.gz"),
    ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE75nnn/GSE75010/suppl/GSE75010_complete_dataset.csv.gz",
     "GSE75010_complete_dataset.csv.gz"),
]


def download(url, name, retries=6):
    path = os.path.join(OUT, name)
    part = path + ".part"
    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code not in (200, 206):
                    print(f"[{name}] HTTP {r.status_code}", flush=True)
                    time.sleep(5)
                    continue
                mode = "ab" if (have and r.status_code == 206) else "wb"
                if mode == "wb":
                    have = 0
                total = r.headers.get("Content-Length")
                total = int(total) + have if total else None
                t0 = time.time()
                with open(part, mode) as f:
                    for chunk in r.iter_content(65536):
                        f.write(chunk)
                        have += len(chunk)
                if os.path.exists(part):
                    os.replace(part, path)
                size = os.path.getsize(path) / 1024 / 1024
                print(f"[{name}] DONE {size:.2f} MB in {time.time()-t0:.0f}s", flush=True)
                return True
        except Exception as e:
            print(f"[{name}] attempt {attempt} failed: {type(e).__name__} {str(e)[:60]}", flush=True)
            time.sleep(3)
    print(f"[{name}] FAILED after {retries} attempts", flush=True)
    return False


if __name__ == "__main__":
    for url, name in FILES:
        download(url, name)
    print("ALL FINISHED", flush=True)
