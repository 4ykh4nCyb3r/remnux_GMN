"""Rebuild labels.csv by re-querying MalwareBazaar only (fast).
Skips VT because we already know it works in memory — the bug was
in case-sensitive dict lookup on disk writeout."""
import csv
import sys
import time
from pathlib import Path
import requests

MB_API = "https://mb-api.abuse.ch/api/v1/"

def query_mb(sha256):
    try:
        r = requests.post(MB_API, data={"query": "get_info", "hash": sha256}, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("query_status") != "ok":
            return None
        entries = data.get("data", [])
        if not entries:
            return None
        sig = entries[0].get("signature")
        return sig.lower() if sig else None
    except Exception:
        return None

samples_dir = Path("database_formation/eval_unpacked")
hashes = sorted(p.name.lower() for p in samples_dir.iterdir()
                if p.is_file() and len(p.name) == 64)

rows = []
for i, sha in enumerate(hashes, 1):
    fam = query_mb(sha)
    if fam:
        rows.append((sha, fam, "mb"))
        print(f"[{i}/{len(hashes)}] {sha[:12]}... -> {fam}")
    else:
        rows.append((sha, "unknown", "unknown"))
        print(f"[{i}/{len(hashes)}] {sha[:12]}... -> unknown")
    time.sleep(0.3)

with open("labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sha256", "family", "source"])
    for r in rows:
        w.writerow(r)

print(f"\nWrote {len(rows)} rows to labels.csv")