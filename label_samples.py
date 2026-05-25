"""
label_samples.py — Label malware samples by family.

Pipeline:
  1. (Optional, skipped by default if --no_mb) Query MalwareBazaar for each
     sha256 — free, no key. Falls back automatically if MB returns errors.
  2. For samples MB didn't resolve, query VirusTotal (free API key required).
     Run AVClass on each VT report individually to get a canonical family.
  3. Write labels.csv with columns: sha256, family, source.
     Writeout is INCREMENTAL — every successful label is appended immediately,
     so if VT takes 25 min and crashes at minute 24, no progress is lost.
  4. Print the family distribution.

Usage:
    export VT_API_KEY=<your_key>
    python label_samples.py \\
        --samples_dir database_formation/eval_unpacked \\
        --output labels.csv \\
        --no_mb            # skip MalwareBazaar entirely, go straight to VT

Requirements:
    pip install requests avclass-malicialab
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import requests

MB_API = "https://mb-api.abuse.ch/api/v1/"
VT_API = "https://www.virustotal.com/api/v3/files/"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# MalwareBazaar
# ---------------------------------------------------------------------------

def query_mb(sha256):
    """Return family name (lowercased) from MalwareBazaar, or None."""
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
    except Exception as e:
        print(f"   [!] MB error for {sha256[:12]}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# VirusTotal
# ---------------------------------------------------------------------------

def query_vt(sha256, api_key):
    """Return raw VT report dict, or None on failure."""
    try:
        r = requests.get(
            VT_API + sha256,
            headers={"x-apikey": api_key},
            timeout=20,
        )
        if r.status_code == 404:
            return None
        if r.status_code == 429:
            print("   [!] VT rate-limited, sleeping 60s...", file=sys.stderr)
            time.sleep(60)
            return query_vt(sha256, api_key)
        if r.status_code != 200:
            print(f"   [!] VT HTTP {r.status_code} for {sha256[:12]}", file=sys.stderr)
            return None
        return r.json()
    except Exception as e:
        print(f"   [!] VT error for {sha256[:12]}: {e}", file=sys.stderr)
        return None


def vt_to_avclass_input(sha256, vt_json):
    """
    Build the AVClass input record for one sample.
    Critical: we put the sha256 in md5/sha1/sha256 fields and AVClass uses
    whichever it finds first as the row ID. We rely on the *sha256* field.
    """
    attrs = vt_json.get("data", {}).get("attributes", {})
    results = attrs.get("last_analysis_results", {})
    av_labels = []
    for engine, res in results.items():
        cat = res.get("category")
        label = res.get("result")
        if cat in ("malicious", "suspicious") and label:
            av_labels.append([engine, label])
    return {
        "md5": attrs.get("md5", "") or "",
        "sha1": attrs.get("sha1", "") or "",
        "sha256": sha256.lower(),
        "av_labels": av_labels,
    }


def run_avclass_one(sha256, avclass_input):
    """
    Run AVClass on a SINGLE sample. Returns family name (lowercased) or None.
    Running one-at-a-time means we can tag the output by the sha256 we passed
    in, eliminating the key-mismatch bug from the previous version.
    """
    if not avclass_input["av_labels"]:
        return None

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(avclass_input) + "\n")
        tmp_in = f.name

    try:
        out = subprocess.run(
            ["avclass", "-f", tmp_in],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in out.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            fam = parts[1].strip()
            if not fam or fam.startswith("SINGLETON"):
                return None
            return fam.lower()
        return None
    except FileNotFoundError:
        print("   [!] 'avclass' command not found. Install: pip install avclass-malicialab",
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"   [!] AVClass error for {sha256[:12]}: {e}", file=sys.stderr)
        return None
    finally:
        try:
            os.unlink(tmp_in)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# CSV — incremental writer
# ---------------------------------------------------------------------------

class IncrementalCSV:
    """
    Open labels.csv and write the header. Each call to .write(...) flushes
    immediately so a Ctrl-C or crash mid-run doesn't lose progress.
    Also de-dupes by sha256 in case a run gets restarted.
    """
    def __init__(self, path):
        self.path = path
        self.seen = set()
        existing = []
        if os.path.exists(path):
            with open(path) as f:
                r = csv.reader(f)
                rows = list(r)
            if rows and rows[0] == ["sha256", "family", "source"]:
                existing = rows[1:]
                for row in existing:
                    if row:
                        self.seen.add(row[0].lower())
                print(f"[*] Resuming: {len(self.seen)} entries already in {path}")
        if not existing:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(["sha256", "family", "source"])
        self.fh = open(path, "a", newline="")
        self.writer = csv.writer(self.fh)

    def write(self, sha256, family, source):
        sha = sha256.lower()
        if sha in self.seen:
            return
        self.writer.writerow([sha, family, source])
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.seen.add(sha)

    def close(self):
        self.fh.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--output", default="labels.csv")
    ap.add_argument("--vt_key", default=os.environ.get("VT_API_KEY"))
    ap.add_argument("--vt_sleep", type=float, default=15.5,
                    help="Seconds between VT calls (free tier: 4/min = 15s)")
    ap.add_argument("--no_mb", action="store_true",
                    help="Skip MalwareBazaar entirely; go straight to VT.")
    args = ap.parse_args()

    samples_dir = Path(args.samples_dir)
    if not samples_dir.is_dir():
        print(f"[-] Not a directory: {samples_dir}")
        sys.exit(1)

    # Collect sha256 filenames (lowercased, validated)
    sample_hashes = sorted(
        p.name.lower() for p in samples_dir.iterdir()
        if p.is_file() and SHA256_RE.match(p.name.lower())
    )
    print(f"[*] Found {len(sample_hashes)} sha256-named samples in {samples_dir}\n")

    if not sample_hashes:
        print("[-] No samples found (expecting 64-char hex filenames).")
        sys.exit(1)

    csv_out = IncrementalCSV(args.output)
    labels = {}     # sha -> (family, source), in-memory mirror for distribution

    # Seed the in-memory mirror from any resumed rows
    if csv_out.seen:
        with open(args.output) as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) >= 3 and row[1] not in ("", "unknown"):
                    labels[row[0].lower()] = (row[1], row[2])

    # ──────────────────────────────────────────────────────────────────
    # Pass 1 — MalwareBazaar (optional)
    # ──────────────────────────────────────────────────────────────────
    if args.no_mb:
        print("[*] --no_mb set: skipping MalwareBazaar.\n")
        mb_misses = [s for s in sample_hashes if s not in labels]
    else:
        print("[1/2] Querying MalwareBazaar...")
        mb_misses = []
        mb_consecutive_errors = 0
        for i, sha in enumerate(sample_hashes, 1):
            if sha in csv_out.seen:
                print(f"   [{i:3d}/{len(sample_hashes)}] {sha[:12]}... (already in CSV, skip)")
                continue
            fam = query_mb(sha)
            if fam:
                labels[sha] = (fam, "mb")
                csv_out.write(sha, fam, "mb")
                mb_consecutive_errors = 0
                print(f"   [{i:3d}/{len(sample_hashes)}] {sha[:12]}... → {fam}  (mb)")
            else:
                mb_misses.append(sha)
                mb_consecutive_errors += 1
                print(f"   [{i:3d}/{len(sample_hashes)}] {sha[:12]}... → no MB record")
                # Bail out of MB if it's clearly down/blocking us
                if mb_consecutive_errors >= 20 and i < 30:
                    print("\n   [!] MB returning no results for 20+ samples in a row; "
                          "skipping rest of MB pass and falling through to VT.")
                    mb_misses.extend(s for s in sample_hashes[i:] if s not in labels)
                    break
            time.sleep(0.3)
        print(f"\n[*] MB resolved: {len(labels)}, missed: {len(mb_misses)}\n")

    # ──────────────────────────────────────────────────────────────────
    # Pass 2 — VirusTotal + AVClass (per-sample)
    # ──────────────────────────────────────────────────────────────────
    if mb_misses and args.vt_key:
        print(f"[2/2] Querying VirusTotal for {len(mb_misses)} samples "
              f"(~{args.vt_sleep:.0f}s/call, ~{len(mb_misses) * args.vt_sleep / 60:.0f} min)...")
        for i, sha in enumerate(mb_misses, 1):
            if sha in csv_out.seen:
                print(f"   [{i:3d}/{len(mb_misses)}] {sha[:12]}... (already in CSV, skip)")
                continue
            print(f"   [{i:3d}/{len(mb_misses)}] {sha[:12]}... ", end="", flush=True)
            vt_json = query_vt(sha, args.vt_key)
            if not vt_json:
                print("VT: not found")
                csv_out.write(sha, "unknown", "unknown")
                time.sleep(args.vt_sleep)
                continue
            payload = vt_to_avclass_input(sha, vt_json)
            if not payload["av_labels"]:
                print("VT: no AV labels")
                csv_out.write(sha, "unknown", "unknown")
                time.sleep(args.vt_sleep)
                continue
            fam = run_avclass_one(sha, payload)
            if fam:
                labels[sha] = (fam, "avclass")
                csv_out.write(sha, fam, "avclass")
                print(f"VT+AVClass → {fam}")
            else:
                csv_out.write(sha, "unknown", "unknown")
                print(f"VT: {len(payload['av_labels'])} AV labels, AVClass → SINGLETON")
            time.sleep(args.vt_sleep)
    elif mb_misses:
        print(f"[*] {len(mb_misses)} samples unresolved (no VT key provided).")
        for sha in mb_misses:
            if sha not in csv_out.seen:
                csv_out.write(sha, "unknown", "unknown")

    csv_out.close()
    print(f"\n[+] {args.output} written.")

    # ──────────────────────────────────────────────────────────────────
    # Distribution summary
    # ──────────────────────────────────────────────────────────────────
    fam_counts = Counter(fam for fam, _ in labels.values())
    n_unknown = len(sample_hashes) - len(labels)

    print("\n" + "=" * 60)
    print("FAMILY DISTRIBUTION")
    print("=" * 60)
    for fam, n in fam_counts.most_common():
        bar = "█" * min(n, 40)
        print(f"  {fam:30s}  {n:3d}  {bar}")
    if n_unknown:
        print(f"  {'(unknown)':30s}  {n_unknown:3d}")
    print("=" * 60)

    fams_2 = sum(1 for n in fam_counts.values() if n >= 2)
    fams_3 = sum(1 for n in fam_counts.values() if n >= 3)
    fams_5 = sum(1 for n in fam_counts.values() if n >= 5)
    print(f"\nFamilies with ≥2 samples: {fams_2}")
    print(f"Families with ≥3 samples: {fams_3}")
    print(f"Families with ≥5 samples: {fams_5}")

    print("\nVerdict:")
    if fams_3 >= 3:
        print("  ✓ Enough for the GMN-vs-TLSH experiment.")
    elif fams_2 >= 2:
        print("  ~ Marginal — experiment will work but AUC will be noisy.")
    else:
        print("  ✗ Not enough labeled diversity.")


if __name__ == "__main__":
    main()