"""
eval_gmn_vs_tlsh.py — Compare GMN function-matching to TLSH on labeled malware.

Pipeline:
  1. Read labels.csv (sha256, family, source).
  2. Drop unknowns and singleton families (need ≥2 samples per family for pairs).
  3. Generate all valid binary pairs, labeled 1 (same family) or 0 (different).
  4. Optionally subsample pairs (--max_pairs) to keep runtime manageable.
  5. For each pair, compute:
        - GMN binary-pair score (mean of best-match-per-function, your existing logic)
        - TLSH distance on raw bytes
  6. Report AUC for GMN (score → higher = same-family)
     and AUC for TLSH (-distance → higher = same-family, since lower TLSH = more similar).
  7. Save per-pair results to pairs.csv for later plotting/inspection.

This isolates the contribution of GMN over the byte-level baseline on a clean,
externally-labeled malware set. The headline output is the two AUCs side by side.

Usage:
    python eval_gmn_vs_tlsh.py \
        --labels labels.csv \
        --samples_dir database_formation/eval_unpacked \
        --db malware_eval.sqlite \
        --checkpoint checkpoints/best_model.pth \
        --min_blocks 5 \
        --top_k 50 \
        --max_pairs 600 \
        --out pairs.csv

Function selection:
    Names in our DB are stripped ('sub_XXXXXX'), so we can't blacklist by name.
    Instead we filter by basic-block count: drop functions with fewer than
    --min_blocks blocks (catches PLT stubs, _init/_fini, thunks, register_tm_clones,
    etc.), then keep the top --top_k largest functions by block count. The largest
    functions are where real malware logic lives (C&C handlers, scanners, crypto,
    DDoS workers). Selection is fully deterministic — same binary → same functions.
"""

import argparse
import csv
import json
import os
import random
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch

from binary_dataset import BinaryDataset
from configure import get_default_config
from evaluation import compute_similarity
from utils import build_model, reshape_and_split_tensor


# ---------------------------------------------------------------------------
# Reused scoring functions — kept faithful to your evaluate_malware.py
# ---------------------------------------------------------------------------

def load_substantive_functions(db_conn, binary_name, min_blocks, top_k):
    """
    Load the top_k largest functions of a binary, filtering out boilerplate.

    Filtering rationale:
      - Names are stripped ('sub_XXXXXX'), so we can't use a name blacklist.
      - Size is the only signal we have: stubs, PLT entries, _init/_fini,
        register_tm_clones etc. all have 1-3 basic blocks.
      - Real malware logic (C&C, scanners, encryption, DDoS workers) has
        many more blocks — typically 10s to 100s.
      - We require >= min_blocks (default 5) AND keep only the top_k largest.

    Deterministic: same binary always returns the same functions.
    Returns: (list_of_graphs, n_total_funcs, n_after_min_blocks)
    """
    rows = db_conn.execute(
        "SELECT func_name, graph_data FROM functions WHERE binary_name=? ORDER BY func_index",
        (binary_name,),
    ).fetchall()

    candidates = []          # (n_blocks, func_name, graph_dict)
    for func_name, gd_json in rows:
        gd = json.loads(gd_json)
        n_blocks = len(gd.get("node_features", []))
        if n_blocks < min_blocks:
            continue
        candidates.append((n_blocks, func_name, gd))

    # Sort by size descending, then by name for stable tie-breaking
    candidates.sort(key=lambda x: (-x[0], x[1]))
    kept = [gd for _, _, gd in candidates[:top_k]]
    return kept, len(rows), len(candidates)


def gmn_score(model, config, device, packer, funcs_a, funcs_b, batch_size):
    """Mean-of-best-match-per-function. Same logic as your evaluate_malware.py."""
    n_a, n_b = len(funcs_a), len(funcs_b)
    if n_a == 0 or n_b == 0:
        return None
    all_pairs = [(i, j) for i in range(n_a) for j in range(n_b)]
    sim_matrix = np.full((n_a, n_b), -np.inf, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(all_pairs), batch_size):
            batch = all_pairs[start:start + batch_size]
            graphs = []
            for i, j in batch:
                graphs.extend([funcs_a[i], funcs_b[j]])

            p = packer._pack_batch(graphs)
            nf = torch.tensor(p.node_features, dtype=torch.float32).to(device)
            ef = torch.tensor(p.edge_features, dtype=torch.float32).to(device)
            fi = torch.tensor(p.from_idx,      dtype=torch.long).to(device)
            ti = torch.tensor(p.to_idx,        dtype=torch.long).to(device)
            gi = torch.tensor(p.graph_idx,     dtype=torch.long).to(device)

            vecs = model(nf, ef, fi, ti, gi, len(graphs))
            x, y = reshape_and_split_tensor(vecs, 2)
            sims = compute_similarity(config, x, y)
            for k, (i, j) in enumerate(batch):
                sim_matrix[i, j] = sims[k].item()

    best_matches = sim_matrix.max(axis=1)   # best B match for each A func
    return float(best_matches.mean())


def tlsh_dist(path_a, path_b):
    try:
        import tlsh
        h_a = tlsh.hash(open(path_a, "rb").read())
        h_b = tlsh.hash(open(path_b, "rb").read())
        if h_a == "TNULL" or h_b == "TNULL":
            return None
        return tlsh.diff(h_a, h_b)
    except Exception as e:
        print(f"   [!] TLSH error: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# AUC — same definition for both scorers (higher score → more likely same-family)
# ---------------------------------------------------------------------------

def auc_score(scores, labels):
    """
    Rank-based AUC. scores higher → more likely positive.
    Equivalent to sklearn.metrics.roc_auc_score.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann–Whitney U-statistic / (n_pos * n_neg)
    n_correct = 0.0
    n_total = 0
    # Vectorized: count, for each pos, fraction of negs it beats (+ 0.5 for ties)
    for p in pos:
        n_correct += np.sum(p > neg) + 0.5 * np.sum(p == neg)
        n_total += len(neg)
    return n_correct / n_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels",      required=True, help="labels.csv from label_samples.py")
    ap.add_argument("--samples_dir", required=True, help="folder of raw binaries (for TLSH)")
    ap.add_argument("--db",          required=True, help="malware_eval.sqlite (for GMN)")
    ap.add_argument("--checkpoint",  required=True, help="model checkpoint .pth")
    ap.add_argument("--min_blocks",  type=int, default=5,
                    help="drop functions with fewer than this many basic blocks "
                         "(filters out PLT stubs, _init/_fini, thunks)")
    ap.add_argument("--top_k",       type=int, default=50,
                    help="keep the top_k largest functions per binary (by block count)")
    ap.add_argument("--max_pairs",   type=int, default=600,
                    help="cap on total pairs to evaluate (split evenly between pos/neg)")
    ap.add_argument("--batch_size",  type=int, default=32)
    ap.add_argument("--out",         default="pairs.csv")
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    # ──────────────────────────────────────────────────────────────────
    # Read labels, drop singletons and unknowns
    # ──────────────────────────────────────────────────────────────────
    fam_of = {}
    with open(args.labels) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fam = row["family"].strip().lower()
            if fam and fam != "unknown":
                fam_of[row["sha256"].lower()] = fam

    by_fam = defaultdict(list)
    for sha, fam in fam_of.items():
        by_fam[fam].append(sha)

    # Drop families with <2 samples (can't form same-family pairs)
    by_fam = {fam: shas for fam, shas in by_fam.items() if len(shas) >= 2}
    if len(by_fam) < 2:
        print(f"[-] Need ≥2 families with ≥2 samples each. Got: {list(by_fam.keys())}")
        sys.exit(1)

    print(f"[*] Usable families: {len(by_fam)}")
    for fam, shas in by_fam.items():
        print(f"      {fam}: {len(shas)} samples")
    total_samples = sum(len(s) for s in by_fam.values())
    print(f"[*] Total usable samples: {total_samples}\n")

    # ──────────────────────────────────────────────────────────────────
    # Confirm samples exist on disk AND in the GMN DB
    # ──────────────────────────────────────────────────────────────────
    samples_dir = Path(args.samples_dir)
    conn = sqlite3.connect(args.db)
    db_binaries = {
        r[0].lower()
        for r in conn.execute("SELECT DISTINCT binary_name FROM functions").fetchall()
    }

    missing_disk, missing_db = [], []
    for fam in list(by_fam.keys()):
        kept = []
        for sha in by_fam[fam]:
            ok_disk = (samples_dir / sha).is_file()
            ok_db = sha.lower() in db_binaries
            if not ok_disk:
                missing_disk.append(sha)
            elif not ok_db:
                missing_db.append(sha)
            else:
                kept.append(sha)
        by_fam[fam] = kept

    if missing_disk:
        print(f"[!] {len(missing_disk)} samples in labels but not in {samples_dir} — skipped")
    if missing_db:
        print(f"[!] {len(missing_db)} samples not in {args.db} (not yet dissected) — skipped")

    by_fam = {fam: shas for fam, shas in by_fam.items() if len(shas) >= 2}
    if len(by_fam) < 2:
        print("[-] Not enough usable samples after disk/DB filtering.")
        sys.exit(1)

    print(f"\n[*] After disk + DB filtering: {sum(len(s) for s in by_fam.values())} samples "
          f"across {len(by_fam)} families\n")

    # ──────────────────────────────────────────────────────────────────
    # Build the pair list — balanced pos/neg
    # ──────────────────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    pos_pairs, neg_pairs = [], []

    # Positive: all within-family combinations
    for fam, shas in by_fam.items():
        for a, b in combinations(shas, 2):
            pos_pairs.append((a, b, fam, fam, 1))

    # Negative: cross-family pairs
    fams = list(by_fam.keys())
    for fam_a, fam_b in combinations(fams, 2):
        for a, b in product(by_fam[fam_a], by_fam[fam_b]):
            neg_pairs.append((a, b, fam_a, fam_b, 0))

    print(f"[*] Available positive pairs: {len(pos_pairs)}")
    print(f"[*] Available negative pairs: {len(neg_pairs)}")

    half = args.max_pairs // 2
    if len(pos_pairs) > half:
        pos_pairs = rng.sample(pos_pairs, half)
    if len(neg_pairs) > half:
        neg_pairs = rng.sample(neg_pairs, half)
    pairs = pos_pairs + neg_pairs
    rng.shuffle(pairs)

    print(f"[*] Evaluating {len(pairs)} pairs "
          f"({len(pos_pairs)} pos, {len(neg_pairs)} neg)\n")

    # ──────────────────────────────────────────────────────────────────
    # Load GMN
    # ──────────────────────────────────────────────────────────────────
    config = get_default_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    model, _ = build_model(config, node_feature_dim=64, edge_feature_dim=1)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()
    packer = BinaryDataset.__new__(BinaryDataset)
    print("[+] Model loaded.\n")

    # ──────────────────────────────────────────────────────────────────
    # Pre-load substantive functions per binary (cache, since each sample
    # appears in many pairs and filtering is deterministic)
    # ──────────────────────────────────────────────────────────────────
    print(f"[*] Filtering functions (min_blocks={args.min_blocks}, "
          f"top_k={args.top_k}, deterministic)...")
    func_cache = {}
    sample_set = set()
    for a, b, _, _, _ in pairs:
        sample_set.add(a); sample_set.add(b)
    for sha in sample_set:
        funcs, n_total, n_kept_floor = load_substantive_functions(
            conn, sha, args.min_blocks, args.top_k)
        func_cache[sha] = funcs
        print(f"   {sha[:12]}...  total={n_total:4d}  "
              f">={args.min_blocks}blocks={n_kept_floor:4d}  kept={len(funcs):3d}")
    avg_kept = np.mean([len(f) for f in func_cache.values()])
    print(f"[*] Average functions kept per binary: {avg_kept:.1f}\n")

    # ──────────────────────────────────────────────────────────────────
    # Score every pair
    # ──────────────────────────────────────────────────────────────────
    results = []   # (sha_a, sha_b, fam_a, fam_b, label, gmn, tlsh)
    t0 = time.time()
    for idx, (a, b, fam_a, fam_b, label) in enumerate(pairs, 1):
        funcs_a = func_cache[a]
        funcs_b = func_cache[b]
        if not funcs_a or not funcs_b:
            print(f"[{idx:4d}/{len(pairs)}] SKIP — no substantive functions "
                  f"for {a[:12] if not funcs_a else b[:12]}")
            continue

        gmn = gmn_score(model, config, device, packer, funcs_a, funcs_b, args.batch_size)
        tl = tlsh_dist(str(samples_dir / a), str(samples_dir / b))
        results.append((a, b, fam_a, fam_b, label, gmn, tl))

        elapsed = time.time() - t0
        rate = idx / elapsed if elapsed > 0 else 0
        eta = (len(pairs) - idx) / rate if rate > 0 else 0
        tag = "POS" if label == 1 else "NEG"
        print(f"[{idx:4d}/{len(pairs)}] {tag} {fam_a}/{fam_b}  "
              f"GMN={gmn:.3f}  TLSH={tl}  ({rate:.2f}/s, ETA {eta/60:.1f}min)")

    conn.close()

    # ──────────────────────────────────────────────────────────────────
    # Write per-pair CSV
    # ──────────────────────────────────────────────────────────────────
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sha_a", "sha_b", "family_a", "family_b", "label",
                    "gmn_score", "tlsh_dist"])
        for row in results:
            w.writerow(row)
    print(f"\n[+] Wrote per-pair results to {args.out}")

    # ──────────────────────────────────────────────────────────────────
    # Compute AUCs
    # ──────────────────────────────────────────────────────────────────
    gmn_scores, gmn_labels = [], []
    tlsh_scores, tlsh_labels = [], []
    for _, _, _, _, lab, gmn, tl in results:
        if gmn is not None:
            gmn_scores.append(gmn);  gmn_labels.append(lab)
        if tl is not None:
            # Lower TLSH distance = more similar, so negate so "higher = same-family"
            tlsh_scores.append(-tl); tlsh_labels.append(lab)

    gmn_auc = auc_score(gmn_scores, gmn_labels) if gmn_scores else float("nan")
    tlsh_auc = auc_score(tlsh_scores, tlsh_labels) if tlsh_scores else float("nan")

    # ──────────────────────────────────────────────────────────────────
    # Per-family-pair breakdown (which family confusions does each method handle?)
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Pairs scored : {len(results)}")
    print(f"  positives  : {sum(1 for r in results if r[4] == 1)}")
    print(f"  negatives  : {sum(1 for r in results if r[4] == 0)}")
    print()
    print(f"  GMN  AUC   : {gmn_auc:.4f}     (n={len(gmn_scores)})")
    print(f"  TLSH AUC   : {tlsh_auc:.4f}     (n={len(tlsh_scores)})")
    print()
    delta = gmn_auc - tlsh_auc
    if delta > 0.02:
        print(f"  → GMN beats TLSH by {delta:+.4f}")
    elif delta < -0.02:
        print(f"  → TLSH beats GMN by {-delta:+.4f}")
    else:
        print(f"  → GMN and TLSH within noise ({delta:+.4f})")
    print("=" * 70)

    # Mean scores per pair type — quick sanity check
    print("\nMean scores by pair type:")
    print(f"  {'pair type':25s}  {'n':>4s}  {'GMN':>8s}  {'TLSH':>8s}")
    type_groups = defaultdict(list)
    for _, _, fa, fb, lab, gmn, tl in results:
        key = (lab, tuple(sorted([fa, fb])))
        type_groups[key].append((gmn, tl))
    for (lab, fams_key), entries in sorted(type_groups.items()):
        gmns = [g for g, _ in entries if g is not None]
        tls  = [t for _, t in entries if t is not None]
        tag = "same:" if lab == 1 else "diff:"
        label_str = tag + "/".join(fams_key)
        gmn_mean = np.mean(gmns) if gmns else float("nan")
        tl_mean  = np.mean(tls)  if tls  else float("nan")
        print(f"  {label_str:25s}  {len(entries):4d}  {gmn_mean:8.3f}  {tl_mean:8.1f}")


if __name__ == "__main__":
    main()