"""
eval_cross_binary.py — Cross-binary semantic-equivalence AUC for the trained GMN.

The experiment:
  - Find function names that appear in BOTH curl and wget (statically-linked
    library code: libc, zlib, openssl, common utils). These are guaranteed
    semantic equivalents — same source, different host binary.
  - Build positive pairs: same function name across the two binaries
        (curl_base64_encode  vs  wget_base64_encode)
  - Build negative pairs: different function names across the two binaries
        (curl_base64_encode  vs  wget_inflate)
  - Compute GMN similarity for each pair, report ROC-AUC.

This isolates the question:
  "Does my model generalize from 'same source, different compilation'
   (its training objective) to 'same source, different host binary'
   (a slightly broader form of semantic equivalence)?"

Boilerplate functions inserted by the linker (_init, _fini, frame_dummy,
register_tm_clones, etc.) are filtered out — they're identical across every
ELF and would inflate AUC trivially.

Usage:
    python eval_cross_binary.py \\
        --db test_database.sqlite \\
        --checkpoint checkpoints/best_model.pth \\
        --prefix_a curl \\
        --prefix_b wget \\
        --min_blocks 5 \\
        --max_pos 500 \\
        --max_neg 500 \\
        --out cross_binary_pairs.csv
"""

import argparse
import csv
import json
import random
import sqlite3
import sys
import time
from collections import defaultdict

import numpy as np
import torch

from binary_dataset import BinaryDataset
from configure import get_default_config
from evaluation import compute_similarity
from utils import build_model, reshape_and_split_tensor


# ---------------------------------------------------------------------------
# Boilerplate filter — linker / C-runtime functions present in every ELF
# ---------------------------------------------------------------------------

LINKER_BOILERPLATE = {
    "_start", "_init", "_fini",
    "__libc_csu_init", "__libc_csu_fini",
    "__libc_start_main",
    "frame_dummy",
    "register_tm_clones", "deregister_tm_clones",
    "__do_global_dtors_aux", "__do_global_ctors_aux",
    "_dl_relocate_static_pie",
    "atexit",
    # Trivial leaf wrappers that look identical everywhere
    "abort",
}


# ---------------------------------------------------------------------------
# Schema-aware fetch (graphs table has columns: function_name, graph_data;
# block count comes from len(node_features) in the JSON graph_data)
# ---------------------------------------------------------------------------

def load_functions_by_stem(conn, prefix, min_blocks):
    """
    Return dict: stem -> list of (function_name, graph_dict, n_blocks).
    `stem` = function_name with the binary prefix stripped.

    Example: function_name='curl_base64_encode', prefix='curl' → stem='base64_encode'
    """
    rows = conn.execute(
        "SELECT function_name, graph_data FROM graphs WHERE function_name LIKE ?",
        (f"{prefix}_%",),
    ).fetchall()

    by_stem = defaultdict(list)
    for fname, gd_json in rows:
        gd = json.loads(gd_json)
        n_blocks = len(gd.get("node_features", []))
        if n_blocks < min_blocks:
            continue
        # Strip the binary prefix to get the stem
        stem = fname[len(prefix) + 1:]  # +1 for the underscore
        if stem in LINKER_BOILERPLATE:
            continue
        # Skip ".part" / ".cold" / ".constprop" compiler artifacts which split one
        # source function into multiple compiled fragments — they share a stem but
        # represent different chunks, so pairing them across binaries is noisy.
        if "." in stem:
            continue
        by_stem[stem].append((fname, gd, n_blocks))
    return by_stem


# ---------------------------------------------------------------------------
# GMN scoring — direct function-vs-function similarity (no aggregation)
# ---------------------------------------------------------------------------

def score_batch(model, config, device, packer, function_pairs, batch_size):
    """
    Take a list of (graph_a, graph_b) pairs, return their GMN similarities
    as a flat list of floats, in the same order.
    """
    out = []
    n = len(function_pairs)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = function_pairs[start:start + batch_size]
            graphs = []
            for ga, gb in batch:
                graphs.extend([ga, gb])

            p = packer._pack_batch(graphs)
            nf = torch.tensor(p.node_features, dtype=torch.float32).to(device)
            ef = torch.tensor(p.edge_features, dtype=torch.float32).to(device)
            fi = torch.tensor(p.from_idx,      dtype=torch.long).to(device)
            ti = torch.tensor(p.to_idx,        dtype=torch.long).to(device)
            gi = torch.tensor(p.graph_idx,     dtype=torch.long).to(device)

            vecs = model(nf, ef, fi, ti, gi, len(graphs))
            x, y = reshape_and_split_tensor(vecs, 2)
            sims = compute_similarity(config, x, y)
            for s in sims:
                out.append(float(s.item()))
    return out


def auc_score(scores, labels):
    """Mann-Whitney U / (n_pos * n_neg). Higher score → more likely positive."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    total = 0.0
    for p in pos:
        total += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return total / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",         required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prefix_a",   default="curl",
                    help="binary prefix for set A (e.g. 'curl' matches curl_*)")
    ap.add_argument("--prefix_b",   default="wget")
    ap.add_argument("--min_blocks", type=int, default=5,
                    help="drop functions with fewer than this many blocks")
    ap.add_argument("--max_pos",    type=int, default=500,
                    help="cap on positive pairs (same source function across A and B)")
    ap.add_argument("--max_neg",    type=int, default=500,
                    help="cap on negative pairs (different source functions)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--out",        default="cross_binary_pairs.csv")
    ap.add_argument("--positive_stems_file", default=None,
                    help="Optional path to a text file with one stem per line. "
                         "If given, ONLY these stems will be used to build positive "
                         "pairs (negatives still drawn from all available stems). "
                         "Useful for restricting to stems that don't appear in train.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    conn = sqlite3.connect(args.db)

    # ─────────────────────────────────────────────────────────────
    # Load and group functions from each binary
    # ─────────────────────────────────────────────────────────────
    print(f"[*] Loading {args.prefix_a}_* functions (min_blocks={args.min_blocks}, "
          f"excluding linker boilerplate)...")
    a_by_stem = load_functions_by_stem(conn, args.prefix_a, args.min_blocks)
    print(f"    {sum(len(v) for v in a_by_stem.values())} function records, "
          f"{len(a_by_stem)} unique stems")

    print(f"[*] Loading {args.prefix_b}_* functions...")
    b_by_stem = load_functions_by_stem(conn, args.prefix_b, args.min_blocks)
    print(f"    {sum(len(v) for v in b_by_stem.values())} function records, "
          f"{len(b_by_stem)} unique stems")

    # ─────────────────────────────────────────────────────────────
    # Find shared stems = the labeled positives we can build
    # ─────────────────────────────────────────────────────────────
    shared = sorted(set(a_by_stem.keys()) & set(b_by_stem.keys()))
    if not shared:
        print(f"[-] No function names appear in both {args.prefix_a} and {args.prefix_b}.")
        sys.exit(1)

    print(f"\n[*] Functions present in both binaries: {len(shared)}")
    print(f"    First 20: {shared[:20]}")

    # Optional whitelist for positives (e.g. stems unseen during training)
    if args.positive_stems_file:
        with open(args.positive_stems_file) as f:
            allowed = {line.strip() for line in f if line.strip()}
        before = len(shared)
        shared = [s for s in shared if s in allowed]
        print(f"[*] --positive_stems_file: {len(allowed)} stems in whitelist, "
              f"{len(shared)} of {before} shared stems retained for POSITIVES.")
        if not shared:
            print("[-] No stems left after applying whitelist. Aborting.")
            sys.exit(1)
        print(f"    Using positive stems: {shared}")

    # ─────────────────────────────────────────────────────────────
    # Build positive pairs:  curl_F  ↔  wget_F   (same stem)
    # ─────────────────────────────────────────────────────────────
    pos_records = []     # (stem_a, stem_b, graph_a, graph_b)
    for stem in shared:
        for fa_name, ga, _ in a_by_stem[stem]:
            for fb_name, gb, _ in b_by_stem[stem]:
                pos_records.append((stem, stem, ga, gb, fa_name, fb_name))

    rng.shuffle(pos_records)
    if len(pos_records) > args.max_pos:
        pos_records = pos_records[:args.max_pos]

    print(f"\n[*] Positive pairs available (curl_F ↔ wget_F): "
          f"{len(pos_records)} (capped at {args.max_pos})")

    # ─────────────────────────────────────────────────────────────
    # Build negative pairs:  curl_F  ↔  wget_G   (different stems)
    # ─────────────────────────────────────────────────────────────
    neg_records = []
    a_stems = list(a_by_stem.keys())
    b_stems = list(b_by_stem.keys())

    attempts = 0
    while len(neg_records) < args.max_neg and attempts < args.max_neg * 20:
        attempts += 1
        sa = rng.choice(a_stems)
        sb = rng.choice(b_stems)
        if sa == sb:
            continue  # would be a positive pair
        fa_name, ga, _ = rng.choice(a_by_stem[sa])
        fb_name, gb, _ = rng.choice(b_by_stem[sb])
        neg_records.append((sa, sb, ga, gb, fa_name, fb_name))

    print(f"[*] Negative pairs sampled (curl_F ↔ wget_G): {len(neg_records)}")
    print()

    # ─────────────────────────────────────────────────────────────
    # Load model
    # ─────────────────────────────────────────────────────────────
    config = get_default_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    model, _ = build_model(config, node_feature_dim=64, edge_feature_dim=1)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()
    packer = BinaryDataset.__new__(BinaryDataset)
    print("[+] Model loaded.\n")

    # ─────────────────────────────────────────────────────────────
    # Score
    # ─────────────────────────────────────────────────────────────
    pos_pairs = [(r[2], r[3]) for r in pos_records]
    neg_pairs = [(r[2], r[3]) for r in neg_records]

    print(f"[*] Scoring {len(pos_pairs)} positive pairs...")
    t0 = time.time()
    pos_scores = score_batch(model, config, device, packer, pos_pairs, args.batch_size)
    print(f"    done in {time.time()-t0:.1f}s")

    print(f"[*] Scoring {len(neg_pairs)} negative pairs...")
    t0 = time.time()
    neg_scores = score_batch(model, config, device, packer, neg_pairs, args.batch_size)
    print(f"    done in {time.time()-t0:.1f}s\n")

    # ─────────────────────────────────────────────────────────────
    # Save per-pair CSV
    # ─────────────────────────────────────────────────────────────
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "stem_a", "stem_b", "func_a", "func_b", "gmn_score"])
        for rec, s in zip(pos_records, pos_scores):
            w.writerow([1, rec[0], rec[1], rec[4], rec[5], s])
        for rec, s in zip(neg_records, neg_scores):
            w.writerow([0, rec[0], rec[1], rec[4], rec[5], s])
    print(f"[+] Per-pair results saved to {args.out}")

    # ─────────────────────────────────────────────────────────────
    # AUC
    # ─────────────────────────────────────────────────────────────
    scores = pos_scores + neg_scores
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    auc = auc_score(scores, labels)

    print("\n" + "=" * 65)
    print(f"CROSS-BINARY SEMANTIC-EQUIVALENCE EVAL  ({args.prefix_a} vs {args.prefix_b})")
    print("=" * 65)
    print(f"  Positive pairs (same stem across binaries) : {len(pos_scores)}")
    print(f"  Negative pairs (different stems)           : {len(neg_scores)}")
    print()
    print(f"  Positive mean score : {np.mean(pos_scores):+.4f}  "
          f"(higher = more similar; 0 is the model's 'positive' anchor)")
    print(f"  Negative mean score : {np.mean(neg_scores):+.4f}")
    print(f"  Gap                 : {np.mean(pos_scores) - np.mean(neg_scores):+.4f}")
    print()
    print(f"  Pair AUC            : {auc:.4f}")
    print("=" * 65)

    # ─────────────────────────────────────────────────────────────
    # Top-confused negatives — useful diagnostic
    # ─────────────────────────────────────────────────────────────
    print("\n[*] Top 10 highest-scoring NEGATIVE pairs (model thinks they're equivalent):")
    neg_with_meta = sorted(
        zip(neg_records, neg_scores),
        key=lambda x: -x[1],
    )[:10]
    for (sa, sb, _, _, fa, fb), s in neg_with_meta:
        print(f"   {s:+.4f}   {args.prefix_a}::{sa[:25]:25s} vs {args.prefix_b}::{sb[:25]:25s}")

    print("\n[*] Bottom 10 lowest-scoring POSITIVE pairs (model misses these equivalences):")
    pos_with_meta = sorted(
        zip(pos_records, pos_scores),
        key=lambda x: x[1],
    )[:10]
    for (sa, _, _, _, fa, fb), s in pos_with_meta:
        print(f"   {s:+.4f}   {sa[:50]}")


if __name__ == "__main__":
    main()