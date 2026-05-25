"""
heatmap_pos_neg.py — Side-by-side GMN similarity heatmaps for one POS and one NEG pair.

For each pair of malware binaries:
  1. Take the top --top_k largest functions (by basic-block count, min --min_blocks).
  2. Compute the GMN similarity score for every cross-pair (k_a x k_b matrix).
  3. BINARIZE at --threshold (default -1.0, the training margin):
        scores > threshold  → SIMILAR    → dark cell
        scores ≤ threshold  → DISSIMILAR → light cell
     This matches the model's own decision boundary from margin-loss training.

Two pairs (one POS, one NEG) are rendered side-by-side on one PNG so the
visual contrast between same-family and different-family is immediately readable.

Usage:
    python heatmap_pos_neg.py \\
        --pos_a <sha256> --pos_b <sha256> \\
        --neg_a <sha256> --neg_b <sha256> \\
        --db malware_eval.sqlite \\
        --checkpoint checkpoints/best_model.pth \\
        --min_blocks 20 \\
        --top_k 50 \\
        --threshold -1.0 \\
        --out heatmap.png
"""

import argparse
import json
import sqlite3
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")              # no display required
import matplotlib.pyplot as plt

from binary_dataset import BinaryDataset
from configure import get_default_config
from evaluation import compute_similarity
from utils import build_model, reshape_and_split_tensor


# ---------------------------------------------------------------------------
# Function loading (same logic as eval_gmn_vs_tlsh.py)
# ---------------------------------------------------------------------------

def load_top_functions(conn, binary_name, min_blocks, top_k):
    """Return list of (func_name, graph_dict), sorted by block count desc, top_k kept."""
    rows = conn.execute(
        "SELECT func_name, graph_data FROM functions WHERE binary_name=? ORDER BY func_index",
        (binary_name,),
    ).fetchall()

    candidates = []
    for func_name, gd_json in rows:
        gd = json.loads(gd_json)
        n_blocks = len(gd.get("node_features", []))
        if n_blocks < min_blocks:
            continue
        candidates.append((n_blocks, func_name, gd))

    candidates.sort(key=lambda x: (-x[0], x[1]))
    return [(name, gd, nb) for nb, name, gd in candidates[:top_k]]


# ---------------------------------------------------------------------------
# Similarity matrix (GMN over every cross-pair)
# ---------------------------------------------------------------------------

def similarity_matrix(model, config, device, packer, funcs_a, funcs_b, batch_size=32):
    """Return n_a × n_b matrix of GMN similarity scores."""
    n_a, n_b = len(funcs_a), len(funcs_b)
    pairs = [(i, j) for i in range(n_a) for j in range(n_b)]
    sim = np.full((n_a, n_b), np.nan, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            graphs = []
            for i, j in batch:
                graphs.extend([funcs_a[i][1], funcs_b[j][1]])

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
                sim[i, j] = sims[k].item()

    return sim


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _short_label(func_name, n_blocks):
    """sha256_sub_4012b0 → sub_4012b0 (52 blocks)"""
    short = func_name.split("_sub_")[-1]
    short = "sub_" + short if not short.startswith("sub_") else short
    return f"{short} ({n_blocks})"


def render_heatmap(ax, sim, funcs_a, funcs_b, title, threshold=-1.0):
    """
    Render the similarity matrix as a binary (two-tone) grayscale heatmap.

    Scores > threshold  → SIMILAR    → dark gray (matches model's "positive" region)
    Scores ≤ threshold  → DISSIMILAR → light gray ("negative" region)

    threshold = -1.0 matches the margin loss used at training (margin=1.0):
    the model is trained to push negatives below -1.0 and pull positives toward 0.
    So this binarization is the model's *own* decision boundary, not an arbitrary cut.

    Per-cell numbers are omitted at 50x50 — they'd be unreadable. The colors carry
    the qualitative story; numerical summaries appear in the figure title and stdout.
    """
    binarized = (sim > threshold).astype(np.float32)   # 1.0 = similar, 0.0 = dissimilar

    im = ax.imshow(
        binarized,
        cmap="gray_r",                # 1 → dark, 0 → light
        vmin=0.0, vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )

    # Axis labels: function names with block counts.
    # At top_k=50 we can't show all labels — show every Nth one so the axis stays readable.
    n = max(len(funcs_a), len(funcs_b))
    step = max(1, n // 25)            # at most ~25 labels per axis
    xticks = list(range(0, len(funcs_b), step))
    yticks = list(range(0, len(funcs_a), step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([_short_label(funcs_b[i][0], funcs_b[i][2]) for i in xticks],
                       rotation=90, fontsize=6)
    ax.set_yticks(yticks)
    ax.set_yticklabels([_short_label(funcs_a[i][0], funcs_a[i][2]) for i in yticks],
                       fontsize=6)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Binary B functions (sorted by size desc)")
    ax.set_ylabel("Binary A functions (sorted by size desc)")

    # Grid showing how much of the matrix the model called "similar"
    pct_similar = 100.0 * binarized.mean()
    ax.text(0.02, 0.98,
            f"{pct_similar:.1f}% similar (>{threshold})",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                      edgecolor="black", linewidth=0.5))

    return im


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_a", required=True, help="sha256 of POS pair binary A (same family as pos_b)")
    ap.add_argument("--pos_b", required=True)
    ap.add_argument("--neg_a", required=True, help="sha256 of NEG pair binary A (different family from neg_b)")
    ap.add_argument("--neg_b", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--min_blocks", type=int, default=20,
                    help="drop functions with fewer than this many basic blocks")
    ap.add_argument("--top_k", type=int, default=50,
                    help="keep the top_k largest functions per binary")
    ap.add_argument("--threshold", type=float, default=-1.0,
                    help="binarization cutoff: scores > threshold render as 'similar' "
                         "(dark). Default -1.0 matches the training margin.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default="heatmap.png")
    ap.add_argument("--pos_label", default="POS: same family")
    ap.add_argument("--neg_label", default="NEG: different family")
    args = ap.parse_args()

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
    # Load top-K functions for all four binaries
    # ─────────────────────────────────────────────────────────────
    conn = sqlite3.connect(args.db)

    def fetch(name, label):
        funcs = load_top_functions(conn, name, args.min_blocks, args.top_k)
        if not funcs:
            print(f"[-] No functions ≥{args.min_blocks} blocks for {label} ({name[:12]}...)")
            sys.exit(1)
        sizes = [b for _, _, b in funcs]
        print(f"[*] {label} ({name[:12]}...): {len(funcs)} funcs, "
              f"sizes={sizes}")
        return funcs

    pos_a_funcs = fetch(args.pos_a, "POS-A")
    pos_b_funcs = fetch(args.pos_b, "POS-B")
    neg_a_funcs = fetch(args.neg_a, "NEG-A")
    neg_b_funcs = fetch(args.neg_b, "NEG-B")

    # ─────────────────────────────────────────────────────────────
    # Compute similarity matrices
    # ─────────────────────────────────────────────────────────────
    print("\n[*] Computing POS similarity matrix...")
    sim_pos = similarity_matrix(model, config, device, packer,
                                pos_a_funcs, pos_b_funcs, args.batch_size)
    print(f"   POS: min={np.nanmin(sim_pos):.3f}  "
          f"max={np.nanmax(sim_pos):.3f}  "
          f"mean={np.nanmean(sim_pos):.3f}")

    print("[*] Computing NEG similarity matrix...")
    sim_neg = similarity_matrix(model, config, device, packer,
                                neg_a_funcs, neg_b_funcs, args.batch_size)
    print(f"   NEG: min={np.nanmin(sim_neg):.3f}  "
          f"max={np.nanmax(sim_neg):.3f}  "
          f"mean={np.nanmean(sim_neg):.3f}")

    # ─────────────────────────────────────────────────────────────
    # Render side-by-side, binarized at the training margin
    # ─────────────────────────────────────────────────────────────
    pct_pos = 100.0 * (sim_pos > args.threshold).mean()
    pct_neg = 100.0 * (sim_neg > args.threshold).mean()
    print(f"\n[*] Binarization threshold: {args.threshold}")
    print(f"    POS: {pct_pos:.1f}% of cells classified as similar (> {args.threshold})")
    print(f"    NEG: {pct_neg:.1f}% of cells classified as similar (> {args.threshold})")

    side = 0.18 * args.top_k + 2.0
    fig, axes = plt.subplots(1, 2, figsize=(2 * side, side))

    render_heatmap(
        axes[0], sim_pos, pos_a_funcs, pos_b_funcs,
        f"{args.pos_label}\nmean similarity = {np.nanmean(sim_pos):+.3f}",
        threshold=args.threshold,
    )
    render_heatmap(
        axes[1], sim_neg, neg_a_funcs, neg_b_funcs,
        f"{args.neg_label}\nmean similarity = {np.nanmean(sim_neg):+.3f}",
        threshold=args.threshold,
    )

    fig.suptitle(
        f"GMN function-level similarity (top {args.top_k} functions, "
        f"min {args.min_blocks} basic blocks)\n"
        f"Dark cell = GMN score > {args.threshold} (similar);  "
        f"Light cell = score ≤ {args.threshold} (dissimilar)",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"\n[+] Saved heatmap to {args.out}")

    # ─────────────────────────────────────────────────────────────
    # Numerical summary
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  POS mean similarity : {np.nanmean(sim_pos):+.4f}")
    print(f"  NEG mean similarity : {np.nanmean(sim_neg):+.4f}")
    print(f"  Gap (POS - NEG)     : {np.nanmean(sim_pos) - np.nanmean(sim_neg):+.4f}")
    print(f"  POS cells similar   : {pct_pos:.1f}%")
    print(f"  NEG cells similar   : {pct_neg:.1f}%")
    print(f"  Visual contrast     : {pct_pos - pct_neg:+.1f} pp")
    print("=" * 60)


if __name__ == "__main__":
    main()