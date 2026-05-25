"""
malware_heatmap.py  —  Function-level GMN similarity heatmap between two malware binaries.

Rows    = top N largest non-boilerplate functions of binary A (sorted by block count)
Columns = top N largest non-boilerplate functions of binary B (sorted by block count)
Cell    = GMN similarity score: darker = more similar (score closer to 0)

Usage:
    python3 malware_heatmap.py \
        --binary_a <sha256> \
        --binary_b <sha256> \
        --db malware_eval.sqlite \
        --checkpoint checkpoints/best_model.pth \
        --output results/heatmap.png \
        [--top_n 50]          # functions per binary (default 50)
        [--min_blocks 5]      # skip boilerplate stubs (default 5)
        [--vmin -30]          # fixed colormap min — use SAME value for both heatmaps
        [--batch_size 32]
"""

import argparse
import json
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import torch

from binary_dataset import BinaryDataset
from configure import get_default_config
from evaluation import compute_similarity
from utils import build_model, reshape_and_split_tensor


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_top_functions(db_path, binary_name, top_n, min_blocks):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT func_name, graph_data FROM functions WHERE binary_name = ? ORDER BY func_index",
        (binary_name,),
    )
    rows = cursor.fetchall()
    conn.close()

    entries = []
    for func_name, graph_data_str in rows:
        g = json.loads(graph_data_str)
        n_blocks = len(g["node_features"])
        if n_blocks >= min_blocks:
            entries.append((func_name, g, n_blocks))

    entries.sort(key=lambda x: x[2], reverse=True)
    entries = entries[:top_n]

    names  = [e[0] for e in entries]
    graphs = [e[1] for e in entries]
    return names, graphs


# ---------------------------------------------------------------------------
# GMN full matrix
# ---------------------------------------------------------------------------

def compute_sim_matrix(model, config, device, packer, funcs_a, funcs_b, batch_size):
    n_a, n_b = len(funcs_a), len(funcs_b)
    all_pairs = [(ia, ib) for ia in range(n_a) for ib in range(n_b)]
    sim_matrix = np.zeros((n_a, n_b), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(all_pairs), batch_size):
            batch_pairs = all_pairs[start : start + batch_size]

            graphs = []
            for ia, ib in batch_pairs:
                graphs.extend([funcs_a[ia], funcs_b[ib]])

            packed    = packer._pack_batch(graphs)
            node_f    = torch.tensor(packed.node_features, dtype=torch.float32).to(device)
            edge_f    = torch.tensor(packed.edge_features, dtype=torch.float32).to(device)
            from_idx  = torch.tensor(packed.from_idx,      dtype=torch.long).to(device)
            to_idx    = torch.tensor(packed.to_idx,        dtype=torch.long).to(device)
            graph_idx = torch.tensor(packed.graph_idx,     dtype=torch.long).to(device)

            graph_vectors = model(node_f, edge_f, from_idx, to_idx, graph_idx, len(graphs))
            x, y = reshape_and_split_tensor(graph_vectors, 2)
            sims  = compute_similarity(config, x, y)

            for k, (ia, ib) in enumerate(batch_pairs):
                sim_matrix[ia, ib] = sims[k].item()

    return sim_matrix


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def short_label(func_name):
    idx = func_name.find("_sub_")
    return func_name[idx + 1:] if idx != -1 else func_name[-12:]


def plot_heatmap(sim_matrix, names_a, names_b, binary_a, binary_b, output_path, vmin_override=None):
    n_a, n_b = sim_matrix.shape

    labels_a = [short_label(n) for n in names_a]
    labels_b = [short_label(n) for n in names_b]

    fig_w = max(10, n_b * 0.18)
    fig_h = max(8,  n_a * 0.18)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Use fixed vmin when provided so two heatmaps can be compared side-by-side
    vmin = vmin_override if vmin_override is not None else min(float(sim_matrix.min()), -0.5)

    im = ax.imshow(sim_matrix, cmap="Greys", vmin=vmin, vmax=0.0, aspect="auto")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("GMN score  (0 = identical · more negative = dissimilar)", fontsize=9)

    def pick_ticks(n, labels, max_ticks=30):
        step = max(1, n // max_ticks)
        idxs = list(range(0, n, step))
        return idxs, [labels[i] for i in idxs]

    y_idxs, y_lbls = pick_ticks(n_a, labels_a)
    x_idxs, x_lbls = pick_ticks(n_b, labels_b)

    ax.set_yticks(y_idxs)
    ax.set_yticklabels(y_lbls, fontsize=6)
    ax.set_xticks(x_idxs)
    ax.set_xticklabels(x_lbls, fontsize=6, rotation=90)

    ax.set_ylabel(f"Binary A — {binary_a[:20]}...  (rows sorted by function size ↓)", fontsize=8)
    ax.set_xlabel(f"Binary B — {binary_b[:20]}...  (columns sorted by function size ↓)", fontsize=8)
    ax.set_title(
        f"Function-Level GMN Similarity Heatmap\n"
        f"Top {n_a} × {n_b} non-boilerplate functions  |  darker = more similar",
        fontsize=11,
    )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[+] Heatmap saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GMN function similarity heatmap for two malware binaries.")
    parser.add_argument("--binary_a",   required=True)
    parser.add_argument("--binary_b",   required=True)
    parser.add_argument("--db",         required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output",     default="results/heatmap.png")
    parser.add_argument("--top_n",      type=int, default=50)
    parser.add_argument("--min_blocks", type=int, default=5)
    parser.add_argument("--vmin",       type=float, default=None,
                        help="Fixed colormap min (e.g. -30). Use SAME value on both heatmaps to compare.")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    config = get_default_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    model, _ = build_model(config, node_feature_dim=64, edge_feature_dim=1)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    print("[+] Model loaded.")

    packer = BinaryDataset.__new__(BinaryDataset)

    print(f"[*] Loading top {args.top_n} functions (≥{args.min_blocks} blocks) for A...")
    names_a, funcs_a = load_top_functions(args.db, args.binary_a, args.top_n, args.min_blocks)
    print(f"[*] Loading top {args.top_n} functions (≥{args.min_blocks} blocks) for B...")
    names_b, funcs_b = load_top_functions(args.db, args.binary_b, args.top_n, args.min_blocks)

    if not funcs_a or not funcs_b:
        print("[-] No functions found. Check names with:")
        print(f"    python3 evaluate_malware.py --db {args.db} --list")
        return

    total = len(funcs_a) * len(funcs_b)
    print(f"[*] {len(funcs_a)} × {len(funcs_b)} = {total} GMN comparisons — computing...")

    sim_matrix = compute_sim_matrix(model, config, device, packer, funcs_a, funcs_b, args.batch_size)

    print(f"[*] Score range:       [{sim_matrix.min():.3f}, {sim_matrix.max():.3f}]")
    print(f"[*] Score percentiles — "
          f"p25: {np.percentile(sim_matrix, 25):.1f}  "
          f"p50: {np.percentile(sim_matrix, 50):.1f}  "
          f"p75: {np.percentile(sim_matrix, 75):.1f}  "
          f"p90: {np.percentile(sim_matrix, 90):.1f}")

    plot_heatmap(sim_matrix, names_a, names_b, args.binary_a, args.binary_b,
                 args.output, vmin_override=args.vmin)


if __name__ == "__main__":
    main()
