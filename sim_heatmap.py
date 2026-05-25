"""
Generates a function-level similarity heatmap between two malware binaries.
Takes the top N largest non-boilerplate functions from each binary,
runs GMN inference on every pair, and plots a grayscale heatmap.

Usage:
  python3 similarity_heatmap.py \
      --binary_a <sha256_hash_A> \
      --binary_b <sha256_hash_B> \
      --db malware_eval.sqlite \
      --top_n 100 \
      --output heatmap.png
"""

import torch
import sqlite3
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from binary_dataset import BinaryDataset
from evaluation import compute_similarity
from utils import reshape_and_split_tensor, build_model
from configure import get_default_config

# ── Boilerplate function name filter ─────────────────────────────────────────
BOILERPLATE_PREFIXES = (
    'sub_',        # unnamed stripped functions are fine, keep them
)
BOILERPLATE_EXACT = {
    # Common compiler/libc boilerplate to skip
    '__libc_start_main', '_start', '__cxa_finalize',
    '__gmon_start__', 'frame_dummy', '__do_global_dtors_aux',
    'register_tm_clones', 'deregister_tm_clones',
    '__init', '_fini', '_init',
}

def is_boilerplate(func_name):
    name = func_name.split('_', 1)[-1] if '_' in func_name else func_name
    if name in BOILERPLATE_EXACT:
        return True
    # Keep sub_XXXX — those are the actual stripped malware functions
    return False

# ── Load top-N largest functions for a binary ────────────────────────────────
def load_top_functions(cursor, binary_name, top_n):
    cursor.execute(
        'SELECT func_name, graph_data FROM functions WHERE binary_name = ?',
        (binary_name,)
    )
    rows = cursor.fetchall()

    funcs = []
    for func_name, graph_data_str in rows:
        if is_boilerplate(func_name):
            continue
        graph = json.loads(graph_data_str)
        n_blocks = len(graph.get('node_features', []))
        n_edges  = len(graph.get('from_idx', []))
        size     = n_blocks + n_edges   # proxy for function complexity/size
        funcs.append((func_name, graph, size))

    # Sort by size descending, take top N
    funcs.sort(key=lambda x: x[2], reverse=True)
    top = funcs[:top_n]

    print(f"  [*] {binary_name[:16]}... → {len(rows)} total funcs, "
          f"{len(funcs)} non-boilerplate, using top {len(top)}")
    return top

# ── GMN inference for one pair ────────────────────────────────────────────────
def infer_pair(model, config, device, packer, graph_a, graph_b):
    try:
        packed = packer._pack_batch([graph_a, graph_b])

        node_f    = torch.tensor(packed.node_features, dtype=torch.float32).to(device)
        edge_f    = torch.tensor(packed.edge_features, dtype=torch.float32).to(device)
        from_idx  = torch.tensor(packed.from_idx,      dtype=torch.long).to(device)
        to_idx    = torch.tensor(packed.to_idx,        dtype=torch.long).to(device)
        graph_idx = torch.tensor(packed.graph_idx,     dtype=torch.long).to(device)

        eval_n_graphs = int(graph_idx.max().item()) + 1

        with torch.no_grad():
            graph_vectors = model(node_f, edge_f, from_idx, to_idx,
                                  graph_idx, eval_n_graphs)
            x, y  = reshape_and_split_tensor(graph_vectors, 2)
            score = compute_similarity(config, x, y).item()
        return score
    except Exception as e:
        return None

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary_a',   required=True,  help='SHA256 hash of binary A')
    parser.add_argument('--binary_b',   required=True,  help='SHA256 hash of binary B')
    parser.add_argument('--db',         default='malware_eval.sqlite')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    parser.add_argument('--top_n',      type=int, default=100)
    parser.add_argument('--output',     default='heatmap.png')
    args = parser.parse_args()

    # -- Load model
    config = get_default_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")

    model, _ = build_model(config, node_feature_dim=64, edge_feature_dim=1)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    print("[*] Model loaded")

    # -- Load functions
    conn   = sqlite3.connect(args.db)
    cursor = conn.cursor()
    packer = BinaryDataset.__new__(BinaryDataset)

    print("[*] Loading functions...")
    funcs_A = load_top_functions(cursor, args.binary_a, args.top_n)
    funcs_B = load_top_functions(cursor, args.binary_b, args.top_n)
    conn.close()

    if not funcs_A or not funcs_B:
        print("❌ One or both binaries returned no functions. Check binary names in DB.")
        return

    nA = len(funcs_A)
    nB = len(funcs_B)
    print(f"[*] Matrix size: {nA} × {nB} = {nA * nB} pairs")
    print("[*] Running inference — this may take a while...\n")

    # -- Build score matrix
    score_matrix = np.zeros((nA, nB), dtype=np.float32)

    for i, (name_a, graph_a, _) in enumerate(funcs_A):
        for j, (name_b, graph_b, _) in enumerate(funcs_B):
            score = infer_pair(model, config, device, packer, graph_a, graph_b)
            score_matrix[i, j] = score if score is not None else -2.0

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{nA} rows done...")

    # -- Normalize scores to [0, 1] for display
    # Scores are in range roughly [-1.5, 0.0]
    # We want: 0.0 (most similar) → white/light, very negative → dark
    # So invert: higher raw score = lighter shade
    vmin = score_matrix[score_matrix > -2.0].min() if (score_matrix > -2.0).any() else -1.5
    vmax = 0.0

    print(f"\n[*] Score range in matrix: {vmin:.4f} to {score_matrix.max():.4f}")

    # -- Plot heatmap
    fig_w = max(14, nB // 5)
    fig_h = max(10, nA // 5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Use gray_r: high score (similar) = dark, low score = light
    # This gives "bolder = more similar" as requested
    im = ax.imshow(
        score_matrix,
        cmap='gray_r',
        vmin=vmin,
        vmax=vmax,
        aspect='auto',
        interpolation='nearest'
    )

    # -- Labels: short function names (strip binary prefix)
    def short_name(full_name, idx):
        parts = full_name.split('_', 1)
        short = parts[1] if len(parts) > 1 else full_name
        return f"{idx}:{short[:18]}"

    labels_A = [short_name(f[0], i) for i, f in enumerate(funcs_A)]
    labels_B = [short_name(f[0], j) for j, f in enumerate(funcs_B)]

    # Only show tick labels if N is small enough to be readable
    if nA <= 40:
        ax.set_yticks(range(nA))
        ax.set_yticklabels(labels_A, fontsize=6)
    else:
        ax.set_yticks(range(0, nA, max(1, nA // 20)))
        ax.set_yticklabels(
            [labels_A[i] for i in range(0, nA, max(1, nA // 20))],
            fontsize=6
        )

    if nB <= 40:
        ax.set_xticks(range(nB))
        ax.set_xticklabels(labels_B, fontsize=6, rotation=90)
    else:
        ax.set_xticks(range(0, nB, max(1, nB // 20)))
        ax.set_xticklabels(
            [labels_B[j] for j in range(0, nB, max(1, nB // 20))],
            fontsize=6, rotation=90
        )

    # -- Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('GMN Similarity Score\n(darker = more similar)', fontsize=9)

    # -- Titles and labels
    ax.set_xlabel(
        f"Binary B — {args.binary_b[:20]}...\n(top {nB} functions by size)",
        fontsize=10
    )
    ax.set_ylabel(
        f"Binary A — {args.binary_a[:20]}...\n(top {nA} functions by size)",
        fontsize=10
    )
    ax.set_title(
        f"Function-Level Similarity Heatmap\n"
        f"TLSH-similar binary pair  |  GMN model  |  {nA}×{nB} function pairs",
        fontsize=12, fontweight='bold', pad=15
    )

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\n[+] Heatmap saved to: {args.output}")

    # -- Print top 5 most similar pairs as a bonus
    flat_indices = np.argsort(score_matrix.ravel())[::-1][:5]
    print("\n[*] Top 5 most similar function pairs:")
    for rank, flat_idx in enumerate(flat_indices):
        i, j = divmod(int(flat_idx), nB)
        print(f"  {rank+1}. {funcs_A[i][0][:30]}  ↔  {funcs_B[j][0][:30]}"
              f"  score={score_matrix[i,j]:.4f}")

if __name__ == '__main__':
    main()
