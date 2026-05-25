"""
Find binary pairs by TLSH similarity from a directory of samples.

Usage:
  python3 find_pairs.py eval_unpacked/ similar random
  python3 find_pairs.py eval_unpacked/ similar top
  python3 find_pairs.py eval_unpacked/ similar bottom
  python3 find_pairs.py eval_unpacked/ dissimilar random
  python3 find_pairs.py eval_unpacked/ dissimilar top
  python3 find_pairs.py eval_unpacked/ dissimilar bottom

  similar   top    → 2 closest pairs (lowest TLSH dist)
  similar   bottom → 2 furthest pairs still within 1-39
  similar   random → 2 random pairs from 1-39
  dissimilar top    → 2 pairs just above threshold (dist 40-60 range)
  dissimilar bottom → 2 most distant pairs
  dissimilar random → 2 random pairs from >= 40
"""

import os
import sys
import random
import tlsh

def compute_hashes(samples_dir):
    hashes = {}
    for fname in os.listdir(samples_dir):
        fpath = os.path.join(samples_dir, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, 'rb') as f:
            data = f.read()
        try:
            h = tlsh.hash(data)
            if h and h != 'TNULL':
                hashes[fname] = h
        except Exception as e:
            print(f"  [!] TLSH failed for {fname[:16]}: {e}")
    return hashes

def find_pairs(samples_dir, mode, selection, n=2):
    print(f"[*] Computing TLSH hashes for files in {samples_dir} ...")
    hashes = compute_hashes(samples_dir)
    print(f"[*] Hashed {len(hashes)} binaries successfully")

    names = list(hashes.keys())
    results = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            try:
                dist = tlsh.diff(hashes[a], hashes[b])
            except Exception:
                continue

            if mode == 'similar' and 1 <= dist <= 39:
                results.append((a, b, dist))
            elif mode == 'dissimilar' and dist >= 40:
                results.append((a, b, dist))

    if not results:
        print(f"[!] No {mode} pairs found.")
        sys.exit(1)

    print(f"[*] Found {len(results)} {mode} pairs total.")

    # Sort ascending by distance
    results.sort(key=lambda x: x[2])

    if selection == 'top':
        # lowest distance (most similar) or just above threshold (least dissimilar)
        chosen = results[:n]
    elif selection == 'bottom':
        # highest distance within the group
        chosen = results[-n:]
    elif selection == 'random':
        chosen = random.sample(results, min(n, len(results)))
    else:
        print(f"Error: selection must be 'top', 'bottom', or 'random'")
        sys.exit(1)

    label_map = {
        ('similar',    'top'):    'closest similar',
        ('similar',    'bottom'): 'furthest similar (still within 1-39)',
        ('similar',    'random'): 'random similar',
        ('dissimilar', 'top'):    'least dissimilar (just above 40)',
        ('dissimilar', 'bottom'): 'most dissimilar',
        ('dissimilar', 'random'): 'random dissimilar',
    }
    label = label_map.get((mode, selection), f'{selection} {mode}')
    print(f"[*] Selection: {label}\n")

    for idx, (a, b, dist) in enumerate(chosen):
        print(f"  Pair {idx + 1}:")
        print(f"    Binary A : {a}")
        print(f"    Binary B : {b}")
        print(f"    TLSH dist: {dist}")
        print()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 find_pairs.py <samples_dir> <similar|dissimilar> <top|bottom|random>")
        sys.exit(1)

    samples_dir = sys.argv[1]
    mode        = sys.argv[2]
    selection   = sys.argv[3]

    if mode not in ('similar', 'dissimilar'):
        print("Error: mode must be 'similar' or 'dissimilar'")
        sys.exit(1)

    if selection not in ('top', 'bottom', 'random'):
        print("Error: selection must be 'top', 'bottom', or 'random'")
        sys.exit(1)

    if not os.path.isdir(samples_dir):
        print(f"Error: '{samples_dir}' is not a directory")
        sys.exit(1)

    find_pairs(samples_dir, mode, selection, n=2)
