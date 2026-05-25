import os
import math

def entropy(filepath):
    with open(filepath, "rb") as f:
        data = f.read()
    if not data:
        return 0
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    e = 0
    for f in freq:
        if f:
            p = f / len(data)
            e -= p * math.log2(p)
    return e

samples_dir = "/home/remnux/GMN/x86_samples"

print(f"{'Entropy':>8}  {'Status':>14}  Filename")
print("-" * 90)

ok = []
suspicious = []
likely_packed = []

for fname in sorted(os.listdir(samples_dir)):
    fpath = os.path.join(samples_dir, fname)
    if not os.path.isfile(fpath):
        continue
    e = entropy(fpath)
    if e > 7.2:
        status = "LIKELY PACKED"
        likely_packed.append(fname)
    elif e > 6.5:
        status = "SUSPICIOUS"
        suspicious.append(fname)
    else:
        status = "ok"
        ok.append(fname)
    print(f"{e:>8.4f}  {status:>14}  {fname}")

print()
print("=" * 90)
print(f"  OK (entropy <= 6.5):        {len(ok)} samples")
print(f"  Suspicious (6.5 - 7.2):     {len(suspicious)} samples")
print(f"  Likely packed (> 7.2):      {len(likely_packed)} samples")
print("=" * 90)

if likely_packed:
    print("\nLikely packed samples:")
    for f in likely_packed:
        print(f"  {f}")

if suspicious:
    print("\nSuspicious samples:")
    for f in suspicious:
        print(f"  {f}")
