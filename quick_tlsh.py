#!/usr/bin/env python3

from sys import argv
import tlsh

h = []
for f in argv[1:]:
    try:
        with open(f, 'rb') as f:
            data = f.read()
        hash = tlsh.hash(data)
        print(hash)
    except:
        hash = f
    h.append(hash)

if len(h) == 2:
    print(f"diff: {tlsh.diff(*h)}")
