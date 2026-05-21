#!/usr/bin/env python3
import os
import subprocess
import glob
from pathlib import Path

def process_binary(binary_path, output_raw, output_ready):
    """Dissect one binary → raw JSON → ready semantic features"""
    print(f"[*] Processing {binary_path.name}")
    
    # 1. Dissect
    subprocess.run([
        "python", "dissect.py", str(binary_path), "-o", output_raw
    ], check=True)
    
    # 2. Extract 64-dim semantic features
    subprocess.run([
        "python", "extract_latest.py", output_raw, output_ready
    ], check=True)

def main():
    base_dir = Path("compiled_binaries")
    splits = ["test"]
    
    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            print(f"⚠️  {split_dir} does not exist, skipping...")
            continue
            
        all_functions = []
        ready_files = []
        
        # # Find all binaries in this split
        # binaries = list(split_dir.glob("*")) + list(split_dir.glob("**/*"))
        # binaries = [b for b in binaries if b.is_file() and not b.name.startswith('.')]
        
        # print(f"\n=== Processing {split.upper()} split ({len(binaries)} binaries) ===")

        # Find all binaries in this split (ignoring subdirectories, hidden files, and JSON outputs)
        binaries = []
        for b in split_dir.glob("*"):
            if b.is_file() and not b.name.startswith('.') and not b.name.endswith('.json'):
                binaries.append(b)
        
        print(f"\n=== Processing {split.upper()} split ({len(binaries)} binaries) ===")
        
        for binary in binaries:
            raw_json = split_dir / f"{binary.name}_raw.json"
            ready_json = split_dir / f"{binary.name}_ready.json"
            
            process_binary(binary, str(raw_json), str(ready_json))
            
            # Load the ready JSON and collect all functions
            import json
            with open(ready_json, 'r') as f:
                funcs = json.load(f)
            all_functions.extend(funcs)
            ready_files.append(ready_json)
        
        # Save combined ready JSON for this split
        combined_file = base_dir / f"{split}_ready.json"
        with open(combined_file, 'w') as f:
            json.dump(all_functions, f, indent=2)
        
        print(f"[+] {split} ready: {len(all_functions)} functions saved to {combined_file}")

if __name__ == "__main__":
    main()