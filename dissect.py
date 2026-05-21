import angr
import json
import argparse
import os

def analyze_binary(binary_path, output_json):
    if not os.path.exists(binary_path):
        print(f"[-] Error: The file '{binary_path}' does not exist.")
        return

    # --- NEW CODE: Extract the prefix from the binary filename ---
    # Example: 'compiled_binaries/busybox_gcc_O2' -> 'busybox_gcc_O2'
    base_filename = os.path.basename(binary_path)
    # Example: 'busybox_gcc_O2' -> 'busybox'
    binary_prefix = base_filename.split('_')[0]
    # -------------------------------------------------------------

    print(f"[*] Loading '{binary_path}' into angr...")
    # auto_load_libs=False makes the analysis much faster and focuses on your code
    proj = angr.Project(binary_path, load_options={'auto_load_libs': False})
    
    print("[*] Generating Control Flow Graph (CFG)...")
    cfg = proj.analyses.CFGFast()

    dissection_data = []

    print("[*] Extracting Blocks and Edges...")
    for addr, func in cfg.kb.functions.items():
        # Filter out compiler boilerplate and library stubs
        if func.is_simprocedure or func.name.startswith('_') or func.is_alignment:
            continue

        # --- NEW CODE: Prepend the prefix to the function name ---
        prefixed_func_name = f"{binary_prefix}_{func.name}"
        # ---------------------------------------------------------

        func_info = {
            "function_name": prefixed_func_name, # Updated to use the new name
            "address": hex(addr),
            "blocks": [],
            "edges": []
        }
        
        # 1. Extract the Nodes (Basic Blocks) and their Instructions
        for block in func.blocks:
            block_info = {
                "address": hex(block.addr),
                "size": block.size,
                "instructions": [str(ins) for ins in block.capstone.insns]
            }
            func_info["blocks"].append(block_info)
        
        # 2. Extract the Edges (The Connections)
        for src_node, dst_node in func.transition_graph.edges():
            edge = [hex(src_node.addr), hex(dst_node.addr)]
            func_info["edges"].append(edge)
        
        # Only save functions that actually contain blocks
        if len(func_info["blocks"]) > 0:
            dissection_data.append(func_info)

    print(f"[*] Saving full graph data to '{output_json}'...")
    with open(output_json, 'w') as f:
        json.dump(dissection_data, f, indent=4)
    print("[+] Done!\n")

if __name__ == "__main__":
    # Set up the command-line arguments
    parser = argparse.ArgumentParser(description="Dissect a binary into CFG JSON for Machine Learning.")
    parser.add_argument("binary", help="Path to the target binary file (e.g., ./v1)")
    parser.add_argument("-o", "--output", help="Output JSON file name", default="angr_dissection.json")
    
    args = parser.parse_args()
    
    # Run the analysis
    analyze_binary(args.binary, args.output)