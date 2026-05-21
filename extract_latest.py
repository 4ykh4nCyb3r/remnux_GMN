import json
import sys
import os

# Set our feature dimension to the "sweet spot"
VECTOR_SIZE = 64

# A robust 64-dimension mapping for x86/x64 assembly
INSTRUCTION_CLASSES = {
    # Data Transfer & Stack (0-7)
    "mov": 0, "movzx": 0, "movsx": 0, "movabs": 0,
    "lea": 1,
    "push": 2, "pushfq": 2,
    "pop": 3, "popfq": 3,
    "xchg": 4, "cmpxchg": 4,
    "cwtl": 5, "cltq": 5, "cqto": 5,
    "bswap": 6,
    "cmov": 7, "cmove": 7, "cmovne": 7, "cmovg": 7, "cmovl": 7, "cmova": 7, "cmovb": 7,

    # Arithmetic (8-19)
    "add": 8, "xadd": 8,
    "sub": 9,
    "inc": 10,
    "dec": 11,
    "imul": 12, "mul": 12,
    "idiv": 13, "div": 13,
    "cmp": 14,
    "test": 15,
    "neg": 16,
    "adc": 17, # Add with carry
    "sbb": 18, # Subtract with borrow
    "nop": 19, # Placed here as a neutral operation

    # Logic & Bitwise (20-29)
    "and": 20,
    "or": 21,
    "xor": 22,
    "not": 23,
    "shl": 24, "sal": 24,
    "shr": 25, "sar": 25,
    "rol": 26, "ror": 26,
    "rcl": 27, "rcr": 27,
    "set": 28, "sete": 28, "setne": 28, "setg": 28, "setl": 28,
    "bt": 29, "bts": 29, "btr": 29, "btc": 29,

    # Control Flow & Branching (30-45)
    "jmp": 30,
    "je": 31, "jz": 31,
    "jne": 32, "jnz": 32,
    "jg": 33, "jnle": 33,
    "jge": 34, "jnl": 34,
    "jl": 35, "jnge": 35,
    "jle": 36, "jng": 36,
    "ja": 37, "jnbe": 37,
    "jae": 38, "jnb": 38, "jnc": 38,
    "jb": 39, "jnae": 39, "jc": 39,
    "jbe": 40, "jna": 40,
    "jo": 41, "jno": 41,
    "js": 42, "jns": 42,
    "call": 43,
    "ret": 44, "repz": 44,
    "leave": 45,

    # Floating Point (FPU) (46-51)
    "fld": 46, "fst": 46, "fstp": 46,
    "fadd": 47, "fsub": 47,
    "fmul": 48, "fdiv": 48,
    "fcom": 49, "fucom": 49,
    "fxch": 50,
    "fnstsw": 51, "fstsw": 51,

    # Vector / SIMD (SSE/AVX) (52-58)
    "movaps": 52, "movups": 52, "movapd": 52, "movdqa": 52, "movdqu": 52, "vmovaps": 52,
    "addps": 53, "addpd": 53, "vaddps": 53,
    "subps": 54, "subpd": 54, "vsubps": 54,
    "mulps": 55, "mulpd": 55, "vmulps": 55,
    "divps": 56, "divpd": 56, "vdivps": 56,
    "xorps": 57, "pxor": 57, "vxorps": 57,
    "ucomiss": 58, "ucomisd": 58,

    # System & Interrupts (59-62)
    "syscall": 59, "sysenter": 59, "int": 59,
    "hlt": 60,
    "cpuid": 61,
    "rdtsc": 62
}
# Anything not explicitly listed above will fall into Bucket 63 (Other)

def extract_features(dissected_json):
    ai_ready_graphs = []
    
    for func in dissected_json:
        # Dictionary to map hex addresses to 0-based integer indexes
        addr_to_idx = {}
        node_features = []
        
        # 1. Process Nodes and build the address map
        for idx, block in enumerate(func.get("blocks", [])):
            addr_to_idx[block["address"]] = idx
            
            # Initialize a 64-dimensional vector of zeros for this block
            feature_vector = [0.0] * VECTOR_SIZE
            
            # 2. Vectorize instructions
            for ins_string in block.get("instructions", []):
                try:
                    # Clean the string: "0x41b930:\tmov\trdx..." -> "mov"
                    # Split by ':' to remove address, strip whitespace, split by spaces/tabs to get mnemonic
                    mnemonic = ins_string.split(":", 1)[1].strip().split()[0].lower()
                    
                    # Look up the mnemonic bucket. Default to 63 ("Other") if not found.
                    class_idx = INSTRUCTION_CLASSES.get(mnemonic, 63)
                    feature_vector[class_idx] += 1.0
                    
                except IndexError:
                    # Failsafe for empty or malformed instruction strings
                    feature_vector[63] += 1.0
                    
            node_features.append(feature_vector)
            
        # 3. Process Edges and map them to integers
        integer_edges = []
        for edge in func.get("edges", []):
            src_hex = edge[0]
            dst_hex = edge[1]
            
            # Only add the edge if both nodes exist in our block list
            # (This ignores calls to external functions/libraries not in this CFG)
            if src_hex in addr_to_idx and dst_hex in addr_to_idx:
                src_idx = addr_to_idx[src_hex]
                dst_idx = addr_to_idx[dst_hex]
                integer_edges.append([src_idx, dst_idx])
                
        # 4. Save the finalized, AI-ready graph
        ai_ready_graphs.append({
            "name": func.get("function_name", "unknown"),
            "node_features": node_features,
            "edges": integer_edges
        })
        
    return ai_ready_graphs

def main():
    # Enforce command line arguments
    if len(sys.argv) != 3:
        print("Usage: python extract_features.py <input_dissected.json> <output_ai_ready.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        sys.exit(1)

    print(f"[*] Reading raw graphs from {input_file}...")
    with open(input_file, "r") as f:
        dissected_json = json.load(f)

    print(f"[*] Extracting 64-dimensional features...")
    ai_ready_graphs = extract_features(dissected_json)

    print(f"[*] Saving {len(ai_ready_graphs)} AI-ready graphs to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(ai_ready_graphs, f, indent=2)
        
    print("[+] Done!")

if __name__ == "__main__":
    main()