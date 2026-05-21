import json
from pathlib import Path

def combine_memory_efficient():
    input_dir = Path("compiled_binaries/train_2")
    output_file = Path("compiled_binaries/train_ready.json")
    
    ready_files = sorted(list(input_dir.glob("*_ready.json")))
    print(f"[*] Combining {len(ready_files)} files into {output_file}...")

    total_functions = 0

    # 1. Open the output file and start the giant JSON array manually
    with open(output_file, 'w') as out_f:
        out_f.write("[\n") 
        
        first_item = True
        
        for file_path in ready_files:
            # 2. Load only ONE file into RAM at a time
            with open(file_path, 'r') as in_f:
                functions = json.load(in_f)
                
            print(f"   ✓ {file_path.name} → {len(functions)} functions")
            total_functions += len(functions)
            
            # 3. Dump each function to the hard drive immediately
            for func in functions:
                if not first_item:
                    out_f.write(",\n")
                json.dump(func, out_f)
                first_item = False
                
            # 4. Explicitly delete the list to free up RAM for the next loop!
            del functions 
            
        # 5. Close the giant array
        out_f.write("\n]\n") 

    print(f"\n✅ DONE: {total_functions} total functions saved safely without crashing!")

if __name__ == "__main__":
    combine_memory_efficient()