import json
import sqlite3
from pathlib import Path

def build_db(input_path, db_path):
    # Ensure input path exists
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"⚠️  Skipping {input_path} (Not found)")
        return

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table and index
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS graphs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            function_name TEXT,
            graph_data TEXT
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_func_name ON graphs(function_name)')
    
    # Check if input is a single file or a directory of files
    if input_path.is_file():
        ready_files = [input_path]
    else:
        ready_files = sorted(list(input_path.glob("*_ready.json")))
        
    print(f"\n[*] Building database {db_path} from {len(ready_files)} files...")
    total_inserted = 0

    for file_path in ready_files:
        with open(file_path, 'r') as in_f:
            functions = json.load(in_f)
            
        batch = [(func['name'], json.dumps(func)) for func in functions]
        cursor.executemany('INSERT INTO graphs (function_name, graph_data) VALUES (?, ?)', batch)
        conn.commit()
        
        total_inserted += len(functions)
        print(f"   ✓ {file_path.name} -> Added {len(functions)} rows")
        del functions 
        
    conn.close()
    print(f"✅ DONE: {total_inserted} total functions saved to {db_path}!")

if __name__ == "__main__":
    # Just list your inputs and desired outputs here!
    datasets = [
        # (Input folder or file, Output Database name)
        ("compiled_binaries/train_2", "compiled_binaries/train_database.sqlite"),
        ("compiled_binaries/val_ready.json", "compiled_binaries/val_database.sqlite"),
        ("compiled_binaries/test_ready.json", "compiled_binaries/test_database.sqlite")
    ]
    
    for in_path, out_db in datasets:
        build_db(in_path, out_db)