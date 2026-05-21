import torch
import numpy as np
import sqlite3
import json
from binary_dataset import BinaryDataset
from evaluation import compute_similarity
from utils import reshape_and_split_tensor, build_model
from configure import get_default_config

def main():
    # 1. Configuration & Setup
    config = get_default_config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"[*] Using device: {device}")
    print("[*] Loading best model...")

    # 2. Build model and load the best checkpoint
    model, _ = build_model(config, node_feature_dim=64, edge_feature_dim=1)
    # Using your updated checkpoint path
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    print("✅ Best model loaded successfully!\n")

    # 3. Load the dataset wrapper (points to your new SQLite DB)
    db_path = "compiled_binaries/test_database.sqlite"
    dataset = BinaryDataset(db_path)

    print("\nAvailable functions (first 50 curl functions):")
    # BinaryDataset now stores names in a list called valid_function_names
    print([name for name in dataset.valid_function_names if name.startswith("curl")][:50])

    print("\nAvailable functions (first 50 wget functions):")
    print([name for name in dataset.valid_function_names if name.startswith("wget")][:50])

    # =====================================================================
    # 4. DEFINE YOUR TEST
    # =====================================================================
    func_name1 = "curl_single_transfer"   #curl_main, curl_single_transfer, curl_ipfs_url_rewrite, curl_curl_easy_setopt, curl_base64_encode, curl_curl_mime_encoder
    func_name2 = "curl_ipfs_url_rewrite" #wget_gethttp, wget_ftp_retrieve_glob, wget_url_parse, wget_main

    print(f"\n==========================================")
    print(f"🔍 COMPARING: '{func_name1}' vs '{func_name2}'")
    print("AI similarity match score range {0.0} --- {-1.2}")
    print(f"==========================================")

    if func_name1 not in dataset.valid_function_names or func_name2 not in dataset.valid_function_names:
        print(f"❌ Error: One or both of the functions were not found in {db_path}!")
        return

    # 5. Fetch all compiled versions from the SQLite Database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all versions of Function 1 (ORDER BY id keeps the original file insertion order)
    cursor.execute('SELECT graph_data FROM graphs WHERE function_name = ? ORDER BY id ASC', (func_name1,))
    versions_A = [json.loads(row[0]) for row in cursor.fetchall()]

    # Get all versions of Function 2
    cursor.execute('SELECT graph_data FROM graphs WHERE function_name = ? ORDER BY id ASC', (func_name2,))
    versions_B = [json.loads(row[0]) for row in cursor.fetchall()]
    
    conn.close()

    # 6. Grab the specific compiled versions safely
    # If the requested index exists, grab it. Otherwise, default to the first available version (index 0).
    idx_A = 0 if len(versions_A) > 1 else 0 #gcc_O1
    idx_B = 0 if len(versions_B) > 5 else 0 #clang_O3
    
    version_A = versions_A[idx_A]
    version_B = versions_B[idx_B]

    # 7. Pack the graphs for the neural network
    batch_graphs = [version_A, version_B]
    packed = dataset._pack_batch(batch_graphs)

    # 8. Run Inference
    with torch.no_grad():
        # Safely convert numpy arrays directly to PyTorch tensors
        node_f = torch.tensor(packed.node_features, dtype=torch.float32).to(device)
        edge_f = torch.tensor(packed.edge_features, dtype=torch.float32).to(device)
        from_idx = torch.tensor(packed.from_idx, dtype=torch.long).to(device)
        to_idx = torch.tensor(packed.to_idx, dtype=torch.long).to(device)
        graph_idx = torch.tensor(packed.graph_idx, dtype=torch.long).to(device)

        # Safely count graphs (matches train.py and evaluate.py logic)
        eval_n_graphs = int(graph_idx.max().item()) + 1

        # Pass through the Graph Matching Network
        graph_vectors = model(node_f, edge_f, from_idx, to_idx, graph_idx, eval_n_graphs)

        # Unzip the 1D output back into Graph A (x) and Graph B (y)
        x, y = reshape_and_split_tensor(graph_vectors, 2)
        
        # Calculate final similarity score
        similarity = compute_similarity(config, x, y)

    # 9. Output Results
    print(f"\n🧠 AI Similarity Score: {similarity.item():.4f}")
    print(f"   (Higher score = More mathematically similar)")
    print(f"==========================================\n")

if __name__ == "__main__":
    main()