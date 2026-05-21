import torch
import numpy as np
from binary_dataset import BinaryDataset 
from utils import get_graph, reshape_and_split_tensor, build_model
from evaluation import compute_similarity, auc
from configure import get_default_config

def main():
    config = get_default_config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"[*] Using device: {device}")
    print("[*] Loading best model...")

    # Build model and load the best checkpoint
    model, _ = build_model(config, node_feature_dim=64, edge_feature_dim=1)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    print("✅ Best model loaded successfully!\n")

    # --- UPDATE: Point this to your new SQLite database! ---
    db_path = "compiled_binaries/test_database.sqlite"
    print(f"[*] Loading test dataset from {db_path}...")
    test_set = BinaryDataset(db_path)

    print("[*] Running final evaluation on TEST set...")

    accumulated_pair_auc = []

    with torch.no_grad():
        # === Pair AUC Evaluation ===
        print("   Running Pair AUC evaluation...")
        
        # Limit the loop so it doesn't run forever
        num_test_batches = 500  # You can increase this for a more thorough test!
        test_generator = test_set.pairs(config['evaluation']['batch_size'])
        
        for step in range(num_test_batches):
            batch = next(test_generator)
            node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
            labels = labels.to(device)

            # Dynamic graph counting to prevent segment.py crash
            eval_n_graphs = int(graph_idx.max().item()) + 1

            graph_vectors = model(node_features.to(device), edge_features.to(device),
                                  from_idx.to(device), to_idx.to(device),
                                  graph_idx.to(device), eval_n_graphs)

            x, y = reshape_and_split_tensor(graph_vectors, 2)
            similarity = compute_similarity(config, x, y)
            pair_auc = auc(similarity, labels)
            accumulated_pair_auc.append(pair_auc)
            
            # Optional: Print progress so you know it hasn't frozen
            if (step + 1) % 10 == 0:
                print(f"   ... Processed {step + 1}/{num_test_batches} batches")

    print("\n" + "="*60)
    print("🎯 FINAL TEST RESULTS")
    print("="*60)
    print(f"Pair AUC          : {np.mean(accumulated_pair_auc):.4f}")
    print("="*60)

if __name__ == "__main__":
    main()