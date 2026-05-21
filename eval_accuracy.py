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

    db_path = "compiled_binaries/test_database.sqlite"
    print(f"[*] Loading test dataset from {db_path}...")
    test_set = BinaryDataset(db_path)

    print("[*] Running final evaluation on TEST set...")

    accumulated_pair_auc = []
    
    # --- NEW: Counters for Accuracy, Precision, and Recall ---
    total_tp = 0  # True Positives
    total_fp = 0  # False Positives
    total_tn = 0  # True Negatives
    total_fn = 0  # False Negatives
    
    # The rejection border we calculated from your CSV data!
    THRESHOLD = -1.20 

    with torch.no_grad():
        print("   Running full metric evaluation...")
        
        num_test_batches = 500 
        test_generator = test_set.pairs(config['evaluation']['batch_size'])
        
        for step in range(num_test_batches):
            batch = next(test_generator)
            node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
            labels = labels.to(device)

            eval_n_graphs = int(graph_idx.max().item()) + 1

            graph_vectors = model(node_features.to(device), edge_features.to(device),
                                  from_idx.to(device), to_idx.to(device),
                                  graph_idx.to(device), eval_n_graphs)

            x, y = reshape_and_split_tensor(graph_vectors, 2)
            similarity = compute_similarity(config, x, y)
            
            # 1. Calculate traditional AUC
            pair_auc = auc(similarity, labels)
            accumulated_pair_auc.append(pair_auc)
            
            # 2. Calculate TP, TN, FP, FN for this batch
            sim_scores = similarity.detach().cpu().numpy()
            true_labels = labels.detach().cpu().numpy()
            
            # Labels are usually 1 for Match, -1 for Mismatch
            is_true_match = (true_labels > 0)
            
            # AI predicts "Match" if the score is higher than the -1.20 threshold
            is_pred_match = (sim_scores > THRESHOLD)
            
            total_tp += np.sum(is_pred_match & is_true_match)
            total_tn += np.sum(~is_pred_match & ~is_true_match)
            total_fp += np.sum(is_pred_match & ~is_true_match)
            total_fn += np.sum(~is_pred_match & is_true_match)
            
            if (step + 1) % 50 == 0:
                print(f"   ... Processed {step + 1}/{num_test_batches} batches")

    # --- NEW: Calculate Final Metrics ---
    total_predictions = total_tp + total_tn + total_fp + total_fn
    accuracy = (total_tp + total_tn) / total_predictions if total_predictions > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    print("\n" + "="*60)
    print("🎯 FINAL TEST RESULTS (64,000 Pairs)")
    print("="*60)
    print(f"Pair AUC          : {np.mean(accumulated_pair_auc):.4f}")
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print("-" * 60)
    print(f"True Positives  (Correct Matches)   : {total_tp}")
    print(f"True Negatives  (Correct Rejects)   : {total_tn}")
    print(f"False Positives (Fake Matches)      : {total_fp}")
    print(f"False Negatives (Missed Matches)    : {total_fn}")
    print("="*60)

if __name__ == "__main__":
    main()