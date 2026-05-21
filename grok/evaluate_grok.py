from binary_dataset import BinaryDataset
from utils import get_graph, reshape_and_split_tensor, build_model
from evaluation import compute_similarity, auc
from configure import get_default_config
import torch
import numpy as np

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

# Load the test dataset
test_set = BinaryDataset("compiled_binaries/test_ready.json")

print("[*] Running final evaluation on TEST set...")

accumulated_pair_auc = []
accumulated_triplet_acc = []

with torch.no_grad():
    # === Pair AUC Evaluation ===
    print("   Running Pair AUC evaluation...")
    for batch in test_set.pairs(config['evaluation']['batch_size']):
        node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
        labels = labels.to(device)

        graph_vectors = model(node_features.to(device), edge_features.to(device),
                              from_idx.to(device), to_idx.to(device),
                              graph_idx.to(device), config['evaluation']['batch_size'] * 2)

        x, y = reshape_and_split_tensor(graph_vectors, 2)
        similarity = compute_similarity(config, x, y)
        pair_auc = auc(similarity, labels)
        accumulated_pair_auc.append(pair_auc)

    # === Triplet Accuracy (optional but useful) ===
    print("   Running Triplet Accuracy evaluation...")
    for batch in test_set.pairs(config['evaluation']['batch_size']):  # reuse pairs for simplicity
        node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
        graph_vectors = model(node_features.to(device), edge_features.to(device),
                              from_idx.to(device), to_idx.to(device),
                              graph_idx.to(device), config['evaluation']['batch_size'] * 2)
        x_1, y, x_2, z = reshape_and_split_tensor(graph_vectors, 4)
        sim_pos = compute_similarity(config, x_1, y)
        sim_neg = compute_similarity(config, x_2, z)
        triplet_acc = torch.mean((sim_pos > sim_neg).float())
        accumulated_triplet_acc.append(triplet_acc.item())

print("\n" + "="*60)
print("🎯 FINAL TEST RESULTS")
print("="*60)
print(f"Pair AUC          : {np.mean(accumulated_pair_auc):.4f}")
print(f"Triplet Accuracy  : {np.mean(accumulated_triplet_acc):.4f}")
print("="*60)