from binary_dataset_1 import BinaryDataset
from evaluation import compute_similarity, auc
from loss import pairwise_loss, triplet_loss
from utils import *
from configure import *
import numpy as np
import torch.nn as nn
import collections
import time
import os

# ==================== CONFIG ====================
config = get_default_config()
for (k, v) in config.items():
    print("%s= %s" % (k, v))

# ==================== DEVICE SETUP ====================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[*] Using device: {device}")    

# Set random seeds
seed = config['seed']
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# ==================== DATASETS ====================
base = "compiled_binaries"
training_set   = BinaryDataset(f"{base}/train_ready.json")
validation_set = BinaryDataset(f"{base}/val_ready.json")
# test_set     = BinaryDataset(f"{base}/test_ready.json")   # used later in evaluate.py

# ==================== MODEL & OPTIMIZER ====================
node_feature_dim = 64   # from extract_latest.py
edge_feature_dim = 1
model, optimizer = build_model(config, node_feature_dim, edge_feature_dim)
model.to(device)

# ==================== CHECKPOINT SETUP ====================
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
best_val_auc = 0.0

print(f"[*] Best model will be saved to: {best_model_path}")

# ==================== TRAINING LOOP ====================
accumulated_metrics = collections.defaultdict(list)
training_n_graphs_in_batch = config['training']['batch_size'] * 2   # pair mode

t_start = time.time()

for i_iter in range(config['training']['n_training_steps']):
    model.train()
    batch = next(training_set.pairs(config['training']['batch_size']))
    node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
    labels = labels.to(device)

    graph_vectors = model(node_features.to(device), edge_features.to(device),
                          from_idx.to(device), to_idx.to(device),
                          graph_idx.to(device), training_n_graphs_in_batch)

    x, y = reshape_and_split_tensor(graph_vectors, 2)
    loss = pairwise_loss(x, y, labels,
                         loss_type=config['training']['loss'],
                         margin=config['training']['margin']).mean()

    # Compute metrics for logging
    sim = compute_similarity(config, x, y)
    is_pos = (labels == 1).float()
    is_neg = (labels == -1).float()
    sim_pos = torch.sum(sim * is_pos) / (torch.sum(is_pos) + 1e-8)
    sim_neg = torch.sum(sim * is_neg) / (torch.sum(is_neg) + 1e-8)

    # Regularizer
    graph_vec_scale = torch.mean(graph_vectors ** 2)
    if config['training']['graph_vec_regularizer_weight'] > 0:
        loss = loss + config['training']['graph_vec_regularizer_weight'] * 0.5 * graph_vec_scale

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(model.parameters(), config['training']['clip_value'])
    optimizer.step()

    # Logging
    accumulated_metrics['loss'].append(loss.item())
    accumulated_metrics['sim_pos'].append(sim_pos.item())
    accumulated_metrics['sim_neg'].append(sim_neg.item())

    # ==================== EVALUATION & CHECKPOINT ====================
    if (i_iter + 1) % config['training']['print_after'] == 0:
        metrics_to_print = {k: np.mean(v) for k, v in accumulated_metrics.items()}
        info_str = ', '.join([f"{k} {v:.4f}" for k, v in metrics_to_print.items()])
        accumulated_metrics = collections.defaultdict(list)

        # Run validation
        if (i_iter + 1) // config['training']['print_after'] % config['training']['eval_after'] == 0:
            model.eval()
            with torch.no_grad():
                accumulated_pair_auc = []
                for batch in validation_set.pairs(config['evaluation']['batch_size']):
                    node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
                    labels = labels.to(device)
                    graph_vectors = model(node_features.to(device), edge_features.to(device),
                                          from_idx.to(device), to_idx.to(device),
                                          graph_idx.to(device), config['evaluation']['batch_size'] * 2)
                    x, y = reshape_and_split_tensor(graph_vectors, 2)
                    similarity = compute_similarity(config, x, y)
                    pair_auc = auc(similarity, labels)
                    accumulated_pair_auc.append(pair_auc)

                current_val_auc = np.mean(accumulated_pair_auc)
                info_str += f", val/pair_auc {current_val_auc:.4f}"

                # Save best checkpoint
                if current_val_auc > best_val_auc:
                    best_val_auc = current_val_auc
                    torch.save(model.state_dict(), best_model_path)
                    print(f"\n💾 NEW BEST MODEL SAVED! Val AUC = {current_val_auc:.4f} (step {i_iter+1})")

            model.train()

        print(f"iter {i_iter+1:5d}, {info_str}, time {time.time()-t_start:.1f}s")
        t_start = time.time()

print("\n🎉 Training finished!")
print(f"Best validation AUC: {best_val_auc:.4f}")
print(f"Best model saved at: {best_model_path}")