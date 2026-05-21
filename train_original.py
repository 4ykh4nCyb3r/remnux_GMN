import os
import time
import random
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from binary_dataset import BinaryDataset
from evaluation import compute_similarity, auc
from loss import pairwise_loss, triplet_loss
from utils import *
from configure import *

# ==================== PYTORCH DATA WRAPPER ====================
class GraphDatabaseWrapper(IterableDataset):
    def __init__(self, db_path, batch_size):
        self.dataset = BinaryDataset(db_path)
        self.batch_size = batch_size

    def __iter__(self):
        return self.dataset.pairs(self.batch_size)

# ==================== CONFIG ====================
config = get_default_config()
for (k, v) in config.items():
    print("%s= %s" % (k, v))

# ==================== DEVICE SETUP ====================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[*] Using device: {device}")    

# Set random seeds for dataset shuffling
seed = config['seed']
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed + 2) 

# --- FAST SETTINGS ACTIVATED ---
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# ==================== DATASETS & LOADERS ====================
base = "compiled_binaries"
train_db = f"{base}/train_database.sqlite"
val_db = f"{base}/val_database.sqlite"

print("[*] Initializing Database Connections...")

train_dataset = GraphDatabaseWrapper(train_db, config['training']['batch_size'])

# You used 6 workers in your last run, so I locked it in here!
num_workers = 10 #changed from 6 

train_loader = DataLoader(
    train_dataset, 
    num_workers=num_workers, 
    batch_size=None,     
    pin_memory=True,
    prefetch_factor=4, #added to speed up data loading, Queues up 40 batches (10 workers * 4) in advance
    persistent_workers=True  #Keeps them alive for maximum speed
)

validation_set = BinaryDataset(val_db)

# ==================== MODEL & OPTIMIZER ====================
node_feature_dim = 64   
edge_feature_dim = 1
model, optimizer = build_model(config, node_feature_dim, edge_feature_dim)
model.to(device)

# ==================== CHECKPOINT SETUP ====================
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
best_val_auc = 0.0

print(f"[*] Best model will be saved to: {best_model_path}")
print(f"[*] Starting training with {num_workers} background workers...\n")

# ==================== TRAINING LOOP ====================
accumulated_metrics = collections.defaultdict(list)

# --- START TENSORBOARD ---
writer = SummaryWriter(log_dir="runs/massive_dataset_run")
print("[*] 📊 TensorBoard is recording! Run 'tensorboard --logdir runs' in another terminal.\n")

t_start = time.time()
total_steps = config['training']['n_training_steps']

# --- ADD TQDM PROGRESS BAR ---
for step, batch_data in tqdm(enumerate(train_loader), total=total_steps, desc="Training"):
    if step >= total_steps:
        break

    model.train()
    
    node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch_data)
    labels = labels.to(device)

    actual_n_graphs = int(graph_idx.max().item()) + 1

    graph_vectors = model(node_features.to(device), edge_features.to(device),
                          from_idx.to(device), to_idx.to(device),
                          graph_idx.to(device), actual_n_graphs) 

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

    # --- LOG LIVE STATS TO TENSORBOARD ---
    writer.add_scalar('Training/Loss', loss.item(), step)
    writer.add_scalar('Training/Pos_Similarity', sim_pos.item(), step)
    writer.add_scalar('Training/Neg_Similarity', sim_neg.item(), step)

    # ==================== EVALUATION & CHECKPOINT ====================
    if (step + 1) % config['training']['print_after'] == 0:
        metrics_to_print = {k: np.mean(v) for k, v in accumulated_metrics.items()}
        
        # We don't need a massive info_str anymore because of the progress bar, 
        # but we reset the accumulated metrics for the next batch.
        accumulated_metrics = collections.defaultdict(list)

        # Run validation
        if (step + 1) // config['training']['print_after'] % config['training']['eval_after'] == 0:
            model.eval()
            with torch.no_grad():
                accumulated_pair_auc = []
                
                num_eval_batches = 10 
                val_generator = validation_set.pairs(config['evaluation']['batch_size'])
                
                for _ in range(num_eval_batches):
                    batch_val_data = next(val_generator)
                    
                    node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch_val_data)
                    labels = labels.to(device)
                    
                    eval_n_graphs = int(graph_idx.max().item()) + 1
                    
                    graph_vectors = model(node_features.to(device), edge_features.to(device),
                                          from_idx.to(device), to_idx.to(device),
                                          graph_idx.to(device), eval_n_graphs) 
                    
                    x, y = reshape_and_split_tensor(graph_vectors, 2)
                    similarity = compute_similarity(config, x, y)
                    pair_auc = auc(similarity, labels)
                    accumulated_pair_auc.append(pair_auc)

                current_val_auc = np.mean(accumulated_pair_auc)
                
                # --- LOG VALIDATION TO TENSORBOARD ---
                writer.add_scalar('Validation/AUC', current_val_auc, step)

                # Save best checkpoint
                if current_val_auc > best_val_auc:
                    best_val_auc = current_val_auc
                    torch.save(model.state_dict(), best_model_path)
                    
                    # Make the progress bar print a message without breaking the visual line!
                    tqdm.write(f"\n💾 NEW BEST MODEL SAVED! Val AUC = {current_val_auc:.4f} (step {step+1})")

            model.train()

writer.close()
print("\n🎉 Training finished!")
print(f"Best validation AUC: {best_val_auc:.4f}")
print(f"Best model saved at: {best_model_path}")