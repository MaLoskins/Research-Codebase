#!/usr/bin/env python3
"""
Simplified DySAT implementation for the PHEME dataset.
Heavily modified with stability improvements and aggressive simplification.
Updated for comprehensive results saving and ablation studies.

Usage:
  Single run:
  python train_dysat_pheme.py --data-dir data_dysat --event germanwings-crash --hidden-dim 64 --lr 5e-4

  Optuna study and aggregation (full ablation):
  python train_dysat_pheme.py --data-dir data_dysat --optuna-study --optuna-trials 50 \
         --optuna-study-name "PHEME_SimpleDySAT_Study" \
         --optuna-event-subset "germanwings-crash,charliehebdo" \
         --events-for-aggregation "all" --epochs 150

  Aggregate with CLI HPs:
  python train_dysat_pheme.py --data-dir data_dysat --generate-aggregated-report-for-cli-hps \
         --events-for-aggregation "germanwings-crash,charliehebdo,ottawashooting" \
         --hidden-dim 128 --lr 1e-4 --dropout 0.4 --epochs 150
"""

import argparse
import gc
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import matplotlib as mpl
mpl.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torchmetrics.classification import Accuracy, F1Score
from tqdm import tqdm

# Optional: Import Optuna for automated HP tuning
try:
    import optuna
    import optuna.visualization as vis
    OPTUNA_AVAILABLE = True
    try:
        import kaleido # For saving plotly to static images
        KALEIDO_AVAILABLE = True
    except ImportError:
        KALEIDO_AVAILABLE = False
        logging.warning("Kaleido not found. Optuna plots will be saved as HTML or skipped if PNG saving fails. `pip install kaleido`")
except ImportError:
    OPTUNA_AVAILABLE = False
    KALEIDO_AVAILABLE = False # Not relevant if Optuna is not available

# ---------------------------------------------------------------------------
# Global Configuration & Setup
# ---------------------------------------------------------------------------
BASE_RESULTS_DIR = Path("RESULTS") / "SimpleDySAT"

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Data loading helpers (Unchanged from original, retained for completeness)
# ---------------------------------------------------------------------------
def load_temporal_data(
    event_dir: Path,
    device: torch.device,
    cpu_offload: bool = False,
    max_nodes: Optional[int] = None
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor, Dict[str, Any]]:
    """Load preprocessed temporal data for an event."""
    X_numpy = np.load(event_dir / "X.npy").astype(np.float32)
    labels_numpy = np.load(event_dir / "labels.npy").astype(np.int64)

    X = torch.from_numpy(X_numpy)
    labels = torch.from_numpy(labels_numpy)
    if labels.ndim == 0:
        labels = labels.unsqueeze(0)

    with open(event_dir / "time_info.json", "r") as f:
        time_info = json.load(f)

    num_time_steps = time_info["num_windows"]

    edge_indices_list = []
    for t in range(num_time_steps):
        edge_index = torch.from_numpy(np.load(event_dir / "edge_indices" / f"t{t}.npy").astype(np.int64))
        edge_indices_list.append(edge_index)

    node_masks_list = []
    for t in range(num_time_steps):
        mask_numpy = np.load(event_dir / "node_masks" / f"t{t}.npy")
        mask = torch.from_numpy(mask_numpy)
        if mask.ndim == 0:
             mask = mask.unsqueeze(0)
        node_masks_list.append(mask)

    if max_nodes is not None and X.size(0) > max_nodes:
        original_num_nodes = X.size(0)
        logging.info(f"Subsampling from {original_num_nodes} to {max_nodes} nodes for event {event_dir.name}")

        perm_idx = torch.randperm(original_num_nodes)[:max_nodes]
        X = X[perm_idx].contiguous()
        labels = labels[perm_idx].contiguous()

        new_idx_map_device = 'cpu'
        new_idx_map = torch.full((original_num_nodes,), -1, dtype=torch.long, device=new_idx_map_device)
        new_idx_map[perm_idx.to(new_idx_map_device)] = torch.arange(max_nodes, dtype=torch.long, device=new_idx_map_device)

        remapped_edge_indices = []
        for t_edge_index in edge_indices_list:
            t_edge_index_cpu = t_edge_index.cpu()
            if t_edge_index_cpu.numel() == 0:
                remapped_edge_indices.append(torch.zeros((2,0), dtype=torch.long))
                continue
            src, dst = t_edge_index_cpu[0], t_edge_index_cpu[1]
            # Clamp indices to be within the valid range of original_num_nodes
            src_clamped = torch.clamp(src, 0, original_num_nodes - 1)
            dst_clamped = torch.clamp(dst, 0, original_num_nodes - 1)

            mask_src_in_subsample = new_idx_map[src_clamped] != -1
            mask_dst_in_subsample = new_idx_map[dst_clamped] != -1
            valid_edge_mask = mask_src_in_subsample & mask_dst_in_subsample

            filtered_src = src[valid_edge_mask]
            filtered_dst = dst[valid_edge_mask]

            remapped_src = new_idx_map[filtered_src]
            remapped_dst = new_idx_map[filtered_dst]
            remapped_edge_indices.append(torch.stack([remapped_src, remapped_dst], dim=0))
        edge_indices_list = remapped_edge_indices

        remapped_node_masks = []
        for t_node_mask in node_masks_list:
            if t_node_mask.ndim == 1 and t_node_mask.shape[0] == original_num_nodes:
                remapped_node_masks.append(t_node_mask[perm_idx].contiguous())
            elif t_node_mask.ndim == 0 and original_num_nodes == 1 and perm_idx.numel() == 1 and perm_idx.item() == 0 :
                remapped_node_masks.append(t_node_mask.unsqueeze(0).contiguous())
            else:
                logging.warning(f"Node mask shape mismatch during subsampling for event {event_dir.name}. Original mask shape: {t_node_mask.shape}, expected ({original_num_nodes},). Creating all-False mask for subsampled size {max_nodes}.")
                remapped_node_masks.append(torch.zeros(max_nodes, dtype=torch.bool))
        node_masks_list = remapped_node_masks

    if not cpu_offload:
        X = X.to(device)
        labels = labels.to(device)
        edge_indices_list = [ei.to(device) for ei in edge_indices_list]
        node_masks_list = [nm.to(device) for nm in node_masks_list]

    return X, edge_indices_list, node_masks_list, labels, time_info

# ---------------------------------------------------------------------------
# SIMPLIFIED MODEL DEFINITION: SimpleDySAT (Unchanged)
# ---------------------------------------------------------------------------
class SimpleAttentionLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.linear_transform = nn.Linear(input_dim, output_dim, bias=False)
        self.attention_weights_mlp = nn.Linear(output_dim * 2, 1, bias=False)
        self.layer_norm = nn.LayerNorm(output_dim)

        self.projection = None
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim, bias=False)

        nn.init.xavier_uniform_(self.linear_transform.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.attention_weights_mlp.weight, gain=nn.init.calculate_gain('relu'))
        if self.projection is not None:
            nn.init.xavier_uniform_(self.projection.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x_input_features, edge_index):
        N = x_input_features.size(0)
        h_transformed = self.linear_transform(x_input_features)
        h_aggregated_messages = torch.zeros_like(h_transformed)

        if edge_index.size(1) > 0:
            src_nodes, dst_nodes = edge_index[0], edge_index[1]
            h_src_for_attn = h_transformed[src_nodes]
            h_dst_for_attn = h_transformed[dst_nodes]
            edge_attn_input = torch.cat([h_src_for_attn, h_dst_for_attn], dim=-1)
            raw_attn_scores = self.attention_weights_mlp(edge_attn_input).squeeze(-1)
            raw_attn_scores = F.leaky_relu(raw_attn_scores, 0.2)

            try:
                from torch_scatter import scatter_softmax, scatter_add
                alpha_for_edges = scatter_softmax(raw_attn_scores, dst_nodes, dim=0, dim_size=N)
                alpha_for_edges = F.dropout(alpha_for_edges, self.dropout, training=self.training)
                messages_to_aggregate = h_transformed[src_nodes] * alpha_for_edges.unsqueeze(-1)
                h_aggregated_messages = scatter_add(messages_to_aggregate, dst_nodes, dim=0, out=h_aggregated_messages, dim_size=N)
            except ImportError:
                for i in range(N):
                    mask_edges_to_i = (dst_nodes == i)
                    if mask_edges_to_i.sum() > 0:
                        scores_for_node_i = raw_attn_scores[mask_edges_to_i]
                        alpha_for_node_i = F.softmax(scores_for_node_i, dim=0)
                        alpha_for_node_i = F.dropout(alpha_for_node_i, self.dropout, training=self.training)
                        source_nodes_for_i = src_nodes[mask_edges_to_i]
                        h_sources_for_node_i = h_transformed[source_nodes_for_i]
                        weighted_sum = (h_sources_for_node_i * alpha_for_node_i.unsqueeze(-1)).sum(dim=0)
                        h_aggregated_messages[i] = weighted_sum

        if self.projection is not None:
            residual_features = self.projection(x_input_features)
        else:
            residual_features = x_input_features

        h_final = h_aggregated_messages + residual_features
        h_final = self.layer_norm(h_final)
        h_final = F.dropout(h_final, self.dropout, training=self.training)
        return h_final

class SimpleDySAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_temporal: bool = True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_temporal = use_temporal

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.structural_attn = SimpleAttentionLayer(hidden_dim, hidden_dim, dropout=self.dropout)

        if use_temporal:
            self.temporal_weight = nn.Parameter(torch.tensor(0.5))
            self.temporal_norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight, gain=nn.init.calculate_gain('relu'))
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('linear'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_original, edge_indices, node_masks):
        N = x_original.size(0)
        num_time_steps = len(edge_indices)
        x_projected = self.input_proj(x_original)
        x_projected = F.elu(x_projected)
        x_projected_dropout = F.dropout(x_projected, self.dropout, training=self.training)

        if not self.use_temporal:
            last_valid_t = -1
            for t_idx in range(num_time_steps - 1, -1, -1):
                if isinstance(node_masks[t_idx], torch.Tensor) and node_masks[t_idx].sum() > 0 and \
                   isinstance(edge_indices[t_idx], torch.Tensor) and edge_indices[t_idx].size(1) > 0:
                    last_valid_t = t_idx
                    break
            if last_valid_t != -1:
                final_embedding = self.structural_attn(x_projected_dropout, edge_indices[last_valid_t])
            else: # Fallback if no valid timestep has edges
                final_embedding = x_projected_dropout
        else:
            timestep_embeddings = []
            for t in range(num_time_steps):
                current_mask = node_masks[t]
                current_edges = edge_indices[t]
                is_mask_valid = isinstance(current_mask, torch.Tensor) and current_mask.numel() > 0
                is_edges_valid = isinstance(current_edges, torch.Tensor)
                if is_mask_valid and current_mask.sum() > 0 :
                    # Process even if no edges, structural_attn handles empty edge_index
                    h_t = self.structural_attn(x_projected_dropout, current_edges)
                    timestep_embeddings.append(h_t)

            if not timestep_embeddings:
                final_embedding = x_projected_dropout
            else:
                last_embedding = timestep_embeddings[-1]
                if len(timestep_embeddings) > 1:
                    all_embeddings_stacked = torch.stack(timestep_embeddings, dim=0)
                    avg_embedding = all_embeddings_stacked.mean(dim=0)
                    temporal_w = torch.sigmoid(self.temporal_weight)
                    final_embedding = temporal_w * last_embedding + (1 - temporal_w) * avg_embedding
                    final_embedding = self.temporal_norm(final_embedding)
                    final_embedding = F.dropout(final_embedding, self.dropout, training=self.training)
                else:
                    final_embedding = last_embedding
        logits = self.classifier(final_embedding)
        return logits

# ---------------------------------------------------------------------------
# Training and evaluation functions (Unchanged logic, minor logging adjustments)
# ---------------------------------------------------------------------------
def create_balanced_splits(labels, node_masks, device=None, event_name=""):
    n_nodes = labels.size(0)
    active_nodes_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device)
    for mask_t in node_masks:
        if isinstance(mask_t, torch.Tensor) and mask_t.numel() > 0 and mask_t.shape[0] == n_nodes:
             active_nodes_mask = active_nodes_mask | mask_t.to(labels.device).bool()
    active_indices = torch.nonzero(active_nodes_mask).squeeze()

    if active_indices.numel() == 0:
        logging.warning(f"[{event_name}] No active nodes found across all timesteps. Splits will be empty.")
        empty_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device)
        return empty_mask.clone(), empty_mask.clone(), empty_mask.clone()

    active_labels = labels[active_indices]
    train_ratio, val_ratio = 0.7, 0.15 # val_ratio is 0.15 of total, so (0.15 / (1-0.7)) = 0.5 of temp
    indices_np = active_indices.cpu().numpy()
    labels_np = active_labels.cpu().numpy()

    min_samples_per_class_train = 2
    unique_labels, counts = np.unique(labels_np, return_counts=True)

    # Check if stratification is possible for train/temp split
    can_stratify_train_temp = len(unique_labels) >= 2 and \
                              np.all(counts >= min_samples_per_class_train) and \
                              len(labels_np) >= (min_samples_per_class_train * len(unique_labels) / (1 - train_ratio))

    if not can_stratify_train_temp:
        logging.warning(f"[{event_name}] Not enough samples or classes for stratified train/temp split "
                        f"({len(unique_labels)} classes, counts {counts}). Using non-stratified split.")
        train_idx, temp_idx = train_test_split(indices_np, test_size=(1-train_ratio), random_state=42, shuffle=True)
    else:
        train_idx, temp_idx, _, _ = train_test_split(
            indices_np, labels_np, test_size=(1-train_ratio), random_state=42, stratify=labels_np
        )

    # For val/test split from temp_idx
    temp_labels_np = labels[torch.from_numpy(temp_idx).to(labels.device).long()].cpu().numpy() # Get labels for temp_idx
    unique_temp_labels, temp_counts = np.unique(temp_labels_np, return_counts=True)
    min_samples_per_class_val_test = 1 # Need at least 1 per class for stratify, but train_test_split needs 2 for stratify
                                      # We are splitting 50/50, so need 2 per class in temp set.

    can_stratify_val_test = len(unique_temp_labels) >= 2 and \
                            np.all(temp_counts >= 2) and \
                            len(temp_labels_np) >= 4 # Need at least 2 per class for 2 classes to stratify

    if not can_stratify_val_test:
         logging.warning(f"[{event_name}] Not enough samples or classes in temp_idx for stratified val/test split "
                         f"({len(unique_temp_labels)} classes, counts {temp_counts}). Using non-stratified.")
         if len(temp_idx) > 0: # Ensure temp_idx is not empty
            val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, shuffle=True)
         else:
            val_idx, test_idx = np.array([]), np.array([])
    else:
         val_idx, test_idx, _, _ = train_test_split(
            temp_idx, temp_labels_np, test_size=0.5, random_state=42, stratify=temp_labels_np
        )

    train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device)

    if len(train_idx) > 0: train_mask[torch.from_numpy(train_idx).to(labels.device).long()] = True
    if len(val_idx) > 0: val_mask[torch.from_numpy(val_idx).to(labels.device).long()] = True
    if len(test_idx) > 0: test_mask[torch.from_numpy(test_idx).to(labels.device).long()] = True

    return train_mask, val_mask, test_mask

def calculate_class_weights(labels, masks=None, device='cpu', event_name=""):
    combined_mask = torch.zeros_like(labels, dtype=torch.bool)
    if masks:
        for m_tensor in masks:
            if isinstance(m_tensor, torch.Tensor) and m_tensor.numel() > 0:
                combined_mask |= m_tensor.to(labels.device).bool()
    else:
        combined_mask = torch.ones_like(labels, dtype=torch.bool)

    subset_labels = labels[combined_mask]
    if subset_labels.numel() == 0:
        logging.warning(f"[{event_name}] No labels found for class weight calculation (e.g. train_mask empty). Using uniform weights.")
        n_classes = int(labels.max().item() + 1) if labels.numel() > 0 else 2
        return torch.ones(n_classes, device=device) / n_classes

    n_classes_actual = int(labels.max().item() + 1) if labels.numel() > 0 else (int(subset_labels.max().item() + 1) if subset_labels.numel() > 0 else 2)
    counts = torch.bincount(subset_labels.long(), minlength=n_classes_actual)
    weights = torch.tensor([1.0 / c if c > 0 else 0 for c in counts], dtype=torch.float, device=device)

    if weights.sum() == 0:
        logging.warning(f"[{event_name}] All class counts zero in subset for weights. Using uniform weights.")
        return torch.ones(n_classes_actual, device=device) / n_classes_actual

    weights = weights / weights.sum() * n_classes_actual # Normalize and scale
    weights = torch.clamp(weights, min=0.1, max=10.0) # Clamp weights
    return weights.to(device)

def train_epoch(model, optimizer, criterion, X, edge_indices, node_masks, labels,
               train_mask, device, clip_val=1.0, event_name=""):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(X, edge_indices, node_masks)

    if not isinstance(train_mask, torch.Tensor) or train_mask.sum() == 0:
      logging.warning(f"[{event_name}] Train mask is empty or invalid, skipping loss calculation.")
      return 0.0

    train_logits = logits[train_mask]
    train_labels = labels[train_mask]

    if train_logits.shape[0] == 0:
        logging.warning(f"[{event_name}] No samples selected by train_mask. Skipping loss calculation.")
        return 0.0

    loss = criterion(train_logits, train_labels)
    if torch.isnan(loss) or torch.isinf(loss):
        logging.warning(f"[{event_name}] NaN or Inf loss detected: {loss.item()}. Skipping backpropagation.")
        return loss.item() # Return the problematic loss value

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
    optimizer.step()
    return loss.item()

def evaluate(model, X, edge_indices, node_masks, labels, mask, acc_metric, f1_metric, event_name=""):
    model.eval()
    with torch.no_grad():
        logits = model(X, edge_indices, node_masks)
        if not isinstance(mask, torch.Tensor) or mask.sum() == 0:
            # logging.warning(f"[{event_name}] Evaluation mask is empty. Returning 0 for metrics.")
            return 0.0, 0.0 # Expected if val or test set is empty

        preds_on_mask = logits[mask].argmax(dim=1).cpu()
        labels_on_mask = labels[mask].cpu()

        if labels_on_mask.numel() == 0: # Double check after masking
            # logging.warning(f"[{event_name}] No samples in mask after selection. Returning 0 for metrics.")
            return 0.0, 0.0

        acc_metric.reset(); f1_metric.reset()
        acc = acc_metric(preds_on_mask, labels_on_mask)
        f1 = f1_metric(preds_on_mask, labels_on_mask)
    return acc.item(), f1.item()

def train_and_evaluate(
    event_name: str, # Added for better logging and context
    X: torch.Tensor, edge_indices: List[torch.Tensor], node_masks: List[torch.Tensor], labels: torch.Tensor,
    hps: Union[argparse.Namespace, Dict], # Hyperparameters
    device: torch.device
):
    # Extract HPs
    hidden_dim = hps.hidden_dim if isinstance(hps, argparse.Namespace) else hps['hidden_dim']
    lr = hps.lr if isinstance(hps, argparse.Namespace) else hps['lr']
    epochs = hps.epochs if isinstance(hps, argparse.Namespace) else hps['epochs']
    use_temporal = not (hps.no_temporal if isinstance(hps, argparse.Namespace) else hps.get('no_temporal', False))
    dropout_rate = hps.dropout if isinstance(hps, argparse.Namespace) else hps['dropout']
    weight_decay_val = hps.weight_decay if isinstance(hps, argparse.Namespace) else hps['weight_decay']
    scheduler_patience = hps.scheduler_patience if isinstance(hps, argparse.Namespace) else hps['scheduler_patience']
    early_stop_patience = hps.early_stop_patience if isinstance(hps, argparse.Namespace) else hps['early_stop_patience']

    train_mask, val_mask, test_mask = create_balanced_splits(labels, node_masks, device=device, event_name=event_name)
    logging.info(f"[{event_name}] Train samples: {train_mask.sum().item()}, Val samples: {val_mask.sum().item()}, Test samples: {test_mask.sum().item()}")

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        logging.error(f"[{event_name}] Train or Validation split is empty. Cannot proceed with training.")
        return None, 0.0, 0.0, 0.0, {}, 0 # model, val_acc, test_acc, f1, history, best_epoch

    class_weights = calculate_class_weights(labels, [train_mask], device=device, event_name=event_name)
    logging.info(f"[{event_name}] Using class weights: {class_weights.cpu().numpy()}")

    n_classes = int(labels.max().item() + 1) if labels.numel() > 0 else 2
    model = SimpleDySAT(
        in_dim=X.size(1),
        hidden_dim=hidden_dim,
        num_classes=n_classes,
        dropout=dropout_rate,
        use_temporal=use_temporal
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay_val)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=scheduler_patience,
        min_lr=1e-7
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    acc_metric = Accuracy(task="multiclass", num_classes=n_classes, average="micro").cpu() # micro for overall acc
    f1_metric = F1Score(task="multiclass", num_classes=n_classes, average="macro").cpu()

    best_val_acc = 0.0
    best_test_acc_at_best_val = 0.0
    best_f1_at_best_val = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {"loss": [], "val_acc": [], "val_f1": [], "test_acc": [], "test_f1": [], "lr": []}
    best_model_state = model.state_dict()

    for epoch in range(epochs):
        loss = train_epoch(model, optimizer, criterion, X, edge_indices, node_masks, labels, train_mask, device, event_name=event_name)
        val_acc, val_f1 = evaluate(model, X, edge_indices, node_masks, labels, val_mask, acc_metric, f1_metric, event_name=event_name)
        test_acc, test_f1 = evaluate(model, X, edge_indices, node_masks, labels, test_mask, acc_metric, f1_metric, event_name=event_name)

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 10 == 0 or epoch == epochs -1 : # Log less frequently
            logging.info(f"[{event_name}] Epoch {epoch:03d}: loss={loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, test_acc={test_acc:.4f}, test_f1={test_f1:.4f}, lr={current_lr:.7f}")

        history["loss"].append(loss); history["val_acc"].append(val_acc); history["val_f1"].append(val_f1)
        history["test_acc"].append(test_acc); history["test_f1"].append(test_f1); history["lr"].append(current_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc; best_test_acc_at_best_val = test_acc
            best_f1_at_best_val = test_f1; best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            logging.info(f"[{event_name}] Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break
        if current_lr <= scheduler.min_lrs[0] + 1e-9 and patience_counter > early_stop_patience // 2:
            logging.info(f"[{event_name}] Learning rate at minimum and no improvement for {patience_counter} epochs. Stopping at epoch {epoch}.")
            break

    if best_model_state:
         model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    logging.info(f"[{event_name}] Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    logging.info(f"[{event_name}] Corresponding test accuracy: {best_test_acc_at_best_val:.4f}, F1: {best_f1_at_best_val:.4f}")
    return model, best_val_acc, best_test_acc_at_best_val, best_f1_at_best_val, history, best_epoch

# ---------------------------------------------------------------------------
# Results Saving Helpers
# ---------------------------------------------------------------------------
def save_training_plots(history: Dict, event_name: str, output_dir: Path):
    """Saves training plots to a file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_file = output_dir / "training_plots.png"

    if not history or not history.get("loss"):
        logging.warning(f"[{event_name}] No history data to plot. Skipping training plots.")
        return

    plt.figure(figsize=(24, 5))
    plt.subplot(1, 4, 1); plt.plot(history["loss"])
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")

    plt.subplot(1, 4, 2); plt.plot(history["val_acc"], label="Val Acc"); plt.plot(history["test_acc"], label="Test Acc")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Acc"); plt.legend()

    plt.subplot(1, 4, 3); plt.plot(history.get("val_f1",[]), label="Val F1"); plt.plot(history.get("test_f1",[]), label="Test F1")
    plt.title("F1 (Macro)"); plt.xlabel("Epoch"); plt.ylabel("F1"); plt.legend()

    plt.subplot(1, 4, 4); plt.plot(history["lr"]); plt.title("LR"); plt.xlabel("Epoch"); plt.ylabel("LR"); plt.yscale("log")

    plt.suptitle(f"Training Curves for {event_name}"); plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(plot_file)
    plt.close()
    logging.info(f"[{event_name}] Training plots saved to {plot_file}")

def save_event_results_md(
    event_name: str,
    hps: Union[argparse.Namespace, Dict],
    metrics: Dict, # Expects {'val_acc': ..., 'test_acc': ..., 'f1': ..., 'best_epoch': ...}
    output_dir: Path
):
    """Saves per-event results summary to a markdown file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    md_file = output_dir / "results.md"

    # Convert hps to dict if it's a namespace
    hps_dict = vars(hps) if isinstance(hps, argparse.Namespace) else hps

    with open(md_file, "w") as f:
        f.write(f"# Results for Event: {event_name}\n\n")
        f.write("## Hyperparameters Used:\n")
        for key, value in hps_dict.items():
            # Skip some less relevant CLI args for this summary
            if key not in ['data_dir', 'device', 'seed', 'optuna_study', 'optuna_trials',
                           'optuna_study_name', 'optuna_event_subset', 'events_for_aggregation',
                           'generate_aggregated_report_for_cli_hps', 'all_events_list_for_agg']: # internal temp var
                f.write(f"- **{key}**: {value}\n")
        f.write("\n## Performance Metrics:\n")
        f.write(f"- Best Validation Accuracy: {metrics.get('val_acc', 0.0):.4f}\n")
        f.write(f"- Test Accuracy (at best val_acc): {metrics.get('test_acc', 0.0):.4f}\n")
        f.write(f"- Test F1-score (at best val_acc): {metrics.get('f1', 0.0):.4f}\n")
        f.write(f"- Best Epoch: {metrics.get('best_epoch', 0)}\n")
    logging.info(f"[{event_name}] Markdown results summary saved to {md_file}")

def save_aggregated_results_md(
    best_hps: Union[argparse.Namespace, Dict],
    event_metrics_list: List[Dict], # List of {'event_name': ..., 'test_acc': ..., 'f1': ...}
    output_dir: Path
):
    """Saves aggregated results summary to a markdown file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    md_file = output_dir / "aggregated_results.md"

    hps_dict = vars(best_hps) if isinstance(best_hps, argparse.Namespace) else best_hps

    avg_test_acc = np.mean([m['test_acc'] for m in event_metrics_list if m['test_acc'] is not None]) if event_metrics_list else 0.0
    avg_f1 = np.mean([m['f1'] for m in event_metrics_list if m['f1'] is not None]) if event_metrics_list else 0.0

    with open(md_file, "w") as f:
        f.write("# Aggregated Results (Best Overall Hyperparameters)\n\n")
        f.write("## Best Hyperparameters Applied:\n")
        for key, value in hps_dict.items():
            if key not in ['data_dir', 'device', 'seed', 'optuna_study', 'optuna_trials',
                           'optuna_study_name', 'optuna_event_subset', 'events_for_aggregation',
                           'generate_aggregated_report_for_cli_hps', 'all_events_list_for_agg']:
                f.write(f"- **{key}**: {value}\n")
        f.write("\n## Performance per Event:\n")
        f.write("| Event Name | Test Accuracy | Test F1-score |\n")
        f.write("|------------|---------------|---------------|\n")
        for metrics in event_metrics_list:
            f.write(f"| {metrics['event_name']} | {metrics.get('test_acc', 0.0):.4f} | {metrics.get('f1', 0.0):.4f} |\n")

        f.write("\n## Average Performance Across Events:\n")
        f.write(f"- Average Test Accuracy: {avg_test_acc:.4f}\n")
        f.write(f"- Average Test F1-score: {avg_f1:.4f}\n")
    logging.info(f"Aggregated results markdown saved to {md_file}")

# ---------------------------------------------------------------------------
# Core Logic for Different Run Modes
# ---------------------------------------------------------------------------

def run_single_event_evaluation(
    event_name: str,
    data_root: Path,
    hps: Union[argparse.Namespace, Dict],
    device: torch.device,
    max_nodes_subsample: Optional[int]
) -> Optional[Dict]:
    """Loads data, trains, evaluates, and saves results for a single event."""
    event_dir = data_root / event_name
    if not event_dir.is_dir():
        logging.warning(f"[{event_name}] Directory not found. Skipping.")
        return None

    logging.info(f"Processing event: {event_name}")
    event_output_dir = BASE_RESULTS_DIR / event_name
    event_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        X, edge_indices, node_masks, labels, _ = load_temporal_data(event_dir, device, False, max_nodes_subsample)
        if X.nelement() == 0 or labels.nelement() == 0 or not any(isinstance(nm, torch.Tensor) and nm.numel()>0 for nm in node_masks):
            logging.warning(f"[{event_name}] Skipped due to empty data or masks.")
            return {'event_name': event_name, 'test_acc': None, 'f1': None, 'error': True}

        # Display label distribution
        if labels.numel() > 0:
            labels_to_count = labels
            active_nodes_overall_mask = torch.zeros_like(labels, dtype=torch.bool)
            for nm_t in node_masks:
                if isinstance(nm_t, torch.Tensor) and nm_t.shape[0] == labels.shape[0]:
                    active_nodes_overall_mask |= nm_t.to(labels.device).bool()
            if active_nodes_overall_mask.sum() > 0:
                labels_to_count = labels[active_nodes_overall_mask]

            if labels_to_count.numel() > 0 and not torch.is_floating_point(labels_to_count) and labels_to_count.min() >= 0:
                 logging.info(f"[{event_name}] Label distribution (active nodes): {torch.bincount(labels_to_count.long()).tolist()}")


        model, val_acc, test_acc, f1, history, best_epoch = train_and_evaluate(
            event_name, X, edge_indices, node_masks, labels, hps, device
        )

        if model is None: # Training failed (e.g., empty splits)
            return {'event_name': event_name, 'val_acc': 0.0, 'test_acc': None, 'f1': None, 'best_epoch': 0, 'error': True}

        save_training_plots(history, event_name, event_output_dir)
        event_metrics = {'val_acc': val_acc, 'test_acc': test_acc, 'f1': f1, 'best_epoch': best_epoch}
        save_event_results_md(event_name, hps, event_metrics, event_output_dir)

        del X, edge_indices, node_masks, labels, model, history; gc.collect(); torch.cuda.empty_cache()
        return {'event_name': event_name, **event_metrics, 'error': False}

    except Exception as e:
        logging.error(f"[{event_name}] Critical error during processing: {e}", exc_info=True)
        return {'event_name': event_name, 'test_acc': None, 'f1': None, 'error': True}


def run_evaluation_and_aggregation(
    hps_to_apply: Union[argparse.Namespace, Dict],
    event_names_to_run: List[str],
    cli_args: argparse.Namespace, # For data_dir, device, max_nodes
    device: torch.device
):
    """
    Runs evaluation with a given set of HPs on multiple events,
    saves individual event results, and then saves an aggregated report.
    """
    data_root = Path(cli_args.data_dir)
    all_event_run_metrics = []

    logging.info(f"Starting evaluation and aggregation with HPs: {hps_to_apply}")

    for event_name in tqdm(event_names_to_run, desc="Aggregating Events"):
        event_metrics = run_single_event_evaluation(
            event_name, data_root, hps_to_apply, device, cli_args.max_nodes_subsample
        )
        if event_metrics and not event_metrics.get('error'):
            all_event_run_metrics.append(event_metrics)
        elif event_metrics: # Error occurred but dict returned
             all_event_run_metrics.append({'event_name': event_name, 'test_acc': 0.0, 'f1': 0.0}) # for table consistency


    if not all_event_run_metrics:
        logging.warning("No events were successfully processed for aggregation.")
        return

    save_aggregated_results_md(hps_to_apply, all_event_run_metrics, BASE_RESULTS_DIR)
    logging.info("Aggregation process complete.")


# ---------------------------------------------------------------------------
# Optuna Integration
# ---------------------------------------------------------------------------
def objective(trial: optuna.trial.Trial, fixed_args: argparse.Namespace, device: torch.device):
    """Optuna objective function."""
    current_hps_dict = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.6, step=0.05),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "scheduler_patience": trial.suggest_int("scheduler_patience", 5, 15),
        "early_stop_patience": trial.suggest_int("early_stop_patience", 10, 30),
        # Fixed parameters for each trial, but part of the HP set
        "epochs": fixed_args.epochs,
        "no_temporal": fixed_args.no_temporal,
    }
    if current_hps_dict["early_stop_patience"] <= current_hps_dict["scheduler_patience"] + 2:
        current_hps_dict["early_stop_patience"] = current_hps_dict["scheduler_patience"] + 3

    logging.info(f"Optuna Trial {trial.number}: Starting with HPs: {current_hps_dict}")
    set_seed(fixed_args.seed) # Ensure same seed for data loading etc. within a trial if needed by dataset
                             # but train_and_evaluate itself will re-set_seed if necessary or use its own split logic

    data_root = Path(fixed_args.data_dir)
    event_names_for_trial = fixed_args.optuna_event_subset.split(',')
    event_list_to_process_paths = []
    for e_name_raw in event_names_for_trial:
        e_name = e_name_raw.strip()
        event_path = data_root / e_name
        if event_path.exists() and event_path.is_dir():
            event_list_to_process_paths.append(event_path)

    if not event_list_to_process_paths:
        logging.error(f"Optuna Trial {trial.number}: No valid events for tuning: {fixed_args.optuna_event_subset}")
        return -1.0 # Bad score

    all_trial_event_metrics = []
    for event_dir_path in event_list_to_process_paths:
        event_name = event_dir_path.name
        try:
            X, edge_indices, node_masks, labels, _ = load_temporal_data(
                event_dir_path, device, False, fixed_args.max_nodes_subsample
            )
            if X.nelement() == 0 or labels.nelement() == 0 or not any(isinstance(nm, torch.Tensor) and nm.numel()>0 for nm in node_masks):
                logging.warning(f"Optuna Trial {trial.number}: Skipping {event_name} due to empty data.")
                continue

            _, val_acc, _, _, _, _ = train_and_evaluate( # We optimize on val_acc for Optuna
                event_name, X, edge_indices, node_masks, labels, current_hps_dict, device
            )
            if val_acc is not None:
                all_trial_event_metrics.append(val_acc)

            del X, edge_indices, node_masks, labels; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Optuna Trial {trial.number}: Error during training for {event_name}: {e}", exc_info=False)
            # Potentially report failure to Optuna or return a very bad score
            # return -1.0 # This would mark trial as failed and potentially stop early if pruner active

    if not all_trial_event_metrics:
        logging.warning(f"Optuna Trial {trial.number}: No events were successfully processed for this trial.")
        return -1.0

    avg_val_acc = np.mean(all_trial_event_metrics)
    logging.info(f"Optuna Trial {trial.number}: Avg Val Acc: {avg_val_acc:.4f} for HPs: {current_hps_dict}")

    trial.set_user_attr("hps", current_hps_dict) # Store full HPs for later retrieval

    return avg_val_acc # Optuna maximizes this

# ---------------------------------------------------------------------------
# Arg Parsing and Main Execution
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SimpleDySAT Trainer with Comprehensive Results Saving")
    # Core arguments
    parser.add_argument("--data-dir", default="data_dysat", help="Path to preprocessed data directory")
    parser.add_argument("--event", type=str, default=None, help="Event name (for single runs), or comma-separated list. If None and not aggregating, processes all events in data-dir.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-nodes-subsample", type=int, default=None, help="Max nodes for subsampling (applied per event if data is large)")

    # Model Hyperparameters (can be tuned by Optuna or set manually)
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=150, help="Maximum number of training epochs")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay for AdamW optimizer")
    parser.add_argument("--scheduler-patience", type=int, default=10, help="Patience for ReduceLROnPlateau scheduler")
    parser.add_argument("--early-stop-patience", type=int, default=25, help="Patience for early stopping")
    parser.add_argument("--no-temporal", action="store_true", help="Disable temporal component (use last snapshot only)")

    # Optuna specific arguments
    parser.add_argument("--optuna-study", action="store_true", help="Run Optuna hyperparameter study and then aggregate results.")
    parser.add_argument("--optuna-trials", type=int, default=50, help="Number of trials for Optuna study")
    parser.add_argument("--optuna-study-name", type=str, default="SimpleDySAT_PHEME_Study", help="Name for Optuna study (and SQLite DB)")
    parser.add_argument("--optuna-event-subset", type=str, default="germanwings-crash,charliehebdo", help="Comma-separated list of events for Optuna's objective evaluation (faster tuning).")

    # Aggregation arguments
    parser.add_argument("--events-for-aggregation", type=str, default="all", help="Comma-separated list of event names for the final aggregation report (e.g., after Optuna, or with --generate-aggregated-report-for-cli-hps). 'all' uses all events in data_dir.")
    parser.add_argument("--generate-aggregated-report-for-cli-hps", action="store_true", help="Generate aggregated report using HPs passed via CLI (not from Optuna).")

    return parser.parse_args()

if __name__ == "__main__":
    cli_args = parse_args()
    set_seed(cli_args.seed)
    BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(cli_args.device if cli_args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Using device: {device}")

    data_root = Path(cli_args.data_dir)
    if not data_root.exists() or not data_root.is_dir():
        logging.error(f"Data directory '{data_root}' not found. Exiting.")
        exit(1)

    # Prepare list of all available events if "all" is specified for aggregation
    all_available_event_names = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    if not all_available_event_names:
        logging.error(f"No event subdirectories found in '{data_root}'. Exiting.")
        exit(1)
    
    # This attribute will store the actual list of event names for aggregation
    cli_args.all_events_list_for_agg = []
    if cli_args.events_for_aggregation.lower() == 'all':
        cli_args.all_events_list_for_agg = all_available_event_names
    else:
        for e_name_agg in cli_args.events_for_aggregation.split(','):
            e_name_agg_stripped = e_name_agg.strip()
            if e_name_agg_stripped in all_available_event_names:
                cli_args.all_events_list_for_agg.append(e_name_agg_stripped)
            else:
                logging.warning(f"Event '{e_name_agg_stripped}' specified for aggregation not found in data_dir. Skipping it.")
    
    if not cli_args.all_events_list_for_agg and \
       (cli_args.optuna_study or cli_args.generate_aggregated_report_for_cli_hps):
        logging.error(f"No valid events selected for aggregation. Check --events-for-aggregation. Exiting.")
        exit(1)


    if cli_args.optuna_study:
        if not OPTUNA_AVAILABLE:
            logging.error("Optuna is not installed. Please install it: `pip install optuna optuna-dashboard plotly kaleido`")
        else:
            logging.info(f"Starting Optuna study: {cli_args.optuna_study_name} for {cli_args.optuna_trials} trials.")
            logging.info(f"Optuna trials will evaluate on event subset: {cli_args.optuna_event_subset}")
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            study_db_path = BASE_RESULTS_DIR / f"{cli_args.optuna_study_name.replace(' ','_')}.db"
            study = optuna.create_study(
                study_name=cli_args.optuna_study_name,
                storage=f"sqlite:///{study_db_path}",
                direction="maximize", # Maximizing average validation accuracy
                load_if_exists=True,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=3),
                sampler=optuna.samplers.TPESampler(seed=cli_args.seed)
            )
            study.optimize(lambda trial: objective(trial, cli_args, device),
                           n_trials=cli_args.optuna_trials, gc_after_trial=True)

            logging.info(f"\nOptuna study '{cli_args.optuna_study_name}' complete.")
            logging.info(f"Number of finished trials: {len(study.trials)}")

            if not study.trials or study.best_trial is None :
                 logging.error("Optuna study finished with no successful trials. Cannot proceed to aggregation.")
            else:
                best_trial = study.best_trial
                logging.info(f"Best trial value (Avg Val Acc on subset): {best_trial.value:.4f}")
                logging.info("Best hyperparameters found by Optuna:")
                best_hps_from_optuna = best_trial.params # These are only the tuned ones
                # Augment with fixed HPs for the final run
                full_best_hps = {
                    **best_hps_from_optuna,
                    "epochs": cli_args.epochs,
                    "no_temporal": cli_args.no_temporal,
                    # Pass other relevant fixed args if train_and_evaluate needs them from hps dict
                }
                for key, value in full_best_hps.items():
                    logging.info(f"  {key}: {value}")

                # Save Optuna study results CSV
                df_results = study.trials_dataframe()
                optuna_csv_path = BASE_RESULTS_DIR / "optuna_study_results.csv"
                df_results.to_csv(optuna_csv_path, index=False)
                logging.info(f"Full Optuna study results saved to {optuna_csv_path}")

                # Save Optuna parallel coordinate plot
                if OPTUNA_AVAILABLE and vis.is_available():
                    try:
                        params_to_plot = list(best_hps_from_optuna.keys()) # Plot only tuned params
                        fig_parallel = vis.plot_parallel_coordinate(study, params=params_to_plot)
                        plot_path_png = BASE_RESULTS_DIR / "optuna_parallel_coordinate_plot.png"
                        plot_path_html = BASE_RESULTS_DIR / "optuna_parallel_coordinate_plot.html"

                        if KALEIDO_AVAILABLE:
                            fig_parallel.write_image(plot_path_png)
                            logging.info(f"Optuna parallel coordinate plot saved to {plot_path_png}")
                        else:
                            fig_parallel.write_html(plot_path_html)
                            logging.warning(f"Kaleido not installed. Optuna parallel coordinate plot saved as HTML to {plot_path_html}")
                    except Exception as e:
                        logging.error(f"Failed to generate Optuna parallel coordinate plot: {e}")

                logging.info(f"To visualize Optuna results: optuna-dashboard sqlite:///{study_db_path}")

                # Proceed to aggregation with best HPs
                logging.info(f"Now running evaluation on events for aggregation ({cli_args.all_events_list_for_agg}) with best Optuna HPs.")
                run_evaluation_and_aggregation(full_best_hps, cli_args.all_events_list_for_agg, cli_args, device)

    elif cli_args.generate_aggregated_report_for_cli_hps:
        logging.info("Generating aggregated report using HPs from CLI arguments.")
        # Construct HPs dict from cli_args
        cli_hps = {
            "hidden_dim": cli_args.hidden_dim, "lr": cli_args.lr, "epochs": cli_args.epochs,
            "dropout": cli_args.dropout, "weight_decay": cli_args.weight_decay,
            "scheduler_patience": cli_args.scheduler_patience, "early_stop_patience": cli_args.early_stop_patience,
            "no_temporal": cli_args.no_temporal
        }
        logging.info(f"Running evaluation on events ({cli_args.all_events_list_for_agg}) with CLI HPs: {cli_hps}")
        run_evaluation_and_aggregation(cli_hps, cli_args.all_events_list_for_agg, cli_args, device)

    else: # Manual run for specific event(s) with CLI HPs, no aggregation here
        logging.info("Running in manual mode (specific HPs on specified events).")
        events_to_run_manual = []
        if cli_args.event:
            events_to_run_manual = [e.strip() for e in cli_args.event.split(',')]
        else: # If --event is not given, run on all available events
            logging.info("--event not specified, running on all available events individually.")
            events_to_run_manual = all_available_event_names

        if not events_to_run_manual:
            logging.warning("No events specified or found to run in manual mode.")
        else:
            # Construct HPs dict from cli_args for run_single_event_evaluation
            manual_hps = {
                "hidden_dim": cli_args.hidden_dim, "lr": cli_args.lr, "epochs": cli_args.epochs,
                "dropout": cli_args.dropout, "weight_decay": cli_args.weight_decay,
                "scheduler_patience": cli_args.scheduler_patience, "early_stop_patience": cli_args.early_stop_patience,
                "no_temporal": cli_args.no_temporal
            }
            successful_manual_runs = 0
            for event_name_manual in tqdm(events_to_run_manual, desc="Manual Event Runs"):
                result = run_single_event_evaluation(
                    event_name_manual, data_root, manual_hps, device, cli_args.max_nodes_subsample
                )
                if result and not result.get('error'):
                    successful_manual_runs +=1
            logging.info(f"Manual run mode complete. Successfully processed {successful_manual_runs}/{len(events_to_run_manual)} events.")

    logging.info("Script finished.")