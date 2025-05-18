#!/usr/bin/env python3
"""
DySAT implementation for the PHEME dataset, incorporating
Multi-Head Attention and Temporal Self-Attention.

Usage:
python train_OG_dysat_pheme.py --data-dir data_dysat_v2 --event germanwings-crash
python train_OG_dysat_pheme.py --run-full-ablation --event all --optuna-trials 30 --optuna-trials-per-event 15
"""

import argparse
import gc
import json
import logging # Keep this import
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import math
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torchmetrics.classification import Accuracy, F1Score
from tqdm import tqdm

# Optional: Import Optuna for automated HP tuning
try:
    import optuna
    OPTUNA_AVAILABLE = True
    try:
        from optuna.visualization import plot_parallel_coordinate, plot_optimization_history
        OPTUNA_VISUALIZATION_AVAILABLE = True
    except ImportError:
        OPTUNA_VISUALIZATION_AVAILABLE = False
except ImportError:
    OPTUNA_AVAILABLE = False
    OPTUNA_VISUALIZATION_AVAILABLE = False

# Optional: Import torch_scatter
# Logging calls for this will happen AFTER basicConfig in the main block
try:
    from torch_scatter import scatter_softmax, scatter_add
    TORCH_SCATTER_AVAILABLE = True
except ImportError:
    TORCH_SCATTER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "DySAT"
BASE_RESULTS_ROOT = Path("RESULTS")
MAIN_MODEL_RESULTS_DIR = BASE_RESULTS_ROOT / MODEL_NAME

# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------
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
# Data loading helpers
# ---------------------------------------------------------------------------
def load_temporal_data(
    event_dir: Path,
    device: torch.device,
    cpu_offload: bool = False,
    max_nodes: Optional[int] = None
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor, Dict[str, Any]]:
    X_numpy = np.load(event_dir / "X.npy").astype(np.float32)
    labels_numpy = np.load(event_dir / "labels.npy").astype(np.int64)
    X = torch.from_numpy(X_numpy)
    labels = torch.from_numpy(labels_numpy) 
    if labels.ndim == 0: labels = labels.unsqueeze(0)
    with open(event_dir / "time_info.json", "r") as f: time_info = json.load(f)
    num_time_steps = time_info["num_windows"]
    edge_indices_list = [torch.from_numpy(np.load(event_dir / "edge_indices" / f"t{t}.npy").astype(np.int64)) for t in range(num_time_steps)]
    node_masks_list = []
    for t in range(num_time_steps):
        mask_numpy = np.load(event_dir / "node_masks" / f"t{t}.npy") 
        mask = torch.from_numpy(mask_numpy)
        if mask.ndim == 0: mask = mask.unsqueeze(0)
        node_masks_list.append(mask)
    if max_nodes is not None and X.size(0) > max_nodes:
        original_num_nodes = X.size(0)
        logging.info(f"Subsampling from {original_num_nodes} to {max_nodes} nodes for event {event_dir.name}")
        perm_idx = torch.randperm(original_num_nodes)[:max_nodes]
        X = X[perm_idx].contiguous(); labels = labels[perm_idx].contiguous()
        new_idx_map_device = 'cpu' 
        new_idx_map = torch.full((original_num_nodes,), -1, dtype=torch.long, device=new_idx_map_device)
        new_idx_map[perm_idx.to(new_idx_map_device)] = torch.arange(max_nodes, dtype=torch.long, device=new_idx_map_device)
        remapped_edge_indices = []
        for t_edge_index in edge_indices_list:
            t_edge_index_cpu = t_edge_index.cpu() 
            if t_edge_index_cpu.numel() == 0: remapped_edge_indices.append(torch.zeros((2,0), dtype=torch.long)); continue
            src, dst = t_edge_index_cpu[0], t_edge_index_cpu[1]
            src_clamped = torch.clamp(src, 0, original_num_nodes - 1); dst_clamped = torch.clamp(dst, 0, original_num_nodes - 1)
            mask_src_in_subsample = new_idx_map[src_clamped] != -1; mask_dst_in_subsample = new_idx_map[dst_clamped] != -1
            valid_edge_mask = mask_src_in_subsample & mask_dst_in_subsample
            filtered_src = src[valid_edge_mask]; filtered_dst = dst[valid_edge_mask]
            remapped_src = new_idx_map[filtered_src]; remapped_dst = new_idx_map[filtered_dst]
            remapped_edge_indices.append(torch.stack([remapped_src, remapped_dst], dim=0)) 
        edge_indices_list = remapped_edge_indices
        remapped_node_masks = []
        for t_node_mask in node_masks_list:
            if t_node_mask.ndim == 1 and t_node_mask.shape[0] == original_num_nodes: remapped_node_masks.append(t_node_mask[perm_idx].contiguous())
            elif t_node_mask.ndim == 0 and original_num_nodes == 1 and perm_idx.numel() == 1 and perm_idx.item() == 0 : remapped_node_masks.append(t_node_mask.unsqueeze(0).contiguous()) 
            else: logging.warning(f"Node mask shape mismatch for {event_dir.name}. Creating all-False mask."); remapped_node_masks.append(torch.zeros(max_nodes, dtype=torch.bool))
        node_masks_list = remapped_node_masks
    if not cpu_offload:
        X = X.to(device); labels = labels.to(device)
        edge_indices_list = [ei.to(device) for ei in edge_indices_list]
        node_masks_list = [nm.to(device) for nm in node_masks_list]
    return X, edge_indices_list, node_masks_list, labels, time_info

# ---------------------------------------------------------------------------
# DySAT Model Components
# ---------------------------------------------------------------------------

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim_per_head, num_heads, dropout, concat=True, residual=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim_per_head = out_dim_per_head
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.concat = concat
        self.residual = residual
        self.W = nn.Linear(in_dim, num_heads * out_dim_per_head, bias=False)
        self.attn_mlp = nn.Linear(2 * out_dim_per_head, 1, bias=False) 
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        if self.residual:
            if in_dim != (num_heads * out_dim_per_head if concat else out_dim_per_head):
                self.res_fc = nn.Linear(in_dim, num_heads * out_dim_per_head if concat else out_dim_per_head, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.res_fc = nn.Identity()
        self.norm = nn.LayerNorm(num_heads * out_dim_per_head if concat else out_dim_per_head)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.attn_mlp.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
        if self.residual and isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_uniform_(self.res_fc.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x, edge_index):
        N = x.size(0)
        h_transformed = self.W(x) 
        h_transformed_heads = h_transformed.view(N, self.num_heads, self.out_dim_per_head)
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        h_src_for_attn = h_transformed_heads[src_nodes]
        h_dst_for_attn = h_transformed_heads[dst_nodes]
        edge_attn_input = torch.cat([h_src_for_attn, h_dst_for_attn], dim=-1)
        attn_scores = self.attn_mlp(edge_attn_input.reshape(-1, 2 * self.out_dim_per_head)) 
        attn_scores = attn_scores.view(edge_index.size(1), self.num_heads, 1) 
        attn_scores = self.leaky_relu(attn_scores)
        attn_scores_T = attn_scores.permute(1,0,2) 

        if TORCH_SCATTER_AVAILABLE:
            dst_nodes_expanded = dst_nodes.unsqueeze(0).expand(self.num_heads, -1)
            alpha = scatter_softmax(attn_scores_T.squeeze(-1), dst_nodes_expanded, dim=1) 
            alpha = self.dropout(alpha) 
        else: 
            alpha = torch.zeros(self.num_heads, edge_index.size(1), device=x.device)
            for head_idx in range(self.num_heads):
                for i in range(N): 
                    mask_edges_to_i = (dst_nodes == i)
                    if mask_edges_to_i.sum() > 0:
                        scores_for_node_i_head = attn_scores_T[head_idx, mask_edges_to_i].squeeze(-1)
                        alpha_for_node_i_head = F.softmax(scores_for_node_i_head, dim=0)
                        alpha[head_idx, mask_edges_to_i] = alpha_for_node_i_head
            alpha = self.dropout(alpha)
        messages = h_transformed_heads[src_nodes].permute(1,0,2) * alpha.unsqueeze(-1) 
        out_h_heads = torch.zeros(self.num_heads, N, self.out_dim_per_head, device=x.device)
        if TORCH_SCATTER_AVAILABLE:
            out_h_heads = scatter_add(messages, dst_nodes_expanded, dim=1, out=out_h_heads) 
        else: 
            for head_idx in range(self.num_heads):
                for i in range(N): 
                    mask_edges_to_i = (dst_nodes == i)
                    if mask_edges_to_i.sum() > 0:
                        msg_for_node_i_head = messages[head_idx, mask_edges_to_i]
                        out_h_heads[head_idx, i] = msg_for_node_i_head.sum(dim=0)
        out_h_heads = out_h_heads.permute(1,0,2)
        if self.concat: out_h = out_h_heads.reshape(N, self.num_heads * self.out_dim_per_head)
        else: out_h = out_h_heads.mean(dim=1)
        if self.residual: out_h = out_h + self.res_fc(x)
        out_h = self.norm(out_h); out_h = F.elu(out_h) 
        return out_h

class TemporalSelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim; self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim); self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout); self.norm = nn.LayerNorm(embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight); self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight); self.out_proj.bias.data.fill_(0)

    def forward(self, x_temporal_sequence, attention_mask=None):
        seq_len, batch_size, _ = x_temporal_sequence.shape
        qkv = self.qkv_proj(x_temporal_sequence)
        qkv = qkv.reshape(seq_len, batch_size, self.num_heads, 3 * self.head_dim).permute(1, 2, 0, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None: attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(1), float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1); attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(seq_len, batch_size, self.embed_dim)
        output = self.out_proj(attn_output)
        output = self.norm(x_temporal_sequence + self.dropout(output)); output = F.elu(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model); position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1); self.register_buffer('pe', pe)
    def forward(self, x_seq): return x_seq + self.pe[:x_seq.size(0), :]

class FullDySAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_struct_heads, num_temporal_heads,
                 num_time_steps, dropout, use_temporal_attn=True):
        super().__init__()
        self.use_temporal_attn = use_temporal_attn; self.num_time_steps = num_time_steps
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.structural_attention = MultiHeadGATLayer(
            in_dim=hidden_dim, out_dim_per_head=hidden_dim // num_struct_heads, 
            num_heads=num_struct_heads, dropout=dropout, concat=True, residual=True
        )
        if self.use_temporal_attn:
            self.pos_encoder = PositionalEncoding(hidden_dim, max_len=num_time_steps + 1) # +1 for safety if num_time_steps is exact
            self.temporal_attention = TemporalSelfAttentionLayer(
                embed_dim=hidden_dim, num_heads=num_temporal_heads, dropout=dropout
            )
        else: 
            self.temporal_weight = nn.Parameter(torch.tensor(0.5))
            self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.dropout_module = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight, gain=nn.init.calculate_gain('relu'))
        if self.input_proj.bias is not None: nn.init.zeros_(self.input_proj.bias)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('linear'))
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x_original, edge_indices, node_masks):
        N = x_original.size(0); T_snapshots = len(edge_indices)
        x_projected = self.input_proj(x_original); x_projected = F.elu(x_projected)
        x_projected_dropout = self.dropout_module(x_projected)
        snapshot_embeddings = []
        for t in range(T_snapshots):
            h_struct_t = self.structural_attention(x_projected_dropout, edge_indices[t])
            snapshot_embeddings.append(h_struct_t)
        if not snapshot_embeddings: final_node_embeddings = x_projected_dropout
        elif self.use_temporal_attn:
            temporal_input_sequence = torch.stack(snapshot_embeddings, dim=0)
            temporal_input_sequence = self.pos_encoder(temporal_input_sequence)
            temporal_output_sequence = self.temporal_attention(temporal_input_sequence)
            final_node_embeddings = temporal_output_sequence[-1, :, :]
        else: 
            last_embedding = snapshot_embeddings[-1]
            if len(snapshot_embeddings) > 1:
                all_embeddings_stacked = torch.stack(snapshot_embeddings, dim=0)
                avg_embedding = all_embeddings_stacked.mean(dim=0)
                temporal_w = torch.sigmoid(self.temporal_weight)
                final_node_embeddings = temporal_w * last_embedding + (1 - temporal_w) * avg_embedding
                final_node_embeddings = self.temporal_norm(final_node_embeddings)
                final_node_embeddings = self.dropout_module(final_node_embeddings)
            else: final_node_embeddings = last_embedding
        logits = self.classifier(final_node_embeddings)
        return logits

# ---------------------------------------------------------------------------
# Training, evaluation
# ---------------------------------------------------------------------------
def create_balanced_splits(labels, node_masks, device=None): 
    n_nodes = labels.size(0); active_nodes_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device) 
    for mask_t in node_masks:
        if isinstance(mask_t, torch.Tensor) and mask_t.numel() > 0 and mask_t.shape[0] == n_nodes: active_nodes_mask = active_nodes_mask | mask_t.to(labels.device).bool()
    active_indices = torch.nonzero(active_nodes_mask).squeeze()
    if active_indices.numel() == 0: logging.warning("No active nodes. Splits empty."); empty_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device); return empty_mask.clone(), empty_mask.clone(), empty_mask.clone()
    active_labels = labels[active_indices]; train_ratio, val_ratio = 0.7, 0.15; indices_np = active_indices.cpu().numpy(); labels_np = active_labels.cpu().numpy()
    min_samples_per_class_train = 2; unique_labels, counts = np.unique(labels_np, return_counts=True)
    if len(unique_labels) < 2 or np.any(counts < min_samples_per_class_train) or len(labels_np) < (min_samples_per_class_train * len(unique_labels) / (1-train_ratio if (1-train_ratio)>0 else 0.01)): # avoid div by zero
        logging.warning(f"Non-stratified train/temp split due to insufficient data for stratification (active_nodes={len(labels_np)}, unique_labels={len(unique_labels)}, counts={counts})."); 
        train_idx, temp_idx = train_test_split(indices_np, test_size=(1-train_ratio), random_state=42, shuffle=True)
    else: train_idx, temp_idx, _, _ = train_test_split(indices_np, labels_np, test_size=(1-train_ratio), random_state=42, stratify=labels_np)
    temp_labels_np = labels_np[np.isin(indices_np, temp_idx)]; unique_temp_labels, temp_counts = np.unique(temp_labels_np, return_counts=True)
    if len(unique_temp_labels) < 2 or np.any(temp_counts < 2) or len(temp_labels_np) < 4:
         logging.warning(f"Non-stratified val/test split due to insufficient data (temp_nodes={len(temp_labels_np)}, unique_labels={len(unique_temp_labels)}, counts={temp_counts})."); 
         val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, shuffle=True) 
    else: val_idx, test_idx, _, _ = train_test_split(temp_idx, temp_labels_np, test_size=0.5, random_state=42, stratify=temp_labels_np)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device); val_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device); test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device)
    if len(train_idx) > 0: train_mask[torch.from_numpy(train_idx).to(labels.device).long()] = True
    if len(val_idx) > 0: val_mask[torch.from_numpy(val_idx).to(labels.device).long()] = True
    if len(test_idx) > 0: test_mask[torch.from_numpy(test_idx).to(labels.device).long()] = True
    return train_mask, val_mask, test_mask

def calculate_class_weights(labels, masks=None, device='cpu'):
    combined_mask = torch.zeros_like(labels, dtype=torch.bool); 
    if masks:
        for m_tensor in masks: 
            if isinstance(m_tensor, torch.Tensor) and m_tensor.numel() > 0: combined_mask |= m_tensor.to(labels.device).bool()
    else: combined_mask = torch.ones_like(labels, dtype=torch.bool)
    subset_labels = labels[combined_mask]
    if subset_labels.numel() == 0: n_classes = int(labels.max().item() + 1) if labels.numel() > 0 else 2; return torch.ones(n_classes, device=device) / n_classes
    n_classes_actual = int(labels.max().item() + 1) if labels.numel() > 0 else (int(subset_labels.max().item() + 1) if subset_labels.numel() > 0 else 2)
    counts = torch.bincount(subset_labels.long(), minlength=n_classes_actual) 
    weights = torch.tensor([1.0 / c if c > 0 else 0 for c in counts], dtype=torch.float, device=device)
    if weights.sum() == 0: return torch.ones(n_classes_actual, device=device) / n_classes_actual
    weights = weights / weights.sum() * n_classes_actual; weights = torch.clamp(weights, min=0.1, max=10.0); return weights.to(device) 

def train_epoch(model, optimizer, criterion, X, edge_indices, node_masks, labels, train_mask, device, clip_val=1.0): 
    model.train(); optimizer.zero_grad(set_to_none=True); logits = model(X, edge_indices, node_masks)
    if not isinstance(train_mask, torch.Tensor) or train_mask.sum() == 0: return 0.0, 0.0 # Loss, num_trained_samples
    train_logits = logits[train_mask]; train_labels = labels[train_mask]
    if train_logits.shape[0] == 0: return 0.0, 0
    loss = criterion(train_logits, train_labels)
    if torch.isnan(loss) or torch.isinf(loss): return loss.item(), train_logits.shape[0]
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val); optimizer.step()
    return loss.item(), train_logits.shape[0]

def evaluate(model, X, edge_indices, node_masks, labels, mask, acc_metric, f1_metric):
    model.eval()
    with torch.no_grad():
        logits = model(X, edge_indices, node_masks)
        if not isinstance(mask, torch.Tensor) or mask.sum() == 0: return 0.0, 0.0, 0 # Acc, F1, num_eval_samples
        preds_on_mask = logits[mask].argmax(dim=1).cpu(); labels_on_mask = labels[mask].cpu()
        if labels_on_mask.numel() == 0: return 0.0, 0.0, 0
        acc_metric.reset(); f1_metric.reset()
        acc = acc_metric(preds_on_mask, labels_on_mask); f1 = f1_metric(preds_on_mask, labels_on_mask)
    return acc.item(), f1.item(), labels_on_mask.numel()

def train_and_evaluate_with_hps(
    hps: Union[argparse.Namespace, Dict], # Hyperparameters
    X: torch.Tensor, 
    edge_indices: List[torch.Tensor], 
    node_masks: List[torch.Tensor], 
    labels: torch.Tensor, 
    num_time_steps_for_pe: int,
    fixed_args: argparse.Namespace # For fixed params like epochs, device
):
    # Extract HPs, falling back to fixed_args defaults if not in hps
    if isinstance(hps, dict): # Optuna trial.params is a dict
        hidden_dim = hps.get('hidden_dim', fixed_args.hidden_dim)
        lr = hps.get('lr', fixed_args.lr)
        dropout_rate = hps.get('dropout', fixed_args.dropout)
        weight_decay_val = hps.get('weight_decay', fixed_args.weight_decay)
        scheduler_patience = hps.get('scheduler_patience', fixed_args.scheduler_patience)
        early_stop_patience = hps.get('early_stop_patience', fixed_args.early_stop_patience)
        num_struct_heads = hps.get('num_struct_heads', fixed_args.num_struct_heads)
        num_temporal_heads = hps.get('num_temporal_heads', fixed_args.num_temporal_heads)
    elif isinstance(hps, argparse.Namespace): # Manual run args
        hidden_dim = hps.hidden_dim
        lr = hps.lr
        dropout_rate = hps.dropout
        weight_decay_val = hps.weight_decay
        scheduler_patience = hps.scheduler_patience
        early_stop_patience = hps.early_stop_patience
        num_struct_heads = hps.num_struct_heads
        num_temporal_heads = hps.num_temporal_heads
    else:
        raise TypeError("hps must be a dict or argparse.Namespace")

    use_temporal_attn = not fixed_args.no_temporal_attn # This is usually a fixed choice per study
    device = torch.device(fixed_args.device if fixed_args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Ensure hidden_dim is divisible by heads
    if hidden_dim % num_struct_heads != 0:
        hidden_dim = num_struct_heads * max(1, (hidden_dim // num_struct_heads))
    if hidden_dim % num_temporal_heads != 0:
        hidden_dim = num_temporal_heads * max(1, (hidden_dim // num_temporal_heads))
    if early_stop_patience <= scheduler_patience +2 :
         early_stop_patience = scheduler_patience + 3
        
    train_mask, val_mask, test_mask = create_balanced_splits(labels, node_masks, device=device)
    
    num_train_samples = train_mask.sum().item()
    num_val_samples = val_mask.sum().item()
    num_test_samples = test_mask.sum().item()
    logging.debug(f"Train: {num_train_samples}, Val: {num_val_samples}, Test: {num_test_samples}")

    if num_train_samples == 0 or num_val_samples == 0 : 
        logging.warning(f"Skipping training due to empty train/val split (Train: {num_train_samples}, Val: {num_val_samples}).")
        return None, 0.0, 0.0, 0.0, {}, 0, 0, 0 # model, val_acc, test_acc, test_f1, history, splits

    class_weights = calculate_class_weights(labels, [train_mask], device=device) 
    logging.debug(f"Class weights: {class_weights.cpu().numpy()}")
    
    n_classes = int(labels.max().item() + 1) if labels.numel() > 0 else 2
    if n_classes < 2 and (num_train_samples > 0 or num_val_samples > 0 or num_test_samples > 0) :
        # If there are labels, ensure n_classes is at least 2 for classification metrics
        logging.warning(f"Computed n_classes={n_classes}, but data exists. Setting n_classes=2.")
        n_classes = 2
    elif n_classes < 2 : # No labels at all
        logging.warning(f"Computed n_classes={n_classes} and no data. Setting n_classes=2. This run will likely fail or be meaningless.")
        n_classes = 2


    model = FullDySAT( 
        in_dim=X.size(1), hidden_dim=hidden_dim, num_classes=n_classes,
        num_struct_heads=num_struct_heads, num_temporal_heads=num_temporal_heads,
        num_time_steps=num_time_steps_for_pe, 
        dropout=dropout_rate, use_temporal_attn=use_temporal_attn
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay_val)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience, min_lr=1e-7) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    acc_metric = Accuracy(task="multiclass", num_classes=n_classes, ignore_index=-1 if n_classes > 0 else None).cpu() 
    f1_metric = F1Score(task="multiclass", num_classes=n_classes, average="macro", ignore_index=-1 if n_classes > 0 else None).cpu()
    
    best_val_acc = 0; best_test_acc_at_best_val = 0; best_f1_at_best_val = 0; best_epoch = 0
    patience_counter = 0
    history = {"loss": [], "val_acc": [], "val_f1": [], "test_acc": [], "test_f1": [], "lr": []}
    best_model_state = model.state_dict() 

    for epoch in range(fixed_args.epochs):
        loss, trained_this_epoch = train_epoch(model, optimizer, criterion, X, edge_indices, node_masks, labels, train_mask, device)
        val_acc, val_f1, eval_val_this_epoch = evaluate(model, X, edge_indices, node_masks, labels, val_mask, acc_metric, f1_metric)
        test_acc, test_f1, eval_test_this_epoch = evaluate(model, X, edge_indices, node_masks, labels, test_mask, acc_metric, f1_metric)
        
        scheduler.step(val_f1) # Optimize for F1 on val set
        current_lr = optimizer.param_groups[0]['lr']
        
        logging.debug(f"Epoch {epoch:03d}: loss={loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, test_acc={test_acc:.4f}, test_f1={test_f1:.4f}, lr={current_lr:.7f}")
        history["loss"].append(loss); history["val_acc"].append(val_acc); history["val_f1"].append(val_f1)
        history["test_acc"].append(test_acc); history["test_f1"].append(test_f1); history["lr"].append(current_lr)
        
        primary_metric = val_f1 # Using val_f1 for early stopping and model selection
        if primary_metric > best_val_acc : # best_val_acc now stores best_val_f1
            best_val_acc = primary_metric
            best_test_acc_at_best_val = test_acc
            best_f1_at_best_val = test_f1
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            logging.info(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch} with Val F1: {best_val_acc:.4f}")
            break
        if current_lr <= scheduler.min_lrs[0] + 1e-9 and patience_counter > early_stop_patience // 2 :
            logging.info(f"LR at min and no improvement for {patience_counter} epochs. Stopping.")
            break
            
    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    else: # Case where training loop might not have run if splits were bad
        logging.warning("No best model state found, possibly due to no valid epochs run.")
        # Return 0s for metrics to indicate failure
        return model, 0.0, 0.0, 0.0, history, num_train_samples, num_val_samples, num_test_samples


    logging.info(f"Best Val F1: {best_val_acc:.4f} at epoch {best_epoch}. Corresponding Test Acc: {best_test_acc_at_best_val:.4f}, Test F1: {best_f1_at_best_val:.4f}")
    return model, best_val_acc, best_test_acc_at_best_val, best_f1_at_best_val, history, num_train_samples, num_val_samples, num_test_samples

# ---------------------------------------------------------------------------
# Args Parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="DySAT trainer for PHEME with HP Tuning and Ablations")
    parser.add_argument("--data-dir", default="data_dysat_v2", help="Path to preprocessed data")
    parser.add_argument("--event", type=str, default=None, help="Event name, comma-separated list, or 'all'.")
    
    # Model Hyperparameters
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension") 
    parser.add_argument("--num-struct-heads", type=int, default=4, help="Number of structural attention heads")
    parser.add_argument("--num-temporal-heads", type=int, default=4, help="Number of temporal attention heads")
    parser.add_argument("--no-temporal-attn", action="store_true", help="Disable temporal self-attention")

    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=0.0007, help="Learning rate") 
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--weight-decay", type=float, default=1.7e-5, help="Weight decay")
    parser.add_argument("--scheduler-patience", type=int, default=14, help="Scheduler patience")
    parser.add_argument("--early-stop-patience", type=int, default=20, help="Early stopping patience")
    
    # General Training Config
    parser.add_argument("--epochs", type=int, default=200, help="Max epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    parser.add_argument("--max-nodes-subsample", type=int, default=None, help="Max nodes for subsampling large graphs")
    
    # Optuna Specific
    parser.add_argument("--optuna-study", action="store_true", help="Run a global Optuna study")
    parser.add_argument("--optuna-trials", type=int, default=30, help="Optuna trials for global study")
    parser.add_argument("--optuna-study-name", type=str, default="FullDySAT_PHEME_Global", help="Global Optuna study name")
    parser.add_argument("--optuna-event-subset", type=str, default="germanwings-crash,charliehebdo", help="Events for global Optuna study (comma-separated)")

    # Full Ablation Workflow
    parser.add_argument("--run-full-ablation", action="store_true", help="Run full ablation workflow")
    parser.add_argument("--optuna-trials-per-event", type=int, default=15, help="Optuna trials for per-event optimization in full ablation")
    
    # Output control for manual runs (less relevant if full ablation is used)
    parser.add_argument("--run-tag", type=str, default="", help="Tag for manual run outputs")

    return parser.parse_args()

# ---------------------------------------------------------------------------
# Optuna Objective Function
# ---------------------------------------------------------------------------
def objective(trial: optuna.trial.Trial, base_args: argparse.Namespace, event_dirs_for_study: List[Path]):
    # Define HPs to tune for this trial
    hps_values = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.05),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "scheduler_patience": trial.suggest_int("scheduler_patience", 7, 20),
        "early_stop_patience": trial.suggest_int("early_stop_patience", 15, 35), # will be adjusted if needed
        "num_struct_heads": trial.suggest_categorical("num_struct_heads", [2, 4, 8]),
        "num_temporal_heads": trial.suggest_categorical("num_temporal_heads", [2, 4, 8]),
    }
    # Ensure hidden_dim is compatible with heads
    if hps_values["hidden_dim"] % hps_values["num_struct_heads"] != 0:
        hps_values["hidden_dim"] = hps_values["num_struct_heads"] * max(1, (hps_values["hidden_dim"] // hps_values["num_struct_heads"]))
    if hps_values["hidden_dim"] % hps_values["num_temporal_heads"] != 0:
        hps_values["hidden_dim"] = hps_values["num_temporal_heads"] * max(1, (hps_values["hidden_dim"] // hps_values["num_temporal_heads"]))
    if hps_values["early_stop_patience"] <= hps_values["scheduler_patience"] + 2:
        hps_values["early_stop_patience"] = hps_values["scheduler_patience"] + 3
    
    trial.set_user_attr("hps", hps_values) # Store chosen HPs

    logging.info(f"Optuna Trial {trial.number} for events {[d.name for d in event_dirs_for_study]}: HPs: {hps_values}")
    set_seed(base_args.seed) # Reset seed for each trial for comparable runs
    device = torch.device(base_args.device if base_args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    all_event_metrics = []
    for event_dir in event_dirs_for_study:
        event_name = event_dir.name
        try:
            X, edge_indices, node_masks, labels, time_info = load_temporal_data(event_dir, device, False, base_args.max_nodes_subsample)
            if X.nelement() == 0 or labels.nelement() == 0 or not any(isinstance(nm, torch.Tensor) and nm.numel()>0 for nm in node_masks):
                logging.warning(f"Trial {trial.number}: Skipping {event_name}, empty data/masks.")
                continue
            
            num_actual_time_steps = time_info.get("num_windows", len(edge_indices))
            if num_actual_time_steps == 0 and not base_args.no_temporal_attn: # No time steps, but temporal attention expects some
                logging.warning(f"Trial {trial.number}: Event {event_name} has 0 time steps. Temporal Attention might error. Setting num_actual_time_steps to 1 for PE.")
                num_actual_time_steps = 1 # For PositionalEncoding

            _, val_f1, test_acc, test_f1, _, _, _, _ = train_and_evaluate_with_hps(
                hps=hps_values, X=X, edge_indices=edge_indices, node_masks=node_masks, labels=labels,
                num_time_steps_for_pe=num_actual_time_steps, fixed_args=base_args
            )
            # The objective for Optuna should be based on validation performance
            all_event_metrics.append({'val_f1': val_f1, 'test_acc': test_acc, 'test_f1': test_f1}) # Store val_f1 for optimization
            del X, edge_indices, node_masks, labels, time_info; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Optuna Trial {trial.number}: Error during training for {event_name}: {e}", exc_info=False)
            # Prune if error occurs, or return a very bad score
            # return -1.0 # or raise optuna.exceptions.TrialPruned()

    if not all_event_metrics:
        logging.warning(f"Optuna Trial {trial.number}: No events successfully processed.")
        return -1.0 # Indicate failure

    avg_val_f1 = np.mean([m['val_f1'] for m in all_event_metrics if m['val_f1'] is not None])
    logging.info(f"Optuna Trial {trial.number}: Avg Val F1: {avg_val_f1:.4f}")
    return avg_val_f1 # Optuna maximizes this

# ---------------------------------------------------------------------------
# Helper functions for saving results
# ---------------------------------------------------------------------------
def save_event_results_md(output_dir: Path, hps: Union[Dict, argparse.Namespace], val_f1: float, test_acc: float, test_f1: float, event_name: str, splits: Tuple[int,int,int]):
    md_content = f"# Results for Event: {event_name}\n\n"
    md_content += "## Hyperparameters Used:\n"
    if isinstance(hps, dict):
        for k, v in hps.items(): md_content += f"- {k}: {v}\n"
    else: # argparse.Namespace
        for k, v in vars(hps).items(): 
            if k not in ['data_dir', 'event', 'device', 'seed', 'epochs', 'run_tag', 'optuna_study', 'optuna_trials', 'optuna_study_name', 'optuna_event_subset', 'run_full_ablation', 'optuna_trials_per_event', 'results_file']: # Filter out non-HP args
                 md_content += f"- {k}: {v}\n"
    md_content += "\n## Performance Metrics:\n"
    md_content += f"- Validation F1-score (macro): {val_f1:.4f}\n"
    md_content += f"- Test Accuracy: {test_acc:.4f}\n"
    md_content += f"- Test F1-score (macro): {test_f1:.4f}\n"
    md_content += "\n## Data Splits:\n"
    md_content += f"- Train samples: {splits[0]}\n"
    md_content += f"- Validation samples: {splits[1]}\n"
    md_content += f"- Test samples: {splits[2]}\n"

    with open(output_dir / "results.md", "w") as f:
        f.write(md_content)
    logging.info(f"Saved results.md for {event_name} to {output_dir}")

def save_training_plots(output_dir: Path, history: Dict, event_name: str, run_tag: str = ""):
    if not history or not history.get("loss"):
        logging.warning(f"No history data to plot for {event_name}.")
        return
        
    plt.figure(figsize=(24, 5))
    plt.subplot(1, 4, 1); plt.plot(history["loss"]); plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.subplot(1, 4, 2); plt.plot(history["val_f1"], label="Val F1"); plt.plot(history.get("test_f1", []), label="Test F1"); plt.title("F1 (Macro)"); plt.xlabel("Epoch"); plt.ylabel("F1"); plt.legend()
    plt.subplot(1, 4, 3); plt.plot(history["val_acc"], label="Val Acc"); plt.plot(history["test_acc"], label="Test Acc"); plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Acc"); plt.legend()
    plt.subplot(1, 4, 4); plt.plot(history["lr"]); plt.title("Learning Rate"); plt.xlabel("Epoch"); plt.ylabel("LR"); plt.yscale("log")
    
    suptitle = f"Training Curves for {event_name}"
    if run_tag: suptitle += f" ({run_tag})"
    plt.suptitle(suptitle); plt.tight_layout(rect=[0,0,1,0.96])
    plot_path = output_dir / "training_plots.png"
    plt.savefig(plot_path); plt.close()
    logging.info(f"Saved training plots for {event_name} to {plot_path}")

def save_optuna_study_results(study: optuna.Study, output_dir: Path, study_name_prefix: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    df_results = study.trials_dataframe()
    csv_path = output_dir / f"{study_name_prefix}_optuna_ablation_results.csv"
    df_results.to_csv(csv_path, index=False)
    logging.info(f"Optuna study results saved to {csv_path}")

    if OPTUNA_VISUALIZATION_AVAILABLE:
        try:
            fig_parallel = plot_parallel_coordinate(study)
            fig_parallel.write_image(output_dir / f"{study_name_prefix}_optuna_parallel_coordinate.png")
            logging.info(f"Optuna parallel coordinate plot saved to {output_dir}")
            
            fig_history = plot_optimization_history(study)
            fig_history.write_image(output_dir / f"{study_name_prefix}_optuna_optimization_history.png")
            logging.info(f"Optuna optimization history plot saved to {output_dir}")
        except Exception as e:
            logging.warning(f"Failed to generate Optuna visualization plots: {e}")
    else:
        logging.info("Optuna visualization module not available. Skipping plot generation.")

# ---------------------------------------------------------------------------
# Core Workflows
# ---------------------------------------------------------------------------
def get_event_dirs(data_root_str: str, event_arg: Optional[str]) -> List[Path]:
    data_root = Path(data_root_str)
    if not data_root.exists():
        logging.error(f"Data directory {data_root} not found.")
        return []
        
    event_list_to_process = []
    if event_arg and event_arg.lower() != 'all':
        event_names_from_arg = event_arg.split(',')
        for e_name in event_names_from_arg:
            event_path = data_root / e_name.strip()
            if event_path.exists() and event_path.is_dir():
                event_list_to_process.append(event_path)
            else:
                logging.warning(f"Event directory {event_path} not found. Skipping.")
    else: # 'all' or None specified
        event_list_to_process = [d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith("_") and not d.name.lower() == "all"] # Skip hidden/special dirs
    
    if not event_list_to_process:
        logging.error(f"No event directories found or specified to process in {data_root}.")
    return event_list_to_process

def evaluate_event_with_fixed_hps(event_dir: Path, hps_config: Union[argparse.Namespace, Dict], base_args: argparse.Namespace, output_dir: Path):
    event_name = event_dir.name
    device = torch.device(base_args.device if base_args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Evaluating event: {event_name} with fixed HPs. Output to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        X, edge_indices, node_masks, labels, time_info = load_temporal_data(event_dir, device, False, base_args.max_nodes_subsample)
        if X.nelement() == 0 or labels.nelement() == 0 or not any(isinstance(nm, torch.Tensor) and nm.numel()>0 for nm in node_masks):
            logging.warning(f"Skipping {event_name}: empty data/masks."); return None, 0.0, 0.0, 0.0, {}
        
        num_actual_time_steps = time_info.get("num_windows", len(edge_indices))
        if num_actual_time_steps == 0 and not base_args.no_temporal_attn:
            num_actual_time_steps = 1 # For PositionalEncoding

        model, val_f1, test_acc, test_f1, history, tr_s, v_s, te_s = train_and_evaluate_with_hps(
            hps=hps_config, X=X, edge_indices=edge_indices, node_masks=node_masks, labels=labels,
            num_time_steps_for_pe=num_actual_time_steps, fixed_args=base_args
        )

        if model is None: # Training failed (e.g. no train/val samples)
            return None, 0.0, 0.0, 0.0, {}

        # Save results for this event
        save_event_results_md(output_dir, hps_config, val_f1, test_acc, test_f1, event_name, (tr_s, v_s, te_s))
        save_training_plots(output_dir, history, event_name, run_tag=getattr(base_args, "run_tag", ""))
        torch.save(model.state_dict(), output_dir / "model.pt")
        
        del X, edge_indices, node_masks, labels, model, history, time_info; gc.collect(); torch.cuda.empty_cache()
        return model, val_f1, test_acc, test_f1, getattr(hps_config, 'params', hps_config) # Return HPs

    except Exception as e:
        logging.error(f"Error processing event {event_name} during fixed HP evaluation: {e}", exc_info=True)
        return None, 0.0, 0.0, 0.0, {}

def run_optuna_for_single_event(event_dir: Path, base_args: argparse.Namespace, num_trials: int, output_dir: Path):
    if not OPTUNA_AVAILABLE:
        logging.error("Optuna not installed. Cannot run Optuna for single event.")
        return None, None

    event_name = event_dir.name
    study_name = f"{MODEL_NAME}_{event_name}_Optuna"
    storage_path = f"sqlite:///{output_dir.parent / (study_name.replace(' ','_') + '.db')}" # Store DB one level up
    
    output_dir.mkdir(parents=True, exist_ok=True) # For CSVs and plots for this event

    logging.info(f"Starting Optuna study for event: {event_name} ({num_trials} trials). DB: {storage_path}")
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Optuna's own logging
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction="maximize", # Maximizing validation F1
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=max(1,num_trials//10), n_warmup_steps=5, n_min_trials=max(1,num_trials//5)),
        sampler=optuna.samplers.TPESampler(seed=base_args.seed, n_startup_trials=max(1,num_trials//5))
    )
    # Pass only the current event_dir to the objective
    study.optimize(lambda trial: objective(trial, base_args, [event_dir]), 
                     n_trials=num_trials, gc_after_trial=True, show_progress_bar=True)

    logging.info(f"Optuna study for {event_name} complete. {len(study.trials)} trials conducted.")
    save_optuna_study_results(study, output_dir, event_name) # Save CSV/plots in event specific folder

    try:
        best_trial = study.best_trial
        logging.info(f"Best trial for {event_name} (Val F1): {best_trial.value:.4f}")
        logging.info(f"Best HPs for {event_name}: {best_trial.params}")
        return best_trial.params, best_trial.value
    except ValueError: # No successful trials
        logging.warning(f"No successful Optuna trials for {event_name}.")
        return None, None

def find_overall_best_hps_with_optuna(base_args: argparse.Namespace, event_dirs_for_global_study: List[Path]):
    if not OPTUNA_AVAILABLE:
        logging.error("Optuna not installed. Cannot run global Optuna study.")
        return None

    study_name = base_args.optuna_study_name
    global_optuna_output_dir = MAIN_MODEL_RESULTS_DIR / f"global_optuna_study_{study_name.replace(' ','_')}"
    storage_path = f"sqlite:///{global_optuna_output_dir / (study_name.replace(' ','_') + '.db')}"
    
    global_optuna_output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting GLOBAL Optuna study: {study_name} ({base_args.optuna_trials} trials) "
                 f"using events: {[d.name for d in event_dirs_for_global_study]}. DB: {storage_path}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction="maximize", # Maximizing average validation F1 across subset
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=max(1,base_args.optuna_trials//10), n_warmup_steps=5, n_min_trials=max(1,base_args.optuna_trials//5)),
        sampler=optuna.samplers.TPESampler(seed=base_args.seed, n_startup_trials=max(1,base_args.optuna_trials//5))
    )
    study.optimize(lambda trial: objective(trial, base_args, event_dirs_for_global_study), 
                     n_trials=base_args.optuna_trials, gc_after_trial=True, show_progress_bar=True)

    logging.info(f"Global Optuna study '{study_name}' complete. {len(study.trials)} trials conducted.")
    save_optuna_study_results(study, global_optuna_output_dir, "global")

    try:
        best_trial = study.best_trial
        logging.info(f"Best trial for GLOBAL study (Avg Val F1): {best_trial.value:.4f}")
        logging.info(f"Overall Best HPs: {best_trial.params}")
        return best_trial.params
    except ValueError:
        logging.warning("No successful trials in Global Optuna study. Cannot determine overall best HPs.")
        return None

def generate_aggregated_report(overall_best_hps: Dict, all_event_dirs: List[Path], base_args: argparse.Namespace):
    MAIN_MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MAIN_MODEL_RESULTS_DIR / "aggregated_results.md"
    
    md_content = f"# {MODEL_NAME} - Aggregated Performance Report\n\n"
    md_content += "This report shows the performance of the model on various event streams "
    md_content += "when using a single set of 'overall best' hyperparameters.\n\n"
    
    md_content += "## Overall Best Hyperparameters Used:\n"
    for k, v in overall_best_hps.items():
        md_content += f"- {k}: {v}\n"
    # Also include fixed parameters like no_temporal_attn
    md_content += f"- use_temporal_attn: {not base_args.no_temporal_attn}\n"
    md_content += "\n## Per-Event Performance (with Overall Best HPs):\n"
    md_content += "| Event Stream         | Test Accuracy | Test F1-score (macro) |\n"
    md_content += "|----------------------|---------------|-----------------------|\n"

    all_test_accs = []
    all_test_f1s = []

    for event_dir in tqdm(all_event_dirs, desc="Generating Aggregated Report Entries"):
        event_name = event_dir.name
        # Note: For aggregated report, we re-evaluate. Plots/individual MD for these runs are NOT saved per event folder
        # The event folders contain results from their *own* best HPs if per-event optuna was run.
        
        device = torch.device(base_args.device if base_args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            X, edge_indices, node_masks, labels, time_info = load_temporal_data(event_dir, device, False, base_args.max_nodes_subsample)
            if X.nelement() == 0 or labels.nelement() == 0 or not any(isinstance(nm, torch.Tensor) and nm.numel()>0 for nm in node_masks):
                md_content += f"| {event_name:<20} | SKIPPED (No data) | SKIPPED (No data) |\n"
                continue

            num_actual_time_steps = time_info.get("num_windows", len(edge_indices))
            if num_actual_time_steps == 0 and not base_args.no_temporal_attn: num_actual_time_steps = 1


            # We need to call train_and_evaluate again here with the overall_best_hps
            # This ensures the metrics are from the *global* HPs
            _, _, test_acc, test_f1, _, _,_,_ = train_and_evaluate_with_hps(
                hps=overall_best_hps, X=X, edge_indices=edge_indices, node_masks=node_masks, labels=labels,
                num_time_steps_for_pe=num_actual_time_steps, fixed_args=base_args
            )
            
            md_content += f"| {event_name:<20} | {test_acc:13.4f} | {test_f1:21.4f} |\n"
            all_test_accs.append(test_acc)
            all_test_f1s.append(test_f1)
            del X, edge_indices, node_masks, labels, time_info; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Error evaluating {event_name} for aggregated report: {e}")
            md_content += f"| {event_name:<20} | ERROR         | ERROR                 |\n"

    md_content += "|----------------------|---------------|-----------------------|\n"
    if all_test_f1s: # Check if list is not empty
        avg_test_acc = np.mean([a for a in all_test_accs if a is not None]) if all_test_accs else 0.0
        avg_test_f1 = np.mean([f for f in all_test_f1s if f is not None]) if all_test_f1s else 0.0
        md_content += f"| **Average**          | **{avg_test_acc:10.4f}** | **{avg_test_f1:18.4f}** |\n"
    else:
        md_content += "| **Average**          | **N/A**       | **N/A**               |\n"

    with open(report_path, "w") as f:
        f.write(md_content)
    logging.info(f"Aggregated results report saved to {report_path}")


def run_full_ablation_workflow(args: argparse.Namespace):
    logging.info("===== Starting Full Ablation Workflow =====")
    MAIN_MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_available_event_dirs = get_event_dirs(args.data_dir, 'all') # All events in data_dir
    
    # Determine which events to use for the global study to find overall best HPs
    if args.optuna_event_subset and args.optuna_event_subset.lower() != 'all':
        event_dirs_for_global_study = get_event_dirs(args.data_dir, args.optuna_event_subset)
    else: # 'all' or not specified, use all available events for global study
        event_dirs_for_global_study = all_available_event_dirs
    
    if not event_dirs_for_global_study:
        logging.error("No events available for global Optuna study. Aborting full ablation.")
        return

    # 1. Find Overall Best Hyperparameters via Global Optuna Study
    logging.info("--- Stage 1: Finding Overall Best Hyperparameters (Global Optuna Study) ---")
    overall_best_hps = find_overall_best_hps_with_optuna(args, event_dirs_for_global_study)
    if overall_best_hps is None:
        logging.error("Failed to find overall best HPs from global Optuna. Using default CLI HPs for aggregated report.")
        # Fallback to CLI args if global Optuna fails (though this might not be "best")
        overall_best_hps = {
            "hidden_dim": args.hidden_dim, "lr": args.lr, "dropout": args.dropout,
            "weight_decay": args.weight_decay, "scheduler_patience": args.scheduler_patience,
            "early_stop_patience": args.early_stop_patience, 
            "num_struct_heads": args.num_struct_heads, "num_temporal_heads": args.num_temporal_heads
        }

    # Determine which events to process for per-event optimization and final reporting
    if args.event and args.event.lower() != 'all':
        events_to_process_individually = get_event_dirs(args.data_dir, args.event)
    else:
        events_to_process_individually = all_available_event_dirs
        
    if not events_to_process_individually:
        logging.error("No events specified or found for per-event optimization and reporting. Aborting stage 2 & 3.")
        return

    # 2. Per-Event Optuna and Reporting
    logging.info("\n--- Stage 2: Per-Event Optuna and Reporting ---")
    for event_dir in tqdm(events_to_process_individually, desc="Processing Events (Per-Event Optuna)"):
        event_name = event_dir.name
        logging.info(f"--- Processing event: {event_name} ---")
        event_output_dir = MAIN_MODEL_RESULTS_DIR / event_name
        event_output_dir.mkdir(parents=True, exist_ok=True)

        # Run Optuna for this single event
        best_hps_for_event, best_val_f1_for_event = run_optuna_for_single_event(
            event_dir, args, args.optuna_trials_per_event, event_output_dir
        )

        if best_hps_for_event:
            logging.info(f"Found best HPs for {event_name} via its Optuna. Evaluating and saving results.")
            # Evaluate with these event-specific best HPs and save results.md, plots.png, model.pt
            evaluate_event_with_fixed_hps(event_dir, best_hps_for_event, args, event_output_dir)
        else:
            logging.warning(f"Optuna failed for {event_name} or found no good HPs. "
                            "Skipping final evaluation for this event's individual report or using CLI defaults.")
            # Optionally, run with default HPs if Optuna fails for an event's individual report
            # evaluate_event_with_fixed_hps(event_dir, args, args, event_output_dir) # Using 'args' as hps_config
            # For now, if per-event optuna fails, its specific results.md and plots might be missing or based on poor HPs.

    # 3. Generate Aggregated Report (using overall_best_hps from Stage 1)
    logging.info("\n--- Stage 3: Generating Aggregated Report (using Overall Best HPs) ---")
    # The aggregated report uses all events available in data_dir, or those specified by --event if not 'all'
    events_for_aggregation = events_to_process_individually # Use the same set as per-event for consistency here
    if overall_best_hps: # Check if we have HPs for aggregation
        generate_aggregated_report(overall_best_hps, events_for_aggregation, args)
    else:
        logging.error("Cannot generate aggregated report as overall best HPs were not determined.")

    logging.info("===== Full Ablation Workflow Complete =====")


# ---------------------------------------------------------------------------
# Main Execution Logic
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
        level=logging.INFO 
    )
    if TORCH_SCATTER_AVAILABLE:
        logging.info("torch_scatter found. Using for GAT acceleration.")
    else:
        logging.warning("torch_scatter not found. GAT layers will use a slower loop-based fallback.")
    if OPTUNA_AVAILABLE:
        logging.info(f"Optuna found (version: {optuna.__version__}). Visualization available: {OPTUNA_VISUALIZATION_AVAILABLE}")
    else:
        logging.warning("Optuna not found. Hyperparameter tuning and full ablation workflow will be limited/unavailable.")

    cli_args = parse_args()
    set_seed(cli_args.seed)
    MAIN_MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True) # Ensure base results dir exists

    # HP Sanity Checks (can be moved into train_and_evaluate_with_hps as well)
    if cli_args.hidden_dim % cli_args.num_struct_heads != 0:
        old_hd = cli_args.hidden_dim
        cli_args.hidden_dim = cli_args.num_struct_heads * max(1, (cli_args.hidden_dim // cli_args.num_struct_heads))
        logging.warning(f"Adjusted hidden_dim from {old_hd} to {cli_args.hidden_dim} for num_struct_heads ({cli_args.num_struct_heads}).")
    if cli_args.hidden_dim % cli_args.num_temporal_heads != 0:
        old_hd = cli_args.hidden_dim
        cli_args.hidden_dim = cli_args.num_temporal_heads * max(1, (cli_args.hidden_dim // cli_args.num_temporal_heads))
        logging.warning(f"Adjusted hidden_dim from {old_hd} to {cli_args.hidden_dim} for num_temporal_heads ({cli_args.num_temporal_heads}).")
    if cli_args.early_stop_patience <= cli_args.scheduler_patience + 2:
        cli_args.early_stop_patience = cli_args.scheduler_patience + 3
        logging.warning(f"Adjusted early_stop_patience to {cli_args.early_stop_patience} to be > scheduler_patience+2.")

    if cli_args.run_full_ablation:
        if not OPTUNA_AVAILABLE:
            logging.error("Optuna is required for --run-full-ablation. Please install optuna.")
        else:
            run_full_ablation_workflow(cli_args)
    
    elif cli_args.optuna_study: # Run only a global Optuna study
        if not OPTUNA_AVAILABLE:
            logging.error("Optuna is required for --optuna-study. Please install optuna.")
        else:
            event_dirs_for_study = get_event_dirs(cli_args.data_dir, cli_args.optuna_event_subset)
            if event_dirs_for_study:
                find_overall_best_hps_with_optuna(cli_args, event_dirs_for_study)
            else:
                logging.error(f"No events found for Optuna study based on subset: {cli_args.optuna_event_subset}")
    
    else: # Manual run with fixed HPs specified via CLI
        logging.info("Running manual evaluation with CLI-specified hyperparameters.")
        event_dirs_to_run = get_event_dirs(cli_args.data_dir, cli_args.event)
        if not event_dirs_to_run:
            logging.error(f"No events found to process for manual run based on --event '{cli_args.event}'.")
        else:
            all_event_metrics = []
            for event_dir in event_dirs_to_run:
                event_name = event_dir.name
                event_output_dir = MAIN_MODEL_RESULTS_DIR / event_name
                event_output_dir.mkdir(parents=True, exist_ok=True)
                
                logging.info(f"Manual run for event: {event_name}")
                _, val_f1, test_acc, test_f1, hps_used = evaluate_event_with_fixed_hps(
                    event_dir, cli_args, cli_args, event_output_dir # cli_args used for both hps_config and base_args
                )
                if test_f1 > 0 or test_acc > 0 : # If some valid result
                     all_event_metrics.append({'event': event_name, 'test_acc': test_acc, 'test_f1': test_f1})

            if all_event_metrics:
                avg_test_acc = np.mean([m['test_acc'] for m in all_event_metrics])
                avg_test_f1 = np.mean([m['test_f1'] for m in all_event_metrics])
                logging.info(f"Manual Run Summary ({len(all_event_metrics)} events): Avg Test Acc: {avg_test_acc:.4f}, Avg Test F1: {avg_test_f1:.4f}")
                # This summary is just for logging, the main aggregated report is more formal.