#!/usr/bin/env python3
"""
DySAT implementation for the PHEME dataset, incorporating
Multi-Head Attention and Temporal Self-Attention.
Can also run comparison with SimpleDySAT.

Usage (Full DySAT):
python train_dysat_pheme_Full.py --data-dir data_dysat_v2 --event germanwings-crash
python train_dysat_pheme_Full.py --run-full-ablation --event all --optuna-trials 30 --optuna-trials-per-event 15

Usage (Comparison with SimpleDySAT):
python train_dysat_pheme_Full.py --compare-with-simple-dysat --event all \
       --data-dir data_dysat_v2 --simple-model-data-dir data_dysat \
       --epochs 100 --device cuda 
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
try:
    from torch_scatter import scatter_softmax, scatter_add
    TORCH_SCATTER_AVAILABLE = True
except ImportError:
    TORCH_SCATTER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "DySAT_Full"
BASE_RESULTS_ROOT = Path("RESULTS")
MAIN_MODEL_RESULTS_DIR = BASE_RESULTS_ROOT / MODEL_NAME
COMPARISON_RESULTS_DIR = BASE_RESULTS_ROOT / "Model_Comparisons"


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
# Data loading helpers (shared)
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
    time_info_path = event_dir / "time_info.json"
    if not time_info_path.exists():
        logging.error(f"time_info.json not found in {event_dir}")
        # Provide default minimal time_info if not present
        time_info = {"num_windows": 0, "start_time": 0, "end_time": 0, "window_size": 0}
        # Attempt to infer num_windows if edge_indices exist
        num_edge_files = len(list((event_dir / "edge_indices").glob("t*.npy")))
        if num_edge_files > 0:
            time_info["num_windows"] = num_edge_files
            logging.warning(f"time_info.json missing, inferred num_windows={num_edge_files} from edge_indices files.")
        else: # If no edge_indices, try node_masks
            num_mask_files = len(list((event_dir / "node_masks").glob("t*.npy")))
            if num_mask_files > 0:
                time_info["num_windows"] = num_mask_files
                logging.warning(f"time_info.json missing, inferred num_windows={num_mask_files} from node_mask files.")

    else:
        with open(time_info_path, "r") as f: time_info = json.load(f)

    num_time_steps = time_info.get("num_windows", 0) # Default to 0 if not found

    edge_indices_list = []
    if (event_dir / "edge_indices").exists():
        for t in range(num_time_steps):
            try:
                edge_index = torch.from_numpy(np.load(event_dir / "edge_indices" / f"t{t}.npy").astype(np.int64))
                edge_indices_list.append(edge_index)
            except FileNotFoundError:
                logging.warning(f"Edge index file t{t}.npy not found in {event_dir / 'edge_indices'}. Appending empty tensor.")
                edge_indices_list.append(torch.empty((2,0), dtype=torch.long)) # Add empty tensor if file missing
    else: # If edge_indices folder doesn't exist, create empty list for all timesteps
        logging.warning(f"edge_indices directory not found in {event_dir}. Assuming no edges for {num_time_steps} timesteps.")
        edge_indices_list = [torch.empty((2,0), dtype=torch.long) for _ in range(num_time_steps)]


    node_masks_list = []
    if (event_dir / "node_masks").exists():
        for t in range(num_time_steps):
            try:
                mask_numpy = np.load(event_dir / "node_masks" / f"t{t}.npy")
                mask = torch.from_numpy(mask_numpy)
                if mask.ndim == 0: mask = mask.unsqueeze(0)
                node_masks_list.append(mask)
            except FileNotFoundError:
                logging.warning(f"Node mask file t{t}.npy not found in {event_dir / 'node_masks'}. Appending False mask for {X.size(0)} nodes.")
                node_masks_list.append(torch.zeros(X.size(0), dtype=torch.bool)) # Add all-false mask
    else: # If node_masks folder doesn't exist, create all-true masks
        logging.warning(f"node_masks directory not found in {event_dir}. Assuming all nodes active for {num_time_steps} timesteps.")
        node_masks_list = [torch.ones(X.size(0), dtype=torch.bool) for _ in range(num_time_steps)]


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

    # Ensure lists are not empty before moving to device, if num_time_steps was 0 initially.
    if not edge_indices_list and num_time_steps > 0:
        edge_indices_list = [torch.empty((2,0), dtype=torch.long) for _ in range(num_time_steps)]
    if not node_masks_list and num_time_steps > 0:
        node_masks_list = [torch.ones(X.size(0), dtype=torch.bool) for _ in range(num_time_steps)]


    if not cpu_offload:
        X = X.to(device); labels = labels.to(device)
        edge_indices_list = [ei.to(device) for ei in edge_indices_list]
        node_masks_list = [nm.to(device) for nm in node_masks_list]
    return X, edge_indices_list, node_masks_list, labels, time_info

# ---------------------------------------------------------------------------
# Full DySAT Model Components
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
        if N == 0: return x # Handle empty graph input

        h_transformed = self.W(x)
        h_transformed_heads = h_transformed.view(N, self.num_heads, self.out_dim_per_head)
        
        if edge_index.numel() == 0: # No edges
            if self.residual:
                out_h = self.res_fc(x)
            else: # Should ideally not happen if residual is standard
                out_h = torch.zeros(N, self.num_heads * self.out_dim_per_head if self.concat else self.out_dim_per_head, device=x.device)
            out_h = self.norm(out_h)
            return F.elu(out_h)

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
            alpha = scatter_softmax(attn_scores_T.squeeze(-1), dst_nodes_expanded, dim=1, dim_size=N)
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
            out_h_heads = scatter_add(messages, dst_nodes_expanded, dim=1, out=out_h_heads, dim_size=N)
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
        self.d_model = d_model
        self.max_len = max_len
        self._pe = self._build_pe(max_len, d_model)
        self.register_buffer('pe', self._pe)

    def _build_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0: # Odd d_model
            pe[:, 1::2] = torch.cos(position * div_term)[:,:d_model//2] # Correct indexing for odd
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(1) # Shape: [max_len, 1, d_model]

    def forward(self, x_seq): # x_seq shape: [seq_len, batch_size, d_model]
        seq_len = x_seq.size(0)
        if seq_len > self.pe.size(0):
            logging.warning(f"PositionalEncoding: x_seq length {seq_len} > max_len {self.pe.size(0)}. Rebuilding PE.")
            self.pe = self._build_pe(seq_len + 10, self.d_model).to(x_seq.device) # Rebuild with some margin
            self.max_len = seq_len + 10
        return x_seq + self.pe[:seq_len, :].to(x_seq.device)


class FullDySATModel(nn.Module):
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
            self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max(num_time_steps + 1, 50))
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
        if N == 0: # Handle empty graph input for the whole model
             # Need to return something with the expected output shape for the classifier
            num_classes = self.classifier[-1].out_features
            return torch.empty((0, num_classes), device=x_original.device)

        x_projected = self.input_proj(x_original); x_projected = F.elu(x_projected)
        x_projected_dropout = self.dropout_module(x_projected)
        snapshot_embeddings = []
        for t in range(T_snapshots):
            current_edge_index = edge_indices[t].to(x_projected_dropout.device)
            h_struct_t = self.structural_attention(x_projected_dropout, current_edge_index)
            snapshot_embeddings.append(h_struct_t)

        if not snapshot_embeddings: # If no snapshots (e.g., T_snapshots = 0)
            final_node_embeddings = x_projected_dropout
        elif self.use_temporal_attn:
            temporal_input_sequence = torch.stack(snapshot_embeddings, dim=0) # Shape: [T, N, D]
            if temporal_input_sequence.size(0) > 0:
                temporal_input_sequence = self.pos_encoder(temporal_input_sequence)
                temporal_output_sequence = self.temporal_attention(temporal_input_sequence)
                final_node_embeddings = temporal_output_sequence[-1, :, :] # Last time step's embeddings
            else:
                final_node_embeddings = x_projected_dropout
        else: # Use weighted average if not using temporal attention but have snapshots
            last_embedding = snapshot_embeddings[-1]
            if len(snapshot_embeddings) > 1:
                all_embeddings_stacked = torch.stack(snapshot_embeddings, dim=0)
                avg_embedding = all_embeddings_stacked.mean(dim=0)
                temporal_w = torch.sigmoid(self.temporal_weight)
                final_node_embeddings = temporal_w * last_embedding + (1 - temporal_w) * avg_embedding
                final_node_embeddings = self.temporal_norm(final_node_embeddings)
                final_node_embeddings = self.dropout_module(final_node_embeddings)
            else: # Only one snapshot
                final_node_embeddings = last_embedding
        
        logits = self.classifier(final_node_embeddings)
        return logits

# ---------------------------------------------------------------------------
# Training, evaluation for Full DySAT
# ---------------------------------------------------------------------------
def create_balanced_splits_for_full_model(labels, node_masks, device=None, event_name=""):
    n_nodes = labels.size(0); active_nodes_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device)
    for mask_t in node_masks:
        if isinstance(mask_t, torch.Tensor) and mask_t.numel() > 0 and mask_t.shape[0] == n_nodes: active_nodes_mask = active_nodes_mask | mask_t.to(labels.device).bool()
    active_indices = torch.nonzero(active_nodes_mask).squeeze()
    if active_indices.numel() == 0: logging.warning(f"[{event_name}] No active nodes for FullModel. Splits empty."); empty_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device); return empty_mask.clone(), empty_mask.clone(), empty_mask.clone()
    active_labels = labels[active_indices]; train_ratio, val_ratio = 0.7, 0.15; indices_np = active_indices.cpu().numpy(); labels_np = active_labels.cpu().numpy()
    min_samples_per_class_train = 2; unique_labels, counts = np.unique(labels_np, return_counts=True)
    can_stratify_train = len(unique_labels) >= 2 and np.all(counts >= min_samples_per_class_train) and \
                         len(labels_np) >= (min_samples_per_class_train * len(unique_labels) / (1 - train_ratio if (1 - train_ratio) > 0 else 0.01))

    if not can_stratify_train:
        logging.warning(f"[{event_name}] Non-stratified train/temp split for FullModel (active_nodes={len(labels_np)}, unique_labels={len(unique_labels)}, counts={counts}).");
        train_idx, temp_idx = train_test_split(indices_np, test_size=(1-train_ratio), random_state=42, shuffle=True)
    else: train_idx, temp_idx, _, _ = train_test_split(indices_np, labels_np, test_size=(1-train_ratio), random_state=42, stratify=labels_np)

    if len(temp_idx) > 0:
        temp_labels_np = labels[torch.from_numpy(temp_idx).to(labels.device).long()].cpu().numpy()
        unique_temp_labels, temp_counts = np.unique(temp_labels_np, return_counts=True)
        can_stratify_val_test = len(unique_temp_labels) >= 2 and np.all(temp_counts >= 2) and len(temp_labels_np) >= 4
        if not can_stratify_val_test:
             logging.warning(f"[{event_name}] Non-stratified val/test split for FullModel in temp (temp_nodes={len(temp_labels_np)}, unique_labels={len(unique_temp_labels)}, counts={temp_counts}).");
             val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, shuffle=True)
        else: val_idx, test_idx, _, _ = train_test_split(temp_idx, temp_labels_np, test_size=0.5, random_state=42, stratify=temp_labels_np)
    else: val_idx, test_idx = np.array([]), np.array([])

    train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device); val_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device); test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device)
    if len(train_idx) > 0: train_mask[torch.from_numpy(train_idx).to(labels.device).long()] = True
    if len(val_idx) > 0: val_mask[torch.from_numpy(val_idx).to(labels.device).long()] = True
    if len(test_idx) > 0: test_mask[torch.from_numpy(test_idx).to(labels.device).long()] = True
    return train_mask, val_mask, test_mask


def calculate_class_weights_for_full_model(labels, masks=None, device='cpu', event_name=""):
    combined_mask = torch.zeros_like(labels, dtype=torch.bool);
    if masks:
        for m_tensor in masks:
            if isinstance(m_tensor, torch.Tensor) and m_tensor.numel() > 0: combined_mask |= m_tensor.to(labels.device).bool()
    else: combined_mask = torch.ones_like(labels, dtype=torch.bool)
    subset_labels = labels[combined_mask]
    if subset_labels.numel() == 0:
        logging.warning(f"[{event_name}] No labels for FullModel class weights. Using uniform.")
        n_classes = int(labels.max().item() + 1) if labels.numel() > 0 else 2; return torch.ones(n_classes, device=device) / n_classes
    n_classes_actual = int(labels.max().item() + 1) if labels.numel() > 0 else (int(subset_labels.max().item() + 1) if subset_labels.numel() > 0 else 2)
    counts = torch.bincount(subset_labels.long(), minlength=n_classes_actual)
    weights = torch.tensor([1.0 / c if c > 0 else 0 for c in counts], dtype=torch.float, device=device)
    if weights.sum() == 0:
        logging.warning(f"[{event_name}] All class counts zero for FullModel weights. Using uniform.")
        return torch.ones(n_classes_actual, device=device) / n_classes_actual
    weights = weights / weights.sum() * n_classes_actual; weights = torch.clamp(weights, min=0.1, max=10.0); return weights.to(device)

def train_epoch_for_full_model(model, optimizer, criterion, X, edge_indices, node_masks, labels, train_mask, device, clip_val=1.0, event_name=""):
    model.train(); optimizer.zero_grad(set_to_none=True)
    X_dev = X.to(device)
    edge_indices_dev = [ei.to(device) for ei in edge_indices]
    logits = model(X_dev, edge_indices_dev, node_masks)
    if not isinstance(train_mask, torch.Tensor) or train_mask.sum() == 0:
        logging.warning(f"[{event_name}] Train mask empty for FullModel. Loss 0."); return 0.0, 0
    train_logits = logits[train_mask]; train_labels = labels[train_mask]
    if train_logits.shape[0] == 0: return 0.0, 0
    loss = criterion(train_logits, train_labels.to(train_logits.device))
    if torch.isnan(loss) or torch.isinf(loss):
        logging.warning(f"[{event_name}] NaN/Inf loss in FullModel: {loss.item()}"); return loss.item(), train_logits.shape[0]
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val); optimizer.step()
    return loss.item(), train_logits.shape[0]

def evaluate_for_full_model(model, X, edge_indices, node_masks, labels, mask, acc_metric, f1_metric, device, event_name=""):
    model.eval()
    with torch.no_grad():
        X_dev = X.to(device)
        edge_indices_dev = [ei.to(device) for ei in edge_indices]
        logits = model(X_dev, edge_indices_dev, node_masks)
        if not isinstance(mask, torch.Tensor) or mask.sum() == 0: return 0.0, 0.0, 0
        mask_dev = mask.to(logits.device)
        labels_dev = labels.to(logits.device)
        preds_on_mask = logits[mask_dev].argmax(dim=1).cpu(); labels_on_mask = labels_dev[mask_dev].cpu()
        if labels_on_mask.numel() == 0: return 0.0, 0.0, 0
        acc_metric.reset(); f1_metric.reset()
        acc = acc_metric(preds_on_mask, labels_on_mask); f1 = f1_metric(preds_on_mask, labels_on_mask)
    return acc.item(), f1.item(), labels_on_mask.numel()


def run_training_for_full_model(
    hps_values: Union[argparse.Namespace, Dict],
    X: torch.Tensor,
    edge_indices: List[torch.Tensor],
    node_masks: List[torch.Tensor],
    labels: torch.Tensor,
    num_time_steps_for_pe: int,
    fixed_args: argparse.Namespace, 
    event_name: str
):
    if isinstance(hps_values, dict):
        hidden_dim = hps_values['hidden_dim']
        lr = hps_values['lr']
        dropout_rate = hps_values['dropout']
        weight_decay_val = hps_values['weight_decay']
        scheduler_patience = hps_values['scheduler_patience']
        early_stop_patience = hps_values['early_stop_patience']
        num_struct_heads = hps_values['num_struct_heads']
        num_temporal_heads = hps_values['num_temporal_heads']
        epochs = hps_values['epochs']
    elif isinstance(hps_values, argparse.Namespace):
        hidden_dim = hps_values.hidden_dim
        lr = hps_values.lr
        dropout_rate = hps_values.dropout
        weight_decay_val = hps_values.weight_decay
        scheduler_patience = hps_values.scheduler_patience
        early_stop_patience = hps_values.early_stop_patience
        num_struct_heads = hps_values.num_struct_heads
        num_temporal_heads = hps_values.num_temporal_heads
        epochs = hps_values.epochs
    else: raise TypeError("hps_values must be a dict or argparse.Namespace")

    use_temporal_attn = not fixed_args.no_temporal_attn
    device = torch.device(fixed_args.device if fixed_args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    if hidden_dim == 0: # Handle case from Optuna if it suggests 0, or ensure categorical is non-zero
        logging.warning(f"[{event_name}-Full] hidden_dim is 0, adjusting to be compatible with heads.")
        hidden_dim = max(num_struct_heads, num_temporal_heads, 1) # Ensure at least 1

    if num_struct_heads == 0 : num_struct_heads = 1 # GAT layer needs at least one head
    if num_temporal_heads == 0: num_temporal_heads = 1 # Temporal layer needs at least one head
    
    if hidden_dim % num_struct_heads != 0: hidden_dim = num_struct_heads * max(1, math.ceil(hidden_dim / num_struct_heads))
    if hidden_dim % num_temporal_heads != 0: hidden_dim = num_temporal_heads * max(1, math.ceil(hidden_dim / num_temporal_heads))
    if early_stop_patience <= scheduler_patience +2 : early_stop_patience = scheduler_patience + 3

    labels_on_device = labels.to(device)
    node_masks_on_device = [nm.to(device) for nm in node_masks]
    train_mask, val_mask, test_mask = create_balanced_splits_for_full_model(labels_on_device, node_masks_on_device, device=device, event_name=f"{event_name}-Full")
    num_train_samples = train_mask.sum().item(); num_val_samples = val_mask.sum().item(); num_test_samples = test_mask.sum().item()
    logging.debug(f"[{event_name}-Full] Train: {num_train_samples}, Val: {num_val_samples}, Test: {num_test_samples}")

    if num_train_samples == 0 or num_val_samples == 0 :
        logging.warning(f"[{event_name}-Full] Skipping training due to empty train/val split. Returning 0 metrics.")
        return None, 0.0, 0.0, 0.0, {"loss": [], "val_acc": [], "val_f1": [], "test_acc": [], "test_f1": [], "lr": []}, 0, 0, 0

    class_weights = calculate_class_weights_for_full_model(labels_on_device, [train_mask], device=device, event_name=f"{event_name}-Full")
    logging.debug(f"[{event_name}-Full] Class weights: {class_weights.cpu().numpy()}")
    n_classes = int(labels_on_device.max().item() + 1) if labels_on_device.numel() > 0 else 2
    if n_classes < 2: n_classes = 2

    model = FullDySATModel(
        in_dim=X.size(1), hidden_dim=hidden_dim, num_classes=n_classes,
        num_struct_heads=num_struct_heads, num_temporal_heads=num_temporal_heads,
        num_time_steps=num_time_steps_for_pe, dropout=dropout_rate, use_temporal_attn=use_temporal_attn
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay_val)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience, min_lr=1e-7)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    acc_metric_cpu = Accuracy(task="multiclass", num_classes=n_classes, average="micro", ignore_index=-1 if n_classes == 0 else None).cpu()
    f1_metric_cpu = F1Score(task="multiclass", num_classes=n_classes, average="macro", ignore_index=-1 if n_classes == 0 else None).cpu()

    best_val_metric = 0; best_test_acc_at_best_val = 0; best_f1_at_best_val = 0; best_epoch = 0
    patience_counter = 0; history = {"loss": [], "val_acc": [], "val_f1": [], "test_acc": [], "test_f1": [], "lr": []}
    best_model_state = None

    for epoch in range(epochs):
        loss, _ = train_epoch_for_full_model(model, optimizer, criterion, X, edge_indices, node_masks, labels_on_device, train_mask, device, event_name=f"{event_name}-Full")
        val_acc, val_f1, _ = evaluate_for_full_model(model, X, edge_indices, node_masks, labels_on_device, val_mask, acc_metric_cpu, f1_metric_cpu, device, event_name=f"{event_name}-Full")
        test_acc, test_f1, _ = evaluate_for_full_model(model, X, edge_indices, node_masks, labels_on_device, test_mask, acc_metric_cpu, f1_metric_cpu, device, event_name=f"{event_name}-Full")
        scheduler.step(val_f1); current_lr = optimizer.param_groups[0]['lr']
        logging.debug(f"[{event_name}-Full] Ep {epoch:03d}: loss={loss:.4f},val_acc={val_acc:.4f},val_f1={val_f1:.4f},test_acc={test_acc:.4f},test_f1={test_f1:.4f},lr={current_lr:.7f}")
        history["loss"].append(loss); history["val_acc"].append(val_acc); history["val_f1"].append(val_f1)
        history["test_acc"].append(test_acc); history["test_f1"].append(test_f1); history["lr"].append(current_lr)
        primary_metric_val = val_f1
        if primary_metric_val > best_val_metric :
            best_val_metric = primary_metric_val; best_test_acc_at_best_val = test_acc; best_f1_at_best_val = test_f1
            best_epoch = epoch; patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else: patience_counter += 1
        if patience_counter >= early_stop_patience: logging.info(f"[{event_name}-Full] Early stopping at epoch {epoch}. Best Val F1: {best_val_metric:.4f}"); break
        if current_lr <= scheduler.min_lrs[0] + 1e-9 and patience_counter > early_stop_patience // 2 : logging.info(f"[{event_name}-Full] LR at min. Stopping."); break
    if best_model_state: model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    else: logging.warning(f"[{event_name}-Full] No best model state found, using last epoch model."); best_val_metric = val_f1 # Use last val_f1 if no improvement
    logging.info(f"[{event_name}-Full] Best Val F1: {best_val_metric:.4f} at epoch {best_epoch}. Test Acc: {best_test_acc_at_best_val:.4f}, Test F1: {best_f1_at_best_val:.4f}")
    return model, best_val_metric, best_test_acc_at_best_val, best_f1_at_best_val, history, num_train_samples, num_val_samples, num_test_samples


# ----- START: Integration of SimpleDySAT components for comparison -----
class SimpleAttentionLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim; self.output_dim = output_dim; self.dropout = dropout
        self.linear_transform = nn.Linear(input_dim, output_dim, bias=False)
        self.attention_weights_mlp = nn.Linear(output_dim * 2, 1, bias=False)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.projection = nn.Linear(input_dim, output_dim, bias=False) if input_dim != output_dim else None
        nn.init.xavier_uniform_(self.linear_transform.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.attention_weights_mlp.weight, gain=nn.init.calculate_gain('relu'))
        if self.projection: nn.init.xavier_uniform_(self.projection.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x_input_features, edge_index):
        N = x_input_features.size(0)
        if N == 0: return x_input_features # Handle empty graph input

        h_transformed = self.linear_transform(x_input_features)
        h_aggregated_messages = torch.zeros_like(h_transformed)
        edge_index = edge_index.to(x_input_features.device)

        if edge_index.numel() > 0 and edge_index.size(1) > 0:
            src_nodes, dst_nodes = edge_index[0], edge_index[1]
            h_src_for_attn = h_transformed[src_nodes]; h_dst_for_attn = h_transformed[dst_nodes]
            edge_attn_input = torch.cat([h_src_for_attn, h_dst_for_attn], dim=-1)
            raw_attn_scores = self.attention_weights_mlp(edge_attn_input).squeeze(-1)
            raw_attn_scores = F.leaky_relu(raw_attn_scores, 0.2)
            if TORCH_SCATTER_AVAILABLE:
                alpha_for_edges = scatter_softmax(raw_attn_scores, dst_nodes, dim=0, dim_size=N)
                alpha_for_edges = F.dropout(alpha_for_edges, self.dropout, training=self.training)
                messages_to_aggregate = h_transformed[src_nodes] * alpha_for_edges.unsqueeze(-1)
                h_aggregated_messages = scatter_add(messages_to_aggregate, dst_nodes, dim=0, out=h_aggregated_messages, dim_size=N)
            else: # Fallback if torch_scatter not available
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
        
        residual_features = self.projection(x_input_features) if self.projection else x_input_features
        h_final = h_aggregated_messages + residual_features
        h_final = self.layer_norm(h_final)
        h_final = F.dropout(h_final, self.dropout, training=self.training)
        return h_final

class SimpleDySATModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1, use_temporal: bool = True):
        super().__init__()
        self.in_dim = in_dim; self.hidden_dim = hidden_dim; self.dropout = dropout; self.use_temporal = use_temporal
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.structural_attn = SimpleAttentionLayer(hidden_dim, hidden_dim, dropout=self.dropout)
        if use_temporal:
            self.temporal_weight = nn.Parameter(torch.tensor(0.5)); self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(hidden_dim, num_classes))
        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight, gain=nn.init.calculate_gain('relu'))
        if self.input_proj.bias is not None: nn.init.zeros_(self.input_proj.bias)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('linear'))
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x_original, edge_indices, node_masks):
        N = x_original.size(0); num_time_steps = len(edge_indices)
        if N == 0:
            num_classes = self.classifier[-1].out_features
            return torch.empty((0, num_classes), device=x_original.device)

        x_projected = self.input_proj(x_original); x_projected = F.elu(x_projected)
        x_projected_dropout = F.dropout(x_projected, self.dropout, training=self.training)

        if not self.use_temporal or num_time_steps == 0:
            # If not using temporal or no time steps, use the last available snapshot if any, or just projected features
            final_embedding = x_projected_dropout # Default
            if num_time_steps > 0:
                last_valid_t = -1
                for t_idx in range(num_time_steps - 1, -1, -1):
                    # Check if node_masks[t_idx] is valid and has active nodes
                    is_mask_valid_for_t = isinstance(node_masks[t_idx], torch.Tensor) and node_masks[t_idx].numel() > 0 and node_masks[t_idx].sum() > 0
                    is_edges_valid_for_t = isinstance(edge_indices[t_idx], torch.Tensor) # Allow empty edge_index
                    if is_mask_valid_for_t: # Process if mask is valid, even if no edges
                        last_valid_t = t_idx; break
                if last_valid_t != -1:
                    final_embedding = self.structural_attn(x_projected_dropout, edge_indices[last_valid_t].to(x_projected_dropout.device))
        else: # Use temporal logic
            timestep_embeddings = []
            for t in range(num_time_steps):
                current_mask = node_masks[t]; current_edges = edge_indices[t]
                is_mask_valid = isinstance(current_mask, torch.Tensor) and current_mask.numel() > 0
                if is_mask_valid and current_mask.sum() > 0 : # Only process if mask indicates active nodes
                    h_t = self.structural_attn(x_projected_dropout, current_edges.to(x_projected_dropout.device))
                    timestep_embeddings.append(h_t)
            
            if not timestep_embeddings: final_embedding = x_projected_dropout
            else:
                last_embedding = timestep_embeddings[-1]
                if len(timestep_embeddings) > 1:
                    all_embeddings_stacked = torch.stack(timestep_embeddings, dim=0)
                    avg_embedding = all_embeddings_stacked.mean(dim=0)
                    temporal_w = torch.sigmoid(self.temporal_weight)
                    final_embedding = temporal_w * last_embedding + (1 - temporal_w) * avg_embedding
                    final_embedding = self.temporal_norm(final_embedding)
                    final_embedding = F.dropout(final_embedding, self.dropout, training=self.training)
                else: final_embedding = last_embedding
        logits = self.classifier(final_embedding)
        return logits

def create_balanced_splits_for_simple_model(labels, node_masks, device=None, event_name=""):
    n_nodes = labels.size(0)
    active_nodes_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device)
    for mask_t in node_masks:
        if isinstance(mask_t, torch.Tensor) and mask_t.numel() > 0 and mask_t.shape[0] == n_nodes:
             active_nodes_mask = active_nodes_mask | mask_t.to(labels.device).bool()
    active_indices = torch.nonzero(active_nodes_mask).squeeze()
    if active_indices.numel() == 0:
        logging.warning(f"[{event_name}-Simple] No active nodes. Splits empty.")
        empty_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device)
        return empty_mask.clone(), empty_mask.clone(), empty_mask.clone()
    active_labels = labels[active_indices]; train_ratio, val_ratio = 0.7, 0.15
    indices_np = active_indices.cpu().numpy(); labels_np = active_labels.cpu().numpy()
    min_samples_per_class_train = 2; unique_labels, counts = np.unique(labels_np, return_counts=True)
    can_stratify_train_temp = len(unique_labels) >= 2 and np.all(counts >= min_samples_per_class_train) and \
                              len(labels_np) >= (min_samples_per_class_train * len(unique_labels) / (1 - train_ratio if (1-train_ratio)>0 else 0.01) )
    if not can_stratify_train_temp:
        logging.warning(f"[{event_name}-Simple] Non-stratified train/temp split ({len(unique_labels)} classes, counts {counts}).")
        train_idx, temp_idx = train_test_split(indices_np, test_size=(1-train_ratio), random_state=42, shuffle=True)
    else:
        train_idx, temp_idx, _, _ = train_test_split(indices_np, labels_np, test_size=(1-train_ratio), random_state=42, stratify=labels_np)

    if len(temp_idx) > 0:
        temp_labels_np = labels[torch.from_numpy(temp_idx).to(labels.device).long()].cpu().numpy()
        unique_temp_labels, temp_counts = np.unique(temp_labels_np, return_counts=True)
        can_stratify_val_test = len(unique_temp_labels) >= 2 and np.all(temp_counts >= 2) and len(temp_labels_np) >= 4
        if not can_stratify_val_test:
             logging.warning(f"[{event_name}-Simple] Non-stratified val/test split in temp_idx ({len(unique_temp_labels)} classes, counts {temp_counts}).")
             val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, shuffle=True)
        else:
             val_idx, test_idx, _, _ = train_test_split(temp_idx, temp_labels_np, test_size=0.5, random_state=42, stratify=temp_labels_np)
    else:
        val_idx, test_idx = np.array([]), np.array([])

    train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device); val_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device); test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=labels.device)
    if len(train_idx) > 0: train_mask[torch.from_numpy(train_idx).to(labels.device).long()] = True
    if len(val_idx) > 0: val_mask[torch.from_numpy(val_idx).to(labels.device).long()] = True
    if len(test_idx) > 0: test_mask[torch.from_numpy(test_idx).to(labels.device).long()] = True
    return train_mask, val_mask, test_mask

def calculate_class_weights_for_simple_model(labels, masks=None, device='cpu', event_name=""):
    combined_mask = torch.zeros_like(labels, dtype=torch.bool)
    if masks:
        for m_tensor in masks:
            if isinstance(m_tensor, torch.Tensor) and m_tensor.numel() > 0: combined_mask |= m_tensor.to(labels.device).bool()
    else: combined_mask = torch.ones_like(labels, dtype=torch.bool)
    subset_labels = labels[combined_mask]
    if subset_labels.numel() == 0:
        logging.warning(f"[{event_name}-Simple] No labels for class weights. Using uniform.")
        n_classes = int(labels.max().item() + 1) if labels.numel() > 0 else 2
        return torch.ones(n_classes, device=device) / n_classes
    n_classes_actual = int(labels.max().item() + 1) if labels.numel() > 0 else (int(subset_labels.max().item() + 1) if subset_labels.numel() > 0 else 2)
    counts = torch.bincount(subset_labels.long(), minlength=n_classes_actual)
    weights = torch.tensor([1.0 / c if c > 0 else 0 for c in counts], dtype=torch.float, device=device)
    if weights.sum() == 0:
        logging.warning(f"[{event_name}-Simple] All class counts zero for weights. Using uniform.")
        return torch.ones(n_classes_actual, device=device) / n_classes_actual
    weights = weights / weights.sum() * n_classes_actual; weights = torch.clamp(weights, min=0.1, max=10.0)
    return weights.to(device)

def train_epoch_for_simple_model(model, optimizer, criterion, X, edge_indices, node_masks, labels, train_mask, device, clip_val=1.0, event_name=""):
    model.train(); optimizer.zero_grad(set_to_none=True)
    X_dev = X.to(device)
    edge_indices_dev = [ei.to(device) for ei in edge_indices]

    logits = model(X_dev, edge_indices_dev, node_masks)
    if not isinstance(train_mask, torch.Tensor) or train_mask.sum() == 0:
      logging.warning(f"[{event_name}-Simple] Train mask empty. Loss 0."); return 0.0
    train_logits = logits[train_mask]; train_labels = labels[train_mask]
    if train_logits.shape[0] == 0: logging.warning(f"[{event_name}-Simple] No samples by train_mask. Loss 0."); return 0.0
    loss = criterion(train_logits, train_labels.to(train_logits.device))
    if torch.isnan(loss) or torch.isinf(loss):
        logging.warning(f"[{event_name}-Simple] NaN/Inf loss: {loss.item()}."); return loss.item()
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val); optimizer.step()
    return loss.item()

def evaluate_for_simple_model(model, X, edge_indices, node_masks, labels, mask, acc_metric, f1_metric, device, event_name=""):
    model.eval()
    with torch.no_grad():
        X_dev = X.to(device)
        edge_indices_dev = [ei.to(device) for ei in edge_indices]
        logits = model(X_dev, edge_indices_dev, node_masks)
        if not isinstance(mask, torch.Tensor) or mask.sum() == 0: return 0.0, 0.0
        mask_dev = mask.to(logits.device); labels_dev = labels.to(logits.device)
        preds_on_mask = logits[mask_dev].argmax(dim=1).cpu(); labels_on_mask = labels_dev[mask_dev].cpu()
        if labels_on_mask.numel() == 0: return 0.0, 0.0
        acc_metric.reset(); f1_metric.reset()
        acc = acc_metric(preds_on_mask, labels_on_mask); f1 = f1_metric(preds_on_mask, labels_on_mask)
    return acc.item(), f1.item()

def run_training_for_simple_model(
    hps_dict: Dict, X: torch.Tensor, edge_indices: List[torch.Tensor], 
    node_masks: List[torch.Tensor], labels: torch.Tensor,
    fixed_args: argparse.Namespace, event_name: str
):
    hidden_dim = hps_dict['hidden_dim']; lr = hps_dict['lr']; epochs = hps_dict['epochs']
    use_temporal = not hps_dict.get('no_temporal', False)
    dropout_rate = hps_dict['dropout']; weight_decay_val = hps_dict['weight_decay']
    scheduler_patience = hps_dict['scheduler_patience']; early_stop_patience = hps_dict['early_stop_patience']
    device = torch.device(fixed_args.device if fixed_args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    labels_on_device = labels.to(device)
    node_masks_on_device = [nm.to(device) for nm in node_masks]
    train_mask, val_mask, test_mask = create_balanced_splits_for_simple_model(labels_on_device, node_masks_on_device, device=device, event_name=f"{event_name}-Simple")
    logging.info(f"[{event_name}-Simple] Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        logging.error(f"[{event_name}-Simple] Train or Val split empty.");
        return None, 0.0, 0.0, 0.0, {"loss": [], "val_acc": [], "val_f1": [], "test_acc": [], "test_f1": [], "lr": []}, 0
    
    class_weights = calculate_class_weights_for_simple_model(labels_on_device, [train_mask], device=device, event_name=f"{event_name}-Simple")
    logging.info(f"[{event_name}-Simple] Class weights: {class_weights.cpu().numpy()}")
    n_classes = int(labels_on_device.max().item() + 1) if labels_on_device.numel() > 0 else 2
    if n_classes < 2: n_classes = 2
    
    model = SimpleDySATModel(in_dim=X.size(1), hidden_dim=hidden_dim, num_classes=n_classes, dropout=dropout_rate, use_temporal=use_temporal).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay_val)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience, min_lr=1e-7)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    acc_metric_cpu = Accuracy(task="multiclass", num_classes=n_classes, average="micro").cpu()
    f1_metric_cpu = F1Score(task="multiclass", num_classes=n_classes, average="macro").cpu()

    best_val_metric = 0.0; best_test_acc_at_best_val = 0.0; best_f1_at_best_val = 0.0; best_epoch = 0
    patience_counter = 0; history = {"loss": [], "val_acc": [], "val_f1": [], "test_acc": [], "test_f1": [], "lr": []}
    best_model_state = None

    for epoch in range(epochs):
        loss = train_epoch_for_simple_model(model, optimizer, criterion, X, edge_indices, node_masks, labels_on_device, train_mask, device, event_name=f"{event_name}-Simple")
        val_acc, val_f1 = evaluate_for_simple_model(model, X, edge_indices, node_masks, labels_on_device, val_mask, acc_metric_cpu, f1_metric_cpu, device, event_name=f"{event_name}-Simple")
        test_acc, test_f1 = evaluate_for_simple_model(model, X, edge_indices, node_masks, labels_on_device, test_mask, acc_metric_cpu, f1_metric_cpu, device, event_name=f"{event_name}-Simple")
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 10 == 0 or epoch == epochs -1:
            logging.info(f"[{event_name}-Simple] Ep {epoch:03d}: loss={loss:.4f},val_acc={val_acc:.4f},val_f1={val_f1:.4f},test_acc={test_acc:.4f},test_f1={test_f1:.4f},lr={current_lr:.7f}")
        history["loss"].append(loss); history["val_acc"].append(val_acc); history["val_f1"].append(val_f1)
        history["test_acc"].append(test_acc); history["test_f1"].append(test_f1); history["lr"].append(current_lr)
        primary_metric_val = val_acc
        if primary_metric_val > best_val_metric:
            best_val_metric = primary_metric_val; best_test_acc_at_best_val = test_acc
            best_f1_at_best_val = test_f1; best_epoch = epoch; patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else: patience_counter += 1
        if patience_counter >= early_stop_patience: logging.info(f"[{event_name}-Simple] Early stopping at epoch {epoch}. Best epoch: {best_epoch}"); break
        if current_lr <= scheduler.min_lrs[0] + 1e-9 and patience_counter > early_stop_patience // 2: logging.info(f"[{event_name}-Simple] LR at min. Stopping."); break
    
    if best_model_state: model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    else: logging.warning(f"[{event_name}-Simple] No best model state, using last epoch model."); best_val_metric = val_acc # Use last val_acc if no improvement

    logging.info(f"[{event_name}-Simple] Best Val Acc: {best_val_metric:.4f} at epoch {best_epoch}. Test Acc: {best_test_acc_at_best_val:.4f}, Test F1: {best_f1_at_best_val:.4f}")
    return model, best_val_metric, best_test_acc_at_best_val, best_f1_at_best_val, history, best_epoch
# ----- END: Integration of SimpleDySAT components -----

# ---------------------------------------------------------------------------
# Args Parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} trainer with HP Tuning, Ablations, and Comparison mode")
    parser.add_argument("--data-dir", default="data_dysat_v2", help="Path to preprocessed data for Full DySAT")
    parser.add_argument("--event", type=str, default=None, help="Event name, comma-separated list, or 'all'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    parser.add_argument("--max-nodes-subsample", type=int, default=None, help="Max nodes for subsampling")
    parser.add_argument("--run-tag", type=str, default="", help="Tag for manual run outputs (Full DySAT)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dim (Full DySAT/Optuna default)")
    parser.add_argument("--num-struct-heads", type=int, default=4, help="Struct heads (Full DySAT/Optuna default)")
    parser.add_argument("--num-temporal-heads", type=int, default=4, help="Temporal heads (Full DySAT/Optuna default)")
    parser.add_argument("--no-temporal-attn", action="store_true", help="Disable temporal self-attn (Full DySAT)")
    parser.add_argument("--lr", type=float, default=0.0007, help="Learning rate (Full DySAT/Optuna default)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (Full DySAT/Optuna default)")
    parser.add_argument("--weight-decay", type=float, default=1.7e-5, help="Weight decay (Full DySAT/Optuna default)")
    parser.add_argument("--scheduler-patience", type=int, default=14, help="Scheduler patience (Full DySAT/Optuna default)")
    parser.add_argument("--early-stop-patience", type=int, default=20, help="Early stopping (Full DySAT/Optuna default)")
    parser.add_argument("--epochs", type=int, default=200, help="Max epochs (used by all modes)")
    parser.add_argument("--optuna-study", action="store_true", help="Run Optuna study (Full DySAT)")
    parser.add_argument("--optuna-trials", type=int, default=30, help="Optuna trials (Full DySAT)")
    parser.add_argument("--optuna-study-name", type=str, default="FullDySAT_PHEME_Global", help="Optuna study name (Full DySAT)")
    parser.add_argument("--optuna-event-subset", type=str, default="germanwings-crash,charliehebdo", help="Events for Optuna (Full DySAT)")
    parser.add_argument("--run-full-ablation", action="store_true", help="Run full ablation (Full DySAT)")
    parser.add_argument("--optuna-trials-per-event", type=int, default=15, help="Optuna trials per event (Full DySAT)")
    parser.add_argument("--compare-with-simple-dysat", action="store_true", help="Run Full vs Simple DySAT comparison.")
    parser.add_argument("--simple-model-data-dir", default="data_dysat", help="Data dir for Simple DySAT (comparison mode).")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Optuna Objective Function (Full DySAT)
# ---------------------------------------------------------------------------
def objective_full_dysat(trial: optuna.trial.Trial, base_args: argparse.Namespace, event_dirs_for_study: List[Path]):
    hps_values = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.05),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "scheduler_patience": trial.suggest_int("scheduler_patience", 7, 20),
        "early_stop_patience": trial.suggest_int("early_stop_patience", 15, 35),
        "num_struct_heads": trial.suggest_categorical("num_struct_heads", [2, 4, 8]),
        "num_temporal_heads": trial.suggest_categorical("num_temporal_heads", [2, 4, 8]),
        "epochs": base_args.epochs
    }
    # Sanitize HPs before passing to training
    if hps_values["hidden_dim"] == 0: hps_values["hidden_dim"] = max(hps_values["num_struct_heads"], hps_values["num_temporal_heads"], 1)
    if hps_values["num_struct_heads"] == 0: hps_values["num_struct_heads"] = 1
    if hps_values["num_temporal_heads"] == 0: hps_values["num_temporal_heads"] = 1
    if hps_values["hidden_dim"] % hps_values["num_struct_heads"] != 0: hps_values["hidden_dim"] = hps_values["num_struct_heads"] * max(1, math.ceil(hps_values["hidden_dim"] / hps_values["num_struct_heads"]))
    if hps_values["hidden_dim"] % hps_values["num_temporal_heads"] != 0: hps_values["hidden_dim"] = hps_values["num_temporal_heads"] * max(1, math.ceil(hps_values["hidden_dim"] / hps_values["num_temporal_heads"]))
    if hps_values["early_stop_patience"] <= hps_values["scheduler_patience"] + 2: hps_values["early_stop_patience"] = hps_values["scheduler_patience"] + 3
    
    trial.set_user_attr("hps", hps_values)
    logging.info(f"Optuna Trial {trial.number} FullDySAT: HPs: {hps_values}")
    set_seed(base_args.seed)
    device = torch.device(base_args.device if base_args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    all_event_metrics = []
    for event_dir in event_dirs_for_study:
        event_name = event_dir.name
        try:
            X, ei, nm, lab, ti = load_temporal_data(event_dir, device, False, base_args.max_nodes_subsample)
            if X.nelement() == 0 or lab.nelement() == 0 or not any(isinstance(m, torch.Tensor) and m.numel()>0 for m in nm):
                logging.warning(f"Trial {trial.number}: Skipping {event_name}, empty data."); continue
            n_ts = ti.get("num_windows", len(ei)); n_ts = max(1, n_ts)
            _, val_f1, _, _, _, _, _, _ = run_training_for_full_model(
                hps_values=hps_values, X=X, edge_indices=ei, node_masks=nm, labels=lab,
                num_time_steps_for_pe=n_ts, fixed_args=base_args, event_name=event_name
            )
            all_event_metrics.append({'val_f1': val_f1})
            del X, ei, nm, lab, ti; gc.collect(); torch.cuda.empty_cache()
        except Exception as e: logging.error(f"Optuna Trial {trial.number} FullDySAT Error ({event_name}): {e}", exc_info=False)
    if not all_event_metrics: logging.warning(f"Optuna Trial {trial.number} FullDySAT: No events processed."); return -1.0
    avg_val_f1 = np.mean([m['val_f1'] for m in all_event_metrics if m['val_f1'] is not None and not np.isnan(m['val_f1'])])
    if np.isnan(avg_val_f1): avg_val_f1 = -1.0 # Handle case where all val_f1s were None/NaN
    logging.info(f"Optuna Trial {trial.number} FullDySAT: Avg Val F1: {avg_val_f1:.4f}")
    return avg_val_f1

# ---------------------------------------------------------------------------
# Helper functions for saving results (Full DySAT)
# ---------------------------------------------------------------------------
def save_event_results_md_for_full_model(output_dir: Path, hps: Union[Dict, argparse.Namespace], val_f1: float, test_acc: float, test_f1: float, event_name: str, splits: Tuple[int,int,int]):
    md_content = f"# Results: {event_name} (Full DySAT)\n\n## HPs:\n"
    hps_dict = vars(hps) if isinstance(hps, argparse.Namespace) else hps
    for k, v in hps_dict.items():
        if k not in ['data_dir', 'event', 'device', 'seed', 'run_tag', 'optuna_study', 'optuna_trials', 'optuna_study_name', 'optuna_event_subset', 'run_full_ablation', 'optuna_trials_per_event', 'results_file', 'compare_with_simple_dysat', 'simple_model_data_dir']:
             md_content += f"- {k}: {v}\n"
    md_content += f"\n## Metrics:\n- Val F1: {val_f1:.4f}\n- Test Acc: {test_acc:.4f}\n- Test F1: {test_f1:.4f}\n"
    md_content += f"\n## Splits:\n- Train: {splits[0]}, Val: {splits[1]}, Test: {splits[2]}\n"
    with open(output_dir / "results_full_dysat.md", "w") as f: f.write(md_content)
    logging.info(f"Saved results_full_dysat.md for {event_name} to {output_dir}")

def save_training_plots_for_full_model(output_dir: Path, history: Dict, event_name: str, run_tag: str = ""):
    if not history or not history.get("loss"): logging.warning(f"No history for Full DySAT {event_name}."); return
    plt.figure(figsize=(24, 5))
    plt.subplot(1, 4, 1); plt.plot(history["loss"]); plt.title("Loss")
    plt.subplot(1, 4, 2); plt.plot(history["val_f1"], label="Val F1"); plt.plot(history.get("test_f1", []), label="Test F1"); plt.title("F1"); plt.legend()
    plt.subplot(1, 4, 3); plt.plot(history["val_acc"], label="Val Acc"); plt.plot(history["test_acc"], label="Test Acc"); plt.title("Acc"); plt.legend()
    plt.subplot(1, 4, 4); plt.plot(history["lr"]); plt.title("LR"); plt.yscale("log")
    suptitle = f"Curves for {event_name} (Full DySAT)";
    if run_tag: suptitle += f" ({run_tag})"
    plt.suptitle(suptitle); plt.tight_layout(rect=[0,0,1,0.96])
    plot_path = output_dir / "training_plots_full_dysat.png"
    plt.savefig(plot_path); plt.close(); logging.info(f"Saved Full DySAT plots for {event_name} to {plot_path}")

def save_optuna_study_results_for_full_model(study: optuna.Study, output_dir: Path, study_name_prefix: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    df_results = study.trials_dataframe(); csv_path = output_dir / f"{study_name_prefix}_optuna_full_dysat_results.csv"
    df_results.to_csv(csv_path, index=False); logging.info(f"Optuna (Full DySAT) results: {csv_path}")
    if OPTUNA_VISUALIZATION_AVAILABLE:
        try:
            fig_p = plot_parallel_coordinate(study); fig_p.write_image(output_dir / f"{study_name_prefix}_optuna_full_dysat_parallel.png")
            fig_h = plot_optimization_history(study); fig_h.write_image(output_dir / f"{study_name_prefix}_optuna_full_dysat_history.png")
            logging.info(f"Optuna (Full DySAT) plots saved to {output_dir}")
        except Exception as e: logging.warning(f"Failed Optuna (Full DySAT) plots: {e}")
    else: logging.info("Optuna viz not available. Skipping plots for Full DySAT study.")

# ---------------------------------------------------------------------------
# Core Workflows (Full DySAT)
# ---------------------------------------------------------------------------
def get_event_dirs(data_root_str: str, event_arg: Optional[str]) -> List[Path]:
    data_root = Path(data_root_str)
    if not data_root.exists(): logging.error(f"Data dir {data_root} not found."); return []
    event_list = []
    if event_arg and event_arg.lower() != 'all':
        for e_name in event_arg.split(','):
            ep = data_root / e_name.strip()
            if ep.exists() and ep.is_dir(): event_list.append(ep)
            else: logging.warning(f"Event dir {ep} not found. Skipping.")
    else: event_list = [d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith("_") and not d.name.lower() == "all"]
    if not event_list: logging.error(f"No event dirs found/specified in {data_root}.")
    return event_list

def evaluate_event_with_fixed_hps_for_full_model(event_dir: Path, hps_config: Union[argparse.Namespace, Dict], base_args: argparse.Namespace, output_dir: Path):
    event_name = event_dir.name; device = torch.device(base_args.device if base_args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Eval Full DySAT on {event_name} with fixed HPs. Output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = None; val_f1, test_acc, test_f1_score = 0.0, 0.0, 0.0 
    history_data = {}
    try:
        X, ei, nm, lab, ti = load_temporal_data(event_dir, device, False, base_args.max_nodes_subsample)
        if X.nelement() == 0 or lab.nelement() == 0 or not any(isinstance(m, torch.Tensor) and m.numel()>0 for m in nm):
            logging.warning(f"Skipping {event_name} (Full DySAT): empty data."); return None, 0.0, 0.0, 0.0, {}
        n_ts = ti.get("num_windows", len(ei)); n_ts = max(1, n_ts)
        
        train_res = run_training_for_full_model(
            hps_values=hps_config, X=X, edge_indices=ei, node_masks=nm, labels=lab,
            num_time_steps_for_pe=n_ts, fixed_args=base_args, event_name=event_name
        )
        model, val_f1, test_acc, test_f1_score, history_data, tr_s, v_s, te_s = train_res

        if model is None: logging.warning(f"No model from training {event_name} (Full DySAT)."); return None, 0.0, 0.0, 0.0, history_data or {}
        save_event_results_md_for_full_model(output_dir, hps_config, val_f1, test_acc, test_f1_score, event_name, (tr_s, v_s, te_s))
        save_training_plots_for_full_model(output_dir, history_data, event_name, run_tag=getattr(base_args, "run_tag", ""))
        torch.save(model.state_dict(), output_dir / "model_full_dysat.pt")
        del X, ei, nm, lab, history_data, ti; gc.collect(); torch.cuda.empty_cache()
        return model, val_f1, test_acc, test_f1_score, getattr(hps_config, 'params', hps_config)
    except Exception as e:
        logging.error(f"Error processing {event_name} (Full DySAT) for fixed HP eval: {e}", exc_info=True)
        return model, val_f1, test_acc, test_f1_score, history_data or {}

# ... (Optuna single event, find overall best HPs, aggregated report for Full DySAT remain largely same but call the corrected functions) ...

def run_optuna_for_single_event_full_model(event_dir: Path, base_args: argparse.Namespace, num_trials: int, output_dir: Path):
    if not OPTUNA_AVAILABLE: logging.error("Optuna not installed."); return None, None
    event_name = event_dir.name; study_name = f"{MODEL_NAME}_{event_name}_Optuna" # Use MODEL_NAME
    storage_path = f"sqlite:///{output_dir.parent / (study_name.replace(' ','_') + '.db')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Optuna study (Full DySAT) for {event_name} ({num_trials} trials). DB: {storage_path}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=study_name, storage=storage_path, direction="maximize", load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=max(1,num_trials//10), n_warmup_steps=5, n_min_trials=max(1,num_trials//5)),
                                sampler=optuna.samplers.TPESampler(seed=base_args.seed, n_startup_trials=max(1,num_trials//5)))
    study.optimize(lambda trial: objective_full_dysat(trial, base_args, [event_dir]), 
                     n_trials=num_trials, gc_after_trial=True, show_progress_bar=True)
    logging.info(f"Optuna study (Full DySAT) for {event_name} done. {len(study.trials)} trials.")
    save_optuna_study_results_for_full_model(study, output_dir, event_name) # Pass correct prefix
    try: 
        best_trial = study.best_trial
        logging.info(f"Best trial for {event_name} (Full DySAT, Val F1): {best_trial.value:.4f}\nBest HPs: {best_trial.params}")
        return best_trial.params, best_trial.value
    except ValueError: 
        logging.warning(f"No successful Optuna trials for {event_name} (Full DySAT).")
        return None, None

def find_overall_best_hps_with_optuna_full_model(base_args: argparse.Namespace, event_dirs_for_global_study: List[Path]):
    if not OPTUNA_AVAILABLE: logging.error("Optuna not installed."); return None
    study_name = base_args.optuna_study_name
    global_optuna_output_dir = MAIN_MODEL_RESULTS_DIR / f"global_optuna_study_{study_name.replace(' ','_')}"
    storage_path = f"sqlite:///{global_optuna_output_dir / (study_name.replace(' ','_') + '.db')}"
    global_optuna_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"GLOBAL Optuna (Full DySAT): {study_name} ({base_args.optuna_trials} trials) on {[d.name for d in event_dirs_for_global_study]}. DB: {storage_path}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=study_name, storage=storage_path, direction="maximize", load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=max(1,base_args.optuna_trials//10), n_warmup_steps=5, n_min_trials=max(1,base_args.optuna_trials//5)),
                                sampler=optuna.samplers.TPESampler(seed=base_args.seed, n_startup_trials=max(1,base_args.optuna_trials//5)))
    study.optimize(lambda trial: objective_full_dysat(trial, base_args, event_dirs_for_global_study), 
                     n_trials=base_args.optuna_trials, gc_after_trial=True, show_progress_bar=True)
    logging.info(f"Global Optuna (Full DySAT) '{study_name}' done. {len(study.trials)} trials.")
    save_optuna_study_results_for_full_model(study, global_optuna_output_dir, "global_full_dysat")
    try: 
        best_trial = study.best_trial
        logging.info(f"Best trial GLOBAL (Full DySAT, Avg Val F1): {best_trial.value:.4f}\nOverall Best HPs: {best_trial.params}")
        return best_trial.params
    except ValueError: 
        logging.warning("No successful trials in Global Optuna (Full DySAT).")
        return None

def generate_aggregated_report_for_full_model(overall_best_hps: Dict, all_event_dirs: List[Path], base_args: argparse.Namespace):
    MAIN_MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MAIN_MODEL_RESULTS_DIR / "aggregated_results_full_dysat.md"
    md_content = f"# {MODEL_NAME} - Aggregated Report (Overall Best HPs)\n\n"
    md_content += "## Overall Best HPs (Full DySAT):\n"
    for k, v in overall_best_hps.items(): md_content += f"- {k}: {v}\n"
    md_content += f"- use_temporal_attn: {not base_args.no_temporal_attn}\n" # from base_args
    md_content += "\n## Per-Event Performance (Full DySAT with Overall Best HPs):\n| Event Stream | Test Acc | Test F1 (macro) |\n|---|---|---|\n"
    all_test_accs, all_test_f1s = [], []
    
    # Ensure epochs from overall_best_hps is used, or fallback to base_args.epochs
    hps_for_agg = {**base_args.__dict__, **overall_best_hps} # Prioritize overall_best_hps
    hps_for_agg.pop('event', None) # Remove event if it was in base_args to avoid conflict

    for event_dir in tqdm(all_event_dirs, desc="Aggregating Full DySAT Report"):
        event_name = event_dir.name
        device = torch.device(base_args.device if base_args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            X, ei, nm, lab, ti = load_temporal_data(event_dir, device, False, base_args.max_nodes_subsample)
            if X.nelement() == 0 or lab.nelement() == 0 or not any(isinstance(m, torch.Tensor) and m.numel()>0 for m in nm):
                md_content += f"| {event_name:<20} | SKIPPED | SKIPPED |\n"; continue
            n_ts = ti.get("num_windows", len(ei)); n_ts = max(1,n_ts)

            # Pass the combined HPs dictionary
            _, _, test_acc, test_f1, _, _,_,_ = run_training_for_full_model(
                hps_values=hps_for_agg, X=X, edge_indices=ei, node_masks=nm, labels=lab,
                num_time_steps_for_pe=n_ts, fixed_args=base_args, event_name=event_name
            )
            md_content += f"| {event_name:<20} | {test_acc:8.4f} | {test_f1:15.4f} |\n"
            if test_acc is not None: all_test_accs.append(test_acc)
            if test_f1 is not None: all_test_f1s.append(test_f1)
            del X, ei, nm, lab, ti; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Error evaluating {event_name} (Full DySAT) for aggregated report: {e}", exc_info=True)
            md_content += f"| {event_name:<20} | ERROR   | ERROR           |\n"
    md_content += "|---|---|---|\n"
    if all_test_f1s: # Check if list is not empty
        avg_test_acc = np.mean([a for a in all_test_accs if a is not None]) if all_test_accs else 0.0
        avg_test_f1 = np.mean([f for f in all_test_f1s if f is not None]) if all_test_f1s else 0.0
        md_content += f"| **Average**          | **{avg_test_acc:5.4f}** | **{avg_test_f1:12.4f}** |\n"
    else: md_content += "| **Average**          | **N/A**   | **N/A**         |\n"
    with open(report_path, "w") as f: f.write(md_content)
    logging.info(f"Aggregated Full DySAT results report saved to {report_path}")


def run_full_ablation_workflow_for_full_model(args: argparse.Namespace):
    logging.info("===== Starting Full Ablation Workflow (Full DySAT) =====")
    MAIN_MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_event_dirs = get_event_dirs(args.data_dir, 'all')
    event_dirs_for_global = get_event_dirs(args.data_dir, args.optuna_event_subset) if args.optuna_event_subset and args.optuna_event_subset.lower() != 'all' else all_event_dirs
    if not event_dirs_for_global: logging.error("No events for global Optuna (Full DySAT). Aborting."); return
    
    logging.info("--- Stage 1: Finding Overall Best HPs (Full DySAT Global Optuna) ---")
    overall_best_hps = find_overall_best_hps_with_optuna_full_model(args, event_dirs_for_global)
    if overall_best_hps is None:
        logging.error("Failed: Overall Best HPs (Full DySAT). Using CLI defaults for aggregated report.")
        overall_best_hps = {
            "hidden_dim": args.hidden_dim, "lr": args.lr, "dropout": args.dropout, 
            "weight_decay": args.weight_decay, "scheduler_patience": args.scheduler_patience, 
            "early_stop_patience": args.early_stop_patience, 
            "num_struct_heads": args.num_struct_heads, "num_temporal_heads": args.num_temporal_heads,
            "epochs": args.epochs # Add epochs here
        }
    else: # Ensure epochs is part of overall_best_hps if found by Optuna, or add from args
        if 'epochs' not in overall_best_hps:
            overall_best_hps['epochs'] = args.epochs


    events_to_process_individually = get_event_dirs(args.data_dir, args.event) if args.event and args.event.lower() != 'all' else all_event_dirs
    if not events_to_process_individually: logging.error("No events for per-event Optuna (Full DySAT). Aborting stage 2 & 3."); return
    
    logging.info("\n--- Stage 2: Per-Event Optuna and Reporting (Full DySAT) ---")
    for event_dir in tqdm(events_to_process_individually, desc="Per-Event Optuna (Full DySAT)"):
        event_name = event_dir.name; logging.info(f"--- Processing event: {event_name} (Full DySAT) ---")
        event_output_dir = MAIN_MODEL_RESULTS_DIR / event_name; event_output_dir.mkdir(parents=True, exist_ok=True)
        best_hps_event, _ = run_optuna_for_single_event_full_model(event_dir, args, args.optuna_trials_per_event, event_output_dir)
        if best_hps_event:
            logging.info(f"Found best HPs for {event_name} (Full DySAT). Evaluating.")
            hps_for_eval = {**best_hps_event, "epochs": args.epochs} # Use CLI epochs
            evaluate_event_with_fixed_hps_for_full_model(event_dir, hps_for_eval, args, event_output_dir)
        else: logging.warning(f"Optuna failed for {event_name} (Full DySAT). Skipping final eval for its report.")
    
    logging.info("\n--- Stage 3: Generating Aggregated Report (Full DySAT with Overall Best HPs) ---")
    if overall_best_hps: 
        generate_aggregated_report_for_full_model(overall_best_hps, events_to_process_individually, args)
    else: logging.error("Cannot gen aggregated report (Full DySAT) - no overall best HPs.")
    logging.info("===== Full Ablation Workflow (Full DySAT) Complete =====")

# ---------------------------------------------------------------------------
# Comparison Mode Functions - MODIFIED FOR AGGREGATION
# ---------------------------------------------------------------------------
def plot_multi_event_comparison_line_curves(
    histories_full_dysat: Dict[str, Dict[str, List[float]]], # {'event_name': history_dict, ...}
    histories_simple_dysat: Dict[str, Dict[str, List[float]]],# {'event_name': history_dict, ...}
    all_event_names: List[str], # Sorted list of event names that were processed
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "comparison_plot_MULTI_EVENT_LINES_Full_vs_Simple.png"

    num_metric_cols = 3 # Loss, Accuracy (Val/Test), F1 (Val/Test)
    fig, axs = plt.subplots(2, num_metric_cols, figsize=(18, 10)) # 2 rows, 3 metric columns

    # Define distinct colors for events
    try:
        event_colors_cmap = mpl.colormaps.get_cmap('tab10')
        if not all_event_names: event_colors_list = []
        else: event_colors_list = [event_colors_cmap(i % event_colors_cmap.N) for i in range(len(all_event_names))]
    except Exception:
        logging.warning("Colormap 'tab10' issue. Using default color cycle for events.")
        event_colors_list = [None] * (len(all_event_names) if all_event_names else 0)

    metric_setups = [
        {"title": "Loss", "key": "loss", "is_single_line": True},
        {"title": "Accuracy", "val_key": "val_acc", "test_key": "test_acc", "is_single_line": False},
        {"title": "F1-score (Macro)", "val_key": "val_f1", "test_key": "test_f1", "is_single_line": False}
    ]

    for row_idx, (model_name_prefix, model_histories_dict) in enumerate([
        ("Full DySAT", histories_full_dysat),
        ("Simple DySAT", histories_simple_dysat)
    ]):
        for col_idx, metric_info in enumerate(metric_setups):
            ax = axs[row_idx, col_idx]
            ax.set_title(f"{model_name_prefix} - {metric_info['title']}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_info['title'])

            for event_idx, event_name in enumerate(all_event_names):
                event_history = model_histories_dict.get(event_name)
                event_color = event_colors_list[event_idx] if event_colors_list and event_idx < len(event_colors_list) else None

                if event_history:
                    if metric_info["is_single_line"]: # For Loss
                        key = metric_info["key"]
                        if key in event_history and event_history[key]:
                            ax.plot(event_history[key], label=f"{event_name}", color=event_color)
                    else: # For Accuracy, F1 (Val/Test)
                        val_key = metric_info["val_key"]
                        test_key = metric_info["test_key"]
                        if val_key in event_history and event_history[val_key]:
                            ax.plot(event_history[val_key], label=f"{event_name} (Val)", color=event_color, linestyle=':')
                        if test_key in event_history and event_history[test_key]:
                            ax.plot(event_history[test_key], label=f"{event_name} (Test)", color=event_color, linestyle='-')
            
            if not metric_info["is_single_line"]: # For Acc/F1, set ylim
                 ax.set_ylim(0, 1.05)
            
            # Create a concise legend if many events, or full legend if few
            handles, labels = ax.get_legend_handles_labels()
            if handles: # Only add legend if there are lines plotted
                # Filter duplicate labels for event names (e.g. event (Val) and event (Test) should have one entry)
                unique_labels_for_legend = {}
                for handle, label in zip(handles, labels):
                    event_part = label.split(" (")[0] # Get event name part
                    if event_part not in unique_labels_for_legend:
                         # For the legend, just show the event name and its color
                         # The linestyle (dotted/solid) will differentiate Val/Test visually
                         unique_labels_for_legend[event_part] = plt.Line2D([0], [0], color=handle.get_color(), lw=2)
                
                if len(unique_labels_for_legend) > 5: # If many events, make legend smaller
                    ax.legend(unique_labels_for_legend.values(), unique_labels_for_legend.keys(), loc='lower right', fontsize='small', ncol=2)
                else:
                    ax.legend(unique_labels_for_legend.values(), unique_labels_for_legend.keys(), loc='best')


            ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f"Training Curves Comparison: Full DySAT vs Simple DySAT", fontsize=16)
    plt.savefig(plot_path); plt.close()
    logging.info(f"Multi-event line comparison plot saved to {plot_path}")


def run_comparison_and_generate_plots(cli_args: argparse.Namespace):
    if not cli_args.event:
        logging.error("--event <LIST_OR_ALL> required for comparison.")
        return

    full_data_root = Path(cli_args.data_dir)
    simple_data_root = Path(cli_args.simple_model_data_dir)
    all_event_names_full = [d.name for d in get_event_dirs(cli_args.data_dir, "all")]
    all_event_names_simple = [d.name for d in get_event_dirs(cli_args.simple_model_data_dir, "all")]
    
    events_to_process_names = []
    if cli_args.event.lower() == "all":
        events_to_process_names = sorted(list(set(all_event_names_full) & set(all_event_names_simple)))
        if not events_to_process_names:
            logging.error(f"No common events between {full_data_root} and {simple_data_root}. Aborting.")
            return
    else:
        requested = [e.strip() for e in cli_args.event.split(',')]
        for rev in requested:
            if rev in all_event_names_full and rev in all_event_names_simple:
                events_to_process_names.append(rev)
            else:
                logging.warning(f"Event '{rev}' not in both data dirs. Skipping for comparison.")
        if not events_to_process_names:
            logging.error("No valid common events for comparison. Aborting.")
            return

    logging.info(f"===== Starting MULTI-EVENT LINE PLOT Comparison for Events: {', '.join(events_to_process_names)} =====")
    COMPARISON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(cli_args.device if cli_args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # --- Optimal HPs ---
    hps_full_optimal = {
        "hidden_dim": 128, "lr": 0.0015361883741367818, "dropout": 0.1,
        "weight_decay": 2.208070847805232e-06, "scheduler_patience": 18, "early_stop_patience": 20,
        "num_struct_heads": 2, "num_temporal_heads": 2, "epochs": cli_args.epochs
    }
    fixed_args_full = argparse.Namespace(device=cli_args.device, seed=cli_args.seed, 
                                         no_temporal_attn=False, max_nodes_subsample=cli_args.max_nodes_subsample)
    hps_simple_optimal = {
        "hidden_dim": 256, "lr": 0.00038084081088686695, "dropout": 0.15,
        "weight_decay": 2.658075949227783e-05, "scheduler_patience": 15, "early_stop_patience": 24,
        "epochs": cli_args.epochs, "no_temporal": False
    }
    fixed_args_simple = argparse.Namespace(device=cli_args.device, seed=cli_args.seed, 
                                           max_nodes_subsample=cli_args.max_nodes_subsample)

    # --- Dictionaries to store training histories for all events ---
    # Structure: {'event_name': history_dict_from_training_func, ...}
    all_histories_full_dysat = {}
    all_histories_simple_dysat = {}

    for event_name in tqdm(events_to_process_names, desc="Comparing Models Across Events (Line Plots)"):
        logging.info(f"\n--- Processing event: {event_name} for multi-event line plot comparison ---")
        set_seed(cli_args.seed)

        # --- Load Data for Full DySAT ---
        full_event_dir = full_data_root / event_name
        X_f, ei_f, nm_f, lab_f, ti_f = load_temporal_data(full_event_dir, device, False, cli_args.max_nodes_subsample)
        n_ts_f = max(1, ti_f.get("num_windows", len(ei_f) if ei_f else 0)) # Handle ei_f potentially being None or empty

        # --- Load Data for Simple DySAT ---
        simple_event_dir = simple_data_root / event_name
        X_s, ei_s, nm_s, lab_s, _ = load_temporal_data(simple_event_dir, device, False, cli_args.max_nodes_subsample)
        
        # --- Train Full DySAT & Store History ---
        logging.info(f"Training Full DySAT on {event_name} for line plot...")
        _, _, _, _, history_f, _, _, _ = run_training_for_full_model(
            hps_full_optimal, X_f, ei_f, nm_f, lab_f, n_ts_f, fixed_args_full, event_name)
        all_histories_full_dysat[event_name] = history_f
        del X_f, ei_f, nm_f, lab_f, ti_f; gc.collect(); torch.cuda.empty_cache()

        # --- Train Simple DySAT & Store History ---
        logging.info(f"Training Simple DySAT on {event_name} for line plot...")
        _, _, _, _, history_s, _ = run_training_for_simple_model(
            hps_simple_optimal, X_s, ei_s, nm_s, lab_s, fixed_args_simple, event_name)
        all_histories_simple_dysat[event_name] = history_s
        del X_s, ei_s, nm_s, lab_s; gc.collect(); torch.cuda.empty_cache()

    # --- Generate and Save Multi-Event Line Comparison Plot ---
    if all_histories_full_dysat and all_histories_simple_dysat:
        plot_multi_event_comparison_line_curves(
            all_histories_full_dysat,
            all_histories_simple_dysat,
            events_to_process_names,
            COMPARISON_RESULTS_DIR
        )
    else:
        logging.error("Failed to obtain training histories. Cannot generate multi-event line comparison plot.")

    logging.info(f"===== MULTI-EVENT LINE PLOT Comparison Complete =====")

# ---------------------------------------------------------------------------
# Main Execution Logic
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
        level=logging.INFO )
    if TORCH_SCATTER_AVAILABLE: logging.info("torch_scatter found.")
    else: logging.warning("torch_scatter not found. GAT fallback.")
    if OPTUNA_AVAILABLE: logging.info(f"Optuna found (v: {optuna.__version__}). Viz: {OPTUNA_VISUALIZATION_AVAILABLE}")
    else: logging.warning("Optuna not found. HP tuning/ablation limited.")

    cli_args = parse_args()
    set_seed(cli_args.seed)

    if cli_args.compare_with_simple_dysat:
        run_comparison_and_generate_plots(cli_args)
    else: # Standard Full DySAT workflows
        MAIN_MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        # Apply HP sanity checks from CLI args for Full DySAT specific runs
        if cli_args.hidden_dim == 0 : cli_args.hidden_dim = max(cli_args.num_struct_heads, cli_args.num_temporal_heads, 1)
        if cli_args.num_struct_heads == 0: cli_args.num_struct_heads = 1
        if cli_args.num_temporal_heads == 0: cli_args.num_temporal_heads = 1
        if cli_args.hidden_dim % cli_args.num_struct_heads != 0:
            cli_args.hidden_dim = cli_args.num_struct_heads * max(1, math.ceil(cli_args.hidden_dim / cli_args.num_struct_heads))
            logging.warning(f"Adjusted hidden_dim to {cli_args.hidden_dim} for struct_heads.")
        if cli_args.hidden_dim % cli_args.num_temporal_heads != 0:
            cli_args.hidden_dim = cli_args.num_temporal_heads * max(1, math.ceil(cli_args.hidden_dim / cli_args.num_temporal_heads))
            logging.warning(f"Adjusted hidden_dim to {cli_args.hidden_dim} for temporal_heads.")
        if cli_args.early_stop_patience <= cli_args.scheduler_patience + 2:
            cli_args.early_stop_patience = cli_args.scheduler_patience + 3
            logging.warning(f"Adjusted early_stop_patience to {cli_args.early_stop_patience}.")

        if cli_args.run_full_ablation:
            if not OPTUNA_AVAILABLE: logging.error("Optuna required for --run-full-ablation.")
            else: run_full_ablation_workflow_for_full_model(cli_args)
        elif cli_args.optuna_study:
            if not OPTUNA_AVAILABLE: logging.error("Optuna required for --optuna-study.")
            else:
                event_dirs = get_event_dirs(cli_args.data_dir, cli_args.optuna_event_subset)
                if event_dirs: find_overall_best_hps_with_optuna_full_model(cli_args, event_dirs)
                else: logging.error(f"No events for Optuna study (Full DySAT) from: {cli_args.optuna_event_subset}")
        else: # Manual run for Full DySAT
            logging.info("Running manual evaluation for Full DySAT with CLI-specified HPs.")
            event_dirs = get_event_dirs(cli_args.data_dir, cli_args.event)
            if not event_dirs: logging.error(f"No events for manual Full DySAT run from --event '{cli_args.event}'.")
            else:
                metrics_summary = []
                for event_dir_item in event_dirs:
                    event_name_item = event_dir_item.name
                    output_dir_item = MAIN_MODEL_RESULTS_DIR / event_name_item; output_dir_item.mkdir(parents=True, exist_ok=True)
                    logging.info(f"Manual Full DySAT run for event: {event_name_item}")
                    
                    # Create HPs dict from CLI args for this specific run type
                    hps_for_manual_run = {
                        "hidden_dim": cli_args.hidden_dim, "lr": cli_args.lr, "dropout": cli_args.dropout,
                        "weight_decay": cli_args.weight_decay, "scheduler_patience": cli_args.scheduler_patience,
                        "early_stop_patience": cli_args.early_stop_patience, 
                        "num_struct_heads": cli_args.num_struct_heads,
                        "num_temporal_heads": cli_args.num_temporal_heads, 
                        "epochs": cli_args.epochs
                    }
                    # Fixed args remain from cli_args directly
                    _, v_f1, t_acc, t_f1, _ = evaluate_event_with_fixed_hps_for_full_model(
                        event_dir_item, hps_for_manual_run, cli_args, output_dir_item
                    )
                    if t_f1 > 0 or t_acc > 0 : metrics_summary.append({'event': event_name_item, 'test_acc': t_acc, 'test_f1': t_f1})
                if metrics_summary:
                    avg_acc = np.mean([m['test_acc'] for m in metrics_summary if m['test_acc'] is not None])
                    avg_f1 = np.mean([m['test_f1'] for m in metrics_summary if m['test_f1'] is not None])
                    logging.info(f"Manual Full DySAT Summary ({len(metrics_summary)} events): Avg Test Acc: {avg_acc:.4f}, Avg Test F1: {avg_f1:.4f}")