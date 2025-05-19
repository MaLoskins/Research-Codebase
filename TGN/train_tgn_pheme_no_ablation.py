#!/usr/bin/env python3
"""
Train a Temporal Graph Network (TGN) for node classification on preprocessed PHEME data.
Includes Optuna integration for hyperparameter optimization and ablation studies.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import json
import math 
import os # For creating directories
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice

# --- Time Encoding Layer ---
class TimeEncode(torch.nn.Module):
    def __init__(self, dimension: int):
        super(TimeEncode, self).__init__()
        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension) 
        
        with torch.no_grad():
            if self.dimension > 0:
                base_freq = torch.arange(0, dimension, 2, dtype=torch.float32) / float(dimension) # Ensure float division
                inv_freq = 1.0 / (10000 ** base_freq)
                
                if dimension % 2 != 0: 
                    init_weight_val = inv_freq.repeat_interleave(2)[:dimension]
                else: 
                    init_weight_val = inv_freq.repeat_interleave(2)

                self.w.weight.data.copy_(init_weight_val.unsqueeze(1))
                self.w.bias.data.copy_(torch.zeros(dimension, dtype=torch.float32))
            else: 
                self.w.weight.data.copy_(torch.empty((0,1), dtype=torch.float32))
                self.w.bias.data.copy_(torch.empty((0), dtype=torch.float32))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1: t = t.unsqueeze(dim=1) 
        t = t.float() 
        if self.dimension == 0: 
            return torch.empty(t.shape[0], 0, device=t.device)
        return torch.cos(self.w(t))

# --- TGN Model ---
class TGN(nn.Module):
    def __init__(self, num_nodes: int, raw_node_feat_dim: int, edge_feat_dim: int,
                 memory_dim: int, time_dim: int, embedding_dim: int, num_classes: int,
                 dropout_rate: float, projector_dropout_rate: float,
                 device: torch.device,
                 project_input_features: bool = False, leaky_relu_slope: float = 0.1,
                 use_layernorm: bool = False): 
        super(TGN, self).__init__()
        self.num_nodes, self.raw_node_feat_dim_orig, self.edge_feat_dim_orig = num_nodes, raw_node_feat_dim, edge_feat_dim
        self.memory_dim, self.embedding_dim, self.device = memory_dim, embedding_dim, device
        self.project_input_features, self.use_layernorm = project_input_features, use_layernorm
        self.leaky_relu_slope = leaky_relu_slope

        self.current_node_feat_dim_for_mlp = self.raw_node_feat_dim_orig
        self.current_edge_feat_dim_for_mlp = self.edge_feat_dim_orig 

        if self.project_input_features:
            # Ensure projected_static_feat_dim_target is reasonable, e.g., memory_dim or embedding_dim
            self.projected_static_feat_dim_target = memory_dim # Or some other sensible value like 256
            
            self.node_feat_projector_fc = nn.Linear(self.raw_node_feat_dim_orig, self.projected_static_feat_dim_target)
            torch.nn.init.xavier_uniform_(self.node_feat_projector_fc.weight)
            if self.node_feat_projector_fc.bias is not None: torch.nn.init.zeros_(self.node_feat_projector_fc.bias)
            
            # Edge features might be 0-dim if no edges, handle this
            if self.edge_feat_dim_orig > 0:
                self.edge_feat_projector_fc = nn.Linear(self.edge_feat_dim_orig, self.projected_static_feat_dim_target)
                torch.nn.init.xavier_uniform_(self.edge_feat_projector_fc.weight)
                if self.edge_feat_projector_fc.bias is not None: torch.nn.init.zeros_(self.edge_feat_projector_fc.bias)
            else:
                self.edge_feat_projector_fc = None # No projection if no edge features
            
            self.projection_activation = nn.LeakyReLU(leaky_relu_slope) 
            self.projection_dropout = nn.Dropout(projector_dropout_rate)
            
            self.current_node_feat_dim_for_mlp = self.projected_static_feat_dim_target
            if self.edge_feat_projector_fc is not None:
                 self.current_edge_feat_dim_for_mlp = self.projected_static_feat_dim_target
            # else it remains self.edge_feat_dim_orig (which is 0)
        
        self.time_encoder = TimeEncode(time_dim).to(device) 
        
        if self.use_layernorm:
            self.memory_norm = nn.LayerNorm(memory_dim)
            self.static_feat_norm = nn.LayerNorm(self.current_node_feat_dim_for_mlp)

        actual_time_dim_for_concat = self.time_encoder.dimension
        # current_edge_feat_dim_for_mlp could be 0 if edge_feat_dim_orig is 0
        # and projection is on but edge_feat_projector_fc is None
        edge_dim_for_concat = self.current_edge_feat_dim_for_mlp if self.current_edge_feat_dim_for_mlp > 0 else 0

        msg_mlp_input_dim = memory_dim + memory_dim + actual_time_dim_for_concat + edge_dim_for_concat
        
        msg_hidden_dim = memory_dim * 2 
        self.source_message_mlp = nn.Sequential(
            nn.Linear(msg_mlp_input_dim, msg_hidden_dim), nn.LeakyReLU(self.leaky_relu_slope), nn.Dropout(dropout_rate),
            nn.Linear(msg_hidden_dim, memory_dim))
        self.destination_message_mlp = nn.Sequential(
            nn.Linear(msg_mlp_input_dim, msg_hidden_dim), nn.LeakyReLU(self.leaky_relu_slope), nn.Dropout(dropout_rate),
            nn.Linear(msg_hidden_dim, memory_dim))

        self.memory_updater = nn.GRUCell(input_size=memory_dim, hidden_size=memory_dim)
        
        embedding_projector_input_dim = memory_dim + self.current_node_feat_dim_for_mlp
        embed_projector_hidden_dim = (embedding_projector_input_dim + embedding_dim) // 2
        self.embedding_projector = nn.Sequential(
            nn.Linear(embedding_projector_input_dim, embed_projector_hidden_dim), 
            nn.LeakyReLU(self.leaky_relu_slope), 
            nn.Dropout(dropout_rate),
            nn.Linear(embed_projector_hidden_dim, embedding_dim) 
        )
        
        self.classifier = nn.Linear(embedding_dim, num_classes)
        if hasattr(self.classifier, 'bias') and self.classifier.bias is not None:
            torch.nn.init.zeros_(self.classifier.bias.data)
        
        self.to(device)

    def _apply_projection(self, features, projector_fc):
        if projector_fc is None: # Handle case where projector might not exist (e.g. 0-dim edge features)
            return features 
        x = projector_fc(features)
        if hasattr(self, 'projection_activation'): x = self.projection_activation(x)
        if hasattr(self, 'projection_dropout'): x = self.projection_dropout(x)
        return x

    def compute_messages_and_update_memory(self, current_memory: torch.Tensor, current_last_update_timestamps: torch.Tensor,
                                           source_nodes: torch.Tensor, destination_nodes: torch.Tensor,
                                           event_timestamps: torch.Tensor, edge_features: torch.Tensor):
        source_nodes_dev, destination_nodes_dev = source_nodes.to(self.device), destination_nodes.to(self.device)
        event_timestamps_dev = event_timestamps.to(self.device).float() 
        
        processed_edge_features_dev = edge_features.to(self.device) 
        if self.project_input_features and self.edge_feat_projector_fc is not None and processed_edge_features_dev.shape[1] > 0:
            processed_edge_features_dev = self._apply_projection(processed_edge_features_dev, self.edge_feat_projector_fc)

        mem_source_prev, mem_destination_prev = current_memory[source_nodes_dev], current_memory[destination_nodes_dev]
        
        time_diff_source = event_timestamps_dev - current_last_update_timestamps[source_nodes_dev].float()
        time_diff_destination = event_timestamps_dev - current_last_update_timestamps[destination_nodes_dev].float()
        
        time_enc_source = self.time_encoder(time_diff_source)
        time_enc_destination = self.time_encoder(time_diff_destination)
        
        source_inputs = [mem_source_prev, mem_destination_prev]
        if self.time_encoder.dimension > 0: 
            source_inputs.append(time_enc_source)
        if processed_edge_features_dev.shape[1] > 0 : # Only concat if edge features exist
            source_inputs.append(processed_edge_features_dev)
        source_mlp_input = torch.cat(source_inputs, dim=1)

        destination_inputs = [mem_destination_prev, mem_source_prev]
        if self.time_encoder.dimension > 0:
            destination_inputs.append(time_enc_destination)
        if processed_edge_features_dev.shape[1] > 0:
            destination_inputs.append(processed_edge_features_dev)
        destination_mlp_input = torch.cat(destination_inputs, dim=1)

        processed_message_for_source = self.source_message_mlp(source_mlp_input)
        processed_message_for_destination = self.destination_message_mlp(destination_mlp_input)
        
        new_memory_batch = current_memory.clone() 
        
        updated_mem_source = self.memory_updater(processed_message_for_source, mem_source_prev)
        updated_mem_destination = self.memory_updater(processed_message_for_destination, mem_destination_prev)
        
        new_memory_batch[source_nodes_dev] = updated_mem_source
        new_memory_batch[destination_nodes_dev] = updated_mem_destination
        
        new_last_update_timestamps_batch = current_last_update_timestamps.clone()
        new_last_update_timestamps_batch[source_nodes_dev] = event_timestamps_dev
        new_last_update_timestamps_batch[destination_nodes_dev] = event_timestamps_dev
        
        return new_memory_batch, new_last_update_timestamps_batch

    def compute_node_embeddings(self, node_indices: torch.Tensor, current_memory: torch.Tensor, raw_node_features: torch.Tensor):
        node_indices_dev = node_indices.to(self.device)
        memory_to_embed = current_memory[node_indices_dev]
        
        current_raw_node_features_batch = raw_node_features[node_indices_dev].to(self.device) 
        
        processed_raw_node_features_for_concat = current_raw_node_features_batch
        if self.project_input_features:
            processed_raw_node_features_for_concat = self._apply_projection(current_raw_node_features_batch, self.node_feat_projector_fc)
        
        if self.use_layernorm:
            memory_to_embed = self.memory_norm(memory_to_embed)
            processed_raw_node_features_for_concat = self.static_feat_norm(processed_raw_node_features_for_concat)
            
        combined_features = torch.cat([memory_to_embed, processed_raw_node_features_for_concat], dim=1)
        return self.embedding_projector(combined_features)

    def predict_logits(self, node_embeddings: torch.Tensor): return self.classifier(node_embeddings)

# --- Data Loading and Utility Functions ---
def load_tgn_data_from_path(data_dir_path: Path): # Device argument removed, data loaded to CPU
    print(f"Loading TGN data from: {data_dir_path}")
    node_features = torch.from_numpy(np.load(data_dir_path / "node_features.npy")).float() 
    labels_all_nodes = torch.from_numpy(np.load(data_dir_path / "labels.npy")).long() 
    
    events_csv_path = data_dir_path / "events.csv"
    edge_features_npy_path = data_dir_path / "edge_features.npy"

    if not events_csv_path.exists() or not edge_features_npy_path.exists():
        print(f"Warning: events.csv or edge_features.npy not found in {data_dir_path}. Assuming no events.")
        source_nodes_all_events = torch.empty(0, dtype=torch.long)
        destination_nodes_all_events = torch.empty(0, dtype=torch.long)
        event_timestamps_all_events = torch.empty(0, dtype=torch.float)
        edge_features_all_events = torch.empty(0, node_features.shape[1] if node_features.ndim > 1 else 0, dtype=torch.float) # Match feature dim
    else:
        events_df = pd.read_csv(events_csv_path)
        if events_df.empty:
            print(f"Warning: events.csv in {data_dir_path} is empty. Assuming no events.")
            source_nodes_all_events = torch.empty(0, dtype=torch.long)
            destination_nodes_all_events = torch.empty(0, dtype=torch.long)
            event_timestamps_all_events = torch.empty(0, dtype=torch.float)
            edge_features_all_events = torch.empty(0, node_features.shape[1] if node_features.ndim > 1 else 0, dtype=torch.float)
        else:
            source_nodes_all_events = torch.from_numpy(events_df["u"].to_numpy()).long()
            destination_nodes_all_events = torch.from_numpy(events_df["i"].to_numpy()).long()
            event_timestamps_all_events = torch.from_numpy(events_df["timestamp"].to_numpy()).float()
            edge_features_all_events = torch.from_numpy(np.load(edge_features_npy_path)).float() 

    with open(data_dir_path / "metadata.json", 'r') as f: metadata = json.load(f)
    num_total_nodes = metadata['num_nodes']

    # Assertions
    if num_total_nodes > 0:
        assert num_total_nodes == node_features.shape[0], f"{data_dir_path}: num_nodes ({num_total_nodes}) != node_features.shape[0] ({node_features.shape[0]})"
        assert num_total_nodes == labels_all_nodes.shape[0], f"{data_dir_path}: num_nodes ({num_total_nodes}) != labels_all_nodes.shape[0] ({labels_all_nodes.shape[0]})"
    
    if len(source_nodes_all_events) > 0 : # Only assert if there are events
        assert len(source_nodes_all_events) == edge_features_all_events.shape[0]

        max_u = source_nodes_all_events.max().item() if len(source_nodes_all_events) > 0 else -1
        max_i = destination_nodes_all_events.max().item() if len(destination_nodes_all_events) > 0 else -1
        max_event_node_idx = max(max_u, max_i)
        if max_event_node_idx != -1 and num_total_nodes > 0:
             assert max_event_node_idx < num_total_nodes, f"{data_dir_path}: Max event node index {max_event_node_idx} out of bounds for {num_total_nodes} nodes."
    
    return (node_features, labels_all_nodes, source_nodes_all_events, destination_nodes_all_events,
            event_timestamps_all_events, edge_features_all_events, num_total_nodes)

def create_event_splits(num_total_events: int, train_ratio: float = 0.7, val_ratio: float = 0.15):
    if num_total_events == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    train_idx_end = int(num_total_events * train_ratio)
    val_idx_end = int(num_total_events * (train_ratio + val_ratio))
    return np.arange(0, train_idx_end), np.arange(train_idx_end, val_idx_end), np.arange(val_idx_end, num_total_events)

def get_active_nodes_in_split(s_nodes_cpu, d_nodes_cpu, indices, node_labels_cpu):
    if len(indices) == 0 or len(s_nodes_cpu) == 0 or len(d_nodes_cpu) == 0: # Added check for empty event arrays
        return torch.tensor([], dtype=torch.long)
    
    # Ensure indices are valid
    max_index_val = indices.max()
    if max_index_val >= len(s_nodes_cpu) or max_index_val >= len(d_nodes_cpu):
        print(f"Warning: Max index in 'indices' ({max_index_val}) is out of bounds for event arrays (s_nodes: {len(s_nodes_cpu)}, d_nodes: {len(d_nodes_cpu)}). Clamping indices.")
        valid_indices_mask = (indices < len(s_nodes_cpu)) & (indices < len(d_nodes_cpu))
        indices = indices[valid_indices_mask]
        if len(indices) == 0: return torch.tensor([], dtype=torch.long)

    active_nodes = torch.cat([s_nodes_cpu[indices], d_nodes_cpu[indices]]).unique()
    
    if len(active_nodes) > 0:
        # Ensure active_nodes are valid indices for node_labels_cpu
        max_active_node_idx = active_nodes.max()
        if max_active_node_idx >= len(node_labels_cpu):
            print(f"Warning: Max index in 'active_nodes' ({max_active_node_idx}) is out of bounds for node_labels_cpu ({len(node_labels_cpu)}). Filtering active nodes.")
            active_nodes = active_nodes[active_nodes < len(node_labels_cpu)]
            if len(active_nodes) == 0: return torch.tensor([], dtype=torch.long)
        
        valid_label_mask = node_labels_cpu[active_nodes] != -1
        return active_nodes[valid_label_mask]
    return torch.tensor([], dtype=torch.long)


# --- Training and Evaluation Logic (Refactored for Optuna) ---
def run_training_epoch(model, raw_node_feats_cpu, optimizer, criterion, s_nodes_cpu, d_nodes_cpu, event_ts_cpu, edge_feats_cpu, 
                       current_epoch_memory_start, current_epoch_last_updates_start, 
                       train_indices, node_labels_cpu, batch_size, clip_norm, device, trial=None, epoch_num=0): 
    model.train(); total_loss = 0.0; num_loss_calcs = 0 
    
    mem_state_for_current_batch = current_epoch_memory_start 
    last_updates_state_for_current_batch = current_epoch_last_updates_start

    raw_node_feats_dev = raw_node_feats_cpu.to(device)
    node_labels_dev = node_labels_cpu.to(device)

    final_mem_of_epoch = current_epoch_memory_start
    final_ts_of_epoch = current_epoch_last_updates_start
    
    # Disable tqdm if trial is not None (Optuna run) to reduce log spam, unless it's the first few epochs
    use_tqdm = trial is None or epoch_num < 2 

    batch_iterator = range(0, len(train_indices), batch_size)
    if use_tqdm:
        batch_iterator = tqdm(batch_iterator, desc="  Training Batches", leave=False)

    for i in batch_iterator:
        optimizer.zero_grad(); batch_idx = train_indices[i : i + batch_size]
        if len(batch_idx) == 0: continue
        
        batch_s, batch_d = s_nodes_cpu[batch_idx], d_nodes_cpu[batch_idx] 
        batch_ts, batch_ef = event_ts_cpu[batch_idx], edge_feats_cpu[batch_idx]

        mem_out_current_batch, last_updates_out_current_batch = model.compute_messages_and_update_memory(
            mem_state_for_current_batch, last_updates_state_for_current_batch, 
            batch_s, batch_d, batch_ts, batch_ef)

        nodes_in_batch_cpu = torch.cat([batch_s, batch_d]).unique()
        
        if len(nodes_in_batch_cpu) > 0 and nodes_in_batch_cpu.max() >= len(node_labels_cpu):
             nodes_in_batch_cpu = nodes_in_batch_cpu[nodes_in_batch_cpu < len(node_labels_cpu)]

        if len(nodes_in_batch_cpu) == 0: 
            mem_state_for_current_batch = mem_out_current_batch.detach()
            last_updates_state_for_current_batch = last_updates_out_current_batch.detach()
            final_mem_of_epoch = mem_out_current_batch 
            final_ts_of_epoch = last_updates_out_current_batch
            continue

        valid_label_mask_batch = node_labels_cpu[nodes_in_batch_cpu] != -1
        nodes_for_loss_cpu = nodes_in_batch_cpu[valid_label_mask_batch]

        if len(nodes_for_loss_cpu) > 0:
            nodes_for_loss_dev = nodes_for_loss_cpu.to(device)
            node_embeds = model.compute_node_embeddings(nodes_for_loss_dev, mem_out_current_batch, raw_node_feats_dev)
            logits = model.predict_logits(node_embeds)
            
            loss = criterion(logits, node_labels_dev[nodes_for_loss_dev])
            if torch.isnan(loss): # Check for NaN loss
                print("Warning: NaN loss detected in training. Skipping batch.")
                # Reset states to before this problematic batch to avoid propagating NaNs
                mem_state_for_current_batch = mem_state_for_current_batch.detach() # from previous iteration
                last_updates_state_for_current_batch = last_updates_state_for_current_batch.detach()
                # final_mem_of_epoch & final_ts_of_epoch remain as they were
                continue

            loss.backward() 
            if clip_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step(); total_loss += loss.item(); num_loss_calcs += 1
        
        mem_state_for_current_batch = mem_out_current_batch.detach()
        last_updates_state_for_current_batch = last_updates_out_current_batch.detach()
        
        final_mem_of_epoch = mem_out_current_batch 
        final_ts_of_epoch = last_updates_out_current_batch
            
    return total_loss / num_loss_calcs if num_loss_calcs > 0 else 0.0, final_mem_of_epoch, final_ts_of_epoch

def run_evaluation(model, raw_node_feats_cpu, s_nodes_cpu, d_nodes_cpu, event_ts_cpu, edge_feats_cpu,
                   mem_start_eval, last_updates_start_eval, 
                   eval_indices, node_labels_cpu, batch_size, active_eval_nodes_cpu, device, 
                   split_name="Eval", trial=None, epoch_num=0): 
    model.eval(); preds_list, true_list = [], []
    
    mem_eval = mem_start_eval.detach().clone() 
    last_updates_eval = last_updates_start_eval.detach().clone()
    
    raw_node_feats_dev = raw_node_feats_cpu.to(device)
    
    use_tqdm = trial is None or epoch_num < 2

    batch_iterator = range(0, len(eval_indices), batch_size)
    if use_tqdm:
        batch_iterator = tqdm(batch_iterator, desc=f"  {split_name} Event Proc", leave=False)


    with torch.no_grad():
        for i in batch_iterator:
            batch_idx = eval_indices[i : i + batch_size]
            if len(batch_idx) == 0: continue
            
            batch_s, batch_d = s_nodes_cpu[batch_idx], d_nodes_cpu[batch_idx]
            batch_ts, batch_ef = event_ts_cpu[batch_idx], edge_feats_cpu[batch_idx]
            
            mem_out, last_updates_out = model.compute_messages_and_update_memory(
                mem_eval, last_updates_eval, batch_s, batch_d, batch_ts, batch_ef)
            mem_eval, last_updates_eval = mem_out, last_updates_out

        if len(active_eval_nodes_cpu) > 0: 
            active_eval_nodes_dev = active_eval_nodes_cpu.to(device)
            node_embeds = model.compute_node_embeddings(active_eval_nodes_dev, mem_eval, raw_node_feats_dev)
            logits = model.predict_logits(node_embeds)
            preds_list.append(torch.argmax(logits, dim=1).cpu().numpy())
            true_list.append(node_labels_cpu[active_eval_nodes_cpu].cpu().numpy()) 
            
    if not preds_list: 
        num_classes = len(torch.unique(node_labels_cpu[node_labels_cpu != -1]))
        if num_classes == 0: num_classes = 2 
        return {"accuracy":0.0,"f1":0.0,"precision":0.0,"recall":0.0,"conf_matrix":np.zeros((num_classes,num_classes)).tolist()}, mem_eval, last_updates_eval

    preds, trues = np.concatenate(preds_list), np.concatenate(true_list)
    
    unique_true_labels = np.unique(trues)
    num_classes_for_cm = 2 
    if len(unique_true_labels) > 0 : 
        max_label_val = unique_true_labels.max()
        if max_label_val >=0 : 
            num_classes_for_cm = int(max_label_val) + 1
    
    avg_metric = 'binary' if num_classes_for_cm <= 2 else 'weighted'
    
    metrics = {
        "accuracy": accuracy_score(trues, preds), 
        "f1": f1_score(trues, preds, average=avg_metric, zero_division=0),
        "precision": precision_score(trues, preds, average=avg_metric, zero_division=0),
        "recall": recall_score(trues, preds, average=avg_metric, zero_division=0),
        "conf_matrix": confusion_matrix(trues, preds, labels=np.arange(num_classes_for_cm)).tolist()
    }
    return metrics, mem_eval, last_updates_eval


def run_single_trial(args_dict, trial=None):
    """
    Runs a single training and evaluation trial.
    Args:
        args_dict (dict): Dictionary of arguments.
        trial (optuna.Trial, optional): Optuna trial object. If None, uses fixed args from args_dict.
    Returns:
        float: The metric to optimize (best validation F1 score).
    """
    # --- Apply Optuna suggested HPs if trial is provided ---
    if trial:
        args_dict["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        args_dict["memory_dim"] = trial.suggest_categorical("memory_dim", [128, 256]) # Reduced options for speed
        args_dict["time_dim"] = trial.suggest_categorical("time_dim", [64, 128])    # Reduced options for speed
        args_dict["embedding_dim"] = trial.suggest_categorical("embedding_dim", [128, 256]) # Reduced options for speed
        args_dict["dropout_rate"] = trial.suggest_float("dropout_rate", 0.1, 0.4)
        args_dict["projector_dropout_rate"] = trial.suggest_float("projector_dropout_rate", 0.05, 0.3)
        args_dict["grad_clip_norm"] = trial.suggest_float("grad_clip_norm", 1.0, 5.0)
        # Optional: tune boolean flags like project_features, use_layernorm
        # args_dict["project_features"] = trial.suggest_categorical("project_features", [True, False])
        # args_dict["use_layernorm"] = trial.suggest_categorical("use_layernorm", [True, False])
        
        # Optuna might suggest a large batch size that causes OOM for smaller datasets
        # Cap batch_size based on dataset size or fixed upper limit
        args_dict["batch_size"] = trial.suggest_categorical("batch_size", [128, 256, 512])


    # --- Setup ---
    torch.manual_seed(args_dict["seed"]); np.random.seed(args_dict["seed"])
    device = torch.device(args_dict["device"])
    if args_dict["device"] == "cuda" and torch.cuda.is_available(): torch.cuda.manual_seed_all(args_dict["seed"])
    
    if trial is None: # Only print for fixed runs, not every Optuna trial
        print(f"Using device: {device}")
        print(f"Running with fixed args: {args_dict}")


    (raw_node_feats_cpu, node_labels_cpu, s_nodes_cpu, d_nodes_cpu, ts_cpu, edge_feats_cpu, n_nodes) = \
        load_tgn_data_from_path(Path(args_dict["data_path"]))

    if n_nodes == 0: 
        print(f"No nodes found in {args_dict['data_path']}. Skipping trial.")
        return 0.0 # Return a poor score for Optuna if data is unusable

    # Handle cases with very few events, which might lead to empty splits or active_nodes lists
    if len(s_nodes_cpu) < 10 : # Arbitrary small number, adjust as needed
        print(f"Warning: Very few events ({len(s_nodes_cpu)}) in {args_dict['data_path']}. Results might be unstable or splits empty.")
        # Potentially return a poor score or handle differently
        # For now, let it run but be aware. If train_idx is empty, it will be handled.
    
    valid_labels_cpu = node_labels_cpu[node_labels_cpu != -1]
    cls_weights = None
    num_classes_data = 2 
    if len(valid_labels_cpu) > 0:
        cls_counts = torch.bincount(valid_labels_cpu)
        if cls_counts.numel() > 0 : 
            num_classes_data = len(cls_counts)
            if trial is None:
                for lbl, cnt in enumerate(cls_counts): 
                    print(f"  L{lbl}: {cnt.item()} ({cnt.item()/len(valid_labels_cpu)*100:.2f}%)")
            
            if num_classes_data > 0 and cls_counts.min() > 0: 
                cls_weights = (1. / cls_counts.float()).to(device)
                if trial is None: print(f"Class weights (unnormalized, for device): {cls_weights.cpu().numpy()}")
            else:
                if trial is None: print("Warning: Not all classes represented or a class has 0 samples in valid labels. No class weights applied.")
                num_classes_data = max(2, num_classes_data) 
        else: 
            if trial is None: print("Warning: cls_counts is empty. No class weights applied.")
            num_classes_data = 2 
    else: 
        if trial is None: print("Warning: No valid labels (all are -1 or empty). Cannot compute class weights or num_classes.")
        # If no valid labels, classification is impossible, return poor score for Optuna
        if trial: return 0.0 


    criterion = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=-1) 
    
    train_idx, val_idx, test_idx = create_event_splits(len(s_nodes_cpu), 
                                                       train_ratio=args_dict.get("train_ratio", 0.7), 
                                                       val_ratio=args_dict.get("val_ratio", 0.15))
    
    if trial is None:
        print(f"Dataset: {args_dict['data_path']}\nNodes: {n_nodes}, Events: {len(s_nodes_cpu)}")
        print(f"Train/Val/Test Events: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    
    active_val_nodes_cpu = get_active_nodes_in_split(s_nodes_cpu, d_nodes_cpu, val_idx, node_labels_cpu)
    active_test_nodes_cpu = get_active_nodes_in_split(s_nodes_cpu, d_nodes_cpu, test_idx, node_labels_cpu)

    if trial is None:
        print(f"Active Val Nodes (for eval): {len(active_val_nodes_cpu)}, Active Test Nodes (for eval): {len(active_test_nodes_cpu)}")

    # Adjust batch size if it's larger than the number of training events (can happen with Optuna)
    current_batch_size = args_dict["batch_size"]
    if len(train_idx) > 0 and args_dict["batch_size"] > len(train_idx):
        current_batch_size = max(1, len(train_idx)) # Ensure batch size is at least 1
        if trial:
            print(f"Optuna Trial {trial.number}: Adjusted batch size from {args_dict['batch_size']} to {current_batch_size} due to small train set ({len(train_idx)} events).")


    model = TGN(n_nodes, raw_node_feats_cpu.shape[1], edge_feats_cpu.shape[1] if len(edge_feats_cpu)>0 else 0, 
                args_dict["memory_dim"], args_dict["time_dim"], args_dict["embedding_dim"], 
                num_classes_data, 
                args_dict["dropout_rate"], args_dict["projector_dropout_rate"],
                device, 
                args_dict["project_features"], 
                args_dict["leaky_relu_slope"],
                args_dict["use_layernorm"]
               )
    if trial is None:
        print(f"Model: project={args_dict['project_features']}, layernorm={args_dict['use_layernorm']}, main_dropout={args_dict['dropout_rate']}")

    optimizer = optim.AdamW(model.parameters(), lr=args_dict["lr"], weight_decay=args_dict["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=args_dict["lr_scheduler_patience"])
    
    best_val_f1_trial = -1.0
    no_improve_epochs_trial = 0
    
    # Initialize memory for the entire run (multiple epochs)
    global_memory_state = torch.zeros((n_nodes, args_dict["memory_dim"]), device=device)
    global_last_update_timestamps = torch.zeros(n_nodes, device=device)

    for epoch in range(1, args_dict["epochs"] + 1):
        if trial is None or epoch <= 2: # Print for fixed runs or first few optuna epochs
            print(f"\nEpoch {epoch}/{args_dict['epochs']} (Current LR: {optimizer.param_groups[0]['lr']:.1e})")
        
        epoch_memory_train_start = global_memory_state.clone() 
        epoch_last_updates_train_start = global_last_update_timestamps.clone()
        
        mem_for_eval_and_next_epoch = epoch_memory_train_start
        ts_for_eval_and_next_epoch = epoch_last_updates_train_start

        if len(train_idx) > 0:
            avg_loss, mem_after_train_epoch_raw, ts_after_train_epoch_raw = run_training_epoch(
                model, raw_node_feats_cpu, optimizer, criterion, 
                s_nodes_cpu, d_nodes_cpu, ts_cpu, edge_feats_cpu, 
                epoch_memory_train_start, epoch_last_updates_train_start, 
                train_idx, node_labels_cpu, current_batch_size, args_dict["grad_clip_norm"], device, trial, epoch)
            if trial is None or epoch <=2 : print(f"  Avg Train Loss: {avg_loss:.4f}")
            mem_for_eval_and_next_epoch = mem_after_train_epoch_raw
            ts_for_eval_and_next_epoch = ts_after_train_epoch_raw
        elif trial is None or epoch <=2 : 
            print("  No training events for this epoch.")
        
        current_trial_val_f1 = -1.0
        if len(val_idx) > 0 and len(active_val_nodes_cpu) > 0:
            val_metrics, _, _ = run_evaluation( 
                model, raw_node_feats_cpu, s_nodes_cpu, d_nodes_cpu, ts_cpu, edge_feats_cpu, 
                mem_for_eval_and_next_epoch, ts_for_eval_and_next_epoch, 
                val_idx, node_labels_cpu, args_dict["batch_size"], active_val_nodes_cpu, device, "Val", trial, epoch)
            if trial is None or epoch <=2 :
                print(f"  Val - Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}")
            current_trial_val_f1 = val_metrics['f1']
            scheduler.step(current_trial_val_f1)
        elif trial is None or epoch <=2 :
            print("  Skipping validation (no validation events or no active validation nodes).")

        if current_trial_val_f1 > best_val_f1_trial:
            best_val_f1_trial = current_trial_val_f1
            no_improve_epochs_trial = 0
            # In Optuna, we don't need to run test set evaluation during optimization usually,
            # but for a fixed run, we do.
            if trial is None and (len(test_idx) > 0 and len(active_test_nodes_cpu) > 0):
                print(f"  New best Val F1: {best_val_f1_trial:.4f}. Evaluating on Test...")
                test_metrics, _, _ = run_evaluation(
                    model, raw_node_feats_cpu, s_nodes_cpu, d_nodes_cpu, ts_cpu, edge_feats_cpu, 
                    mem_for_eval_and_next_epoch, ts_for_eval_and_next_epoch, 
                    test_idx, node_labels_cpu, args_dict["batch_size"], active_test_nodes_cpu, device, "Test", trial, epoch)
                print(f"  Test - Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, P: {test_metrics['precision']:.4f}, R: {test_metrics['recall']:.4f}")
        else:
            no_improve_epochs_trial += 1
        
        global_memory_state = mem_for_eval_and_next_epoch.detach()
        global_last_update_timestamps = ts_for_eval_and_next_epoch.detach()

        if no_improve_epochs_trial >= args_dict["early_stopping_patience"] and best_val_f1_trial != -1.0 :
            if trial is None or epoch <=2 : print(f"Early stopping after {no_improve_epochs_trial} epochs with no improvement."); 
            break
        
        # Optuna Pruning
        if trial:
            trial.report(best_val_f1_trial, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    if trial is None:
        print("\n--- Training Complete (Fixed Run) ---")
        print(f"Best Val F1: {best_val_f1_trial:.4f}")
        # Final test evaluation for fixed run using the best validation epoch's logic would require saving best model state.
        # For simplicity, current fixed run evaluates test when val F1 improves.

    return best_val_f1_trial # Optuna maximizes this


def objective(trial, base_args_dict):
    """Optuna objective function."""
    # Create a mutable copy of base_args_dict for this trial
    current_trial_args = base_args_dict.copy()
    
    # Set the data_path for this specific trial if it's part of the study's goal
    # This would typically be passed into base_args_dict by the calling loop
    
    # `run_single_trial` will further modify current_trial_args with trial.suggest_
    val_f1 = run_single_trial(current_trial_args, trial=trial)
    return val_f1


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TGN for Node Classification on PHEME data with Optuna.")
    # --- Basic TGN parameters (can be overridden by Optuna) ---
    parser.add_argument("--base_data_dir", type=str, default="data_tgn_fixed", help="Base directory containing processed event folders (e.g., all, charliehebdo).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs per trial/run.") # Reduced for Optuna speed
    parser.add_argument("--lr", type=float, default=1e-3) 
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=256) 
    parser.add_argument("--memory_dim", type=int, default=256) 
    parser.add_argument("--time_dim", type=int, default=128)   
    parser.add_argument("--embedding_dim", type=int, default=256) 
    parser.add_argument("--dropout_rate", type=float, default=0.15) 
    parser.add_argument("--projector_dropout_rate", type=float, default=0.2) 
    parser.add_argument("--grad_clip_norm", type=float, default=2.0) 
    parser.add_argument("--lr_scheduler_patience", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=10) # Reduced for Optuna
    parser.add_argument("--project_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_layernorm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--leaky_relu_slope", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)


    # --- Optuna specific parameters ---
    parser.add_argument("--run_optuna", action="store_true", help="Enable Optuna hyperparameter search.")
    parser.add_argument("--study_name_prefix", type=str, default="tgn_pheme_study", help="Prefix for Optuna study names.")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials.") # Small default for quick test
    parser.add_argument("--plot_dir", type=str, default="optuna_plots", help="Directory to save Optuna plots.")
    parser.add_argument("--ablation_target", type=str, default="all", 
                        help="Dataset to run on: 'all', a specific event name (e.g., 'charliehebdo'), or 'all_individual_events' to loop through all found event folders.")

    args = parser.parse_args()
    args_dict = vars(args) # Convert Namespace to dict for easier modification

    if args.run_optuna:
        Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
        
        target_event_folders = []
        base_path = Path(args.base_data_dir)

        if args.ablation_target == "all_individual_events":
            target_event_folders = [d.name for d in base_path.iterdir() if d.is_dir()]
            print(f"Found individual event folders for ablation: {target_event_folders}")
        elif args.ablation_target == "all":
            target_event_folders = ["all"]
        else: # Specific event name
            if (base_path / args.ablation_target).is_dir():
                target_event_folders = [args.ablation_target]
            else:
                print(f"Error: Specified ablation target '{args.ablation_target}' not found in {args.base_data_dir}. Exiting.")
                exit(1)

        for event_folder_name in target_event_folders:
            print(f"\n--- Starting Optuna Study for Event: {event_folder_name} ---")
            current_data_path = str(base_path / event_folder_name)
            study_args_dict = args_dict.copy()
            study_args_dict["data_path"] = current_data_path # Set data_path for this study

            study_name = f"{args.study_name_prefix}_{event_folder_name}"
            # Use a persistent SQLite database for studies to allow continuation
            storage_name = f"sqlite:///{study_name}.db"
            study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize", load_if_exists=True)
            
            study.optimize(lambda trial: objective(trial, study_args_dict), n_trials=args.n_trials, gc_after_trial=True)

            print(f"\n--- Optuna Study for {event_folder_name} Complete ---")
            print(f"Best trial for {event_folder_name}:")
            best_trial = study.best_trial
            print(f"  Value (Val F1): {best_trial.value}")
            print("  Params: ")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")

            # Save plots
            event_plot_dir = Path(args.plot_dir) / f"{event_folder_name}_study"
            event_plot_dir.mkdir(parents=True, exist_ok=True)

            try:
                fig_history = plot_optimization_history(study)
                fig_history.write_image(event_plot_dir / "optimization_history.png")
                
                fig_importance = plot_param_importances(study)
                fig_importance.write_image(event_plot_dir / "param_importances.png")
                
                # Slice plot for each parameter
                for param_name in best_trial.params.keys():
                    fig_slice = plot_slice(study, params=[param_name])
                    fig_slice.write_image(event_plot_dir / f"slice_{param_name}.png")
                print(f"Optuna plots saved to {event_plot_dir}")
            except Exception as e:
                print(f"Error generating or saving Optuna plots for {event_folder_name}: {e}")
                print("Ensure you have plotly and kaleido installed: pip install plotly kaleido")

    else: # Run a single fixed trial
        print("Running a single fixed trial (Optuna disabled).")
        # Ensure data_path is set correctly if ablation_target was used for a single run
        if args.ablation_target and args.ablation_target != "all_individual_events":
             args_dict["data_path"] = str(Path(args.base_data_dir) / args.ablation_target)
        elif "data_path" not in args_dict : # If not specified by ablation_target, try to use a default like 'all'
            args_dict["data_path"] = str(Path(args.base_data_dir) / "all")
            print(f"Defaulting data_path to {args_dict['data_path']} for fixed run.")


        if not Path(args_dict["data_path"]).exists():
            print(f"Error: data_path '{args_dict['data_path']}' does not exist for fixed run. Exiting.")
            exit(1)
            
        best_val_f1 = run_single_trial(args_dict, trial=None)
        print(f"Fixed run completed. Best Validation F1: {best_val_f1:.4f}")