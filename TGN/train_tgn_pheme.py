#!/usr/bin/env python3
"""
Train a Temporal Graph Network (TGN) for node classification on preprocessed PHEME data.
Includes Optuna integration for hyperparameter optimization and ablation studies,
with comprehensive results saving.
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
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- Time Encoding Layer ---
class TimeEncode(torch.nn.Module):
    def __init__(self, dimension: int):
        super(TimeEncode, self).__init__()
        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        with torch.no_grad():
            if self.dimension > 0:
                base_freq = torch.arange(0, dimension, 2, dtype=torch.float32) / float(dimension)
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
            self.projected_static_feat_dim_target = memory_dim
            self.node_feat_projector_fc = nn.Linear(self.raw_node_feat_dim_orig, self.projected_static_feat_dim_target)
            torch.nn.init.xavier_uniform_(self.node_feat_projector_fc.weight)
            if self.node_feat_projector_fc.bias is not None: torch.nn.init.zeros_(self.node_feat_projector_fc.bias)

            if self.edge_feat_dim_orig > 0:
                self.edge_feat_projector_fc = nn.Linear(self.edge_feat_dim_orig, self.projected_static_feat_dim_target)
                torch.nn.init.xavier_uniform_(self.edge_feat_projector_fc.weight)
                if self.edge_feat_projector_fc.bias is not None: torch.nn.init.zeros_(self.edge_feat_projector_fc.bias)
            else:
                self.edge_feat_projector_fc = None

            self.projection_activation = nn.LeakyReLU(leaky_relu_slope)
            self.projection_dropout = nn.Dropout(projector_dropout_rate)

            self.current_node_feat_dim_for_mlp = self.projected_static_feat_dim_target
            if self.edge_feat_projector_fc is not None:
                 self.current_edge_feat_dim_for_mlp = self.projected_static_feat_dim_target

        self.time_encoder = TimeEncode(time_dim).to(device)

        if self.use_layernorm:
            self.memory_norm = nn.LayerNorm(memory_dim)
            self.static_feat_norm = nn.LayerNorm(self.current_node_feat_dim_for_mlp)

        actual_time_dim_for_concat = self.time_encoder.dimension
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
        if projector_fc is None: return features
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
        if self.time_encoder.dimension > 0: source_inputs.append(time_enc_source)
        if processed_edge_features_dev.shape[1] > 0 : source_inputs.append(processed_edge_features_dev)
        source_mlp_input = torch.cat(source_inputs, dim=1)

        destination_inputs = [mem_destination_prev, mem_source_prev]
        if self.time_encoder.dimension > 0: destination_inputs.append(time_enc_destination)
        if processed_edge_features_dev.shape[1] > 0: destination_inputs.append(processed_edge_features_dev)
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
def load_tgn_data_from_path(data_dir_path: Path):
    # print(f"Loading TGN data from: {data_dir_path}") # Less verbose
    node_features = torch.from_numpy(np.load(data_dir_path / "node_features.npy")).float()
    labels_all_nodes = torch.from_numpy(np.load(data_dir_path / "labels.npy")).long()

    events_csv_path = data_dir_path / "events.csv"
    edge_features_npy_path = data_dir_path / "edge_features.npy"

    if not events_csv_path.exists() or not edge_features_npy_path.exists():
        # print(f"Warning: events.csv or edge_features.npy not found in {data_dir_path}. Assuming no events.")
        source_nodes_all_events = torch.empty(0, dtype=torch.long)
        destination_nodes_all_events = torch.empty(0, dtype=torch.long)
        event_timestamps_all_events = torch.empty(0, dtype=torch.float)
        edge_features_all_events = torch.empty(0, node_features.shape[1] if node_features.ndim > 1 and node_features.shape[1] > 0 else 0, dtype=torch.float)
    else:
        events_df = pd.read_csv(events_csv_path)
        if events_df.empty:
            # print(f"Warning: events.csv in {data_dir_path} is empty. Assuming no events.")
            source_nodes_all_events = torch.empty(0, dtype=torch.long)
            destination_nodes_all_events = torch.empty(0, dtype=torch.long)
            event_timestamps_all_events = torch.empty(0, dtype=torch.float)
            edge_features_all_events = torch.empty(0, node_features.shape[1] if node_features.ndim > 1 and node_features.shape[1] > 0 else 0, dtype=torch.float)
        else:
            source_nodes_all_events = torch.from_numpy(events_df["u"].to_numpy()).long()
            destination_nodes_all_events = torch.from_numpy(events_df["i"].to_numpy()).long()
            event_timestamps_all_events = torch.from_numpy(events_df["timestamp"].to_numpy()).float()
            edge_feat_data = np.load(edge_features_npy_path)
            if edge_feat_data.ndim == 1 and len(edge_feat_data) == 0 :
                 edge_features_all_events = torch.empty(0,0, dtype=torch.float)
            elif edge_feat_data.ndim == 1 and len(source_nodes_all_events) > 0 :
                 edge_features_all_events = torch.from_numpy(edge_feat_data).float().unsqueeze(1) if len(edge_feat_data) > 0 else torch.empty(len(source_nodes_all_events), 0, dtype=torch.float)
            else:
                 edge_features_all_events = torch.from_numpy(edge_feat_data).float()


    with open(data_dir_path / "metadata.json", 'r') as f: metadata = json.load(f)
    num_total_nodes = metadata['num_nodes']

    if num_total_nodes > 0:
        assert num_total_nodes == node_features.shape[0], f"{data_dir_path}: num_nodes ({num_total_nodes}) != node_features.shape[0] ({node_features.shape[0]})"
        assert num_total_nodes == labels_all_nodes.shape[0], f"{data_dir_path}: num_nodes ({num_total_nodes}) != labels_all_nodes.shape[0] ({labels_all_nodes.shape[0]})"

    if len(source_nodes_all_events) > 0 :
        assert len(source_nodes_all_events) == edge_features_all_events.shape[0], f"{data_dir_path}: num_events mismatch {len(source_nodes_all_events)} vs {edge_features_all_events.shape[0]}"
        max_u = source_nodes_all_events.max().item() if len(source_nodes_all_events) > 0 else -1
        max_i = destination_nodes_all_events.max().item() if len(destination_nodes_all_events) > 0 else -1
        max_event_node_idx = max(max_u, max_i)
        if max_event_node_idx != -1 and num_total_nodes > 0:
             assert max_event_node_idx < num_total_nodes, f"{data_dir_path}: Max event node index {max_event_node_idx} out of bounds for {num_total_nodes} nodes."

    if edge_features_all_events.ndim == 1 and edge_features_all_events.shape[0] > 0 :
        edge_features_all_events = edge_features_all_events.unsqueeze(1)
    elif edge_features_all_events.ndim == 1 and edge_features_all_events.shape[0] == 0 :
        edge_features_all_events = torch.empty(0,0, dtype=torch.float)


    return (node_features, labels_all_nodes, source_nodes_all_events, destination_nodes_all_events,
            event_timestamps_all_events, edge_features_all_events, num_total_nodes)

def create_event_splits(num_total_events: int, train_ratio: float = 0.7, val_ratio: float = 0.15):
    if num_total_events == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    train_idx_end = int(num_total_events * train_ratio)
    val_idx_end = int(num_total_events * (train_ratio + val_ratio))
    return np.arange(0, train_idx_end), np.arange(train_idx_end, val_idx_end), np.arange(val_idx_end, num_total_events)

def get_active_nodes_in_split(s_nodes_cpu, d_nodes_cpu, indices, node_labels_cpu):
    if len(indices) == 0 or len(s_nodes_cpu) == 0 or len(d_nodes_cpu) == 0:
        return torch.tensor([], dtype=torch.long)

    max_index_val = indices.max()
    if max_index_val >= len(s_nodes_cpu) or max_index_val >= len(d_nodes_cpu):
        valid_indices_mask = (indices < len(s_nodes_cpu)) & (indices < len(d_nodes_cpu))
        indices = indices[valid_indices_mask]
        if len(indices) == 0: return torch.tensor([], dtype=torch.long)

    active_nodes = torch.cat([s_nodes_cpu[indices], d_nodes_cpu[indices]]).unique()

    if len(active_nodes) > 0:
        max_active_node_idx = active_nodes.max()
        if max_active_node_idx >= len(node_labels_cpu):
            active_nodes = active_nodes[active_nodes < len(node_labels_cpu)]
            if len(active_nodes) == 0: return torch.tensor([], dtype=torch.long)

        valid_label_mask = node_labels_cpu[active_nodes] != -1
        return active_nodes[valid_label_mask]
    return torch.tensor([], dtype=torch.long)

# --- Training and Evaluation Logic ---
def run_training_epoch(model, raw_node_feats_cpu, optimizer, criterion, s_nodes_cpu, d_nodes_cpu, event_ts_cpu, edge_feats_cpu,
                       current_epoch_memory_start, current_epoch_last_updates_start,
                       train_indices, node_labels_cpu, batch_size, clip_norm, device, trial=None, epoch_num=0, event_name_for_log=""):
    model.train(); total_loss = 0.0; num_loss_calcs = 0
    mem_state_for_current_batch = current_epoch_memory_start
    last_updates_state_for_current_batch = current_epoch_last_updates_start
    raw_node_feats_dev = raw_node_feats_cpu.to(device)
    node_labels_dev = node_labels_cpu.to(device)
    final_mem_of_epoch = current_epoch_memory_start
    final_ts_of_epoch = current_epoch_last_updates_start

    use_tqdm = trial is None or epoch_num < 2 # Reduced tqdm for Optuna, more for fixed/report runs
    if event_name_for_log.endswith(("_PerfReport", "_FixedRun", "_BestHPs_Overall", "_BestHPs_Individual")) :
        use_tqdm = True # Always use tqdm for these specific run types

    desc_prefix = f"Epoch {epoch_num} ({event_name_for_log}) Train" if trial is None else f"Trial Opt ({event_name_for_log}) E{epoch_num} Train"
    batch_iterator = range(0, len(train_indices), batch_size)
    if use_tqdm: batch_iterator = tqdm(batch_iterator, desc=desc_prefix, leave=False)

    for i in batch_iterator:
        optimizer.zero_grad(); batch_idx = train_indices[i : i + batch_size]
        if len(batch_idx) == 0: continue

        batch_s, batch_d = s_nodes_cpu[batch_idx], d_nodes_cpu[batch_idx]
        batch_ts, batch_ef = event_ts_cpu[batch_idx], edge_feats_cpu[batch_idx]

        mem_out_current_batch, last_updates_out_current_batch = model.compute_messages_and_update_memory(
            mem_state_for_current_batch, last_updates_state_for_current_batch, batch_s, batch_d, batch_ts, batch_ef)

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
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected in training for {event_name_for_log} at epoch {epoch_num}, batch start {i}. Skipping batch.")
                mem_state_for_current_batch = mem_state_for_current_batch.detach() # Use previous state
                last_updates_state_for_current_batch = last_updates_state_for_current_batch.detach()
                continue # Skip optimizer step and state update for this batch
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
                   split_name="Eval", trial=None, epoch_num=0, event_name_for_log=""):
    model.eval(); preds_list, true_list = [], []
    mem_eval = mem_start_eval.detach().clone()
    last_updates_eval = last_updates_start_eval.detach().clone()
    raw_node_feats_dev = raw_node_feats_cpu.to(device)

    use_tqdm = trial is None or epoch_num < 2 # Reduced tqdm for Optuna
    if event_name_for_log.endswith(("_PerfReport", "_FixedRun", "_BestHPs_Overall", "_BestHPs_Individual")) or "Test-Epoch" in split_name or "Test-Final" in split_name:
        use_tqdm = True # Always use tqdm for these specific run types or epoch-wise/final test

    desc_prefix = f"Epoch {epoch_num} ({event_name_for_log}) {split_name}" if trial is None else f"Trial Opt ({event_name_for_log}) E{epoch_num} {split_name}"
    batch_iterator = range(0, len(eval_indices), batch_size)
    if use_tqdm: batch_iterator = tqdm(batch_iterator, desc=f"{desc_prefix} Event Proc", leave=False)

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

    if not preds_list: # No predictions were made (e.g. no active_eval_nodes_cpu)
        num_classes = len(torch.unique(node_labels_cpu[node_labels_cpu != -1]))
        if num_classes == 0: num_classes = 2 # Default for binary case if no labels
        # print(f"Warning ({event_name_for_log}, {split_name}, E{epoch_num}): No predictions made. Returning zero metrics.")
        return {"accuracy":0.0,"f1":0.0,"precision":0.0,"recall":0.0,"conf_matrix":np.zeros((num_classes,num_classes)).tolist()}, mem_eval, last_updates_eval

    preds, trues = np.concatenate(preds_list), np.concatenate(true_list)
    unique_true_labels = np.unique(trues)
    num_classes_for_cm = 2 # Default
    if len(unique_true_labels) > 0 :
        max_label_val = unique_true_labels.max()
        if max_label_val >=0 : num_classes_for_cm = int(max_label_val) + 1
    
    # Ensure labels for confusion matrix are at least [0, 1] if only one class is present in trues/preds
    # This can happen if all trues are e.g. 0 and all preds are 0.
    present_labels = np.unique(np.concatenate((trues, preds)))
    max_present_label = 0
    if len(present_labels) > 0:
        max_present_label = present_labels.max()
    
    # Ensure confusion matrix labels cover all actual unique labels present in trues and preds, and at least up to num_classes_for_cm-1
    cm_labels_list = list(range(max(num_classes_for_cm, int(max_present_label) + 1)))


    avg_metric = 'binary' if num_classes_for_cm <= 2 else 'weighted'

    metrics = {
        "accuracy": accuracy_score(trues, preds), "f1": f1_score(trues, preds, average=avg_metric, zero_division=0),
        "precision": precision_score(trues, preds, average=avg_metric, zero_division=0),
        "recall": recall_score(trues, preds, average=avg_metric, zero_division=0),
        "conf_matrix": confusion_matrix(trues, preds, labels=cm_labels_list).tolist()
    }
    return metrics, mem_eval, last_updates_eval

def run_single_trial(args_dict_passed, trial=None, event_name_for_log="event"):
    args_dict = args_dict_passed.copy() 

    if trial: 
        args_dict["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        args_dict["memory_dim"] = trial.suggest_categorical("memory_dim", [128, 256, 512])
        args_dict["time_dim"] = trial.suggest_categorical("time_dim", [64, 128, 256])
        args_dict["embedding_dim"] = trial.suggest_categorical("embedding_dim", [128, 256, 512])
        args_dict["dropout_rate"] = trial.suggest_float("dropout_rate", 0.05, 0.5)
        args_dict["projector_dropout_rate"] = trial.suggest_float("projector_dropout_rate", 0.05, 0.4)
        args_dict["grad_clip_norm"] = trial.suggest_float("grad_clip_norm", 0.5, 10.0)
        args_dict["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        args_dict["project_features"] = trial.suggest_categorical("project_features", [True, False])
        args_dict["use_layernorm"] = trial.suggest_categorical("use_layernorm", [True, False])
        args_dict["leaky_relu_slope"] = trial.suggest_float("leaky_relu_slope", 0.01, 0.3)

    torch.manual_seed(args_dict["seed"]); np.random.seed(args_dict["seed"])
    device = torch.device(args_dict["device"])
    if args_dict["device"] == "cuda" and torch.cuda.is_available(): torch.cuda.manual_seed_all(args_dict["seed"])

    is_detailed_log_main = trial is None and not event_name_for_log.endswith("_OptunaTrial") # Detailed log for non-optuna
    
    if is_detailed_log_main:
        print(f"[{event_name_for_log}] Using device: {device}")
        hps_to_print = {k: args_dict[k] for k in ['lr', 'memory_dim', 'time_dim', 'embedding_dim', 'dropout_rate', 'projector_dropout_rate', 'grad_clip_norm', 'batch_size', 'project_features', 'use_layernorm', 'leaky_relu_slope'] if k in args_dict}
        print(f"[{event_name_for_log}] Running with HPs: {hps_to_print}")

    best_run_details_for_this_trial = {"hyperparameters": args_dict.copy()} 

    (raw_node_feats_cpu, node_labels_cpu, s_nodes_cpu, d_nodes_cpu, ts_cpu, edge_feats_cpu, n_nodes) = \
        load_tgn_data_from_path(Path(args_dict["data_path"]))

    if n_nodes == 0 or (s_nodes_cpu.nelement() == 0 and d_nodes_cpu.nelement() == 0):
        msg = f"No nodes ({n_nodes}) or events (s:{s_nodes_cpu.nelement()},d:{d_nodes_cpu.nelement()}) in data for {args_dict['data_path']}."
        # print(f"[{event_name_for_log}] {msg} Skipping trial.") # Can be too verbose for Optuna
        best_run_details_for_this_trial["error"] = msg
        return {
            "value_for_optuna": 0.0,
            "best_run_details": best_run_details_for_this_trial,
            "training_history": None
        }

    valid_labels_cpu = node_labels_cpu[node_labels_cpu != -1]
    cls_weights = None; num_classes_data = 2
    if len(valid_labels_cpu) > 0:
        cls_counts = torch.bincount(valid_labels_cpu)
        if cls_counts.numel() > 0 :
            num_classes_data = len(cls_counts)
            if is_detailed_log_main: print(f"  [{event_name_for_log}] Num classes: {num_classes_data}. Counts: {cls_counts.tolist()}")
            if num_classes_data > 0 and cls_counts.min() > 0 and args_dict.get("use_class_weights", True):
                cls_weights = (1. / cls_counts.float()).to(device)
                if is_detailed_log_main: print(f"  [{event_name_for_log}] Using class weights: {cls_weights.cpu().numpy()}")
            # else: num_classes_data = max(2, num_classes_data) # Ensure at least 2 classes for binary case. Already handled by init.
    else:
        msg = "No valid labels (all are -1 or empty)."
        if is_detailed_log_main: print(f"[{event_name_for_log}] Warning: {msg} Defaulting to 2 classes, no weights.")
        if trial: 
            best_run_details_for_this_trial["error"] = msg
            return { "value_for_optuna": 0.0, "best_run_details": best_run_details_for_this_trial, "training_history": None}

    criterion = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=-1)
    train_idx, val_idx, test_idx = create_event_splits(len(s_nodes_cpu), args_dict["train_ratio"], args_dict["val_ratio"])

    if len(train_idx) == 0 and len(val_idx) == 0 and len(s_nodes_cpu) > 0 : # If there are events but not enough for splits
         msg = f"Not enough events to create train/val splits ({len(s_nodes_cpu)} total) for {args_dict['data_path']}."
         # print(f"[{event_name_for_log}] {msg} Skipping.")
         best_run_details_for_this_trial["error"] = msg
         return {
            "value_for_optuna": 0.0,
            "best_run_details": best_run_details_for_this_trial,
            "training_history": None
        }

    if is_detailed_log_main:
        print(f"  [{event_name_for_log}] Dataset: {Path(args_dict['data_path']).name}\n  Nodes: {n_nodes}, Events: {len(s_nodes_cpu)}")
        print(f"  [{event_name_for_log}] Train/Val/Test Events: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    active_val_nodes_cpu = get_active_nodes_in_split(s_nodes_cpu, d_nodes_cpu, val_idx, node_labels_cpu)
    active_test_nodes_cpu = get_active_nodes_in_split(s_nodes_cpu, d_nodes_cpu, test_idx, node_labels_cpu)
    if is_detailed_log_main: print(f"  [{event_name_for_log}] Active Val Nodes: {len(active_val_nodes_cpu)}, Active Test Nodes: {len(active_test_nodes_cpu)}")

    current_batch_size = args_dict["batch_size"]
    if len(train_idx) > 0 and args_dict["batch_size"] > len(train_idx):
        current_batch_size = max(1, len(train_idx)) 
        log_msg = f"Adjusted batch size from {args_dict['batch_size']} to {current_batch_size} due to fewer training samples ({len(train_idx)}) than original batch size."
        if trial: pass # print(f"Optuna Trial {trial.number} ({event_name_for_log}): {log_msg}") # Too verbose for optuna
        elif is_detailed_log_main: print(f"[{event_name_for_log}] {log_msg}")


    edge_feat_dim_model = 0
    if edge_feats_cpu.ndim > 1 and edge_feats_cpu.shape[0] > 0: 
        edge_feat_dim_model = edge_feats_cpu.shape[1]
    elif edge_feats_cpu.ndim == 1 and edge_feats_cpu.shape[0] > 0: 
        edge_feat_dim_model = 1 

    model = TGN(n_nodes, raw_node_feats_cpu.shape[1], edge_feat_dim_model,
                args_dict["memory_dim"], args_dict["time_dim"], args_dict["embedding_dim"],
                num_classes_data, args_dict["dropout_rate"], args_dict["projector_dropout_rate"],
                device, args_dict["project_features"], args_dict["leaky_relu_slope"], args_dict["use_layernorm"])

    optimizer = optim.AdamW(model.parameters(), lr=args_dict["lr"], weight_decay=args_dict["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=args_dict["lr_scheduler_patience"])

    best_val_f1_for_run = -1.0
    no_improve_epochs_for_run = 0
    
    epoch_nums_history, train_losses_history = [], []
    val_f1s_history, val_accs_history, val_precs_history, val_recalls_history = [], [], [], []
    test_f1s_history, test_accs_history, test_precs_history, test_recalls_history = [], [], [], []
    
    global_memory_state = torch.zeros((n_nodes, args_dict["memory_dim"]), device=device)
    global_last_update_timestamps = torch.zeros(n_nodes, device=device)
    num_epochs_to_run = args_dict["epochs"]
    collect_epoch_test_metrics_flag = args_dict.get("collect_epoch_wise_test_metrics", False)

    for epoch in range(1, num_epochs_to_run + 1):
        # Log verbosity for epoch details (non-optuna, or first few optuna epochs, or specific report types)
        is_epoch_detail_log_active = is_detailed_log_main or (trial is not None and epoch <=2) or \
                                     event_name_for_log.endswith(("_PerfReport", "_FixedRun", "_BestHPs_Overall", "_BestHPs_Individual"))
        
        epoch_memory_train_start = global_memory_state.clone() 
        epoch_last_updates_train_start = global_last_update_timestamps.clone()
        
        mem_for_eval_and_next_epoch = epoch_memory_train_start 
        ts_for_eval_and_next_epoch = epoch_last_updates_train_start 

        avg_train_loss = 0.0
        if len(train_idx) > 0:
            avg_train_loss, mem_after_train, ts_after_train = run_training_epoch(
                model, raw_node_feats_cpu, optimizer, criterion, s_nodes_cpu, d_nodes_cpu, ts_cpu, edge_feats_cpu,
                epoch_memory_train_start, epoch_last_updates_train_start, train_idx, node_labels_cpu,
                current_batch_size, args_dict["grad_clip_norm"], device, trial, epoch, event_name_for_log)
            if is_epoch_detail_log_active: print(f"  [{event_name_for_log}] Epoch {epoch}/{num_epochs_to_run} Avg Train Loss: {avg_train_loss:.4f}")
            mem_for_eval_and_next_epoch = mem_after_train
            ts_for_eval_and_next_epoch = ts_after_train
        elif is_epoch_detail_log_active: 
            print(f"  [{event_name_for_log}] Epoch {epoch}/{num_epochs_to_run} No training events.")
        
        epoch_nums_history.append(epoch) 
        train_losses_history.append(avg_train_loss)

        current_epoch_val_f1 = -1.0; current_epoch_val_metrics = None
        if len(val_idx) > 0 and len(active_val_nodes_cpu) > 0:
            val_metrics, _, _ = run_evaluation( 
                model, raw_node_feats_cpu, s_nodes_cpu, d_nodes_cpu, ts_cpu, edge_feats_cpu, 
                mem_for_eval_and_next_epoch, ts_for_eval_and_next_epoch, 
                val_idx, node_labels_cpu, args_dict["batch_size"], active_val_nodes_cpu, device, 
                "Val", trial, epoch, event_name_for_log)
            if is_epoch_detail_log_active:
                print(f"  [{event_name_for_log}] Epoch {epoch}/{num_epochs_to_run} Val - Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}")
            current_epoch_val_f1 = val_metrics['f1']
            current_epoch_val_metrics = val_metrics
            scheduler.step(float(current_epoch_val_f1) if current_epoch_val_f1 is not None else 0.0)
            val_f1s_history.append(float(val_metrics.get('f1', 0.0)))
            val_accs_history.append(float(val_metrics.get('accuracy', 0.0)))
            val_precs_history.append(float(val_metrics.get('precision', 0.0)))
            val_recalls_history.append(float(val_metrics.get('recall', 0.0)))
        else:
            if is_epoch_detail_log_active: print(f"  [{event_name_for_log}] Epoch {epoch}/{num_epochs_to_run} Skipping validation (no val events or no active val nodes).")
            val_f1s_history.append(0.0); val_accs_history.append(0.0); val_precs_history.append(0.0); val_recalls_history.append(0.0)

        if collect_epoch_test_metrics_flag:
            if len(test_idx) > 0 and len(active_test_nodes_cpu) > 0:
                epoch_test_metrics, _, _ = run_evaluation(
                    model, raw_node_feats_cpu, s_nodes_cpu, d_nodes_cpu, ts_cpu, edge_feats_cpu,
                    mem_for_eval_and_next_epoch, ts_for_eval_and_next_epoch,
                    test_idx, node_labels_cpu, args_dict["batch_size"], active_test_nodes_cpu, device,
                    "Test-Epoch", trial, epoch, event_name_for_log)
                if epoch_test_metrics:
                    if is_epoch_detail_log_active: print(f"  [{event_name_for_log}] Epoch {epoch}/{num_epochs_to_run} Test (Epoch-wise) - Acc: {epoch_test_metrics['accuracy']:.4f}, F1: {epoch_test_metrics['f1']:.4f}")
                    test_f1s_history.append(float(epoch_test_metrics.get('f1', 0.0)))
                    test_accs_history.append(float(epoch_test_metrics.get('accuracy', 0.0)))
                    test_precs_history.append(float(epoch_test_metrics.get('precision', 0.0)))
                    test_recalls_history.append(float(epoch_test_metrics.get('recall', 0.0)))
                else:
                    test_f1s_history.append(0.0); test_accs_history.append(0.0); test_precs_history.append(0.0); test_recalls_history.append(0.0)
            else:
                if is_epoch_detail_log_active: print(f"  [{event_name_for_log}] Epoch {epoch}/{num_epochs_to_run} Skipping epoch-wise test (no test events or no active test nodes).")
                test_f1s_history.append(0.0); test_accs_history.append(0.0); test_precs_history.append(0.0); test_recalls_history.append(0.0)
        
        current_epoch_final_test_metrics = None 
        if current_epoch_val_f1 is not None and float(current_epoch_val_f1) > best_val_f1_for_run : 
            best_val_f1_for_run = float(current_epoch_val_f1) 
            no_improve_epochs_for_run = 0
            best_run_details_for_this_trial["best_epoch"] = epoch
            if current_epoch_val_metrics:
                best_run_details_for_this_trial["val_metrics"] = current_epoch_val_metrics
            
            if trial is None and (len(test_idx) > 0 and len(active_test_nodes_cpu) > 0): 
                if is_epoch_detail_log_active: print(f"  [{event_name_for_log}] Epoch {epoch}/{num_epochs_to_run} New best Val F1: {best_val_f1_for_run:.4f}. Evaluating on Test for this best model...")
                test_metrics_output, _, _ = run_evaluation(
                    model, raw_node_feats_cpu, s_nodes_cpu, d_nodes_cpu, ts_cpu, edge_feats_cpu, 
                    mem_for_eval_and_next_epoch, ts_for_eval_and_next_epoch, 
                    test_idx, node_labels_cpu, args_dict["batch_size"], active_test_nodes_cpu, device, 
                    "Test-Final", trial, epoch, event_name_for_log) 
                if is_epoch_detail_log_active:
                    print(f"  [{event_name_for_log}] Epoch {epoch}/{num_epochs_to_run} Test (Corresp. Best Val) - Acc: {test_metrics_output['accuracy']:.4f}, F1: {test_metrics_output['f1']:.4f}")
                current_epoch_final_test_metrics = test_metrics_output 
            if current_epoch_final_test_metrics: 
                best_run_details_for_this_trial["test_metrics"] = current_epoch_final_test_metrics
        else:
            no_improve_epochs_for_run += 1
        
        global_memory_state = mem_for_eval_and_next_epoch.detach()
        global_last_update_timestamps = ts_for_eval_and_next_epoch.detach()

        if no_improve_epochs_for_run >= args_dict["early_stopping_patience"] and best_val_f1_for_run > -0.5 : # -0.5 to allow for 0.0 F1s
            if is_epoch_detail_log_active: print(f"[{event_name_for_log}] Early stopping after {epoch} epochs."); 
            break
        
        if trial:
            trial.report(best_val_f1_for_run if best_val_f1_for_run != -1.0 else 0.0, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if best_val_f1_for_run == -1.0 and "error" not in best_run_details_for_this_trial: 
         if not (len(val_idx) > 0 and len(active_val_nodes_cpu) > 0): 
            best_run_details_for_this_trial["warning"] = "Validation was not possible (no validation data/active nodes). Final F1 is based on initial state."
         else: # Validation was possible but never improved
            best_run_details_for_this_trial["error"] = "No successful validation epoch improved from initial F1 (val_f1 remained at -1.0 or was never computed meaningfully)."


    if is_detailed_log_main: 
        print(f"\n--- [{event_name_for_log}] Training Complete (Fixed/Final Run) ---")
        if "error" in best_run_details_for_this_trial:
            print(f"Run Issue for {event_name_for_log}: {best_run_details_for_this_trial['error']}")
        elif "val_metrics" in best_run_details_for_this_trial : 
            print(f"Best Val F1 for {event_name_for_log}: {best_val_f1_for_run:.4f} at epoch {best_run_details_for_this_trial.get('best_epoch', 'N/A')}")
            if best_run_details_for_this_trial.get("test_metrics"):
                tm = best_run_details_for_this_trial["test_metrics"]
                print(f"Corresp. Test Metrics: Acc: {tm['accuracy']:.4f}, F1: {tm['f1']:.4f}, Prec: {tm['precision']:.4f}, Rec: {tm['recall']:.4f}")
            else:
                 print(f"Test metrics (for best val epoch) were not generated for {event_name_for_log} (e.g., no test data or test eval skipped).")
        elif "warning" in best_run_details_for_this_trial:
             print(f"Run Warning for {event_name_for_log}: {best_run_details_for_this_trial['warning']}")
        else: 
            print(f"No validation metrics properly recorded for {event_name_for_log}. Check data and training progression. Best val F1: {best_val_f1_for_run}")

    training_history_dict = None
    # Pad history lists if early stopping occurred for consistent plotting up to num_epochs_to_run
    # This is important for non-Optuna runs or specific report runs where all plots should have same x-axis length
    if trial is None or collect_epoch_test_metrics_flag: 
        current_num_run_epochs = len(epoch_nums_history)
        if current_num_run_epochs < num_epochs_to_run and current_num_run_epochs > 0:
            last_train_loss = train_losses_history[-1] if train_losses_history else 0.0
            last_val_f1 = val_f1s_history[-1] if val_f1s_history else 0.0
            last_val_acc = val_accs_history[-1] if val_accs_history else 0.0
            last_val_prec = val_precs_history[-1] if val_precs_history else 0.0
            last_val_recall = val_recalls_history[-1] if val_recalls_history else 0.0
            
            last_test_f1 = test_f1s_history[-1] if collect_epoch_test_metrics_flag and test_f1s_history else 0.0
            last_test_acc = test_accs_history[-1] if collect_epoch_test_metrics_flag and test_accs_history else 0.0
            last_test_prec = test_precs_history[-1] if collect_epoch_test_metrics_flag and test_precs_history else 0.0
            last_test_recall = test_recalls_history[-1] if collect_epoch_test_metrics_flag and test_recalls_history else 0.0

            for e_i in range(current_num_run_epochs + 1, num_epochs_to_run + 1):
                epoch_nums_history.append(e_i)
                train_losses_history.append(last_train_loss)
                val_f1s_history.append(last_val_f1)
                val_accs_history.append(last_val_acc)
                val_precs_history.append(last_val_prec)
                val_recalls_history.append(last_val_recall)
                if collect_epoch_test_metrics_flag:
                    test_f1s_history.append(last_test_f1)
                    test_accs_history.append(last_test_acc)
                    test_precs_history.append(last_test_prec)
                    test_recalls_history.append(last_test_recall)
        
        # Ensure all lists are initialized even if current_num_run_epochs is 0 (e.g. skipped trial)
        # but num_epochs_to_run > 0
        if not epoch_nums_history and num_epochs_to_run > 0:
             epoch_nums_history = list(range(1, num_epochs_to_run + 1))
             train_losses_history = [0.0] * num_epochs_to_run
             val_f1s_history = [0.0] * num_epochs_to_run
             val_accs_history = [0.0] * num_epochs_to_run
             val_precs_history = [0.0] * num_epochs_to_run
             val_recalls_history = [0.0] * num_epochs_to_run
             if collect_epoch_test_metrics_flag:
                 test_f1s_history = [0.0] * num_epochs_to_run
                 test_accs_history = [0.0] * num_epochs_to_run
                 test_precs_history = [0.0] * num_epochs_to_run
                 test_recalls_history = [0.0] * num_epochs_to_run


        training_history_dict = {
            "epochs": epoch_nums_history, "train_loss": train_losses_history,
            "val_f1": val_f1s_history, "val_acc": val_accs_history, 
            "val_precision": val_precs_history, "val_recall": val_recalls_history
        }
        if collect_epoch_test_metrics_flag:
            training_history_dict["test_f1"] = test_f1s_history
            training_history_dict["test_acc"] = test_accs_history
            training_history_dict["test_precision"] = test_precs_history
            training_history_dict["test_recall"] = test_recalls_history

    return {
        "value_for_optuna": best_val_f1_for_run if best_val_f1_for_run != -1.0 else 0.0,
        "best_run_details": best_run_details_for_this_trial,
        "training_history": training_history_dict
    }
    
def objective(trial, base_args_dict, event_name_for_log):
    current_trial_args = base_args_dict.copy()
    current_trial_args["epochs"] = current_trial_args["epochs_optuna_trial"] 
    current_trial_args["collect_epoch_wise_test_metrics"] = False 
    results = run_single_trial(current_trial_args, trial=trial, event_name_for_log=f"{event_name_for_log}_OptunaTrial")
    return results["value_for_optuna"]

# --- Plotting and Saving ---
def plot_training_history(history, save_path, event_name):
    """Plots individual training history: Loss, Val metrics, and Test metrics if available."""
    if history is None or not history.get("epochs") or (not history.get("train_loss") and not history.get("val_f1")):
        print(f"No or incomplete training history to plot for {event_name}.")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, "No training data available for plotting.", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"Training History for {event_name} (No Data)")
        try: plt.savefig(save_path); plt.close(fig)
        except Exception as e: print(f"Error saving placeholder plot: {e}")
        return

    epochs = history["epochs"]
    num_subplots = 1
    if any(k in history for k in ["val_f1", "val_acc", "val_precision", "val_recall"]): num_subplots +=1
    if any(k in history for k in ["test_f1", "test_acc", "test_precision", "test_recall"]): num_subplots +=1
    
    if num_subplots == 1 and not history.get("train_loss"): # Only epochs, no actual data
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, "Only epoch numbers available, no metric data.", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"Training History for {event_name} (Minimal Data)")
        try: plt.savefig(save_path); plt.close(fig)
        except Exception as e: print(f"Error saving minimal data plot: {e}")
        return

    fig, axs = plt.subplots(num_subplots, 1, figsize=(12, 5 * num_subplots), sharex=True)
    if num_subplots == 1: axs = [axs] # Make it iterable
    fig.suptitle(f"Training History for Event: {event_name}", fontsize=16)
    
    plot_idx = 0
    # Plot Training Loss
    if "train_loss" in history and history["train_loss"]:
        axs[plot_idx].plot(epochs, history["train_loss"], label="Training Loss", color="royalblue", marker='.')
        axs[plot_idx].set_ylabel("Loss")
        axs[plot_idx].set_title("Training Loss")
        axs[plot_idx].legend()
        axs[plot_idx].grid(True, linestyle='--', alpha=0.7)
        axs[plot_idx].xaxis.set_major_locator(MaxNLocator(integer=True))
        plot_idx += 1

    # Plot Validation Metrics
    val_metrics_plotted = False
    if any(k in history and history[k] for k in ["val_f1", "val_acc", "val_precision", "val_recall"]):
        ax_val = axs[plot_idx]
        if "val_f1" in history and history["val_f1"]:
            ax_val.plot(epochs, history["val_f1"], label="Val F1", color="forestgreen", marker='.')
            val_metrics_plotted = True
        if "val_acc" in history and history["val_acc"]:
            ax_val.plot(epochs, history["val_acc"], label="Val Acc", color="coral", linestyle="--", marker='x')
            val_metrics_plotted = True
        # Add precision/recall if needed, or keep it concise
        if val_metrics_plotted:
            ax_val.set_ylabel("Metric Value")
            ax_val.set_title("Validation Metrics")
            ax_val.legend()
            ax_val.grid(True, linestyle='--', alpha=0.7)
            ax_val.set_ylim(-0.05, 1.05)
            ax_val.xaxis.set_major_locator(MaxNLocator(integer=True))
            plot_idx +=1
        elif plot_idx < len(axs): # No val metrics but subplot was created
            axs[plot_idx].text(0.5, 0.5, "Validation metrics not available.", ha='center', va='center', transform=axs[plot_idx].transAxes)
            axs[plot_idx].set_title("Validation Metrics (No Data)")
            plot_idx +=1


    # Plot Test Metrics (if available, e.g. from collect_epoch_wise_test_metrics)
    test_metrics_plotted = False
    if any(k in history and history[k] for k in ["test_f1", "test_acc", "test_precision", "test_recall"]):
        ax_test = axs[plot_idx]
        if "test_f1" in history and history["test_f1"]:
            ax_test.plot(epochs, history["test_f1"], label="Test F1", color="purple", marker='.')
            test_metrics_plotted = True
        if "test_acc" in history and history["test_acc"]:
            ax_test.plot(epochs, history["test_acc"], label="Test Acc", color="brown", linestyle="--", marker='x')
            test_metrics_plotted = True
        if test_metrics_plotted:
            ax_test.set_ylabel("Metric Value")
            ax_test.set_title("Test Metrics (Epoch-wise)")
            ax_test.legend()
            ax_test.grid(True, linestyle='--', alpha=0.7)
            ax_test.set_ylim(-0.05, 1.05)
            ax_test.xaxis.set_major_locator(MaxNLocator(integer=True))
            plot_idx +=1
        elif plot_idx < len(axs): # No test metrics but subplot was created
            axs[plot_idx].text(0.5, 0.5, "Epoch-wise test metrics not available.", ha='center', va='center', transform=axs[plot_idx].transAxes)
            axs[plot_idx].set_title("Test Metrics (No Data)")
            plot_idx +=1

    if epochs: axs[-1].set_xlabel("Epoch")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    try:
        plt.savefig(save_path); print(f"Training plot saved to {save_path}")
    except Exception as e: print(f"Error saving training plot to {save_path}: {e}")
    plt.close(fig)


def _get_plot_colors(num_items):
    if num_items <= 10: cmap = plt.cm.get_cmap('tab10')
    elif num_items <= 20: cmap = plt.cm.get_cmap('tab20')
    else: cmap = plt.cm.get_cmap('viridis', num_items)
    return [cmap(i) for i in range(num_items)]

def _format_hps_for_title(fixed_hps_for_title, max_len=100):
    hp_str_parts = []
    for k_hp, v_hp in fixed_hps_for_title.items():
        if isinstance(v_hp, float): hp_str_parts.append(f"{k_hp}={v_hp:.3g}")
        else: hp_str_parts.append(f"{k_hp}={v_hp}")
    hp_title_str = ", ".join(hp_str_parts)
    if len(hp_title_str) > max_len:
        hp_title_str = hp_title_str[:max_len-3] + "..."
    return hp_title_str

def _plot_epoch_trends(event_histories_map, save_path, fixed_hps_for_title, max_epochs_x_axis=None):
    valid_event_histories = {
        name: hist for name, hist in event_histories_map.items() 
        if hist and hist.get("epochs") and len(hist["epochs"]) > 0 and not hist.get("error")
    }
    valid_event_names = sorted(list(valid_event_histories.keys()))

    if not valid_event_names:
        print("No valid event histories with epoch data for epoch trends plot.")
        # Create a placeholder plot
        fig, _ = plt.subplots(1,1,figsize=(12,6))
        fig.text(0.5, 0.5, "No data for epoch trends.", ha='center', va='center')
        hp_title_str = _format_hps_for_title(fixed_hps_for_title)
        fig.suptitle(f"TGN Epoch-wise Performance Trends (No Data)\nFixed HPs: {hp_title_str}", fontsize=14)
        plt.savefig(save_path, bbox_inches='tight'); plt.close(fig)
        return

    event_colors = _get_plot_colors(len(valid_event_names))
    
    metrics_to_plot = [
        ("F1-Score", "val_f1", "test_f1"),
        ("Accuracy", "val_acc", "test_acc"),
        ("Precision", "val_precision", "test_precision"),
        ("Recall", "val_recall", "test_recall"),
        ("Training Loss", "train_loss", None) # Special case for loss
    ]
    
    num_metric_groups = len(metrics_to_plot)
    fig, axs = plt.subplots(num_metric_groups, 1, figsize=(15, 5 * num_metric_groups), sharex=True)
    if num_metric_groups == 1: axs = [axs]

    max_epoch_observed = 0

    for i, (metric_name, val_key, test_key) in enumerate(metrics_to_plot):
        axs[i].set_title(metric_name, fontsize=13)
        axs[i].set_ylabel(metric_name if "Loss" not in metric_name else "Loss", fontsize=11)
        axs[i].grid(True, linestyle='--', alpha=0.6)
        if "Loss" not in metric_name: axs[i].set_ylim(-0.02, 1.02)

        for event_idx, event_name in enumerate(valid_event_names):
            history = valid_event_histories[event_name]
            epochs = history["epochs"]
            if not epochs: continue
            max_epoch_observed = max(max_epoch_observed, epochs[-1] if epochs else 0)
            
            color = event_colors[event_idx]
            
            num_epochs_data = len(epochs)
            mark_every_val = max(1, num_epochs_data // 10 if num_epochs_data > 10 else 1)

            common_plot_args = {'marker': '.', 'markersize': 5, 'linewidth': 1.2, 'markevery': mark_every_val}

            if val_key and val_key in history and history[val_key] and any(v is not None for v in history[val_key]):
                axs[i].plot(epochs, history[val_key], label=f"{event_name} Val", color=color, linestyle=":", **common_plot_args)
            
            if test_key and test_key in history and history[test_key] and any(v is not None for v in history[test_key]):
                axs[i].plot(epochs, history[test_key], label=f"{event_name} Test", color=color, linestyle="-", **common_plot_args)
        
        if i == 0 and len(valid_event_names) > 0 : # Legend only on the first plot for clarity if many events
             axs[i].legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize='small', title="Events")
        axs[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    if max_epochs_x_axis is None and max_epoch_observed > 0 : max_epochs_x_axis = max_epoch_observed
    if max_epochs_x_axis : axs[-1].set_xlim(0, max_epochs_x_axis + 1) # +1 for padding
    
    axs[-1].set_xlabel("Epoch", fontsize=12)
    hp_title_str = _format_hps_for_title(fixed_hps_for_title)
    fig.suptitle(f"TGN Epoch-wise Performance Trends\n(Fixed HPs: {hp_title_str})", fontsize=15, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 0.90 if len(valid_event_names)>0 else 1, 0.95]) # Adjust right margin for legend

    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Epoch trends plot saved to {save_path}")
    except Exception as e: print(f"Error saving epoch trends plot: {e}")
    plt.close(fig)

def _plot_summary_bars(event_summary_details_map, save_path, fixed_hps_for_title):
    valid_event_data = {}
    for name, details in event_summary_details_map.items():
        if details and not details.get("error") and details.get("test_metrics"):
            valid_event_data[name] = details["test_metrics"] 
            # Ensure all base metrics are present, default to 0 if missing
            for metric_key in ['f1', 'accuracy', 'precision', 'recall']:
                if metric_key not in valid_event_data[name]:
                     valid_event_data[name][metric_key] = 0.0
        # else:
            # print(f"Event {name} excluded from summary bar plot due to error or missing test_metrics: {details.get('error', 'N/A')}")


    valid_event_names = sorted(list(valid_event_data.keys()))

    if not valid_event_names:
        print("No valid event data for summary bar plot.")
        fig, _ = plt.subplots(1,1,figsize=(12,6))
        fig.text(0.5, 0.5, "No data for summary bars.", ha='center', va='center')
        hp_title_str = _format_hps_for_title(fixed_hps_for_title)
        fig.suptitle(f"TGN Performance Summary (No Data)\nFixed HPs: {hp_title_str}", fontsize=14)
        plt.savefig(save_path, bbox_inches='tight'); plt.close(fig)
        return

    metrics_to_plot = [
        ("F1-Score", "f1"), ("Accuracy", "accuracy"),
        ("Precision", "precision"), ("Recall", "recall")
    ]
    num_metrics = len(metrics_to_plot)
    fig, axs = plt.subplots(num_metrics, 1, figsize=(max(10, len(valid_event_names)*0.5), 5 * num_metrics))
    if num_metrics == 1: axs = [axs]

    event_colors = _get_plot_colors(len(valid_event_names))

    for i, (metric_name, metric_key) in enumerate(metrics_to_plot):
        metric_values = [valid_event_data[name].get(metric_key, 0.0) for name in valid_event_names]
        
        bars = axs[i].bar(valid_event_names, metric_values, color=event_colors)
        axs[i].set_title(f"Test {metric_name}", fontsize=13)
        axs[i].set_ylabel(metric_name, fontsize=11)
        axs[i].set_ylim(0, 1.05)
        axs[i].grid(True, linestyle='--', alpha=0.6, axis='y')

        for bar in bars: # Annotate bars
            yval = bar.get_height()
            axs[i].text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom', fontsize=8)

        if metric_values: # Plot average line
            avg_metric = np.mean([val for val in metric_values if val is not None]) # handle Nones if any snuck in
            axs[i].axhline(avg_metric, color='red', linestyle='dashed', linewidth=1.2, label=f"Avg: {avg_metric:.3f}")
            axs[i].legend(fontsize='small')
        
        if len(valid_event_names) > 5 : # Rotate x-labels if many events
             axs[i].tick_params(axis='x', rotation=45, labelsize=9, ha="right")
        else:
             axs[i].tick_params(axis='x', labelsize=10)


    axs[-1].set_xlabel("Event Stream", fontsize=12)
    hp_title_str = _format_hps_for_title(fixed_hps_for_title)
    fig.suptitle(f"TGN Performance Summary Across Events\n(Test metrics from best validation epoch. Fixed HPs: {hp_title_str})", fontsize=15, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    
    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Summary bar plot saved to {save_path}")
    except Exception as e: print(f"Error saving summary bar plot: {e}")
    plt.close(fig)

def _plot_best_val_f1_vs_epoch(event_summary_details_map, save_path, fixed_hps_for_title):
    plot_data = []
    for name, details in event_summary_details_map.items():
        if details and not details.get("error") and \
           details.get("val_metrics") and 'f1' in details["val_metrics"] and \
           details.get("best_epoch") is not None:
            plot_data.append({
                "name": name,
                "best_val_f1": details["val_metrics"]["f1"],
                "best_epoch": details["best_epoch"]
            })
    
    if not plot_data:
        print("No valid event data for Best Val F1 vs. Epoch scatter plot.")
        fig, _ = plt.subplots(1,1,figsize=(10,6))
        fig.text(0.5, 0.5, "No data for Val F1 vs Epoch.", ha='center', va='center')
        hp_title_str = _format_hps_for_title(fixed_hps_for_title)
        fig.suptitle(f"Best Validation F1 vs. Optimal Epoch (No Data)\nFixed HPs: {hp_title_str}", fontsize=14)
        plt.savefig(save_path, bbox_inches='tight'); plt.close(fig)
        return

    names = [d["name"] for d in plot_data]
    best_val_f1s = [d["best_val_f1"] for d in plot_data]
    best_epochs = [d["best_epoch"] for d in plot_data]
    
    colors = _get_plot_colors(len(names))

    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(best_epochs, best_val_f1s, c=colors, s=100, alpha=0.7)

    # Annotate points with event names
    for i, name in enumerate(names):
        ax.annotate(name, (best_epochs[i], best_val_f1s[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    ax.set_xlabel("Epoch of Best Validation F1", fontsize=12)
    ax.set_ylabel("Best Validation F1-Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    if best_epochs:
        ax.set_xlim(0, max(best_epochs) + max(1, int(0.1 * max(best_epochs)))) # ensure xlim starts at 0 and has some padding
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # Ensure integer ticks for epochs
    ax.grid(True, linestyle='--', alpha=0.6)
    
    hp_title_str = _format_hps_for_title(fixed_hps_for_title)
    fig.suptitle(f"Best Validation F1 vs. Optimal Epoch Across Events\n(Fixed HPs: {hp_title_str})", fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Best Val F1 vs. Epoch scatter plot saved to {save_path}")
    except Exception as e: print(f"Error saving Best Val F1 vs. Epoch scatter plot: {e}")
    plt.close(fig)


def generate_multi_event_report_visualizations(event_histories_map, event_summary_details_map, 
                                               output_dir: Path, fixed_hps_for_title, max_epochs_x_axis=None):
    """
    Generates a suite of plots for the multi-event performance report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Generating Multi-Event Report Visualizations in {output_dir} ---")

    # Plot 1: Epoch Trends (Val/Test F1, Acc, Prec, Recall, Loss)
    _plot_epoch_trends(
        event_histories_map, 
        output_dir / "report_epoch_trends.png", 
        fixed_hps_for_title,
        max_epochs_x_axis
    )
    
    # Plot 2: Summary Bar Charts (Test F1, Acc, Prec, Recall from best val epoch)
    _plot_summary_bars(
        event_summary_details_map, 
        output_dir / "report_performance_summary_bars.png", 
        fixed_hps_for_title
    )

    # Plot 3: Best Val F1 vs. Epoch Scatter Plot
    _plot_best_val_f1_vs_epoch(
        event_summary_details_map, 
        output_dir / "report_best_val_f1_vs_epoch.png", 
        fixed_hps_for_title
    )
    print("--- Multi-Event Report Visualizations Generation Complete ---")


def save_event_specific_results(event_name, results_base_dir, run_details, training_history, hps_used, optuna_study, args_for_the_run):
    event_dir = Path(results_base_dir) / event_name
    event_dir.mkdir(parents=True, exist_ok=True)

    md_content = f"# TGN Results for Event: {event_name}\n\n"
    actual_hps_used_in_run = run_details.get("hyperparameters", {}) if run_details else {}

    if run_details and "error" in run_details:
        md_content += f"**Run failed or was skipped:** {run_details['error']}\n\n"
        md_content += "## Hyperparameters Used (Attempted):\n"
        hps_to_display_on_error = actual_hps_used_in_run if actual_hps_used_in_run else hps_used
        for k, v in hps_to_display_on_error.items():
            if k in ['lr', 'memory_dim', 'time_dim', 'embedding_dim', 'dropout_rate', 'batch_size', 'project_features', 'use_layernorm', 'leaky_relu_slope', 'projector_dropout_rate', 'grad_clip_norm']:
                 md_content += f"- {k}: {v}\n"
    else:
        md_content += "## Hyperparameters Used (for this successful run):\n"
        # Use hps_used for reporting, as actual_hps_used_in_run might be a subset from Optuna trial
        # or args_dict for fixed runs. hps_used (passed as best_overall_hps or best_trial_event_params or fixed_run_args) is more canonical.
        hps_to_report_on_success = hps_used 
        for k, v in hps_to_report_on_success.items():
            if k in ['lr', 'memory_dim', 'time_dim', 'embedding_dim', 'dropout_rate', 'projector_dropout_rate', 'grad_clip_norm', 'batch_size', 'project_features', 'use_layernorm', 'leaky_relu_slope']:
                md_content += f"- {k}: {v}\n"
        md_content += "\n"

        if run_details and "val_metrics" in run_details and run_details["val_metrics"]:
            md_content += "## Performance Metrics (at best validation epoch):\n"
            md_content += f"- Best Epoch: {run_details.get('best_epoch', 'N/A')}\n"
            vm = run_details["val_metrics"]
            md_content += f"- Validation Accuracy: {vm.get('accuracy', 0.0):.4f}\n"
            md_content += f"- Validation F1-Score: {vm.get('f1', 0.0):.4f}\n"
            md_content += f"- Validation Precision: {vm.get('precision', 0.0):.4f}\n"
            md_content += f"- Validation Recall: {vm.get('recall', 0.0):.4f}\n"
            if "conf_matrix" in vm: md_content += f"- Validation Confusion Matrix:\n```\n{np.array(vm['conf_matrix'])}\n```\n"
        elif run_details and "warning" in run_details:
            md_content += f"**Run Warning:** {run_details['warning']}\n"
            md_content += "Validation metrics might be absent or based on initial state.\n"
        else:
            md_content += "Validation metrics not available for this run.\n"


        if run_details and "test_metrics" in run_details and run_details["test_metrics"]:
            tm = run_details["test_metrics"]
            md_content += "\n### Corresponding Test Metrics (for best validation model state):\n"
            md_content += f"- Test Accuracy: {tm.get('accuracy', 0.0):.4f}\n"
            md_content += f"- Test F1-Score: {tm.get('f1', 0.0):.4f}\n"
            md_content += f"- Test Precision: {tm.get('precision', 0.0):.4f}\n"
            md_content += f"- Test Recall: {tm.get('recall', 0.0):.4f}\n"
            if "conf_matrix" in tm: md_content += f"- Test Confusion Matrix:\n```\n{np.array(tm['conf_matrix'])}\n```\n"
        else:
            md_content += "\nTest metrics (for best validation model state) not available or not evaluated for this run.\n"

    with open(event_dir / "results_summary.md", "w") as f: # Renamed to avoid conflict with potential Optuna results.md
        f.write(md_content)
    print(f"Results summary saved to {event_dir / 'results_summary.md'}")

    # Plot individual training history for non-PerfReport runs, or if PerfReport but no aggregate plot is made.
    # The PerfReport runs will have their epoch trends plotted by generate_multi_event_report_visualizations.
    # This plot_training_history is more for Optuna best runs or single fixed runs.
    if not event_name.endswith("_PerfReport"): 
        plot_path = event_dir / "training_plots.png"
        plot_training_history(training_history, plot_path, event_name)

    if optuna_study:
        csv_path = event_dir / "optuna_study_results.csv"
        try:
            optuna_study.trials_dataframe().to_csv(csv_path, index=False)
            print(f"Optuna study results saved to {csv_path}")
        except Exception as e:
            print(f"Warning: Could not save Optuna study CSV for {event_name}: {e}")

        try:
            if len(optuna_study.trials) > 1 : 
                fig_parallel = plot_parallel_coordinate(optuna_study)
                fig_parallel.write_image(str(event_dir / "optuna_parallel_coordinate_plot.png")) # Ensure str for older plotly
                print(f"Optuna parallel coordinate plot saved to {event_dir / 'optuna_parallel_coordinate_plot.png'}")

            if optuna_study.trials: 
                fig_history = plot_optimization_history(optuna_study)
                fig_history.write_image(str(event_dir / "optuna_optimization_history.png"))
                print(f"Optuna optimization history plot saved to {event_dir / 'optuna_optimization_history.png'}")

            if len(optuna_study.trials) > 1 and any(t.state == optuna.trial.TrialState.COMPLETE for t in optuna_study.trials):
                try:
                    fig_importance = plot_param_importances(optuna_study)
                    fig_importance.write_image(str(event_dir / "optuna_param_importances.png"))
                    print(f"Optuna param importances plot saved to {event_dir / 'optuna_param_importances.png'}")
                except (ValueError, IndexError) as ve: 
                    print(f"Could not generate param importances plot for {event_name}: {ve}. (Often needs multiple completed trials with varied params)")
        except ImportError:
             print("Plotly/Kaleido not installed. Skipping Optuna plot generation. pip install plotly kaleido")
        except Exception as e:
            print(f"Warning: Could not generate/save some Optuna plots for {event_name}: {e}")

def save_aggregated_results_md(results_dir, event_summary_details_map, best_overall_hps):
    md_path = Path(results_dir) / "aggregated_results_report.md"
    md_content = "# TGN Aggregated Performance Report\n\n"
    
    hps_subset_for_report = ['lr', 'memory_dim', 'time_dim', 'embedding_dim', 'dropout_rate', 
                             'projector_dropout_rate', 'grad_clip_norm', 'batch_size', 
                             'project_features', 'use_layernorm', 'leaky_relu_slope']
    
    md_content += "## Hyperparameters Used for this Report (from 'all' dataset Optuna or fixed):\n"
    if best_overall_hps:
        for k, v in best_overall_hps.items():
             if k in hps_subset_for_report:
                md_content += f"- {k}: {v}\n"
    else:
        md_content += "- Not available or 'all' dataset study was not run/successful, or fixed HPs were used directly.\n"
    md_content += "\n"

    md_content += "## Performance on Individual Event Streams (Test Metrics from Best Validation Epoch):\n\n"
    md_content += "| Event Stream     | Test Accuracy | Test F1-score | Test Precision | Test Recall | Best Val Epoch |\n"
    md_content += "|------------------|---------------|---------------|----------------|-------------|----------------|\n"

    avg_metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}
    processed_event_count = 0
    
    if not event_summary_details_map:
        md_content += "| No events processed or no successful runs for aggregation | N/A | N/A | N/A | N/A | N/A |\n"
    else:
        for event_name, details in sorted(event_summary_details_map.items()):
            error_msg = details.get("error")
            warning_msg = details.get("warning")
            test_metrics = details.get("test_metrics")
            best_epoch = details.get("best_epoch", "N/A")

            if test_metrics and not error_msg: # Prioritize error over warning for table
                md_content += f"| {event_name:<17} | {test_metrics.get('accuracy', 0.0):.4f}         | {test_metrics.get('f1', 0.0):.4f}         | {test_metrics.get('precision', 0.0):.4f}           | {test_metrics.get('recall', 0.0):.4f}       | {best_epoch:<14} |\n"
                if warning_msg: md_content += f"| `↳ Warn: {warning_msg[:60]}...` | | | | | |\n" # Indent warning
                
                for key in avg_metrics:
                    if key in test_metrics and isinstance(test_metrics[key], (int, float)): 
                         avg_metrics[key].append(test_metrics[key])
                processed_event_count +=1
            elif error_msg:
                md_content += f"| {event_name:<17} | Error         | {error_msg[:15]}...    | Error          | Error       | {best_epoch:<14} |\n"
            elif warning_msg : # No error, but warning, and maybe no test_metrics
                 md_content += f"| {event_name:<17} | Warn          | {warning_msg[:15]}...  | Warn           | Warn        | {best_epoch:<14} |\n"
            else: 
                md_content += f"| {event_name:<17} | N/A           | N/A           | N/A              | N/A         | {best_epoch:<14} |\n"


    md_content += f"\n## Average Performance (over {processed_event_count} successfully processed events with Test Metrics)\n"
    if processed_event_count == 0 or not any(lst for lst in avg_metrics.values() if lst):
        md_content += "- No data to average (all individual event runs may have failed or yielded no usable test metrics).\n"
    else:
        for key in avg_metrics:
            if avg_metrics[key]: 
                md_content += f"- Average Test {key.capitalize()}: {np.mean(avg_metrics[key]):.4f}\n"
            else:
                md_content += f"- Average Test {key.capitalize()}: N/A (no data for this metric among successful runs)\n"


    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Aggregated results report saved to {md_path}")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TGN for Node Classification with Optuna and result saving.")
    parser.add_argument("--base_data_dir", type=str, default="data_tgn_fixed", help="Base directory of processed event folders.")
    parser.add_argument("--results_dir", type=str, default="RESULTS/TGN", help="Base directory to save all results.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs_optuna_trial", type=int, default=30, help="Max epochs per Optuna trial.")
    parser.add_argument("--epochs_for_final_run", type=int, default=50, help="Max epochs for final runs (best HPs or fixed).")
    # Default HPs (can be overridden by Optuna or specific report HPs)
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
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--project_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_layernorm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--leaky_relu_slope", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=True)


    parser.add_argument("--run_optuna", action="store_true", help="Enable Optuna hyperparameter search.")
    parser.add_argument("--study_name_prefix", type=str, default="tgn_pheme_study", help="Prefix for Optuna study names.")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--ablation_target", type=str, default="all",
                        help="Dataset to run on: 'all', a specific event name, or 'all_individual_events'.")
    # This flag is for generating an MD report using best HPs from 'all' dataset Optuna.
    parser.add_argument("--generate_aggregated_md_report", action="store_true", help="Generate aggregated MD report using best HPs from 'all' dataset Optuna.")
    
    parser.add_argument("--run_multi_event_performance_report", action="store_true",
                        help="Run on all individual events with specified fixed HPs and generate comprehensive multi-event performance plots and MD report.")

    args = parser.parse_args()
    args_dict = vars(args)

    RESULTS_BASE_DIR = Path(args.results_dir)
    RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    base_data_path_obj = Path(args.base_data_dir)
    
    all_individual_event_names = sorted([d.name for d in base_data_path_obj.iterdir() if d.is_dir() and d.name != "all"])
    if not all_individual_event_names and (args.ablation_target == "all_individual_events" or args.generate_aggregated_md_report or args.run_multi_event_performance_report):
        print(f"Warning: No individual event subdirectories found in {args.base_data_dir}. Some operations might be affected.")


    if args.run_multi_event_performance_report:
        print("\n--- Running Multi-Event Performance Report Mode ---")
        
        # Hyperparameters specified in the prompt/CLI for this report
        # These will override any defaults from argparse for this specific mode.
        fixed_hps_for_report_mode = {
            "lr": args_dict.get("lr", 0.00688), # Use CLI provided if available, else default
            "memory_dim": args_dict.get("memory_dim", 256),
            "time_dim": args_dict.get("time_dim", 64),
            "embedding_dim": args_dict.get("embedding_dim", 256),
            "dropout_rate": args_dict.get("dropout_rate", 0.20021),
            "projector_dropout_rate": args_dict.get("projector_dropout_rate", 0.21446),
            "grad_clip_norm": args_dict.get("grad_clip_norm", 4.9712),
            "batch_size": args_dict.get("batch_size", 128),
            # Ensure other relevant HPs from args_dict are included or use sensible defaults
            "project_features": args_dict.get("project_features", True),
            "use_layernorm": args_dict.get("use_layernorm", True),
            "leaky_relu_slope": args_dict.get("leaky_relu_slope", 0.1),
            "weight_decay": args_dict.get("weight_decay", 1e-5),
            "lr_scheduler_patience": args_dict.get("lr_scheduler_patience", 5),
            "early_stopping_patience": args_dict.get("early_stopping_patience", 10),
        }
        print(f"Using fixed HPs for this report: {fixed_hps_for_report_mode}")
        
        current_run_args_template = args_dict.copy()
        current_run_args_template.update(fixed_hps_for_report_mode) 
        current_run_args_template["epochs"] = args_dict["epochs_for_final_run"] 
        current_run_args_template["collect_epoch_wise_test_metrics"] = True # Crucial for epoch trend plots

        report_event_histories = {} # For epoch-wise data
        report_event_summary_details = {} # For best_run_details including best test metrics

        # Directory for individual run outputs of this report mode
        perf_report_runs_dir = RESULTS_BASE_DIR / "multi_event_performance_report_runs"
        perf_report_runs_dir.mkdir(parents=True, exist_ok=True)
        
        if not all_individual_event_names:
            print("No individual event datasets found to generate the performance report.")
        else:
            print(f"Will process events for report: {all_individual_event_names}")
            for event_name in all_individual_event_names:
                print(f"\nProcessing event: {event_name} for performance report...")
                event_data_path = base_data_path_obj / event_name
                if not event_data_path.exists() or not event_data_path.is_dir():
                    print(f"Data for event {event_name} not found or not a directory at {event_data_path}. Skipping.")
                    report_event_histories[event_name] = {"epochs": [], "error": "Data not found/not a dir"}
                    report_event_summary_details[event_name] = {"error": "Data not found/not a dir"}
                    continue

                current_run_args_event = current_run_args_template.copy()
                current_run_args_event["data_path"] = str(event_data_path)
                
                event_report_save_name_prefix = f"{event_name}_PerfReport" # For logging and folder name
                run_output = run_single_trial(current_run_args_event, trial=None, event_name_for_log=event_report_save_name_prefix)
                
                # Store epoch-wise history
                if run_output and run_output["training_history"]:
                    report_event_histories[event_name] = run_output["training_history"]
                else: # Handle missing history (e.g. trial skipped early)
                    error_msg_hist = "Run failed or no history produced."
                    if run_output and run_output.get("best_run_details", {}).get("error"):
                        error_msg_hist = run_output["best_run_details"]["error"]
                    report_event_histories[event_name] = {"epochs": [], "error": error_msg_hist}
                    print(f"Warning: Could not get full training history for {event_name}: {error_msg_hist}")

                # Store summary details (best_run_details)
                if run_output and run_output.get("best_run_details"):
                    report_event_summary_details[event_name] = run_output["best_run_details"]
                else:
                    report_event_summary_details[event_name] = {"error": "run_single_trial returned no best_run_details"}
                    print(f"Warning: Failed to get best_run_details for {event_name}")

                # Save individual event's detailed results (MD file, but skip individual plot_training_history)
                # Pass the fixed HPs used for this run, not the original args_dict HPs
                save_event_specific_results(
                    event_name=event_report_save_name_prefix, # Folder name for this specific run's artifacts
                    results_base_dir=perf_report_runs_dir,
                    run_details=run_output.get("best_run_details"),
                    training_history=run_output.get("training_history"), # Pass history for potential future use, even if not plotted here
                    hps_used=current_run_args_event, # These are the HPs actually used
                    optuna_study=None, 
                    args_for_the_run=current_run_args_event
                )


            # After processing all events, generate the aggregate plots and MD report
            if report_event_histories or report_event_summary_details:
                plot_dir = RESULTS_BASE_DIR / "multi_event_performance_report_visuals" # Centralized plots
                plot_dir.mkdir(parents=True, exist_ok=True)
                
                generate_multi_event_report_visualizations(
                    event_histories_map=report_event_histories,
                    event_summary_details_map=report_event_summary_details,
                    output_dir=plot_dir,
                    fixed_hps_for_title=fixed_hps_for_report_mode,
                    max_epochs_x_axis=args.epochs_for_final_run
                )
                
                # Also generate an aggregated MD report for this run
                save_aggregated_results_md(
                    results_dir=RESULTS_BASE_DIR, # Save MD report at the top level of results_dir
                    event_summary_details_map=report_event_summary_details,
                    best_overall_hps=fixed_hps_for_report_mode # Pass the HPs used for this report
                )
            else:
                print("No data collected from event runs, skipping aggregated plot and MD report generation.")

        print("\n--- Multi-event performance report generation finished. ---")

    else: # Original operational modes (Optuna, single fixed run, or Optuna + aggregated MD)
        best_overall_hps_from_all_study = None

        if args.run_optuna and args.generate_aggregated_md_report:
            print("\n--- STAGE 1: Optuna Study for 'all' dataset (for Best Overall Hyperparameters) ---")
            event_name_for_overall_hp = "all"
            path_for_overall_hp_data = base_data_path_obj / event_name_for_overall_hp
            
            if not path_for_overall_hp_data.exists():
                print(f"Error: '{event_name_for_overall_hp}' dataset directory not found at {path_for_overall_hp_data}. Cannot determine best overall HPs. Skipping aggregation.")
                args.generate_aggregated_md_report = False 
            else:
                study_args_for_all = args_dict.copy()
                study_args_for_all["data_path"] = str(path_for_overall_hp_data)
                study_args_for_all["epochs"] = args_dict["epochs_optuna_trial"] # Use optuna epochs
                study_args_for_all["collect_epoch_wise_test_metrics"] = False # No need for Optuna trials


                study_name_all = f"{args.study_name_prefix}_{event_name_for_overall_hp}_overall_hp_search"
                storage_name_all = f"sqlite:///{RESULTS_BASE_DIR / study_name_all}.db"
                
                study_all = optuna.create_study(study_name=study_name_all, storage=storage_name_all, direction="maximize", load_if_exists=True)
                
                try:
                    study_all.optimize(lambda trial: objective(trial, study_args_for_all, event_name_for_overall_hp), n_trials=args.n_trials, gc_after_trial=True)
                except ValueError as e: # Handle Optuna's specific error for changed categorical params
                    if "CategoricalDistribution does not support dynamic value space" in str(e):
                        print(f"Optuna Error for study '{study_name_all}': {e}\nThis usually means the choices for a categorical hyperparameter have changed since the study was first created.\nPlease delete the study database file '{storage_name_all}' or use a different study_name_prefix and try again.")
                        args.generate_aggregated_md_report = False 
                    else: raise e 
                
                if args.generate_aggregated_md_report and hasattr(study_all, 'best_trial') and study_all.best_trial: 
                    best_overall_hps_from_all_study = study_all.best_trial.params
                    print(f"Best overall HPs from '{event_name_for_overall_hp}' dataset study: {best_overall_hps_from_all_study}")

                    args_for_all_rerun = study_args_for_all.copy() 
                    args_for_all_rerun.update(best_overall_hps_from_all_study) 
                    args_for_all_rerun["epochs"] = args_dict["epochs_for_final_run"] # Final run epochs
                    args_for_all_rerun["collect_epoch_wise_test_metrics"] = True # Collect for detailed report of 'all'
                    
                    print(f"Rerunning '{event_name_for_overall_hp}' dataset with its best HPs for detailed results...")
                    all_dataset_final_run_results = run_single_trial(args_for_all_rerun, trial=None, event_name_for_log=f"{event_name_for_overall_hp}_BestHPs_Overall")
                    save_event_specific_results(
                        event_name=event_name_for_overall_hp, results_base_dir=RESULTS_BASE_DIR,
                        run_details=all_dataset_final_run_results["best_run_details"],
                        training_history=all_dataset_final_run_results["training_history"],
                        hps_used=best_overall_hps_from_all_study, optuna_study=study_all, args_for_the_run=args_for_all_rerun)
                elif args.generate_aggregated_md_report: 
                    print(f"Optuna study for '{event_name_for_overall_hp}' did not yield a best trial. Cannot determine best overall HPs.")
                    args.generate_aggregated_md_report = False

        target_event_folders_for_study = []
        if args.run_optuna: 
            if args.ablation_target == "all_individual_events":
                target_event_folders_for_study = all_individual_event_names
                if not target_event_folders_for_study:
                     print("Warning: --ablation_target is 'all_individual_events' but no individual event folders found (excluding 'all').")
            elif (base_data_path_obj / args.ablation_target).is_dir():
                target_event_folders_for_study = [args.ablation_target]
            else: # Specific target not 'all' and not a dir
                if args.ablation_target != "all": # If 'all' was specified, it's fine if it's not in individual events list
                    print(f"Error: Specified ablation target '{args.ablation_target}' not found in {args.base_data_dir} or is not a directory.")
                    exit(1) 
            
            if not target_event_folders_for_study and args.ablation_target != "all":
                print(f"No valid target event folders found for ablation based on '{args.ablation_target}'.")

            if target_event_folders_for_study:
                print(f"\n--- STAGE 2: Running Optuna Studies for Target Events: {target_event_folders_for_study} ---")
                for event_name in target_event_folders_for_study:
                    if event_name == "all" and args.generate_aggregated_md_report: 
                        print(f"Skipping Optuna study for 'all' in STAGE 2 as it was handled in STAGE 1 for overall HPs.")
                        continue

                    print(f"\n--- Optuna Study for Event: {event_name} ---")
                    event_data_path = base_data_path_obj / event_name
                    if not event_data_path.exists():
                        print(f"Warning: Data for event {event_name} not found at {event_data_path}. Skipping.")
                        continue

                    study_args_event = args_dict.copy()
                    study_args_event["data_path"] = str(event_data_path)
                    study_args_event["epochs"] = args_dict["epochs_optuna_trial"]
                    study_args_event["collect_epoch_wise_test_metrics"] = False


                    study_name_event = f"{args.study_name_prefix}_{event_name}"
                    storage_name_event = f"sqlite:///{RESULTS_BASE_DIR / study_name_event}.db"
                    study_event = optuna.create_study(study_name=study_name_event, storage=storage_name_event, direction="maximize", load_if_exists=True)
                    
                    try:
                        study_event.optimize(lambda trial: objective(trial, study_args_event, event_name), n_trials=args.n_trials, gc_after_trial=True)
                    except ValueError as e:
                        if "CategoricalDistribution does not support dynamic value space" in str(e):
                            print(f"Optuna Error for study '{study_name_event}': {e}\nPlease delete the study database file '{storage_name_event}' or use a different study_name_prefix and try again.")
                            continue 
                        else: raise e

                    if hasattr(study_event, 'best_trial') and study_event.best_trial:
                        best_trial_event_params = study_event.best_trial.params
                        args_for_event_rerun = study_args_event.copy()
                        args_for_event_rerun.update(best_trial_event_params)
                        args_for_event_rerun["epochs"] = args_dict["epochs_for_final_run"]
                        args_for_event_rerun["collect_epoch_wise_test_metrics"] = True # For detailed report
                        
                        print(f"Rerunning {event_name} with its best HPs for detailed results...")
                        event_final_run_results = run_single_trial(args_for_event_rerun, trial=None, event_name_for_log=f"{event_name}_BestHPs_Individual")
                        save_event_specific_results(
                            event_name=event_name, results_base_dir=RESULTS_BASE_DIR,
                            run_details=event_final_run_results["best_run_details"],
                            training_history=event_final_run_results["training_history"],
                            hps_used=best_trial_event_params, optuna_study=study_event, args_for_the_run=args_for_event_rerun)
                    else:
                        print(f"Optuna study for {event_name} did not yield a best trial. No results to save for best HPs.")
                        save_event_specific_results( # Save a record of the failed/empty study attempt
                            event_name=event_name, results_base_dir=RESULTS_BASE_DIR,
                            run_details={"error": "Optuna study did not find a best trial.", "hyperparameters": study_args_event},
                            training_history=None, hps_used=study_args_event, optuna_study=study_event, args_for_the_run=study_args_event
                        )
        elif not args.run_optuna and not args.run_multi_event_performance_report: # Single fixed run
            print("\n--- Single Fixed Run Mode (Not Optuna, Not Multi-Event Report) ---")
            event_name_fixed = args.ablation_target
            if args.ablation_target == "all_individual_events":
                print("Error: For a single fixed run (--run_optuna not set), --ablation_target cannot be 'all_individual_events'. Please specify a single event name or 'all'.")
                exit(1)
            
            fixed_run_data_path = base_data_path_obj / event_name_fixed
            if not fixed_run_data_path.exists():
                print(f"Error: Data for event {event_name_fixed} not found at {fixed_run_data_path}. Exiting.")
                exit(1)

            fixed_run_args = args_dict.copy() # Starts with CLI args as base
            fixed_run_args["data_path"] = str(fixed_run_data_path)
            fixed_run_args["epochs"] = args_dict["epochs_for_final_run"] 
            fixed_run_args["collect_epoch_wise_test_metrics"] = True # For detailed report


            print(f"Running fixed configuration for event: {event_name_fixed}")
            fixed_run_results = run_single_trial(fixed_run_args, trial=None, event_name_for_log=f"{event_name_fixed}_FixedRun")
            save_event_specific_results(
                event_name=event_name_fixed, results_base_dir=RESULTS_BASE_DIR,
                run_details=fixed_run_results["best_run_details"],
                training_history=fixed_run_results["training_history"],
                hps_used=fixed_run_args, # These are the HPs that were used
                optuna_study=None, args_for_the_run=fixed_run_args)

        if args.generate_aggregated_md_report: # This is for the Optuna-driven 'all' HPs report
            if not best_overall_hps_from_all_study:
                print("\n--- STAGE 3: Skipping Aggregated MD Report ---")
                print("Reason: Best overall hyperparameters were not determined (e.g., 'all' dataset study failed, was not run, or yielded no best trial).")
            elif not all_individual_event_names:
                print("\n--- STAGE 3: Skipping Aggregated MD Report ---")
                print("Reason: No individual event datasets found to run aggregation on.")
            else:
                print("\n--- STAGE 3: Generating Aggregated MD Report using Best Overall Hyperparameters from 'all' Study ---")
                aggregated_event_summary_details_list_for_md = {} # Use a dict for save_aggregated_results_md
                for event_name in all_individual_event_names:
                    print(f"  Evaluating event: {event_name} with best overall HPs for aggregation MD report...")
                    event_data_path_for_agg = base_data_path_obj / event_name
                    if not event_data_path_for_agg.exists():
                        print(f"Warning: Data for event {event_name} not found at {event_data_path_for_agg}. Skipping for aggregation.")
                        aggregated_event_summary_details_list_for_md[event_name] = {"error": "Data not found"}
                        continue

                    args_for_agg_run = args_dict.copy() 
                    args_for_agg_run.update(best_overall_hps_from_all_study) 
                    args_for_agg_run["data_path"] = str(event_data_path_for_agg)
                    args_for_agg_run["epochs"] = args_dict["epochs_for_final_run"]
                    args_for_agg_run["collect_epoch_wise_test_metrics"] = False # No need for epoch-wise test for this specific MD eval

                    event_run_for_agg_output = run_single_trial(args_for_agg_run, trial=None, event_name_for_log=f"{event_name}_AggMD_Eval")
                    
                    run_details_agg = event_run_for_agg_output.get("best_run_details")
                    if run_details_agg:
                        aggregated_event_summary_details_list_for_md[event_name] = run_details_agg
                    else:
                        aggregated_event_summary_details_list_for_md[event_name] = {"error": "No best_run_details from trial"}
                        print(f"Warning: Could not retrieve valid run details for {event_name} during aggregation MD run.")

                if aggregated_event_summary_details_list_for_md: 
                    save_aggregated_results_md(
                        results_dir=RESULTS_BASE_DIR,
                        event_summary_details_map=aggregated_event_summary_details_list_for_md,
                        best_overall_hps=best_overall_hps_from_all_study
                    )
                else:
                    print("No individual event metrics were collected for the aggregation MD report.")
        
        print("\n--- All processing finished. ---")