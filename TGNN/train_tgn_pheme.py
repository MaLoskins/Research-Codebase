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
        edge_features_all_events = torch.empty(0, node_features.shape[1] if node_features.ndim > 1 and node_features.shape[1] > 0 else 0, dtype=torch.float)
    else:
        events_df = pd.read_csv(events_csv_path)
        if events_df.empty:
            print(f"Warning: events.csv in {data_dir_path} is empty. Assuming no events.")
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

    use_tqdm = trial is None or epoch_num < 2
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
                print(f"Warning: NaN loss detected in training for {event_name_for_log}. Skipping batch.")
                mem_state_for_current_batch = mem_state_for_current_batch.detach()
                last_updates_state_for_current_batch = last_updates_state_for_current_batch.detach()
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
                   split_name="Eval", trial=None, epoch_num=0, event_name_for_log=""):
    model.eval(); preds_list, true_list = [], []
    mem_eval = mem_start_eval.detach().clone()
    last_updates_eval = last_updates_start_eval.detach().clone()
    raw_node_feats_dev = raw_node_feats_cpu.to(device)

    use_tqdm = trial is None or epoch_num < 2
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

    if not preds_list:
        num_classes = len(torch.unique(node_labels_cpu[node_labels_cpu != -1]))
        if num_classes == 0: num_classes = 2
        return {"accuracy":0.0,"f1":0.0,"precision":0.0,"recall":0.0,"conf_matrix":np.zeros((num_classes,num_classes)).tolist()}, mem_eval, last_updates_eval

    preds, trues = np.concatenate(preds_list), np.concatenate(true_list)
    unique_true_labels = np.unique(trues)
    num_classes_for_cm = 2
    if len(unique_true_labels) > 0 :
        max_label_val = unique_true_labels.max()
        if max_label_val >=0 : num_classes_for_cm = int(max_label_val) + 1
    avg_metric = 'binary' if num_classes_for_cm <= 2 else 'weighted'

    metrics = {
        "accuracy": accuracy_score(trues, preds), "f1": f1_score(trues, preds, average=avg_metric, zero_division=0),
        "precision": precision_score(trues, preds, average=avg_metric, zero_division=0),
        "recall": recall_score(trues, preds, average=avg_metric, zero_division=0),
        "conf_matrix": confusion_matrix(trues, preds, labels=np.arange(num_classes_for_cm)).tolist()
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

    if trial is None:
        print(f"[{event_name_for_log}] Using device: {device}")
        hps_to_print = {k: args_dict[k] for k in ['lr', 'memory_dim', 'time_dim', 'embedding_dim', 'dropout_rate', 'batch_size', 'project_features', 'use_layernorm', 'leaky_relu_slope'] if k in args_dict}
        print(f"[{event_name_for_log}] Running with HPs: {hps_to_print}")

    best_run_details_for_this_trial = {"hyperparameters": args_dict.copy()} # Initialize without default error

    (raw_node_feats_cpu, node_labels_cpu, s_nodes_cpu, d_nodes_cpu, ts_cpu, edge_feats_cpu, n_nodes) = \
        load_tgn_data_from_path(Path(args_dict["data_path"]))

    if n_nodes == 0 or (s_nodes_cpu.nelement() == 0 and d_nodes_cpu.nelement() == 0):
        msg = f"No nodes ({n_nodes}) or events (s:{s_nodes_cpu.nelement()},d:{d_nodes_cpu.nelement()}) in data for {args_dict['data_path']}."
        print(f"[{event_name_for_log}] {msg} Skipping trial.")
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
            if trial is None: print(f"  [{event_name_for_log}] Num classes: {num_classes_data}. Counts: {cls_counts.tolist()}")
            if num_classes_data > 0 and cls_counts.min() > 0 and args_dict.get("use_class_weights", True):
                cls_weights = (1. / cls_counts.float()).to(device)
                if trial is None: print(f"  [{event_name_for_log}] Using class weights: {cls_weights.cpu().numpy()}")
            else: num_classes_data = max(2, num_classes_data)
    else:
        msg = "No valid labels (all are -1 or empty)."
        if trial is None: print(f"[{event_name_for_log}] Warning: {msg} Defaulting to 2 classes, no weights.")
        # For Optuna, if no valid labels, it's a failure for this trial configuration
        if trial: 
            best_run_details_for_this_trial["error"] = msg
            return { "value_for_optuna": 0.0, "best_run_details": best_run_details_for_this_trial, "training_history": None}
        # For fixed run, it might proceed but likely won't train meaningfully. We'll let it try and report error later if no val.
        # Or, we can add the error here for fixed runs too.
        # best_run_details_for_this_trial["error"] = msg # Optionally add for fixed runs too

    criterion = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=-1)
    train_idx, val_idx, test_idx = create_event_splits(len(s_nodes_cpu), args_dict["train_ratio"], args_dict["val_ratio"])

    if len(train_idx) == 0 and len(val_idx) == 0: 
         msg = f"Not enough events to create train/val splits ({len(s_nodes_cpu)} total) for {args_dict['data_path']}."
         print(f"[{event_name_for_log}] {msg} Skipping.")
         best_run_details_for_this_trial["error"] = msg
         return {
            "value_for_optuna": 0.0,
            "best_run_details": best_run_details_for_this_trial,
            "training_history": None
        }

    if trial is None:
        print(f"  [{event_name_for_log}] Dataset: {args_dict['data_path']}\n  Nodes: {n_nodes}, Events: {len(s_nodes_cpu)}")
        print(f"  [{event_name_for_log}] Train/Val/Test Events: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    active_val_nodes_cpu = get_active_nodes_in_split(s_nodes_cpu, d_nodes_cpu, val_idx, node_labels_cpu)
    active_test_nodes_cpu = get_active_nodes_in_split(s_nodes_cpu, d_nodes_cpu, test_idx, node_labels_cpu)
    if trial is None: print(f"  [{event_name_for_log}] Active Val Nodes: {len(active_val_nodes_cpu)}, Active Test Nodes: {len(active_test_nodes_cpu)}")

    current_batch_size = args_dict["batch_size"]
    if len(train_idx) > 0 and args_dict["batch_size"] > len(train_idx):
        current_batch_size = max(1, len(train_idx))
        if trial: print(f"Optuna Trial {trial.number} ({event_name_for_log}): Adjusted batch size to {current_batch_size}.")

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
    epoch_nums_history, train_losses_history, val_f1s_history, val_accs_history = [], [], [], []
    
    global_memory_state = torch.zeros((n_nodes, args_dict["memory_dim"]), device=device)
    global_last_update_timestamps = torch.zeros(n_nodes, device=device)
    num_epochs_to_run = args_dict["epochs"]

    for epoch in range(1, num_epochs_to_run + 1):
        is_detailed_log_epoch = trial is None or epoch <= 2 
        
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
            if is_detailed_log_epoch: print(f"  [{event_name_for_log}] Avg Train Loss: {avg_train_loss:.4f}")
            mem_for_eval_and_next_epoch = mem_after_train
            ts_for_eval_and_next_epoch = ts_after_train
        elif is_detailed_log_epoch: 
            print(f"  [{event_name_for_log}] No training events for this epoch.")
        
        if trial is None:
            epoch_nums_history.append(epoch)
            train_losses_history.append(avg_train_loss)

        current_epoch_val_f1 = -1.0; current_epoch_val_metrics = None
        if len(val_idx) > 0 and len(active_val_nodes_cpu) > 0:
            val_metrics, _, _ = run_evaluation( 
                model, raw_node_feats_cpu, s_nodes_cpu, d_nodes_cpu, ts_cpu, edge_feats_cpu, 
                mem_for_eval_and_next_epoch, ts_for_eval_and_next_epoch, 
                val_idx, node_labels_cpu, args_dict["batch_size"], active_val_nodes_cpu, device, 
                "Val", trial, epoch, event_name_for_log)
            if is_detailed_log_epoch:
                print(f"  [{event_name_for_log}] Val - Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            current_epoch_val_f1 = val_metrics['f1']
            current_epoch_val_metrics = val_metrics
            scheduler.step(float(current_epoch_val_f1) if current_epoch_val_f1 is not None else 0.0)
            if trial is None: 
                val_f1s_history.append(float(current_epoch_val_f1) if current_epoch_val_f1 is not None else 0.0)
                val_accs_history.append(float(val_metrics['accuracy']) if val_metrics and 'accuracy' in val_metrics else 0.0)
        else:
            if is_detailed_log_epoch: print(f"  [{event_name_for_log}] Skipping validation (no val events or no active val nodes).")
            if trial is None: 
                val_f1s_history.append(0.0)
                val_accs_history.append(0.0)

        current_epoch_test_metrics = None 
        if current_epoch_val_f1 is not None and float(current_epoch_val_f1) > best_val_f1_for_run : 
            best_val_f1_for_run = float(current_epoch_val_f1) 
            no_improve_epochs_for_run = 0
            best_run_details_for_this_trial["best_epoch"] = epoch
            if current_epoch_val_metrics:
                best_run_details_for_this_trial["val_metrics"] = current_epoch_val_metrics
            
            if trial is None and (len(test_idx) > 0 and len(active_test_nodes_cpu) > 0):
                if is_detailed_log_epoch: print(f"  [{event_name_for_log}] New best Val F1: {best_val_f1_for_run:.4f}. Evaluating on Test...")
                test_metrics_output, _, _ = run_evaluation(
                    model, raw_node_feats_cpu, s_nodes_cpu, d_nodes_cpu, ts_cpu, edge_feats_cpu, 
                    mem_for_eval_and_next_epoch, ts_for_eval_and_next_epoch, 
                    test_idx, node_labels_cpu, args_dict["batch_size"], active_test_nodes_cpu, device, 
                    "Test", trial, epoch, event_name_for_log)
                if is_detailed_log_epoch:
                    print(f"  [{event_name_for_log}] Test - Acc: {test_metrics_output['accuracy']:.4f}, F1: {test_metrics_output['f1']:.4f}")
                current_epoch_test_metrics = test_metrics_output 
            if current_epoch_test_metrics: 
                best_run_details_for_this_trial["test_metrics"] = current_epoch_test_metrics
        else:
            no_improve_epochs_for_run += 1
        
        global_memory_state = mem_for_eval_and_next_epoch.detach()
        global_last_update_timestamps = ts_for_eval_and_next_epoch.detach()

        if no_improve_epochs_for_run >= args_dict["early_stopping_patience"] and best_val_f1_for_run > -1.0 : 
            if is_detailed_log_epoch: print(f"[{event_name_for_log}] Early stopping after {epoch} epochs."); 
            break
        
        if trial:
            trial.report(best_val_f1_for_run if best_val_f1_for_run != -1.0 else 0.0, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if best_val_f1_for_run == -1.0 and "error" not in best_run_details_for_this_trial:
        best_run_details_for_this_trial["error"] = "No successful validation epoch (val_f1 remained at -1.0 or was never computed)."

    if trial is None: 
        print(f"\n--- [{event_name_for_log}] Training Complete (Fixed/Final Run) ---")
        if "error" in best_run_details_for_this_trial:
            print(f"Run Issue for {event_name_for_log}: {best_run_details_for_this_trial['error']}")
        elif "val_metrics" in best_run_details_for_this_trial : 
            print(f"Best Val F1 for {event_name_for_log}: {best_val_f1_for_run:.4f} at epoch {best_run_details_for_this_trial.get('best_epoch', 'N/A')}")
            if best_run_details_for_this_trial.get("test_metrics"):
                tm = best_run_details_for_this_trial["test_metrics"]
                print(f"Corresp. Test Metrics: Acc: {tm['accuracy']:.4f}, F1: {tm['f1']:.4f}")
            else:
                 print(f"Test metrics were not generated for {event_name_for_log} (e.g., no test data or test eval skipped).")
        else: 
            print(f"No validation metrics recorded for {event_name_for_log}. Check data and training progression.")

    training_history = None
    if trial is None:
        training_history = {
            "epochs": epoch_nums_history, "train_loss": train_losses_history,
            "val_f1": val_f1s_history, "val_acc": val_accs_history
        }
        if len(epoch_nums_history) < num_epochs_to_run and len(epoch_nums_history) > 0:
            last_train_loss = train_losses_history[-1] if train_losses_history else 0
            last_val_f1 = val_f1s_history[-1] if val_f1s_history else 0
            last_val_acc = val_accs_history[-1] if val_accs_history else 0
            for e_i in range(len(epoch_nums_history) + 1, num_epochs_to_run + 1):
                epoch_nums_history.append(e_i)
                train_losses_history.append(last_train_loss)
                val_f1s_history.append(last_val_f1)
                val_accs_history.append(last_val_acc)

    return {
        "value_for_optuna": best_val_f1_for_run if best_val_f1_for_run != -1.0 else 0.0,
        "best_run_details": best_run_details_for_this_trial,
        "training_history": training_history
    }
    
def objective(trial, base_args_dict, event_name_for_log):
    current_trial_args = base_args_dict.copy()
    current_trial_args["epochs"] = current_trial_args["epochs_optuna_trial"] 
    results = run_single_trial(current_trial_args, trial=trial, event_name_for_log=f"{event_name_for_log}_OptunaTrial")
    return results["value_for_optuna"]

# --- Plotting and Saving ---
def plot_training_history(history, save_path, event_name):
    if history is None or not history.get("epochs") or not history.get("train_loss") : # Added check for train_loss
        print(f"No or incomplete training history to plot for {event_name}.")
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.text(0.5, 0.5, "No training data available for plotting.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f"Training History for {event_name} (No Data)")
        try:
            plt.savefig(save_path)
        except Exception as e:
            print(f"Error saving placeholder plot: {e}")
        plt.close(fig)
        return

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True) # sharex for common epoch axis
    fig.suptitle(f"Training History for Event: {event_name}", fontsize=16)

    epochs = history["epochs"]
    
    axs[0].plot(epochs, history["train_loss"], label="Training Loss", color="royalblue", marker='.')
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Training Loss over Epochs")
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.7)

    if history.get("val_f1") and history.get("val_acc"): # Check if val metrics exist
        axs[1].plot(epochs, history["val_f1"], label="Validation F1-Score", color="forestgreen", marker='.')
        axs[1].plot(epochs, history["val_acc"], label="Validation Accuracy", color="coral", linestyle="--", marker='.')
        axs[1].set_ylabel("Metric Value")
        axs[1].set_title("Validation Metrics over Epochs")
        axs[1].legend()
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].set_ylim(0, 1.05) 
    else:
        axs[1].text(0.5, 0.5, "Validation metrics not available.", horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        axs[1].set_title("Validation Metrics over Epochs (No Data)")
        
    axs[1].set_xlabel("Epoch") # Set xlabel only for the bottom plot due to sharex

    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    try:
        plt.savefig(save_path)
        print(f"Training plots saved to {save_path}")
    except Exception as e:
        print(f"Error saving training plot to {save_path}: {e}")
    plt.close(fig)

def save_event_specific_results(event_name, results_base_dir, run_details, training_history, hps_used, optuna_study, args_for_the_run):
    event_dir = Path(results_base_dir) / event_name
    event_dir.mkdir(parents=True, exist_ok=True)

    md_content = f"# TGN Results for Event: {event_name}\n\n"
    actual_hps_used_in_run = run_details.get("hyperparameters", {}) if run_details else {}

    if run_details and "error" in run_details:
        md_content += f"**Run failed or was skipped:** {run_details['error']}\n\n"
        md_content += "## Hyperparameters Used (Attempted):\n"
        # Use HPs from the run_details if available, otherwise fall back to hps_used (which might be Optuna's best or initial args)
        hps_to_display_on_error = actual_hps_used_in_run if actual_hps_used_in_run else hps_used
        for k, v in hps_to_display_on_error.items():
            if k in ['lr', 'memory_dim', 'time_dim', 'embedding_dim', 'dropout_rate', 'batch_size', 'project_features', 'use_layernorm', 'leaky_relu_slope', 'projector_dropout_rate', 'grad_clip_norm']:
                 md_content += f"- {k}: {v}\n"
    else:
        md_content += "## Hyperparameters Used (for this successful run):\n"
        # For successful runs, actual_hps_used_in_run should be populated from the run_single_trial
        for k, v in actual_hps_used_in_run.items():
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
        else:
            md_content += "Validation metrics not available for this run.\n"


        if run_details and "test_metrics" in run_details and run_details["test_metrics"]:
            tm = run_details["test_metrics"]
            md_content += "\n### Corresponding Test Metrics:\n"
            md_content += f"- Test Accuracy: {tm.get('accuracy', 0.0):.4f}\n"
            md_content += f"- Test F1-Score: {tm.get('f1', 0.0):.4f}\n"
            md_content += f"- Test Precision: {tm.get('precision', 0.0):.4f}\n"
            md_content += f"- Test Recall: {tm.get('recall', 0.0):.4f}\n"
            if "conf_matrix" in tm: md_content += f"- Test Confusion Matrix:\n```\n{np.array(tm['conf_matrix'])}\n```\n"
        else:
            md_content += "\nTest metrics not available or not evaluated for this run.\n"

    with open(event_dir / "results.md", "w") as f:
        f.write(md_content)
    print(f"Results summary saved to {event_dir / 'results.md'}")

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
                fig_parallel.write_image(event_dir / "parallel_coordinate_plot.png")
                print(f"Optuna parallel coordinate plot saved to {event_dir / 'parallel_coordinate_plot.png'}")

            fig_history = plot_optimization_history(optuna_study)
            fig_history.write_image(event_dir / "optuna_optimization_history.png")
            print(f"Optuna optimization history plot saved to {event_dir / 'optuna_optimization_history.png'}")

            if len(optuna_study.trials) > 1 and any(t.state == optuna.trial.TrialState.COMPLETE for t in optuna_study.trials):
                try:
                    fig_importance = plot_param_importances(optuna_study)
                    fig_importance.write_image(event_dir / "optuna_param_importances.png")
                    print(f"Optuna param importances plot saved to {event_dir / 'optuna_param_importances.png'}")
                except (ValueError, IndexError) as ve: # IndexError for some plotly versions with single param
                    print(f"Could not generate param importances plot for {event_name}: {ve}")
        except ImportError:
             print("Plotly/Kaleido not installed. Skipping Optuna plot generation. pip install plotly kaleido")
        except Exception as e:
            print(f"Warning: Could not generate/save some Optuna plots for {event_name}: {e}")

def save_aggregated_results_md(results_dir, event_metrics_list, best_overall_hps):
    md_path = Path(results_dir) / "results.md"
    md_content = "# TGN Aggregated Results\n\n"
    md_content += "## Best Overall Hyperparameters (from 'all' dataset study):\n"
    if best_overall_hps:
        for k, v in best_overall_hps.items():
             if k in ['lr', 'memory_dim', 'time_dim', 'embedding_dim', 'dropout_rate', 'projector_dropout_rate', 'grad_clip_norm', 'batch_size', 'project_features', 'use_layernorm', 'leaky_relu_slope']:
                md_content += f"- {k}: {v}\n"
    else:
        md_content += "- Not available or 'all' dataset study was not run/successful.\n"
    md_content += "\n"

    md_content += "## Performance on Individual Events (using Best Overall HPs)\n\n"
    md_content += "| Event Name        | Test Accuracy | Test F1-Score | Test Precision | Test Recall |\n"
    md_content += "|-------------------|---------------|---------------|----------------|-------------|\n"

    avg_metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}
    if not event_metrics_list:
        md_content += "| No events processed or no successful runs for aggregation | N/A | N/A | N/A | N/A |\n"
    else:
        for item in event_metrics_list:
            event_name = item["event_name"]
            metrics = item.get("metrics") # Use .get for safety
            if metrics and isinstance(metrics, dict): 
                md_content += f"| {event_name:<17} | {metrics.get('accuracy', 0.0):.4f}        | {metrics.get('f1', 0.0):.4f}         | {metrics.get('precision', 0.0):.4f}           | {metrics.get('recall', 0.0):.4f}        |\n"
                for key in avg_metrics:
                    if key in metrics and isinstance(metrics[key], (int, float)): # Ensure value is numeric
                         avg_metrics[key].append(metrics[key])
            else:
                md_content += f"| {event_name:<17} | N/A           | N/A           | N/A              | N/A         |\n"


    md_content += "\n## Average Performance (using Best Overall HPs)\n"
    if not event_metrics_list or not any(lst for lst in avg_metrics.values() if lst):
        md_content += "- No data to average (all individual event runs may have failed or yielded no metrics).\n"
    else:
        for key in avg_metrics:
            if avg_metrics[key]: 
                md_content += f"- Average Test {key.capitalize()}: {np.mean(avg_metrics[key]):.4f}\n"
            else:
                md_content += f"- Average Test {key.capitalize()}: N/A (no data for this metric)\n"


    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Aggregated results saved to {md_path}")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TGN for Node Classification with Optuna and result saving.")
    parser.add_argument("--base_data_dir", type=str, default="data_tgn_fixed", help="Base directory of processed event folders.")
    parser.add_argument("--results_dir", type=str, default="RESULTS/TGN", help="Base directory to save all results.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs_optuna_trial", type=int, default=30, help="Max epochs per Optuna trial.")
    parser.add_argument("--epochs_for_final_run", type=int, default=50, help="Max epochs for final runs (best HPs or fixed).")
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
    parser.add_argument("--generate_aggregated_report", action="store_true", help="Generate aggregated report using best HPs from 'all' dataset.")

    args = parser.parse_args()
    args_dict = vars(args)

    RESULTS_BASE_DIR = Path(args.results_dir)
    RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    base_data_path_obj = Path(args.base_data_dir)
    
    all_individual_event_names = [d.name for d in base_data_path_obj.iterdir() if d.is_dir() and d.name != "all"]
    if not all_individual_event_names and args.ablation_target == "all_individual_events": # Check if needed
        print(f"Warning: No individual event subdirectories found in {args.base_data_dir} for 'all_individual_events' target.")


    best_overall_hps_from_all_study = None

    if args.run_optuna and args.generate_aggregated_report:
        print("\n--- STAGE 1: Optuna Study for 'all' dataset (for Best Overall Hyperparameters) ---")
        event_name_for_overall_hp = "all"
        path_for_overall_hp_data = base_data_path_obj / event_name_for_overall_hp
        
        if not path_for_overall_hp_data.exists():
            print(f"Error: '{event_name_for_overall_hp}' dataset directory not found at {path_for_overall_hp_data}. Cannot determine best overall HPs. Skipping aggregation.")
            args.generate_aggregated_report = False 
        else:
            study_args_for_all = args_dict.copy()
            study_args_for_all["data_path"] = str(path_for_overall_hp_data)
            study_args_for_all["epochs"] = args_dict["epochs_optuna_trial"]

            study_name_all = f"{args.study_name_prefix}_{event_name_for_overall_hp}_overall_hp_search"
            storage_name_all = f"sqlite:///{RESULTS_BASE_DIR / study_name_all}.db"
            # To avoid CategoricalDistribution error when re-running with changed choices, delete old db or change study name.
            # For simplicity in automated runs, let's just ensure load_if_exists=True handles it, but recommend deleting for major changes.
            study_all = optuna.create_study(study_name=study_name_all, storage=storage_name_all, direction="maximize", load_if_exists=True)
            
            # Check if study has incompatible parameters if it's old
            if study_all.trials:
                first_trial_params = study_all.trials[0].params.keys()
                # Example check for one categorical param, you might need more robust checks
                # This is a bit tricky to auto-detect perfectly without knowing exact previous structure
                # For now, user is advised to delete DB if CategoricalDistribution error occurs.
                pass

            try:
                study_all.optimize(lambda trial: objective(trial, study_args_for_all, event_name_for_overall_hp), n_trials=args.n_trials, gc_after_trial=True)
            except ValueError as e:
                if "CategoricalDistribution does not support dynamic value space" in str(e):
                    print(f"Optuna Error for study '{study_name_all}': {e}")
                    print(f"This usually means the choices for a categorical hyperparameter (e.g., memory_dim) have changed since the study was first created.")
                    print(f"Please delete the study database file '{storage_name_all}' or use a different study_name_prefix and try again.")
                    args.generate_aggregated_report = False # Cannot proceed
                else:
                    raise e # Re-raise other ValueErrors
            
            if args.generate_aggregated_report and study_all.best_trial: # Check flag again as it might have been set to False
                best_overall_hps_from_all_study = study_all.best_trial.params
                print(f"Best overall HPs from '{event_name_for_overall_hp}' dataset study: {best_overall_hps_from_all_study}")

                args_for_all_rerun = study_args_for_all.copy()
                args_for_all_rerun.update(best_overall_hps_from_all_study)
                args_for_all_rerun["epochs"] = args_dict["epochs_for_final_run"] 
                
                print(f"Rerunning '{event_name_for_overall_hp}' dataset with its best HPs for detailed results...")
                all_dataset_final_run_results = run_single_trial(args_for_all_rerun, trial=None, event_name_for_log=f"{event_name_for_overall_hp}_BestHPs_Overall")
                save_event_specific_results(
                    event_name=event_name_for_overall_hp, results_base_dir=RESULTS_BASE_DIR,
                    run_details=all_dataset_final_run_results["best_run_details"],
                    training_history=all_dataset_final_run_results["training_history"],
                    hps_used=best_overall_hps_from_all_study, optuna_study=study_all, args_for_the_run=args_for_all_rerun)
            elif args.generate_aggregated_report: # if flag was true but no best_trial
                print(f"Optuna study for '{event_name_for_overall_hp}' did not yield a best trial. Cannot determine best overall HPs.")
                args.generate_aggregated_report = False

    target_event_folders_for_study = []
    if args.run_optuna:
        if args.ablation_target == "all_individual_events":
            target_event_folders_for_study = all_individual_event_names
            if not target_event_folders_for_study:
                 print("Warning: --ablation_target is 'all_individual_events' but no individual event folders found (excluding 'all').")
        elif (base_data_path_obj / args.ablation_target).is_dir():
            target_event_folders_for_study = [args.ablation_target]
        else:
            print(f"Error: Specified ablation target '{args.ablation_target}' not found in {args.base_data_dir} or is not a directory. Exiting.")
            if args.ablation_target != "all": # "all" could be handled by overall HP search if generate_aggregated_report is true
                 exit(1)
        
        if not target_event_folders_for_study and args.ablation_target != "all": # If target was specific but not found
            print(f"No valid target event folders found for ablation based on '{args.ablation_target}'.")

        if target_event_folders_for_study: # Only proceed if there are targets
            print(f"\n--- STAGE 2: Running Optuna Studies for Target Events: {target_event_folders_for_study} ---")
            for event_name in target_event_folders_for_study:
                if event_name == "all" and args.generate_aggregated_report:
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

                study_name_event = f"{args.study_name_prefix}_{event_name}"
                storage_name_event = f"sqlite:///{RESULTS_BASE_DIR / study_name_event}.db"
                study_event = optuna.create_study(study_name=study_name_event, storage=storage_name_event, direction="maximize", load_if_exists=True)
                
                try:
                    study_event.optimize(lambda trial: objective(trial, study_args_event, event_name), n_trials=args.n_trials, gc_after_trial=True)
                except ValueError as e:
                    if "CategoricalDistribution does not support dynamic value space" in str(e):
                        print(f"Optuna Error for study '{study_name_event}': {e}")
                        print(f"Please delete the study database file '{storage_name_event}' or use a different study_name_prefix and try again.")
                        continue # Skip to next event
                    else:
                        raise e

                if study_event.best_trial:
                    best_trial_event_params = study_event.best_trial.params
                    args_for_event_rerun = study_args_event.copy()
                    args_for_event_rerun.update(best_trial_event_params)
                    args_for_event_rerun["epochs"] = args_dict["epochs_for_final_run"]
                    
                    print(f"Rerunning {event_name} with its best HPs for detailed results...")
                    event_final_run_results = run_single_trial(args_for_event_rerun, trial=None, event_name_for_log=f"{event_name}_BestHPs_Individual")
                    save_event_specific_results(
                        event_name=event_name, results_base_dir=RESULTS_BASE_DIR,
                        run_details=event_final_run_results["best_run_details"],
                        training_history=event_final_run_results["training_history"],
                        hps_used=best_trial_event_params, optuna_study=study_event, args_for_the_run=args_for_event_rerun)
                else:
                    print(f"Optuna study for {event_name} did not yield a best trial. No results to save for best HPs.")
                    save_event_specific_results(
                        event_name=event_name, results_base_dir=RESULTS_BASE_DIR,
                        run_details={"error": "Optuna study did not find a best trial.", "hyperparameters": study_args_event},
                        training_history=None, hps_used=study_args_event, optuna_study=study_event, args_for_the_run=study_args_event
                    )
    elif not args.run_optuna: 
        print("\n--- STAGE 2: Single Fixed Run ---")
        event_name_fixed = args.ablation_target
        if args.ablation_target == "all_individual_events":
            print("Error: For a single fixed run (--run_optuna not set), --ablation_target cannot be 'all_individual_events'. Please specify a single event name or 'all'.")
            exit(1)
        
        fixed_run_data_path = base_data_path_obj / event_name_fixed
        if not fixed_run_data_path.exists():
            print(f"Error: Data for event {event_name_fixed} not found at {fixed_run_data_path}. Exiting.")
            exit(1)

        fixed_run_args = args_dict.copy()
        fixed_run_args["data_path"] = str(fixed_run_data_path)
        fixed_run_args["epochs"] = args_dict["epochs_for_final_run"] 

        print(f"Running fixed configuration for event: {event_name_fixed}")
        fixed_run_results = run_single_trial(fixed_run_args, trial=None, event_name_for_log=f"{event_name_fixed}_FixedRun")
        save_event_specific_results(
            event_name=event_name_fixed, results_base_dir=RESULTS_BASE_DIR,
            run_details=fixed_run_results["best_run_details"],
            training_history=fixed_run_results["training_history"],
            hps_used=fixed_run_args, 
            optuna_study=None, args_for_the_run=fixed_run_args)

    if args.generate_aggregated_report:
        if not best_overall_hps_from_all_study:
            print("\n--- STAGE 3: Skipping Aggregated Report ---")
            print("Reason: Best overall hyperparameters were not determined (e.g., 'all' dataset study failed, was not run, or yielded no best trial).")
        elif not all_individual_event_names:
            print("\n--- STAGE 3: Skipping Aggregated Report ---")
            print("Reason: No individual event datasets found to run aggregation on.")
        else:
            print("\n--- STAGE 3: Generating Aggregated Report using Best Overall Hyperparameters ---")
            aggregated_event_metrics_list = []
            for event_name in all_individual_event_names:
                print(f"  Evaluating event: {event_name} with best overall HPs...")
                event_data_path_for_agg = base_data_path_obj / event_name
                if not event_data_path_for_agg.exists():
                    print(f"Warning: Data for event {event_name} not found at {event_data_path_for_agg}. Skipping for aggregation.")
                    aggregated_event_metrics_list.append({"event_name": event_name, "metrics": None}) 
                    continue

                args_for_agg_run = args_dict.copy()
                args_for_agg_run.update(best_overall_hps_from_all_study) 
                args_for_agg_run["data_path"] = str(event_data_path_for_agg)
                args_for_agg_run["epochs"] = args_dict["epochs_for_final_run"]

                event_run_for_agg_output = run_single_trial(args_for_agg_run, trial=None, event_name_for_log=f"{event_name}_AggEval")
                
                current_event_test_metrics = None
                # Check if the run was successful and test metrics are available
                if event_run_for_agg_output and event_run_for_agg_output.get("best_run_details") and \
                   "error" not in event_run_for_agg_output["best_run_details"] and \
                   "test_metrics" in event_run_for_agg_output["best_run_details"] and \
                   event_run_for_agg_output["best_run_details"]["test_metrics"] is not None:
                    current_event_test_metrics = event_run_for_agg_output["best_run_details"]["test_metrics"]
                
                if current_event_test_metrics:
                     aggregated_event_metrics_list.append({
                        "event_name": event_name,
                        "metrics": current_event_test_metrics
                    })
                else:
                    err_msg = event_run_for_agg_output.get("best_run_details", {}).get("error", "Unknown issue, no test metrics.")
                    print(f"Warning: Could not retrieve valid test metrics for {event_name} during aggregation run. Details: {err_msg}")
                    aggregated_event_metrics_list.append({"event_name": event_name, "metrics": None, "error": err_msg}) 

            if aggregated_event_metrics_list: # Check if list is not empty before saving
                save_aggregated_results_md(
                    results_dir=RESULTS_BASE_DIR,
                    event_metrics_list=aggregated_event_metrics_list,
                    best_overall_hps=best_overall_hps_from_all_study
                )
            else:
                print("No individual event metrics were collected for the aggregation report (e.g., all individual runs failed or no individual datasets).")
    
    print("\n--- All processing finished. ---")