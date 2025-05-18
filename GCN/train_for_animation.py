#!/usr/bin/env python3
"""
GNN Trainer for PHEME (GCN/GAT/GATv2) - Modified for Saving Animation Embeddings
and Hyperparameter Tuning with Optuna.

This script trains GNN models with specific hyperparameters and saves 
penultimate layer embeddings at specified epoch intervals for later animation.
It can also run an Optuna study to find good HPs for generating animation data.
"""
from __future__ import annotations

import argparse
# import itertools # No longer needed for grid_search in this script's main path
import json
import logging
import random
from pathlib import Path
import shutil 
from typing import Dict, List, Tuple, Any
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv
from torchmetrics.classification import Accuracy, F1Score
from tqdm import tqdm

# Optional: Import Optuna for automated HP tuning
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    # logging.warning("Optuna not found...") # Logging configured later


# ... (set_seed, Model definitions, extract_penultimate_for_animation, split_masks, _make_model, History type alias) ...
# These functions remain the same as the previous correct version.

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
# Model definitions (Aligned with train_gnns_pheme.py)
# ---------------------------------------------------------------------------
class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, num_classes)) 
        self.dropout_rate = dropout 
        self.num_classes = num_classes

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1: 
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

class _BaseGAT(nn.Module):
    ConvLayer = None 
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float, heads: int = 4):
        super().__init__()
        assert self.ConvLayer is not None, "ConvLayer must be defined in subclass"
        self.convs = nn.ModuleList()
        self.convs.append(self.ConvLayer(in_dim, hidden_dim, heads=heads, concat=False, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(self.ConvLayer(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout))
        self.convs.append(self.ConvLayer(hidden_dim, num_classes, heads=1, concat=False, dropout=dropout))
        self.dropout_rate = dropout
        self.num_classes = num_classes

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

class GAT(_BaseGAT): ConvLayer = GATConv
class GATv2(_BaseGAT): ConvLayer = GATv2Conv

@torch.no_grad()
def extract_penultimate_for_animation(model: nn.Module, data: Data, device: torch.device) -> np.ndarray:
    model.eval()
    x_feat, ei = data.x.to(device), data.edge_index.to(device)
    if not (hasattr(model, 'convs') and isinstance(model.convs, nn.ModuleList) and len(model.convs) > 0):
        logging.warning(f"Model {type(model).__name__} has no 'convs'. Returning raw features.")
        return x_feat.cpu().numpy()
    current_features = x_feat
    if len(model.convs) > 1:
        for i in range(len(model.convs) - 1): 
            conv_layer = model.convs[i]
            current_features = conv_layer(current_features, ei)
            if isinstance(model, GCN): current_features = F.relu(current_features)
            else: current_features = F.elu(current_features)
            if hasattr(model, 'dropout_rate') and model.dropout_rate > 0:
                current_features = F.dropout(current_features, p=model.dropout_rate, training=False)
        penultimate_features = current_features
    else: 
        logging.warning(f"Model {type(model).__name__} has only 1 conv layer (classifier). Using raw features for 'penultimate'.")
        penultimate_features = x_feat
    return penultimate_features.cpu().numpy()

def split_masks(n_nodes: int, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42):
    idx = np.arange(n_nodes)
    train_idx, temp_idx = train_test_split(idx, test_size=1 - train_ratio, random_state=seed, shuffle=True)
    val_relative = val_ratio / (1 - train_ratio) if (1 - train_ratio) > 1e-6 else 0.5 
    if temp_idx.size > 0 and 0 < val_relative < 1:
        val_idx, test_idx = train_test_split(temp_idx, test_size=1 - val_relative, random_state=seed, shuffle=True)
    elif temp_idx.size > 0 : 
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed, shuffle=True)
    else: val_idx, test_idx = np.array([]), np.array([])
    mask = lambda arr: torch.zeros(n_nodes, dtype=torch.bool).scatter_(0, torch.from_numpy(arr.astype(np.int64)), True)
    return mask(train_idx), mask(val_idx), mask(test_idx)

def _make_model(arch: str, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float, heads: int):
    arch = arch.lower()
    if arch == "gcn": return GCN(in_dim, hidden_dim, num_layers, num_classes, dropout)
    elif arch == "gat": return GAT(in_dim, hidden_dim, num_layers, num_classes, dropout, heads=heads)
    elif arch == "gatv2": return GATv2(in_dim, hidden_dim, num_layers, num_classes, dropout, heads=heads)
    else: raise ValueError(f"Unknown architecture '{arch}'.")

History = Dict[str, List[float]] 

def run_one_experiment(
    data: Data, config: Dict, device: torch.device, arch: str, heads: int, epochs: int = 200,
    event_name: str = "unknown_event", base_animation_save_dir: Optional[Path] = None, 
    save_embedding_every_n_epochs: int = 5,
    is_optuna_trial: bool = False 
) -> Tuple[float, float, Dict, History, Optional[nn.Module]]: 
    
    animation_run_dir = None
    info_data = {} # Initialize info_data
    if base_animation_save_dir and not is_optuna_trial : 
        animation_run_dir = base_animation_save_dir / arch / event_name
        if animation_run_dir.exists(): shutil.rmtree(animation_run_dir)
        animation_run_dir.mkdir(parents=True, exist_ok=True)
        np.save(animation_run_dir / "labels.npy", data.y.cpu().numpy())
        np.save(animation_run_dir / "epoch_000_embeddings.npy", data.x.cpu().numpy()) 
        info_data = {"event_name": event_name, "arch": arch, "config": config, 
                       "total_epochs_trained_for_animation": 0, "saved_every_n_epochs": save_embedding_every_n_epochs,
                       "num_nodes": data.num_nodes, "num_features_raw": data.num_node_features}
    
    num_classes = int(data.y.max().item() + 1) if data.y.numel() > 0 else 2
    model = _make_model(arch=arch, in_dim=data.num_node_features, hidden_dim=config["hidden_dim"],
                        num_layers=config["num_layers"], num_classes=num_classes,
                        dropout=config["dropout"], heads=heads).to(device)

    optimiser = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="max", factor=0.5, patience=config.get("scheduler_patience", 20), min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()
    acc_metric = Accuracy(task="multiclass", num_classes=num_classes, average="micro").to(device)
    f1_metric = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    train_mask, val_mask, test_mask = split_masks(data.num_nodes, seed=config.get("split_seed", 42))
    data = data.to(device)

    best_val_acc = 0.0; best_test_acc = 0.0; best_f1 = 0.0; best_model_state = None; best_epoch = -1
    history: History = {"epoch": [], "train_loss": [], "val_acc": [], "test_acc": [], "lr": []}
    early_stop_patience = config.get("early_stop_patience", 30) 
    patience_counter = 0

    loop_desc = f"Training {arch.upper()} on {event_name}" if not is_optuna_trial else f"Optuna Trial {config.get('trial_num', '')}"
    for epoch in tqdm(range(epochs), desc=loop_desc, leave=is_optuna_trial): # Adjust leave for Optuna
        model.train(); optimiser.zero_grad()
        out = model(data.x, data.edge_index)
        if train_mask.sum() == 0: loss = torch.tensor(0.0, device=device, requires_grad=True)
        else: loss = criterion(out[train_mask], data.y[train_mask])
        if loss.requires_grad: loss.backward(); optimiser.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index); preds = logits.argmax(dim=1)
            val_acc = acc_metric(preds[val_mask], data.y[val_mask]) if val_mask.sum() > 0 else torch.tensor(0.0, device=device)
            test_acc_epoch = acc_metric(preds[test_mask], data.y[test_mask]) if test_mask.sum() > 0 else torch.tensor(0.0, device=device)
            scheduler.step(val_acc.item()) 
            current_val_acc = val_acc.item()
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc; best_test_acc = test_acc_epoch.item()
                best_f1 = f1_metric(preds[test_mask], data.y[test_mask]).item() if test_mask.sum() > 0 else 0.0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch; patience_counter = 0
            else: patience_counter += 1

        if animation_run_dir and not is_optuna_trial and \
           ((epoch + 1) % save_embedding_every_n_epochs == 0 or epoch == epochs - 1 or epoch == best_epoch):
            intermediate_embeddings = extract_penultimate_for_animation(model, data, device)
            np.save(animation_run_dir / f"epoch_{epoch+1:03d}_embeddings.npy", intermediate_embeddings)
            if info_data: # Check if info_data was initialized
                info_data["total_epochs_trained_for_animation"] = epoch + 1
                if "num_features_penultimate" not in info_data or info_data.get("num_features_penultimate") != intermediate_embeddings.shape[1]:
                     info_data["num_features_penultimate"] = intermediate_embeddings.shape[1]
                with open(animation_run_dir / "info.json", "w") as f: json.dump(info_data, f, indent=2)
        
        history["epoch"].append(epoch); history["train_loss"].append(loss.item())
        history["val_acc"].append(current_val_acc); history["test_acc"].append(test_acc_epoch.item())
        history["lr"].append(optimiser.param_groups[0]["lr"])

        if patience_counter >= early_stop_patience: logging.debug(f"Early stopping at epoch {epoch}."); break
        if optimiser.param_groups[0]['lr'] < scheduler.min_lrs[0] + 1e-9 and patience_counter > early_stop_patience // 2 : logging.debug(f"LR at min, no improvement. Stopping."); break

    final_model_to_return = None
    if best_model_state: 
        model.load_state_dict(best_model_state)
        final_model_to_return = model.cpu() 
    return best_val_acc, best_test_acc, {"f1": best_f1}, history, final_model_to_return

# REMOVED run_fixed_config_for_animation as it's redundant if main calls run_one_experiment directly

def objective_for_animation_hps(trial: optuna.trial.Trial, args: argparse.Namespace, data: Data, device: torch.device):
    # (Same as previous, but pass trial.number to config for logging)
    config = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 2, 3), # Typically 2-3 for these GNNs
        "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1), # Adjusted range
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True), # Adjusted range
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True), # Adjusted range
        "split_seed": args.seed, 
        "scheduler_patience": trial.suggest_int("scheduler_patience", 10, 20), # Adjusted range
        "early_stop_patience": trial.suggest_int("early_stop_patience", 15, 30), # Adjusted range
        "trial_num": trial.number # For logging
    }
    if args.arch in ["gat", "gatv2"]: # Add heads for GAT models
        config["heads"] = trial.suggest_categorical("heads", [2, 4, 8])
    else:
        config["heads"] = args.heads # Use default/fixed for GCN

    if config["early_stop_patience"] <= config["scheduler_patience"] :
        config["early_stop_patience"] = config["scheduler_patience"] + 5
    
    # Ensure hidden_dim is divisible by heads for GAT variants
    if args.arch in ["gat", "gatv2"]:
        if config["hidden_dim"] % config["heads"] != 0:
            # Adjust hidden_dim to be smallest multiple of heads >= original suggestion
            config["hidden_dim"] = config["heads"] * (config["hidden_dim"] // config["heads"] + (1 if config["hidden_dim"] % config["heads"] != 0 else 0) )
            if config["hidden_dim"] == 0 : config["hidden_dim"] = config["heads"] # Ensure not zero
            logging.debug(f"Trial {trial.number}: Adjusted hidden_dim to {config['hidden_dim']} for {config['heads']} heads.")


    logging.info(f"Optuna Trial {trial.number} for {args.event}/{args.arch} with config: {config}")
    val_acc, _, _, _, _ = run_one_experiment(
        data, config, device, args.arch, config.get("heads", args.heads), args.epochs, # Pass tuned heads
        event_name=args.event, 
        base_animation_save_dir=None, 
        save_embedding_every_n_epochs=args.epochs + 10, # Disable saving during HP search
        is_optuna_trial=True 
    )
    # Optuna pruner might need intermediate values. For simplicity, we return final val_acc.
    # trial.report(val_acc, step=args.epochs-1) # Report final val_acc
    # if trial.should_prune():
    #    raise optuna.exceptions.TrialPruned()
    return val_acc


def parse_args_train_anim():
    p = argparse.ArgumentParser(description="GNN trainer for PHEME, saves embeddings for animation. Can also run Optuna HP search.")
    p.add_argument("--data-dir", default="data", help="Path to preprocessed data (e.g., GCN/data relative to this script's location)")
    p.add_argument("--event", type=str, required=True, help="Specific event name to train on")
    p.add_argument("--arch", choices=["gcn", "gat", "gatv2"], default="gatv2", help="Architecture to train")
    p.add_argument("--epochs", type=int, default=150, help="Max epochs for training/Optuna trials")
    p.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension (GAT/GATv2: this is total, will be div by heads)")
    p.add_argument("--num-layers", type=int, default=2, help="Number of GNN layers (total, including output)")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    p.add_argument("--lr", type=float, default=0.0007, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1.7e-5, help="Weight decay for AdamW")
    p.add_argument("--heads", type=int, default=4, help="Attention heads for GAT/GATv2")
    p.add_argument("--scheduler-patience", type=int, default=20)
    p.add_argument("--early-stop-patience", type=int, default=30)
    p.add_argument("--animation-save-dir", default="../animation_data_gnn", help="Base directory to save epoch-wise embeddings")
    p.add_argument("--save-every-epochs", type=int, default=5, help="Save embeddings every N epochs (for non-Optuna runs)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optuna-study", action="store_true", help="Run Optuna HP study for the specified event and arch.")
    p.add_argument("--optuna-trials", type=int, default=30, help="Number of trials for Optuna study.")
    p.add_argument("--optuna-study-name-suffix", type=str, default="_anim_hps_study", help="Suffix for Optuna study name.") # Changed default
    return p.parse_args()

def main_train_for_animation():
    args = parse_args_train_anim()
    set_seed(args.seed) 
    logging.basicConfig(format="%(asctime)s | %(levelname)-7s | %(message)s", level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else ("cpu" if args.device == "cpu" else args.device))
    logging.info(f"Using device: {device}")

    script_dir = Path(__file__).resolve().parent
    data_root = (script_dir / args.data_dir).resolve()
    animation_save_base = (script_dir / args.animation_save_dir).resolve()

    event_dir = data_root / args.event
    if not event_dir.is_dir(): logging.error(f"Event directory not found: {event_dir}"); sys.exit(1)

    logging.info(f"--- Loading data for event: {args.event} from {event_dir} ---")
    try:
        X_np = np.load(event_dir / "X.npy").astype(np.float32)
        edge_index_np = np.load(event_dir / "edge_index.npy").astype(np.int64)
        y_np = np.load(event_dir / "labels.npy").astype(np.int64)
        data = Data(x=torch.from_numpy(X_np), edge_index=torch.from_numpy(edge_index_np), y=torch.from_numpy(y_np))
        data.num_nodes = data.x.shape[0]; data.num_node_features = data.x.shape[1]
    except FileNotFoundError as e: logging.error(f"Could not load data for {args.event} from {event_dir}: {e}"); sys.exit(1)

    if args.optuna_study:
        if not OPTUNA_AVAILABLE: logging.error("Optuna not installed. Cannot run study."); sys.exit(1)
        study_name = f"{args.arch}_{args.event}{args.optuna_study_name_suffix}"
        storage_name = f"sqlite:///{study_name.replace(' ','_')}.db"
        logging.info(f"Starting Optuna study '{study_name}' for {args.optuna_trials} trials.")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize", load_if_exists=True,
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=max(10, args.epochs // 10)), # Prune based on epochs
                                    sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=10))
        study.optimize(lambda trial: objective_for_animation_hps(trial, args, data, device), 
                       n_trials=args.optuna_trials, gc_after_trial=True)
        try:
            best_trial = study.best_trial
            logging.info(f"\nOptuna study '{study_name}' complete. Best trial value (Val Acc): {best_trial.value:.4f}")
            logging.info("Best HPs found:"); 
            for key, value in best_trial.params.items(): logging.info(f"  --{key.replace('_','-')} {value}")
            logging.info(f"To generate animation data with these HPs, run again without --optuna-study and pass these values.")
        except ValueError: logging.warning("No successful Optuna trials to determine best HPs.")
    else: 
        logging.info(f"\n===== TRAINING {args.arch.upper()} FOR {args.event} (FOR ANIMATION - FIXED HPs) =====")
        config_to_run = {
            "hidden_dim": args.hidden_dim, "num_layers": args.num_layers, "dropout": args.dropout,
            "lr": args.lr, "weight_decay": args.weight_decay, "split_seed": args.seed,
            "scheduler_patience": args.scheduler_patience, "early_stop_patience": args.early_stop_patience
        }
        # Heads argument is passed directly to run_one_experiment, not via config dict for _make_model
        
        # Ensure hidden_dim is suitable for GAT/GATv2 heads if they use concat=True internally before averaging
        # Based on your _BaseGAT, hidden_dim is the *output after averaging*, so heads divide this for GATConv input.
        # However, the GATConv in _BaseGAT with concat=False actually means hidden_dim IS the out_channels per head effectively.
        # The _make_model will use args.heads.
        # The config_to_run["hidden_dim"] should be the dimension the GAT layer aims to output *after potential averaging of heads*.
        # For your _BaseGAT: hidden_dim is used as out_channels for GATConv with concat=False, so it's fine.

        val_acc, test_acc, metrics, history, model_instance = run_one_experiment(
            data, config_to_run, device, args.arch, args.heads, args.epochs,
            event_name=args.event, base_animation_save_dir=animation_save_base,
            save_embedding_every_n_epochs=args.save_every_epochs
        )
        logging.info(f"Finished training {args.arch.upper()} for {args.event}.")
        logging.info(f"Config used: {config_to_run}")
        logging.info(f"Final Metrics: Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, F1: {metrics.get('f1',0):.4f}")
        logging.info(f"Embeddings for animation saved in: {animation_save_base / args.arch / args.event}")
        if model_instance: # Save the best model from this run
             model_save_path = animation_save_base / args.arch / args.event / "best_model_for_animation.pth"
             torch.save(model_instance.state_dict(), model_save_path)
             logging.info(f"Saved best model from this animation run to {model_save_path}")


if __name__ == "__main__":
    main_train_for_animation()