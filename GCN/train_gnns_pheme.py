#!/usr/bin/env python3
"""
Enhanced trainer for GCN, GAT, and GATv2 on the PHEME rumours dataset.

Key improvements (May 2025)
---------------------------
1. **Per‑architecture results files** – each trained model now writes to
   ``results_<arch>.json`` in the corresponding event folder, keeping the
   outputs tidy and preventing accidental overwrites. (Updated to new structure)
2. **Sequential multi‑architecture runs** – run *gcn*, *gat*, and *gatv2*
   back‑to‑back with a single command (default behaviour) or pass a subset
   via ``--archs``.
3. **Richer training logs** – epoch‑level history of training loss,
   validation accuracy, test accuracy, and learning‑rate evolution is
   stored in the JSON results for later visualisation. (Used for new plots)
4. **Automatic comparison plots** – once every architecture has finished,
   the script builds bar‑plots of the best **test accuracy** and **F1‑score**
   across events, saved under ``plots/`` in *PNG* and *PDF* formats. (Adapted and moved)
5. **Model checkpointing** - the best model during training is saved to disk
   as a PyTorch checkpoint file that can be reused by visualization tools. (Path updated)

NEW FEATURES (User Request - Iteration 2):
------------------------------------------
1. Comprehensive results saving: RESULTS/<arch>/<event>/[results.md, training_history_plots.png, ablation_study_results.json, model.pth]
2. Aggregated results: RESULTS/<arch>/results_aggregated.md (best overall HPs applied to each event, then averaged)
3. Enhanced Comparison training plots: RESULTS/comparison_training_plots/<event>_training_comparison.png (2x2 grid: Loss, Val Acc, Val F1, Test Acc)
4. Confusion Matrix plots: RESULTS/confusion_matrices_plots/<event>_confusion_matrices.png (grid of CMs for best model of each arch on test set)
5. Deterministic data splitting for fairer hyperparameter tuning and consistent CM evaluation.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import collections
import hashlib # For deterministic per-event seeds

import matplotlib as mpl
mpl.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn
from torch import nn, optim
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv
from torchmetrics.classification import Accuracy, F1Score
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Model definitions (Unchanged)
# ---------------------------------------------------------------------------

class GCN(nn.Module):
    """Standard Graph Convolutional Network."""
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, num_classes))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class _BaseGAT(nn.Module):
    """Shared utilities for GAT‑style models (GAT & GATv2)."""
    ConvLayer = None
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float, heads: int = 4):
        super().__init__()
        assert self.ConvLayer is not None, "ConvLayer must be defined in subclass"
        self.convs = nn.ModuleList()
        self.convs.append(self.ConvLayer(in_dim, hidden_dim, heads=heads, concat=False, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(self.ConvLayer(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout))
        self.convs.append(self.ConvLayer(hidden_dim, num_classes, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GAT(_BaseGAT):
    ConvLayer = GATConv


class GATv2(_BaseGAT):
    ConvLayer = GATv2Conv


# ---------------------------------------------------------------------------
# Training + evaluation routines (MODIFIED for fixed splits and richer history)
# ---------------------------------------------------------------------------

def split_masks(n_nodes: int, train_ratio: float = 0.7, val_ratio: float = 0.15, split_seed: int = 42):
    """Generates deterministic train/val/test masks based on a seed."""
    idx = np.arange(n_nodes)
    # Use a consistent seed for the first split
    train_idx, test_idx = train_test_split(idx, test_size=1 - train_ratio, stratify=None, random_state=split_seed)
    val_relative = val_ratio / (1 - train_ratio)
    # Use a related, but different, consistent seed for the second split
    val_idx, test_idx = train_test_split(test_idx, test_size=1 - val_relative, stratify=None, random_state=split_seed + 1)
    
    mask_fn = lambda arr: torch.zeros(n_nodes, dtype=torch.bool).scatter_(0, torch.tensor(arr), True)
    return mask_fn(train_idx), mask_fn(val_idx), mask_fn(test_idx)


def _make_model(
    arch: str, in_dim: int, hidden_dim: int, num_layers: int,
    num_classes: int, dropout: float, heads: int,
):
    arch = arch.lower()
    if arch == "gcn":
        return GCN(in_dim, hidden_dim, num_layers, num_classes, dropout)
    elif arch == "gat":
        return GAT(in_dim, hidden_dim, num_layers, num_classes, dropout, heads=heads)
    elif arch == "gatv2":
        return GATv2(in_dim, hidden_dim, num_layers, num_classes, dropout, heads=heads)
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Expected one of: gcn, gat, gatv2.")

History = Dict[str, List[float]]

def run_one_experiment(
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    config: Dict,
    device: torch.device,
    arch: str,
    heads: int,
    epochs: int = 200,
) -> Tuple[float, float, Dict, History, nn.Module]:
    num_classes = int(data.y.max().item() + 1)
    model = _make_model(
        arch=arch, in_dim=data.num_node_features, hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"], num_classes=num_classes,
        dropout=config["dropout"], heads=heads,
    ).to(device)

    optimiser = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="max", factor=0.5, patience=20, min_lr=1e-6)

    criterion = nn.CrossEntropyLoss()
    acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1_metric = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_f1 = 0.0 # This will be test_f1 at the epoch with best_val_acc
    best_model_state = None

    data = data.to(device) # Ensure data.y is also on device for metric calculation

    history: History = {
        "epoch": [], "train_loss": [], "val_acc": [], "test_acc": [],
        "lr": [], "val_f1": [], "test_f1": []
    }

    for epoch in range(epochs):
        model.train()
        optimiser.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimiser.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            preds = logits.argmax(dim=1)
            
            val_acc = acc_metric(preds[val_mask], data.y[val_mask])
            val_f1_epoch = f1_metric(preds[val_mask], data.y[val_mask]).item()
            
            test_acc_epoch = acc_metric(preds[test_mask], data.y[test_mask]).item()
            test_f1_epoch = f1_metric(preds[test_mask], data.y[test_mask]).item()
            
            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc.item()
                best_test_acc = test_acc_epoch # Store test_acc from this best val epoch
                best_f1 = test_f1_epoch      # Store test_f1 from this best val epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history["epoch"].append(epoch)
        history["train_loss"].append(loss.item())
        history["val_acc"].append(val_acc.item())
        history["val_f1"].append(val_f1_epoch)
        history["test_acc"].append(test_acc_epoch)
        history["test_f1"].append(test_f1_epoch)
        history["lr"].append(optimiser.param_groups[0]["lr"])
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    model = model.cpu()
    return best_val_acc, best_test_acc, {"f1": best_f1}, history, model


# ---------------------------------------------------------------------------
# Grid search driver (MODIFIED to accept masks)
# ---------------------------------------------------------------------------
AblationRunResult = Tuple[Dict, float, float, Dict[str, float]] # config, val_acc, test_acc, metrics

def grid_search(
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    device: torch.device,
    arch: str,
    heads: int,
    epochs: int,
    search_space: Dict[str, List]
) -> Tuple[Dict, Dict, nn.Module, List[AblationRunResult]]:
    keys, values = zip(*search_space.items())
    best_cfg = None
    best_score = -1 # Based on val_acc
    best_metrics_dict = None
    best_history: History | None = None
    best_model: nn.Module | None = None
    
    all_run_results: List[AblationRunResult] = []

    for combo in tqdm(list(itertools.product(*values)), desc=f"Grid Search ({arch})", leave=False):
        cfg = dict(zip(keys, combo))
        val_acc, test_acc, metrics, history, model_run = run_one_experiment(
            data, train_mask, val_mask, test_mask, cfg, device, arch, heads, epochs
        )
        logging.debug("Config %s | val %.4f test %.4f F1 %.4f", cfg, val_acc, test_acc, metrics.get("f1", 0))
        
        current_run_metrics = {"val_acc": val_acc, "test_acc": test_acc, **metrics}
        all_run_results.append((cfg, val_acc, test_acc, metrics.copy()))

        if val_acc > best_score:
            best_score = val_acc
            best_cfg = cfg
            best_metrics_dict = current_run_metrics # This now contains test_acc and test_f1 from the best val_acc epoch
            best_history = history
            best_model = model_run # model_run already has best_model_state loaded

    if not best_cfg or not best_metrics_dict or not best_history or not best_model:
        logging.error(f"Grid search failed to find a best model for {arch}. This might happen with very few epochs or a problematic search space.")
        if all_run_results:
            last_cfg, last_val_acc, last_test_acc, last_metrics = all_run_results[-1]
            _, _, _, last_history, last_model = run_one_experiment(data, train_mask, val_mask, test_mask, last_cfg, device, arch, heads, epochs)
            best_cfg = last_cfg
            best_metrics_dict = {"val_acc": last_val_acc, "test_acc": last_test_acc, **last_metrics}
            best_history = last_history
            best_model = last_model
        else:
            raise RuntimeError(f"Grid search yielded no results for arch {arch}.")

    best_metrics_dict["history"] = best_history
    return best_cfg, best_metrics_dict, best_model, all_run_results


# ---------------------------------------------------------------------------
# Checkpoint and Results Saving Helpers (Unchanged)
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, event_results_dir: Path, arch: str):
    ckpt_path = event_results_dir / f"model_{arch}.pth"
    torch.save(model.state_dict(), ckpt_path)
    logging.info("Saved checkpoint to %s", ckpt_path)

def save_event_results_md(
    event_results_dir: Path, arch: str, event_name: str,
    best_cfg: Dict, best_metrics: Dict
):
    md_path = event_results_dir / "results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Results for {arch.upper()} on {event_name}\n\n")
        f.write("## Best Hyperparameters:\n")
        f.write("```json\n")
        json.dump(best_cfg, f, indent=2)
        f.write("\n```\n\n")
        f.write("## Performance Metrics (on test set, from epoch with best validation accuracy):\n")
        f.write(f"- Best Validation Accuracy: {best_metrics['val_acc']:.4f}\n")
        f.write(f"- Corresponding Test Accuracy: {best_metrics['test_acc']:.4f}\n")
        f.write(f"- Corresponding F1 Score (Macro): {best_metrics['f1']:.4f}\n")
    logging.info("Saved event results summary to %s", md_path)

def save_ablation_results_json(
    all_run_results: List[AblationRunResult], 
    event_results_dir: Path
):
    json_path = event_results_dir / "ablation_study_results.json"
    output_data = []
    for cfg, val_acc, test_acc, metrics in all_run_results:
        output_data.append({
            "config": cfg,
            "val_acc": val_acc,
            "test_acc_at_best_val_epoch": test_acc, # Clarify name
            "f1_at_best_val_epoch": metrics.get("f1")
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    logging.info("Saved ablation study results to %s", json_path)

def save_arch_aggregated_results_md(
    arch_dir: Path, arch: str, best_overall_config: Dict,
    event_results_with_overall_config: List[Dict], avg_metrics: Dict
):
    md_path = arch_dir / "results_aggregated.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Aggregated Results for {arch.upper()}\n\n")
        f.write("## Best Overall Hyperparameters (selected by average validation accuracy across events):\n")
        f.write("```json\n")
        json.dump(best_overall_config, f, indent=2)
        f.write("\n```\n\n")
        f.write("## Performance on Each Event (using best overall hyperparameters):\n")
        
        df = pd.DataFrame(event_results_with_overall_config)
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Average Performance (using best overall hyperparameters):\n")
        f.write(f"- Average Test Accuracy: {avg_metrics['avg_test_acc']:.4f}\n")
        f.write(f"- Average F1 Score (Macro): {avg_metrics['avg_f1']:.4f}\n")
    logging.info("Saved aggregated architecture results to %s", md_path)

# ---------------------------------------------------------------------------
# Plotting helpers (MODIFIED and NEW)
# ---------------------------------------------------------------------------

def plot_single_training_history(history: History, out_path: Path, title: str):
    if not history or not history.get("epoch"):
        logging.warning(f"History is empty or malformed for {title}. Skipping plot.")
        return

    epochs_list = history["epoch"]
    fig, axs = plt.subplots(2, 3, figsize=(18, 10)) # Now 2x3 for 6 plots
    fig.suptitle(title, fontsize=16)

    plot_specs = [
        ("train_loss", "Training Loss", "Loss"),
        ("val_acc", "Validation Accuracy", "Accuracy"),
        ("val_f1", "Validation F1", "F1 Score"),
        ("test_acc", "Test Accuracy (epoch-wise)", "Accuracy"),
        ("test_f1", "Test F1 (epoch-wise)", "F1 Score"),
        ("lr", "Learning Rate", "Learning Rate"),
    ]

    for i, (key, plot_title, ylabel) in enumerate(plot_specs):
        ax = axs[i // 3, i % 3]
        if key in history and history[key]:
            ax.plot(epochs_list, history[key], label=plot_title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(plot_title)
            ax.legend()
            ax.grid(True)
            if key == "lr":
                ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, f"No data for\n{plot_title}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(plot_title)
            ax.set_xticks([])
            ax.set_yticks([])


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path)
    plt.close(fig)
    logging.info(f"Saved training history plot to {out_path}")


def plot_event_comparison_training_history(
    arch_histories: Dict[str, History], out_path: Path, title: str
):
    """Plots a 2x2 grid: Train Loss, Val Acc, Val F1, Test Acc for comparison."""
    if not arch_histories:
        logging.warning(f"No architecture histories provided for {title}. Skipping plot.")
        return

    arch_colors = {"gcn": "blue", "gat": "green", "gatv2": "red", "default": "purple"}
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12)) # 2x2 grid
    fig.suptitle(title, fontsize=16)

    plot_details = [
        ("train_loss", "Training Loss Comparison", "Loss", axs[0,0]),
        ("val_acc", "Validation Accuracy Comparison", "Accuracy", axs[0,1]),
        ("val_f1", "Validation F1 Comparison", "F1 Score", axs[1,0]),
        ("test_acc", "Test Accuracy (epoch-wise) Comparison", "Accuracy", axs[1,1])
    ]

    for metric_key, plot_title, y_label, ax in plot_details:
        has_data_for_metric = False
        for arch, history in arch_histories.items():
            if history and history.get("epoch") and history.get(metric_key) and len(history[metric_key]) == len(history["epoch"]):
                ax.plot(history["epoch"], history[metric_key], label=f"{arch.upper()}", color=arch_colors.get(arch.lower(), arch_colors["default"]))
                has_data_for_metric = True
            elif history and history.get("epoch"): # Log if specific metric is missing but epoch data exists
                 logging.debug(f"Metric '{metric_key}' missing or length mismatch for arch '{arch}' in '{title}'. Epochs: {len(history['epoch'])}, Metric: {len(history.get(metric_key, []))}")


        if has_data_for_metric:
            ax.set_xlabel("Epoch")
            ax.set_ylabel(y_label)
            ax.set_title(plot_title)
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No data available\nfor this metric.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(plot_title)


    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path)
    plt.close(fig)
    logging.info(f"Saved event training comparison plot to {out_path}")

def plot_event_confusion_matrices(
    event_name: str,
    arch_test_results: Dict[str, Tuple[torch.Tensor, torch.Tensor]], # arch -> (preds, trues)
    num_classes: int,
    class_names: List[str] | None,
    out_path: Path
):
    """Plots a grid of confusion matrices for an event, one for each architecture."""
    archs = list(arch_test_results.keys())
    if not archs:
        logging.warning(f"No test results to plot confusion matrices for event {event_name}.")
        return

    n_archs = len(archs)
    # Adjust layout: try to make it squarish, or max 3 columns
    ncols = min(n_archs, 3)
    nrows = (n_archs + ncols - 1) // ncols 

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5), squeeze=False)
    fig.suptitle(f"Confusion Matrices for Event: {event_name} (Test Set)", fontsize=16)

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    arch_idx = 0
    for r in range(nrows):
        for c in range(ncols):
            if arch_idx < n_archs:
                ax = axs[r, c]
                arch = archs[arch_idx]
                preds, trues = arch_test_results[arch]

                cm = confusion_matrix(trues.numpy(), preds.numpy(), labels=list(range(num_classes)))
                
                seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=class_names, yticklabels=class_names,
                                cbar=arch_idx % ncols == ncols -1) # Show cbar only for last col
                ax.set_title(f"{arch.upper()}")
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                arch_idx += 1
            else:
                axs[r,c].axis('off') # Hide unused subplots

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path)
    plt.close(fig)
    logging.info(f"Saved confusion matrices plot to {out_path}")


def _plot_comparison(metric: str, results: Dict[str, Dict[str, float]], out_dir: Path):
    """Bar‑plot *metric* across events + architectures."""
    valid_results = {arch: res for arch, res in results.items() if res}
    if not valid_results:
        logging.warning(f"No data to plot for metric {metric}. Skipping bar plot.")
        return

    first_valid_arch = next(iter(valid_results))
    events = sorted(valid_results[first_valid_arch].keys())
    if not events:
        logging.warning(f"No events found in results for metric {metric}. Skipping bar plot.")
        return

    archs = sorted(valid_results.keys())
    bar_width = 0.25
    x = np.arange(len(events))

    plt.figure(figsize=(max(8, len(events) * 1.5), 6))
    for i, arch in enumerate(archs):
        vals = [valid_results[arch].get(evt, {}).get(metric, np.nan) for evt in events]
        plt.bar(x + i * bar_width, vals, width=bar_width, label=arch.upper())

    plt.xlabel("Event", fontsize=12)
    plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
    plt.title(f"PHEME – Best {metric.replace('_', ' ').title()} per Architecture (from individual best runs)", fontsize=14)
    plt.xticks(x + bar_width * (len(archs) -1) / 2, events, rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--')
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.tight_layout()
        plt.savefig(out_dir / f"best_{metric}_comparison.{ext}")
    plt.close()
    logging.info(f"Saved summary bar plot for {metric} to {out_dir}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GNN trainer for PHEME events (GCN/GAT/GATv2)")
    p.add_argument("--data-dir", default="data", help="Path to the root data directory containing event subfolders")
    p.add_argument("--results-dir", default="RESULTS", help="Directory to save all outputs")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument(
        "--archs", nargs="+", choices=["gcn", "gat", "gatv2"],
        default=["gcn", "gat", "gatv2"], help="Architectures to train sequentially"
    )
    p.add_argument("--heads", type=int, default=4, help="Attention heads (for GAT/GATv2)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    logging.info(f"Using device: {device}")

    data_root = Path(args.data_dir)
    if not data_root.exists():
        logging.error(f"Data directory {data_root} not found.")
        return
        
    dataset_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and (d / "X.npy").exists()])
    if not dataset_dirs:
        logging.error(f"No event subdirectories with X.npy found in {data_root}.")
        return

    results_root_dir = Path(args.results_dir)
    results_root_dir.mkdir(parents=True, exist_ok=True)

    search_space = {
        "hidden_dim": [64, 128],
        "num_layers": [2, 3],
        "dropout": [0.0, 0.5],
        "lr": [1e-2, 5e-3],
        "weight_decay": [0, 5e-4],
    }

    # {event_name: {arch: history_for_best_run}}
    all_event_arch_best_histories: Dict[str, Dict[str, History]] = collections.defaultdict(dict)
    # {event_name: {arch: (test_preds, test_trues)}} for CMs
    all_event_arch_test_details: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = collections.defaultdict(dict)
    # {event_name: num_classes}
    event_num_classes: Dict[str, int] = {}


    summary_bar_plot_data: Dict[str, Dict[str, Dict[str, float]]] = collections.defaultdict(lambda: collections.defaultdict(dict))

    for arch in args.archs:
        logging.info(f"\n===== PROCESSING ARCHITECTURE: {arch.upper()} =====")
        arch_results_dir = results_root_dir / arch
        arch_results_dir.mkdir(parents=True, exist_ok=True)
        current_arch_all_grid_run_details: List[Tuple[Any, float, str, float, float]] = []

        for ds_dir in dataset_dirs:
            event_name = ds_dir.name
            logging.info(f"--- Dataset: {event_name} (Arch: {arch.upper()}) ---")
            
            event_arch_results_dir = arch_results_dir / event_name
            event_arch_results_dir.mkdir(parents=True, exist_ok=True)

            try:
                X = np.load(ds_dir / "X.npy").astype(np.float32)
                edge_index = torch.from_numpy(np.load(ds_dir / "edge_index.npy").astype(np.int64))
                y = torch.from_numpy(np.load(ds_dir / "labels.npy").astype(np.int64))
                data = Data(x=torch.from_numpy(X), edge_index=edge_index, y=y)
                num_classes = int(data.y.max().item() + 1)
                event_num_classes[event_name] = num_classes
            except FileNotFoundError as e:
                logging.error(f"Missing data file in {ds_dir}: {e}. Skipping event {event_name}.")
                continue
            
            # Generate deterministic splits for this event
            event_split_seed_str = f"{args.seed}_{event_name}"
            event_split_seed = int(hashlib.md5(event_split_seed_str.encode()).hexdigest(), 16) % (2**32 -1)
            logging.info(f"Using split_seed {event_split_seed} for event {event_name}")
            train_mask, val_mask, test_mask = split_masks(data.num_nodes, split_seed=event_split_seed)

            best_cfg, best_metrics_for_event, best_model_for_event, all_run_results_for_event = grid_search(
                data, train_mask, val_mask, test_mask, device, arch, args.heads, args.epochs, search_space
            )
            
            logging.info(f"Best config for {event_name} ({arch.upper()}): {best_cfg}")
            logging.info(f"Metrics (Test set @ best Val Acc): ValAcc={best_metrics_for_event['val_acc']:.4f}, TestAcc={best_metrics_for_event['test_acc']:.4f}, TestF1={best_metrics_for_event['f1']:.4f}")

            save_event_results_md(event_arch_results_dir, arch, event_name, best_cfg, best_metrics_for_event)
            plot_single_training_history(
                best_metrics_for_event["history"],
                event_arch_results_dir / "training_history_plots.png",
                f"Training History: {arch.upper()} on {event_name} (Best Run)"
            )
            save_ablation_results_json(all_run_results_for_event, event_arch_results_dir)
            save_checkpoint(best_model_for_event, event_arch_results_dir, arch)

            all_event_arch_best_histories[event_name][arch] = best_metrics_for_event["history"]
            summary_bar_plot_data[arch][event_name] = {
                "test_acc": best_metrics_for_event["test_acc"],
                "f1": best_metrics_for_event["f1"],
            }

            # Get predictions from best model on the fixed test set for CM
            best_model_for_event.to(device).eval()
            with torch.no_grad():
                logits_cm = best_model_for_event(data.x.to(device), data.edge_index.to(device))
                preds_cm = logits_cm.argmax(dim=1)[test_mask.to(device)].cpu() # Apply mask after predictions
                trues_cm = data.y[test_mask].cpu()
            all_event_arch_test_details[event_name][arch] = (preds_cm, trues_cm)


            for run_cfg, run_val_acc, run_test_acc, run_metrics in all_run_results_for_event:
                cfg_tuple = tuple(sorted(run_cfg.items()))
                current_arch_all_grid_run_details.append(
                    (cfg_tuple, run_val_acc, event_name, run_test_acc, run_metrics.get("f1", 0.0))
                )
        
        if not current_arch_all_grid_run_details:
            logging.warning(f"No grid search runs recorded for arch {arch}. Skipping aggregated results for this arch.")
            continue

        config_avg_val_performance = collections.defaultdict(lambda: {"total_val_acc": 0.0, "count": 0})
        for cfg_tuple, val_acc, _, _, _ in current_arch_all_grid_run_details:
            config_avg_val_performance[cfg_tuple]["total_val_acc"] += val_acc
            config_avg_val_performance[cfg_tuple]["count"] += 1
        
        best_overall_cfg_tuple = None
        max_avg_val_acc = -1
        if config_avg_val_performance:
            for cfg_tuple, data_dict_cfg in config_avg_val_performance.items():
                avg_val_acc = data_dict_cfg["total_val_acc"] / data_dict_cfg["count"]
                if avg_val_acc > max_avg_val_acc:
                    max_avg_val_acc = avg_val_acc
                    best_overall_cfg_tuple = cfg_tuple
        
        if best_overall_cfg_tuple is None:
            logging.warning(f"Could not determine best overall config for arch {arch}. Skipping aggregated results.")
        else:
            best_overall_config = dict(best_overall_cfg_tuple)
            logging.info(f"Best overall config for {arch.upper()}: {best_overall_config} (Avg Val Acc across events: {max_avg_val_acc:.4f})")
            overall_config_event_run_results = []
            total_test_acc_overall, total_f1_overall, num_events_for_overall = 0, 0, 0

            for ds_dir_rerun in dataset_dirs:
                event_name_rerun = ds_dir_rerun.name
                try:
                    X_r, e_idx_r, y_r = (np.load(ds_dir_rerun / "X.npy").astype(np.float32),
                                        torch.from_numpy(np.load(ds_dir_rerun / "edge_index.npy").astype(np.int64)),
                                        torch.from_numpy(np.load(ds_dir_rerun / "labels.npy").astype(np.int64)))
                    data_rerun = Data(x=torch.from_numpy(X_r), edge_index=e_idx_r, y=y_r)
                    
                    # Use the same deterministic splits for reruns
                    event_split_seed_str_rerun = f"{args.seed}_{event_name_rerun}"
                    event_split_seed_rerun = int(hashlib.md5(event_split_seed_str_rerun.encode()).hexdigest(), 16) % (2**32-1)
                    tm_r, vm_r, tsm_r = split_masks(data_rerun.num_nodes, split_seed=event_split_seed_rerun)

                except FileNotFoundError: continue
                
                logging.info(f"Re-running {event_name_rerun} with best overall config for {arch.upper()}...")
                _, test_acc_re, metrics_rerun, _, _ = run_one_experiment(
                    data_rerun, tm_r, vm_r, tsm_r, best_overall_config, device, arch, args.heads, args.epochs,
                )
                overall_config_event_run_results.append({
                    "Event": event_name_rerun, "Test Accuracy": f"{test_acc_re:.4f}", "F1 Score": f"{metrics_rerun['f1']:.4f}"
                })
                total_test_acc_overall += test_acc_re
                total_f1_overall += metrics_rerun['f1']
                num_events_for_overall += 1
            
            avg_metrics_overall = {"avg_test_acc": (total_test_acc_overall / num_events_for_overall if num_events_for_overall > 0 else 0.0),
                                   "avg_f1": (total_f1_overall / num_events_for_overall if num_events_for_overall > 0 else 0.0)}
            save_arch_aggregated_results_md(arch_results_dir, arch, best_overall_config, overall_config_event_run_results, avg_metrics_overall)

    # After all architectures and events:
    comparison_plots_dir = results_root_dir / "comparison_training_plots"
    comparison_plots_dir.mkdir(parents=True, exist_ok=True)
    
    cm_plots_dir = results_root_dir / "confusion_matrices_plots"
    cm_plots_dir.mkdir(parents=True, exist_ok=True)

    class_names_map = {0: "Non-Rumour", 1: "Rumour"} # Example, adjust if your labels differ or are more numerous

    for event_name in all_event_arch_best_histories: # Iterate over events that have data
        if all_event_arch_best_histories[event_name]:
            plot_event_comparison_training_history(
                all_event_arch_best_histories[event_name],
                comparison_plots_dir / f"{event_name}_training_comparison.png",
                f"Training Dynamics Comparison for {event_name}"
            )
        
        if all_event_arch_test_details[event_name] and event_name in event_num_classes:
            num_classes_for_event = event_num_classes[event_name]
            # Generate class names like ["Class 0", "Class 1", ...] or use a predefined map
            current_class_names = [class_names_map.get(i, f"Class {i}") for i in range(num_classes_for_event)]

            plot_event_confusion_matrices(
                event_name,
                all_event_arch_test_details[event_name],
                num_classes=num_classes_for_event,
                class_names=current_class_names,
                out_path=cm_plots_dir / f"{event_name}_confusion_matrices.png"
            )

    summary_plot_dir = results_root_dir / "summary_plots"
    summary_plot_dir.mkdir(parents=True, exist_ok=True)
    final_summary_bar_plot_data = {arch_key: dict(events_data) for arch_key, events_data in summary_bar_plot_data.items()}
    if any(final_summary_bar_plot_data.values()):
        _plot_comparison("test_acc", final_summary_bar_plot_data, summary_plot_dir)
        _plot_comparison("f1", final_summary_bar_plot_data, summary_plot_dir)
        logging.info(f"Saved summary bar plots to {summary_plot_dir.resolve()}")
    else:
        logging.warning("No data collected for summary bar plots.")

    logging.info("All processing finished. Results are in %s", results_root_dir.resolve())


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s", 
        level=logging.INFO,
        handlers=[logging.StreamHandler()]
    )
    main()