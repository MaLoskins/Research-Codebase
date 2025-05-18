#!/usr/bin/env python3
"""
Robust PHEME Embeddings Dashboard with DySAT Support
This script creates a reliable dashboard for visualizing PHEME embeddings
from different GNN architectures (GCN, GAT, GATv2) and FullDySAT.
Uses t-SNE for dimensionality reduction with GPU acceleration if available.
Also saves static PNG images for each plot combination.
"""
import argparse
import json
import logging
import os
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.manifold import TSNE # Fallback
from torch_geometric.data import Data as StaticData # For GCN/GAT/GATv2
from tqdm import tqdm

# --- Configure logging early ---
logging.basicConfig(format="%(asctime)s | %(levelname)-7s | %(message)s", level=logging.INFO)

# --- Determine Project Root (CWD) and Adjust sys.path for module imports ---
# This allows 'from DySAT...' and 'from GCN...' if these are direct subdirectories of CWD
CWD_PROJECT_ROOT = Path.cwd() 
logging.info(f"Project root (CWD) for path resolution and imports: {CWD_PROJECT_ROOT}")
if str(CWD_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(CWD_PROJECT_ROOT))
    logging.debug(f"Added CWD {CWD_PROJECT_ROOT} to sys.path.")

# --- Import DySAT components ---
try:
    from DySAT.train_OG_dysat_pheme import FullDySAT, load_temporal_data as load_dysat_event_data, set_seed as dysat_set_seed
    HAS_DYSAT_SCRIPT = True
    logging.info(f"Successfully imported FullDySAT from DySAT.train_OG_dysat_pheme")
except ImportError as e:
    logging.warning(f"Could not import FullDySAT from DySAT.train_OG_dysat_pheme: {e}. DySAT embeddings will not be available.")
    HAS_DYSAT_SCRIPT = False; FullDySAT = None; load_dysat_event_data = None; dysat_set_seed = lambda x: None

# --- Import Static GNN components ---
try:
    from GCN.train_gnns_pheme import GCN, GAT, GATv2, _make_model, set_seed as static_gnn_set_seed
    HAS_GNN_SCRIPT = True
    logging.info("Successfully imported GCN, GAT, GATv2 from GCN.train_gnns_pheme")
except ImportError as e:
    logging.warning(f"❌ Could not import from GCN.train_gnns_pheme – GCN/GAT/GATv2 embeddings will not be available: {e}")
    HAS_GNN_SCRIPT = False; GCN, GAT, GATv2, _make_model, static_gnn_set_seed = None, None, None, None, lambda x: None

try: from cuml.manifold import TSNE as cuTSNE; has_cuml = True
except ImportError: has_cuml = False
try: import plotly.graph_objects as go
except ImportError: raise ImportError("Plotly required: pip install plotly kaleido") 
try: import pandas as pd
except ImportError: raise ImportError("Pandas required: pip install pandas")


# (Rest of the functions: load_static_event_data, extract_penultimate_static_gnn, 
#  extract_penultimate_dysat, load_checkpoints, run_tsne, load_dysat_hps_from_json, 
#  process_model_data, create_dashboard, parse_args - remain exactly the same as the previous version)
# ... PASTE THE FULL SET OF HELPER FUNCTIONS, process_model_data, create_dashboard, parse_args HERE ...
# For brevity, I am omitting them as they were correct in the previous full script you provided.
# Make sure the definition of extract_penultimate_dysat is present.
# The key change is ensuring PROJECT_ROOT is consistently CWD for path resolution.

def load_static_event_data(event_dir: Path) -> Optional[StaticData]:
    try:
        x_path, ei_path, y_path = event_dir / "X.npy", event_dir / "edge_index.npy", event_dir / "labels.npy"
        if not (x_path.exists() and ei_path.exists() and y_path.exists()):
            logging.warning(f"Missing one or more data files (X, edge_index, labels) in {event_dir}. Skipping.")
            return None
        x = torch.from_numpy(np.load(x_path).astype(np.float32))
        edge_index = torch.from_numpy(np.load(ei_path).astype(np.int64))
        y = torch.from_numpy(np.load(y_path).astype(np.int64))
        num_classes_val = int(y.max().item() + 1) if y.numel() > 0 else 2
        return StaticData(x=x, edge_index=edge_index, y=y, num_classes=num_classes_val, event_name=event_dir.name) 
    except Exception as e:
        logging.error(f"Error loading static data from {event_dir}: {e}")
        return None

@torch.no_grad()
def extract_penultimate_static_gnn(model: nn.Module, data: StaticData, device: torch.device, target_dim: int) -> Optional[np.ndarray]:
    if not HAS_GNN_SCRIPT: return None
    model = model.to(device); model.eval()
    x_feat, ei = data.x.to(device), data.edge_index.to(device) 
    
    penultimate_features = None
    num_conv_layers = len(model.convs) if hasattr(model, 'convs') and isinstance(model.convs, nn.ModuleList) else 0

    if num_conv_layers == 0:
        logging.warning(f"Model {type(model).__name__} for event {data.event_name} has no 'convs'. Using raw features.");
        penultimate_features = x_feat 
    elif num_conv_layers == 1: 
        last_conv = model.convs[0]
        is_classifier = False
        expected_out_channels_last_conv = last_conv.out_channels
        if isinstance(model, (GAT, GATv2)) and hasattr(last_conv, 'concat') and last_conv.concat: 
            expected_out_channels_last_conv *= last_conv.heads
        
        if hasattr(model, 'num_classes') and expected_out_channels_last_conv == model.num_classes and not hasattr(model, 'lin'):
            is_classifier = True
        
        if is_classifier:
            logging.warning(f"Model {type(model).__name__} for event {data.event_name} has 1 conv as classifier. Using raw features.")
            penultimate_features = x_feat
        else: 
            current_features = last_conv(x_feat, ei)
            current_features = F.relu(current_features) if isinstance(model, GCN) else F.elu(current_features)
            if hasattr(model, 'dropout') and model.dropout > 0: current_features = F.dropout(current_features, p=float(model.dropout), training=False)
            penultimate_features = current_features
    else: 
        current_features = x_feat
        for i in range(num_conv_layers - 1): 
            conv_layer = model.convs[i]
            current_features = conv_layer(current_features, ei)
            if isinstance(model, GCN): current_features = F.relu(current_features)
            else: current_features = F.elu(current_features) 
            if hasattr(model, 'dropout') and model.dropout > 0:
                current_features = F.dropout(current_features, p=float(model.dropout), training=False)
        penultimate_features = current_features
        logging.debug(f"{type(model).__name__} for event {data.event_name}: Took features from before last classifying conv. Shape: {penultimate_features.shape}")
            
    if penultimate_features is None: logging.error(f"Penultimate extraction failed for {type(model).__name__} for {data.event_name}. Fallback to raw."); penultimate_features = x_feat
    if penultimate_features.shape[1] != target_dim:
        if penultimate_features.shape[1] > 0:
            logging.info(f"Projecting {type(model).__name__} penultimate for event {data.event_name} from {penultimate_features.shape[1]}D to {target_dim}D.")
            projection_layer = nn.Linear(penultimate_features.shape[1], target_dim).to(device)
            nn.init.xavier_uniform_(projection_layer.weight); 
            if projection_layer.bias is not None: nn.init.zeros_(projection_layer.bias)
            penultimate_features = projection_layer(penultimate_features)
        else: logging.error(f"Cannot project 0-feat emb for {type(model).__name__} for {data.event_name}. Ret zeros."); return np.zeros((data.num_nodes, target_dim))
    logging.debug(f"Final extracted penultimate for {type(model).__name__} for event {data.event_name} shape: {penultimate_features.shape}")
    return penultimate_features.cpu().numpy()

@torch.no_grad()
def extract_penultimate_dysat(model: FullDySAT, X_orig, edge_indices_list, node_masks_list, device: torch.device) -> np.ndarray:
    model = model.to(device); model.eval()
    X_orig_dev = X_orig.to(device)
    edge_indices_dev = [ei.to(device) for ei in edge_indices_list]
    x_projected = model.input_proj(X_orig_dev); x_projected = F.elu(x_projected)
    x_projected_dropout = model.dropout_module(x_projected)
    snapshot_embeddings = []
    T_snapshots = len(edge_indices_dev)
    for t in range(T_snapshots):
        current_edges = edge_indices_dev[t]
        if not isinstance(current_edges, torch.Tensor) or current_edges.dim() != 2 : current_edges = torch.empty((2,0), dtype=torch.long, device=device)
        elif current_edges.numel() == 0 and current_edges.dim() == 2 and current_edges.shape[0] !=2 : current_edges = torch.empty((2,0), dtype=torch.long, device=device)
        h_struct_t = model.structural_attention(x_projected_dropout, current_edges)
        snapshot_embeddings.append(h_struct_t)
    if not snapshot_embeddings: logging.warning("No snapshot embeddings for DySAT."); return x_projected_dropout.cpu().numpy()
    final_node_embeddings_before_classifier = None
    if model.use_temporal_attn:
        temporal_input_sequence = torch.stack(snapshot_embeddings, dim=0)
        temporal_input_sequence = model.pos_encoder(temporal_input_sequence)
        temporal_output_sequence = model.temporal_attention(temporal_input_sequence)
        final_node_embeddings_before_classifier = temporal_output_sequence[-1, :, :]
    else: 
        if not hasattr(model, 'temporal_weight') or not hasattr(model, 'temporal_norm'):
             logging.error("DySAT simple temporal mode missing params."); final_node_embeddings_before_classifier = snapshot_embeddings[-1]
        else:
            last_embedding = snapshot_embeddings[-1]
            if len(snapshot_embeddings) > 1:
                all_embeddings_stacked = torch.stack(snapshot_embeddings, dim=0); avg_embedding = all_embeddings_stacked.mean(dim=0)
                temporal_w = torch.sigmoid(model.temporal_weight); combined = temporal_w * last_embedding + (1 - temporal_w) * avg_embedding
                combined = model.temporal_norm(combined); final_node_embeddings_before_classifier = model.dropout_module(combined)
            else: final_node_embeddings_before_classifier = last_embedding
    return final_node_embeddings_before_classifier.cpu().numpy()

def load_checkpoint_static_gnn(model: nn.Module, event_dir: Path, arch: str) -> bool:
    ckpt = event_dir / f"model_{arch}.pth";
    if ckpt.exists():
        try: model.load_state_dict(torch.load(ckpt, map_location="cpu")); logging.info(f"Loaded GNN ckpt {ckpt.name}"); return True
        except RuntimeError as e: logging.error(f"Error GNN ckpt {ckpt.name}: {e}"); return False
    logging.warning(f"Static GNN ckpt not found: {ckpt}"); return False

def load_checkpoint_dysat(model: FullDySAT, dysat_results_dir: Path, event_name: str, run_tag: str) -> bool:
    ckpt = dysat_results_dir / run_tag / event_name / "model.pt"
    if ckpt.exists():
        try: model.load_state_dict(torch.load(ckpt, map_location="cpu")); logging.info(f"Loaded DySAT ckpt {ckpt}"); return True
        except RuntimeError as e: logging.error(f"Error DySAT ckpt {ckpt}: {e}"); return False
    logging.warning(f"DySAT ckpt not found: {ckpt}"); return False

def run_tsne(data, n_components=3, perplexity=30, learning_rate='auto', n_iter=1000, seed=42, use_cuml_if_available=True):
    n_samples, n_features = data.shape
    actual_n_components = min(n_components, n_features) if n_features > 0 else n_components 
    if n_samples <= actual_n_components : actual_n_components = max(1,n_samples-1)
    if actual_n_components <=0: logging.warning(f"Cannot run t-SNE: bad dims."); return np.zeros((n_samples, n_components)) 
    effective_perplexity = perplexity
    if n_samples <= perplexity +1 : 
        effective_perplexity = max(1, n_samples - 2); 
        if effective_perplexity <=0 : effective_perplexity = 1
        logging.warning(f"Perplexity {perplexity} for n_samples={n_samples}. Adjusted to {effective_perplexity}.")
    if use_cuml_if_available and has_cuml and n_samples > actual_n_components: 
        if n_features < actual_n_components: logging.warning(f"cuML t-SNE: n_features ({n_features}) < target_components ({actual_n_components}).")
        tsne_cuml = cuTSNE(n_components=actual_n_components, perplexity=effective_perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=seed, method='barnes_hut')
        logging.info(f"Running cuML t-SNE: data {data.shape} to {actual_n_components}D, perplexity {effective_perplexity}...")
        transformed_data = tsne_cuml.fit_transform(data)
    else:
        tsne_init = 'pca' if n_features >= actual_n_components and n_features > 3 else 'random'
        logging.info(f"Running scikit-learn t-SNE: data {data.shape} to {actual_n_components}D, init='{tsne_init}', perplexity {effective_perplexity}...")
        tsne_sklearn = TSNE(n_components=actual_n_components, perplexity=effective_perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=seed, n_jobs=-1, init=tsne_init)
        transformed_data = tsne_sklearn.fit_transform(data)
    if transformed_data.shape[1] < n_components:
        logging.warning(f"t-SNE produced {transformed_data.shape[1]}D, padding to {n_components}D.")
        padding = np.zeros((transformed_data.shape[0], n_components - transformed_data.shape[1]))
        transformed_data = np.hstack((transformed_data, padding))
    return transformed_data

def load_dysat_hps_from_json(hps_json_path: Optional[Path]) -> Optional[Dict]: # Changed to Path object
    if not hps_json_path: logging.debug("No DySAT HPs JSON path provided."); return None
    # Path is already resolved in main
    if hps_json_path.exists():
        try:
            with open(hps_json_path, 'r') as f: hps = json.load(f)
            logging.info(f"Successfully loaded DySAT HPs from {hps_json_path}"); return hps
        except Exception as e: logging.error(f"Error loading DySAT HPs from {hps_json_path}: {e}")
    else: logging.error(f"DySAT HPs JSON file not found at resolved path: {hps_json_path}")
    return None

def process_model_data(
    model_type: str, arch_name: str, 
    static_gnn_data_path: Optional[Path], dysat_data_path: Optional[Path],
    dysat_results_path: Optional[Path], dysat_run_tag_for_model: Optional[str],
    dysat_hps_config: Optional[Dict], 
    sample_size: int, seed: int, perplexity_val: int, n_iter_val: int,
    static_gnn_target_embed_dim: int = 64 
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_embeddings_list, all_event_names_list, all_binary_labels_list = [], [], []
    display_model_name = arch_name.upper()

    if model_type == "raw_sbert":
        if not static_gnn_data_path or not static_gnn_data_path.exists(): logging.error(f"Static GNN data path for raw SBERT not found: {static_gnn_data_path}"); return pd.DataFrame()
        event_iterator_dirs = sorted([d for d in static_gnn_data_path.iterdir() if d.is_dir() and d.name != "all"], key=lambda p:p.name)
        if not event_iterator_dirs: logging.warning(f"No event subdirs in {static_gnn_data_path}"); return pd.DataFrame()
        display_model_name = "Raw SBERT"; static_gnn_set_seed(seed)
        for event_dir in tqdm(event_iterator_dirs, desc=f"Processing {display_model_name}"):
            data_obj = load_static_event_data(event_dir)
            if data_obj and data_obj.x is not None:
                all_embeddings_list.append(data_obj.x.cpu().numpy())
                all_event_names_list.extend([data_obj.event_name] * data_obj.x.shape[0])
                all_binary_labels_list.extend(data_obj.y.cpu().numpy().tolist())

    elif model_type == "static_gnn":
        if not HAS_GNN_SCRIPT: logging.error("GNN training script unavailable."); return pd.DataFrame()
        if not static_gnn_data_path or not static_gnn_data_path.exists(): logging.error(f"Static GNN data path not found: {static_gnn_data_path}"); return pd.DataFrame()
        event_iterator_dirs = sorted([d for d in static_gnn_data_path.iterdir() if d.is_dir() and d.name != "all"], key=lambda p:p.name)
        if not event_iterator_dirs: logging.warning(f"No event subdirs in {static_gnn_data_path}"); return pd.DataFrame()
        static_gnn_set_seed(seed)
        for event_dir in tqdm(event_iterator_dirs, desc=f"Processing {display_model_name}"):
            data_obj = load_static_event_data(event_dir)
            if not data_obj: continue
            results_file = event_dir / f"results_{arch_name}.json"
            cfg = {}
            num_classes = data_obj.num_classes
            if not results_file.exists(): 
                logging.warning(f"No results JSON for {data_obj.event_name}/{arch_name}. Using default HPs: hidden_dim={static_gnn_target_embed_dim}")
                cfg = {"hidden_dim": static_gnn_target_embed_dim, "num_layers": 2, "dropout": 0.5, "heads": 4}
            else: cfg = json.loads(results_file.read_text())["best_cfg"]
            
            model_hidden_dim = cfg["hidden_dim"] 
            model = _make_model(arch=arch_name, in_dim=data_obj.num_node_features, hidden_dim=model_hidden_dim, 
                                num_layers=cfg["num_layers"], num_classes=num_classes,
                                dropout=cfg["dropout"], heads=cfg.get("heads", 4))
            model.num_classes = num_classes 
            if not load_checkpoint_static_gnn(model, event_dir, arch_name): continue
            emb = extract_penultimate_static_gnn(model, data_obj, device, target_dim=static_gnn_target_embed_dim) 
            if emb is not None and emb.shape[1] == static_gnn_target_embed_dim:
                all_embeddings_list.append(emb); all_event_names_list.extend([data_obj.event_name]*emb.shape[0]); all_binary_labels_list.extend(data_obj.y.cpu().numpy().tolist())
            elif emb is not None: logging.warning(f"Static GNN {arch_name} for {data_obj.event_name} emb shape {emb.shape} != target {static_gnn_target_embed_dim}. Skipping.")
    
    elif model_type == "fulldysat":
        if not HAS_DYSAT_SCRIPT: logging.error("DySAT training script unavailable."); return pd.DataFrame()
        if not dysat_data_path or not dysat_results_path or not dysat_run_tag_for_model: logging.error("DySAT paths/run_tag missing."); return pd.DataFrame()
        if not dysat_data_path.exists(): logging.error(f"DySAT data path not found: {dysat_data_path}"); return pd.DataFrame()
        event_iterator_dirs = sorted([d for d in dysat_data_path.iterdir() if d.is_dir() and d.name != "all"], key=lambda p:p.name)
        if not event_iterator_dirs: logging.warning(f"No event subdirs in {dysat_data_path}"); return pd.DataFrame()
        dysat_set_seed(seed); display_model_name = f"FullDySAT ({dysat_run_tag_for_model[:15]})"
        hps_to_use = dysat_hps_config 
        if not hps_to_use:
            logging.error(f"No HPs for DySAT {dysat_run_tag_for_model}. Using defaults (likely WRONG).")
            hps_to_use = { "hidden_dim": 128, "num_struct_heads": 4, "num_temporal_heads": 4, "dropout": 0.2, "use_temporal_attn": True }
        logging.info(f"Using HPs for DySAT {dysat_run_tag_for_model}: {hps_to_use}")
        for event_dir in tqdm(event_iterator_dirs, desc=f"Processing {display_model_name}"):
            try:
                X_d, EI_d, NM_d, L_d, TI_d = load_dysat_event_data(event_dir, 'cpu') 
                if X_d is None or X_d.nelement() == 0: logging.warning(f"No data in {event_dir} for DySAT."); continue
                num_actual_time_steps = TI_d.get("num_windows", len(EI_d))
                current_hidden_dim = hps_to_use.get("hidden_dim", 128)
                num_struct_heads = hps_to_use.get("num_struct_heads", 4)
                num_temporal_heads = hps_to_use.get("num_temporal_heads", 4)
                if num_struct_heads <= 0 : num_struct_heads = 1 
                if num_temporal_heads <=0: num_temporal_heads = 1 
                if current_hidden_dim <=0 : current_hidden_dim = max(num_struct_heads, num_temporal_heads)

                if current_hidden_dim % num_struct_heads != 0: current_hidden_dim = num_struct_heads * max(1, current_hidden_dim // num_struct_heads)
                if current_hidden_dim % num_temporal_heads != 0: current_hidden_dim = num_temporal_heads * max(1, current_hidden_dim // num_temporal_heads)
                if current_hidden_dim != hps_to_use.get("hidden_dim"): logging.info(f"Adjusted DySAT hidden_dim to {current_hidden_dim} for {event_dir.name}.")
                
                model = FullDySAT(in_dim=X_d.size(1), hidden_dim=current_hidden_dim, num_classes=int(L_d.max().item()+1) if L_d.numel()>0 else 2,
                                  num_struct_heads=num_struct_heads, num_temporal_heads=num_temporal_heads,
                                  num_time_steps=num_actual_time_steps, dropout=hps_to_use.get("dropout",0.2),
                                  use_temporal_attn=hps_to_use.get("use_temporal_attn", True))
                if not load_checkpoint_dysat(model, dysat_results_path, event_dir.name, dysat_run_tag_for_model): continue
                emb = extract_penultimate_dysat(model, X_d, EI_d, NM_d, device)
                if emb is not None: all_embeddings_list.append(emb); all_event_names_list.extend([event_dir.name]*emb.shape[0]); all_binary_labels_list.extend(L_d.cpu().numpy().tolist())
            except Exception as e: logging.error(f"Error event {event_dir.name} for DySAT {dysat_run_tag_for_model}: {e}", exc_info=True); continue
    else: logging.error(f"Unknown model type: {model_type}"); return pd.DataFrame()

    if not all_embeddings_list: logging.warning(f"No embeddings collected for {display_model_name}."); return pd.DataFrame()
    first_emb_shape_feat = all_embeddings_list[0].shape[1]
    for i, emb_arr in enumerate(all_embeddings_list):
        if emb_arr.shape[1] != first_emb_shape_feat: logging.error(f"FATAL: Dim mismatch for vstack in {display_model_name}! Emb {i} shape {emb_arr.shape}, expected {first_emb_shape_feat} features. Skipping."); return pd.DataFrame()
    combined_embeddings = np.vstack(all_embeddings_list); combined_event_names = np.array(all_event_names_list); combined_binary_labels = np.array(all_binary_labels_list)
    if sample_size > 0 and combined_embeddings.shape[0] > sample_size:
        logging.info(f"Sampling {sample_size} points from {combined_embeddings.shape[0]} for {display_model_name}")
        np.random.seed(seed); sel = np.random.choice(combined_embeddings.shape[0], sample_size, replace=False)
        combined_embeddings = combined_embeddings[sel]; combined_event_names = combined_event_names[sel]; combined_binary_labels = combined_binary_labels[sel]
    effective_perplexity = perplexity_val
    if combined_embeddings.shape[0] <= effective_perplexity +1: effective_perplexity = max(1, combined_embeddings.shape[0] - 2)
    if effective_perplexity <=0 : effective_perplexity = 1 
    if combined_embeddings.shape[0] <= 3 : logging.warning(f"Too few samples ({combined_embeddings.shape[0]}) for t-SNE for {display_model_name}."); return pd.DataFrame()
    coords = run_tsne(combined_embeddings, 3, effective_perplexity, 'auto', n_iter_val, seed)
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2], "event": combined_event_names, "rumor": np.where(combined_binary_labels == 1, "Rumor", "Non-rumor"), "model": display_model_name})
    return df

def create_dashboard(model_dfs, output_path_html, static_image_dir: Path, height=800, width=1200):
    # (This function remains the same as the refined version from the previous response)
    fig = go.Figure()
    fig.update_layout(
        title={"text": "PHEME Embeddings Explorer (t-SNE)", "font_size": 20, "x": 0.5, "y":0.95, "xanchor": "center", "yanchor":"top"},
        scene={"xaxis_title_text": "t-SNE Dimension 1", "yaxis_title_text": "t-SNE Dimension 2", "zaxis_title_text": "t-SNE Dimension 3",
               "aspectmode": "cube", "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.5}}},
        margin={"l": 20, "r": 20, "b": 20, "t": 80}, 
        height=height, width=width, template="plotly_white",
        legend_title_text='Display Options',
        legend=dict(orientation="h", yanchor="bottom", y=1.00, xanchor="center", x=0.5, font_size=10) 
    )
    
    all_models_sorted = sorted(list(model_dfs.keys()))
    default_model_name = all_models_sorted[0] if all_models_sorted else None
    traces_info = [] 
    static_image_dir.mkdir(parents=True, exist_ok=True) 

    event_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    rumor_colorscale = [[0, 'dodgerblue'], [1, 'orangered']] 

    for model_idx, model_name in enumerate(all_models_sorted):
        df = model_dfs[model_name]
        if df.empty: continue
        unique_events = sorted(df["event"].unique())
        event_color_map = {event: event_colors[i % len(event_colors)] for i, event in enumerate(unique_events)}
        
        marker_style = dict(size=3, opacity=0.6, line=dict(width=0.35, color='rgba(40,40,40,0.8)'))

        is_default_model_event_trace = (model_name == default_model_name)
        
        trace_event_data = dict(
            x=df["x"], y=df["y"], z=df["z"], mode="markers",
            marker={**marker_style, "color": df["event"].map(event_color_map), 
                   },
            customdata=np.stack((df["event"], df["rumor"]), axis=-1),
            hovertemplate="<b>Event:</b> %{customdata[0]}<br><b>Status:</b> %{customdata[1]}<br><b>x:</b> %{x:.2f}, <b>y:</b> %{y:.2f}, <b>z:</b> %{z:.2f}<extra></extra>",
            name=f"{model_name} by Event", legendgroup=model_name, visible=is_default_model_event_trace
        )
        fig.add_trace(go.Scatter3d(trace_event_data)); 
        traces_info.append({"model": model_name, "color_by": "event", "trace_index": len(fig.data)-1})
        
        trace_rumor_data = dict(
            x=df["x"], y=df["y"], z=df["z"], mode="markers",
            marker={**marker_style, "color": df["rumor"].map({"Rumor": 1, "Non-rumor": 0}), "colorscale": rumor_colorscale,
                    "colorbar": {"title": "Status", "tickvals": [0, 1], "ticktext": ["Non-rumor", "Rumor"],
                                 "lenmode":"fraction", "len":0.55, "thickness":12, "x": 1.02, "y": 0.15, "bgcolor": "rgba(255,255,255,0.7)"}},
            customdata=np.stack((df["event"], df["rumor"]), axis=-1),
            hovertemplate="<b>Event:</b> %{customdata[0]}<br><b>Status:</b> %{customdata[1]}<br><b>x:</b> %{x:.2f}, <b>y:</b> %{y:.2f}, <b>z:</b> %{z:.2f}<extra></extra>",
            name=f"{model_name} by Rumor", legendgroup=model_name, visible=False 
        )
        fig.add_trace(go.Scatter3d(trace_rumor_data)); 
        traces_info.append({"model": model_name, "color_by": "rumor", "trace_index": len(fig.data)-1})

        # Generate static PNGs
        for color_by_static, trace_data_for_static in [("Event", trace_event_data), ("Rumor", trace_rumor_data)]:
            static_fig = go.Figure(data=[go.Scatter3d(trace_data_for_static)]) 
            static_fig.update_layout(fig.layout) 
            static_fig.update_layout(
                title=f"t-SNE: {model_name} (Colored by {color_by_static})", 
                showlegend=False, 
                scene_camera=fig.layout.scene.camera 
            ) 
            if 'colorbar' in trace_data_for_static['marker']:
                static_fig.update_traces(marker_showscale=True, selector=dict(name=trace_data_for_static['name'])) # Not ideal, name might not be unique

            safe_model_name = model_name.replace(' ','_').replace('(','').replace(')','').replace('/','_').replace(':','_').replace('.','_')
            png_filename = static_image_dir / f"{safe_model_name}_by_{color_by_static}.png"
            try: static_fig.write_image(png_filename, width=1000, height=750, scale=2.5); logging.info(f"Saved static image: {png_filename}")
            except Exception as e_png: logging.error(f"Failed to save static PNG {png_filename}: {e_png}")
            
    if traces_info and default_model_name:
        for i in range(len(fig.data)): fig.data[i].visible = False
        default_trace_idx = next((t['trace_index'] for t in traces_info if t['model'] == default_model_name and t['color_by'] == 'event'), 0)
        if default_trace_idx < len(fig.data): fig.data[default_trace_idx].visible = True

    buttons = []; active_button_index = 0; btn_counter = 0
    for model_name_btn in all_models_sorted:
        for color_by_btn in ["event", "rumor"]:
            visibility_mask = [False] * len(fig.data)
            target_trace_idx = next((t['trace_index'] for t in traces_info if t['model'] == model_name_btn and t['color_by'] == color_by_btn), None)
            if target_trace_idx is not None: visibility_mask[target_trace_idx] = True
            buttons.append(dict(label=f"{model_name_btn} (by {color_by_btn.capitalize()})", method="update", args=[{"visible": visibility_mask}]))
            if model_name_btn == default_model_name and color_by_btn == "event": active_button_index = btn_counter
            btn_counter +=1
    camera_buttons = [dict(label=lbl, method="relayout", args=[{"scene.camera": cam}]) for lbl, cam in [
        ("Default", {"eye": {"x":1.5,"y":1.5,"z":1.5}}), ("Top", {"eye": {"x":0,"y":0,"z":2.5}}),
        ("Side XZ", {"eye": {"x":2.5,"y":0,"z":0}}), ("Side YZ", {"eye": {"x":0,"y":2.5,"z":0}})]]
    fig.update_layout(updatemenus=[
        dict(type="dropdown", direction="down", showactive=True, x=0.01, y=0.99, xanchor="left", yanchor="top", buttons=buttons, active=active_button_index, font_size=10, bgcolor='rgba(240,240,240,0.8)'),
        dict(type="buttons", direction="right", showactive=True, x=0.01, y=0.90, xanchor="left", yanchor="top", buttons=camera_buttons, pad={"r":3,"t":3}, font_size=10, bgcolor='rgba(240,240,240,0.8)')])
    logging.info(f"Saving dashboard HTML to {output_path_html}"); fig.write_html(output_path_html, include_plotlyjs="cdn", full_html=True, config={"displayModeBar":True,"responsive":True,"scrollZoom":True, "toImageButtonOptions": {"format": "png", "filename": "dashboard_export", "height": height, "width": width, "scale": 2}})
    return fig

def parse_args():
    p = argparse.ArgumentParser(description="Create PHEME embeddings dashboard with t-SNE")
    p.add_argument("--static-gnn-data-dir", default="GCN/data", help="Dir for GCN/GAT/GATv2 data (relative to project root)")
    p.add_argument("--dysat-data-dir", default="DySAT/data_dysat_v2", help="Dir for DySAT data (relative to project root)")
    p.add_argument("--dysat-results-dir", default="DySAT/results_dysat_full", help="Dir for DySAT model checkpoints (relative to project root)")
    p.add_argument("--dysat-run-tag", type=str, default=None, help="Specific run_tag for DySAT models")
    p.add_argument("--dysat-hps-json", type=str, default=None, help="Path to JSON with HPs for DySAT run_tag (relative to project root)")
    p.add_argument("--static-gnn-emb-dim", type=int, default=64, help="Target embedding dimension for static GNNs for t-SNE input")
    p.add_argument("--output-html", default="pheme_embeddings_dashboard_combined.html", help="Output HTML dashboard file")
    p.add_argument("--output-static-img-dir", default="dashboard_static_images", help="Directory to save static PNG images of plots")
    p.add_argument("--archs", nargs="+", default=["raw", "gcn", "gat", "gatv2", "FullDySAT"], choices=["raw", "gcn", "gat", "gatv2", "FullDySAT"])
    p.add_argument("--seed", type=int, default=42); p.add_argument("--sample", type=int, default=3000)
    p.add_argument("--height", type=int, default=900); p.add_argument("--width", type=int, default=1400)
    p.add_argument("--perplexity", type=int, default=30); p.add_argument("--n-iter", type=int, default=1000)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    if not args.quiet: logging.getLogger().setLevel(logging.INFO)
    else: logging.getLogger().setLevel(logging.WARNING)
    if has_cuml: logging.info("Using GPU-accelerated t-SNE from RAPIDS cuML.")
    else: logging.info("RAPIDS cuML not available, falling back to scikit-learn t-SNE (CPU).")

    static_gnn_data_path = CWD_PROJECT_ROOT / args.static_gnn_data_dir if args.static_gnn_data_dir else None
    dysat_data_path = CWD_PROJECT_ROOT / args.dysat_data_dir if args.dysat_data_dir else None
    dysat_results_path = CWD_PROJECT_ROOT / args.dysat_results_dir if args.dysat_results_dir else None
    dysat_hps_json_abs_path = CWD_PROJECT_ROOT / args.dysat_hps_json if args.dysat_hps_json else None
            
    dysat_hps_config = None
    if args.dysat_hps_json and dysat_hps_json_abs_path: 
        dysat_hps_config = load_dysat_hps_from_json(dysat_hps_json_abs_path) # Pass Path object
    
    if "fulldysat" in [a.lower() for a in args.archs] and not dysat_hps_config : 
         logging.warning(f"Default DySAT HPs used as JSON was not loaded from '{args.dysat_hps_json}' or path was not specified correctly.")
    
    start_time = time.time(); model_dfs = {}
    for arch_arg in args.archs:
        arch_lower = arch_arg.lower(); logging.info(f"--- Processing {arch_arg.upper()} embeddings ---"); df = pd.DataFrame() 
        if arch_lower == "raw":
            df = process_model_data("raw_sbert", "raw", static_gnn_data_path, None, None, None, None,
                                    args.sample, args.seed, args.perplexity, args.n_iter, args.static_gnn_emb_dim)
            if not df.empty: model_dfs["Raw SBERT"] = df
        elif arch_lower in ["gcn", "gat", "gatv2"]:
            df = process_model_data("static_gnn", arch_lower, static_gnn_data_path, None, None, None, None,
                                    args.sample, args.seed, args.perplexity, args.n_iter, args.static_gnn_emb_dim)
            if not df.empty: model_dfs[arch_arg.upper()] = df
        elif arch_lower == "fulldysat":
            if not args.dysat_run_tag: logging.error("--dysat-run-tag required for FullDySAT. Skipping."); continue
            df = process_model_data("fulldysat", "FullDySAT", None, dysat_data_path, dysat_results_path, 
                                    args.dysat_run_tag, dysat_hps_config, 
                                    args.sample, args.seed, args.perplexity, args.n_iter, args.static_gnn_emb_dim) 
            if not df.empty: model_dfs[f"FullDySAT ({args.dysat_run_tag[:15]})"] = df
        else: logging.warning(f"Unsupported architecture '{arch_arg}'. Skipping.")
            
    if not model_dfs: logging.error("No valid embeddings processed."); sys.exit(1)
    
    output_html_path = CWD_PROJECT_ROOT / args.output_html; output_html_path.parent.mkdir(parents=True, exist_ok=True)
    static_image_output_dir = CWD_PROJECT_ROOT / args.output_static_img_dir 
    
    create_dashboard(model_dfs, output_html_path, static_image_output_dir, args.height, args.width)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Dashboard & static images created in {elapsed_time:.1f}s.")
    logging.info(f"HTML dashboard: {output_html_path}")
    logging.info(f"Static images: {static_image_output_dir}")
    if not args.quiet and "DISPLAY" in os.environ and os.environ["DISPLAY"] != "":
        try: import webbrowser; webbrowser.open(f"file://{output_html_path}"); logging.info("Opening dashboard...")
        except Exception as e: logging.warning(f"Failed to open browser: {e}")
                
if __name__ == "__main__":
    main()