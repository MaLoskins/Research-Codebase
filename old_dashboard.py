#!/usr/bin/env python3
"""
Robust PHEME Embeddings Dashboard

This script creates a reliable dashboard for visualizing PHEME embeddings
from different GNN architectures with proper color scheme toggling.
Uses t-SNE for dimensionality reduction with GPU acceleration if available.
"""
import argparse
import json
import logging
import os
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch import nn
from torch_geometric.data import Data
from tqdm import tqdm

# Try to import cuml for GPU-accelerated t-SNE
try:
    from cuml.manifold import TSNE as cuTSNE
    has_cuml = True
    logging.info("Using GPU-accelerated t-SNE from RAPIDS cuML")
except ImportError:
    has_cuml = False
    logging.info("RAPIDS cuML not available, falling back to scikit-learn t-SNE")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError("Plotly is required. Install with: pip install plotly")

try:
    import pandas as pd
except ImportError:
    raise ImportError("Pandas is required. Install with: pip install pandas")

try:
    from GCN.train_gnns_pheme import (
        GCN, GAT, GATv2, _make_model, set_seed,
    )
except ImportError as e:
    raise SystemExit("❌ Could not import train_gnns_pheme.py – "
                    "make sure it's on PYTHONPATH") from e

# ----------------------------------------------------------------------
# Core Functions
# ----------------------------------------------------------------------

def load_event_data(event_dir: Path) -> Data:
    """Load `X.npy`, `edge_index.npy`, `labels.npy`."""
    x = torch.from_numpy(np.load(event_dir / "X.npy").astype(np.float32))
    edge_index = torch.from_numpy(np.load(event_dir / "edge_index.npy").astype(np.int64))
    y = torch.from_numpy(np.load(event_dir / "labels.npy").astype(np.int64))
    return Data(x=x, edge_index=edge_index, y=y)


@torch.no_grad()
def extract_penultimate(model: nn.Module, data: Data, device: torch.device) -> np.ndarray:
    """Return embeddings before the final classifier layer."""
    model = model.to(device)
    model.eval()
    x, ei = data.x.to(device), data.edge_index.to(device)
    for conv in model.convs[:-1]:
        x = conv(x, ei)
        x = F.relu(x) if isinstance(model, GCN) else F.elu(x)
    return x.cpu().numpy()


def load_checkpoint(model: nn.Module, event_dir: Path, arch: str) -> bool:
    """Load pre-trained model checkpoint."""
    ckpt = event_dir / f"model_{arch}.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        logging.info(f"Loaded {ckpt.name}")
        return True
    return False


def run_tsne(data, n_components=3, perplexity=30, learning_rate='auto', n_iter=1000, seed=42):
    """Run t-SNE with GPU acceleration if available."""
    if has_cuml:
        # Use GPU-accelerated t-SNE
        tsne = cuTSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=seed
        )
    else:
        # Fall back to scikit-learn t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=seed,
            n_jobs=-1  # Use all CPU cores
        )
    
    logging.info(f"Running t-SNE on data with shape {data.shape}...")
    return tsne.fit_transform(data)


def process_model_data(data_dir: Path, arch: str, 
                      sample_size: int = 10000, 
                      seed: int = 42,
                      is_raw: bool = False) -> pd.DataFrame:
    """Process data for a specific model and return a DataFrame."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find event directories
    dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name != "all"]
    dirs.sort(key=lambda p: p.name)
    
    if is_raw:  # Process raw SBERT embeddings
        X_list, event_lbls, binary_lbls = [], [], []
        
        for d in tqdm(dirs, desc="Loading raw data"):
            data = load_event_data(d)
            X_list.append(data.x.numpy())
            event_lbls.extend([d.name] * data.x.shape[0])
            binary_lbls.extend(data.y.cpu().numpy().tolist())
        
        X_raw = np.vstack(X_list)
        event_lbls = np.array(event_lbls)
        binary_lbls = np.array(binary_lbls)
        
        # Sample if needed
        if sample_size and X_raw.shape[0] > sample_size:
            logging.info(f"Sampling {sample_size} points from {X_raw.shape[0]} total")
            np.random.seed(seed)
            sel = np.random.choice(X_raw.shape[0], sample_size, replace=False)
            X_raw = X_raw[sel]
            event_lbls = event_lbls[sel]
            binary_lbls = binary_lbls[sel]
        
        # Apply t-SNE instead of PCA
        logging.info(f"Applying t-SNE to raw embeddings...")
        # Adjust perplexity based on data size
        perplexity = min(30, X_raw.shape[0] // 4)
        coords = run_tsne(X_raw, n_components=3, perplexity=perplexity, seed=seed)
        
        # Create DataFrame
        df = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "event": event_lbls,
            "rumor": np.where(binary_lbls == 1, "Rumor", "Non-rumor"),
            "model": "Raw SBERT"
        })
        
        return df
        
    else:  # Process GNN embeddings
        all_coords = []
        all_events = []
        all_binary = []
        
        # Process each event separately to avoid dimension mismatch
        for event_dir in tqdm(dirs, desc=f"Processing {arch}"):
            try:
                data = load_event_data(event_dir)
                results_file = event_dir / f"results_{arch}.json"
                
                if not results_file.exists():
                    logging.warning(f"Missing {results_file.name}; skipping {event_dir.name}")
                    continue
                
                # Load model configuration
                cfg = json.loads(results_file.read_text())["best_cfg"]
                model = _make_model(
                    arch=arch, 
                    in_dim=data.num_node_features, 
                    hidden_dim=cfg["hidden_dim"],
                    num_layers=cfg["num_layers"], 
                    num_classes=int(data.y.max().item() + 1),
                    dropout=cfg["dropout"], 
                    heads=cfg.get("heads", 4)
                )
                
                # Load pre-trained model
                if not load_checkpoint(model, event_dir, arch):
                    logging.warning(f"No checkpoint for {event_dir.name}/{arch} - skipping")
                    continue
                
                # Extract embeddings
                emb = extract_penultimate(model, data, device)
                
                # Apply t-SNE instead of PCA
                # Adjust perplexity based on data size
                perplexity = min(30, emb.shape[0] // 4)
                coords_event = run_tsne(emb, n_components=3, perplexity=perplexity, seed=seed)
                
                # Store
                all_coords.append(coords_event)
                all_events.extend([event_dir.name] * emb.shape[0])
                all_binary.extend(data.y.cpu().numpy().tolist())
                
            except Exception as e:
                logging.error(f"Error processing {event_dir.name} with {arch}: {e}")
                continue
                
        if not all_coords:
            logging.warning(f"No valid embeddings for {arch}")
            return pd.DataFrame()  # Empty DataFrame
            
        # Combine all event data
        coords_combined = np.vstack(all_coords)
        events_combined = np.array(all_events)
        binary_combined = np.array(all_binary)
        
        # Sample if needed
        if sample_size and coords_combined.shape[0] > sample_size:
            logging.info(f"Sampling {sample_size} points from {coords_combined.shape[0]} total")
            np.random.seed(seed)
            sel = np.random.choice(coords_combined.shape[0], sample_size, replace=False)
            coords_combined = coords_combined[sel]
            events_combined = events_combined[sel]
            binary_combined = binary_combined[sel]
            
        # Create DataFrame
        df = pd.DataFrame({
            "x": coords_combined[:, 0],
            "y": coords_combined[:, 1],
            "z": coords_combined[:, 2],
            "event": events_combined,
            "rumor": np.where(binary_combined == 1, "Rumor", "Non-rumor"),
            "model": arch.upper()
        })
        
        return df


def create_dashboard(model_dfs, output_path, height=800, width=1000):
    """Create an interactive dashboard from model DataFrames."""
    # Create figure
    fig = go.Figure()
    
    # Set title and layout
    fig.update_layout(
        title={
            "text": "PHEME Embeddings Explorer (t-SNE)",
            "font": {"size": 24},
            "x": 0.5,
        },
        scene={
            "xaxis": {"title": "t-SNE Dimension 1"},
            "yaxis": {"title": "t-SNE Dimension 2"},
            "zaxis": {"title": "t-SNE Dimension 3"},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.5}},
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 80},
        height=height,
        width=width,
        template="plotly_white",
    )
    
    # Track the models and default states
    models = []
    model_trace_indices = {}  # Maps model name to (event_trace_idx, rumor_trace_idx)
    
    # Set default model (first in list) and coloring (event)
    default_model = next(iter(model_dfs.keys()))
    default_coloring = "event"  # "event" or "rumor"
    
    # Add traces for each model
    trace_idx = 0
    for model_name, df in model_dfs.items():
        if df.empty:
            continue
            
        models.append(model_name)
        
        # Is this the default model to show?
        is_default = (model_name == default_model)
        
        # Create trace for event coloring
        event_trace = go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode="markers",
            marker={
                "size": 5,
                "color": df["event"].astype("category").cat.codes,
                "colorscale": "rainbow",
                "opacity": 0.7,
                "colorbar": {
                    "title": "Event", 
                    "tickvals": list(range(len(df["event"].unique()))),
                    "ticktext": sorted(df["event"].unique()),
                    "lenmode": "fraction",
                    "len": 0.75,
                    "thickness": 15,
                    "x": 0.95,
                },
            },
            customdata=np.stack((
                df["event"],
                df["rumor"],
            ), axis=-1),
            hovertemplate=(
                "<b>Event:</b> %{customdata[0]}<br>"
                "<b>Status:</b> %{customdata[1]}<br>"
                "<b>x:</b> %{x:.2f}<br>"
                "<b>y:</b> %{y:.2f}<br>"
                "<b>z:</b> %{z:.2f}<br>"
                "<extra></extra>"
            ),
            name=f"{model_name}",
            legendgroup=model_name,
            visible=(is_default and default_coloring == "event"),
        )
        fig.add_trace(event_trace)
        event_trace_idx = trace_idx
        trace_idx += 1
        
        # Create trace for rumor coloring
        rumor_trace = go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode="markers",
            marker={
                "size": 5,
                "color": df["rumor"].map({"Rumor": 1, "Non-rumor": 0}),
                "colorscale": [[0, "green"], [1, "red"]],
                "opacity": 0.7,
                "colorbar": {
                    "title": "Rumor Status", 
                    "tickvals": [0, 1],
                    "ticktext": ["Non-rumor", "Rumor"],
                    "lenmode": "fraction",
                    "len": 0.75,
                    "thickness": 15,
                    "x": 0.95,
                },
            },
            customdata=np.stack((
                df["event"],
                df["rumor"],
            ), axis=-1),
            hovertemplate=(
                "<b>Event:</b> %{customdata[0]}<br>"
                "<b>Status:</b> %{customdata[1]}<br>"
                "<b>x:</b> %{x:.2f}<br>"
                "<b>y:</b> %{y:.2f}<br>"
                "<b>z:</b> %{z:.2f}<br>"
                "<extra></extra>"
            ),
            name=f"{model_name}",
            legendgroup=model_name,
            visible=(is_default and default_coloring == "rumor"),
        )
        fig.add_trace(rumor_trace)
        rumor_trace_idx = trace_idx
        trace_idx += 1
        
        # Store the trace indices for this model
        model_trace_indices[model_name] = (event_trace_idx, rumor_trace_idx)
    
    # Create arrays storing all possible button states
    button_states = {}
    
    # For each model and coloring combination, create a visibility array
    for model_name in models:
        button_states[model_name] = {}
        event_idx, rumor_idx = model_trace_indices[model_name]
        
        # Event coloring for this model
        event_visibility = [False] * len(fig.data)
        event_visibility[event_idx] = True
        button_states[model_name]["event"] = event_visibility
        
        # Rumor coloring for this model
        rumor_visibility = [False] * len(fig.data)
        rumor_visibility[rumor_idx] = True
        button_states[model_name]["rumor"] = rumor_visibility
    
    # Create model buttons - each preserves the current coloring scheme
    model_buttons = []
    for model_name in models:
        model_buttons.append(
            dict(
                label=model_name,
                method="update",
                args=[{"visible": button_states[model_name][default_coloring]}],
            )
        )
    
    # Create coloring buttons - each preserves the current model selection
    coloring_buttons = []
    
    # Button for event coloring
    coloring_buttons.append(
        dict(
            label="Color by Event",
            method="update",
            args=[{"visible": button_states[default_model]["event"]}],
        )
    )
    
    # Button for rumor coloring
    coloring_buttons.append(
        dict(
            label="Color by Rumor Status",
            method="update",
            args=[{"visible": button_states[default_model]["rumor"]}],
        )
    )
    
    # Create a more advanced menu system using buttons + annotations
    # The key is to create buttons for EVERY possible combination
    all_buttons = []
    
    # Create a complete button for each model + coloring combination
    for model_name in models:
        for coloring in ["event", "rumor"]:
            visibility = button_states[model_name][coloring]
            button_label = f"{model_name} - {coloring.capitalize()}"
            all_buttons.append(
                dict(
                    label=button_label,
                    method="update",
                    args=[{"visible": visibility}],
                )
            )
    
    # Add camera angle presets
    camera_buttons = [
        dict(
            label="Default View",
            method="relayout", 
            args=[{"scene.camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.5}}}],
        ),
        dict(
            label="Top View",
            method="relayout", 
            args=[{"scene.camera": {"eye": {"x": 0, "y": 0, "z": 2}}}],
        ),
        dict(
            label="Side View",
            method="relayout", 
            args=[{"scene.camera": {"eye": {"x": 2, "y": 0, "z": 0}}}],
        ),
    ]
    
    # Add control menus - the standard approach
    fig.update_layout(
        updatemenus=[

            # Camera controls
            dict(
                buttons=camera_buttons,
                direction="right",
                showactive=True,
                x=0.2,
                y=1.01,
                xanchor="left",
                yanchor="top",
                type="buttons",
                name="Camera",
            ),
            # THE FIX: Add a new dropdown with all possible combinations
            # This ensures we can get to any model+color combination
            dict(
                buttons=all_buttons,
                direction="down",
                showactive=True,
                x=0.9,
                y=1.01, 
                xanchor="right",
                yanchor="top",
                type="dropdown",
                name="All Combinations",
            ),
        ]
    )
    
    # Add annotations for menus
    fig.update_layout(
        annotations=[

            dict(
                text="Camera Angle:",
                x=0.08,
                y=1.00,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14),
            ),
            dict(
                text="All Options:",
                x=0.88,
                y=1.1,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14),
            ),
        ]
    )
    
    # Add explanatory text at the bottom
    footer_text = (
        "• Click and drag to rotate the view, scroll to zoom, right-click to pan<br>"
        "• Hover over points to see detailed information<br>"
        "• Double-click a point to center the view, double-click again to reset"
    )
    
    fig.add_annotation(
        text=footer_text,
        x=0.5,
        y=0,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12),
        align="center",
        bordercolor="lightgray",
        borderwidth=1,
        borderpad=4,
        bgcolor="white",
        opacity=0.8,
    )
    
    # Save to HTML
    logging.info(f"Saving dashboard to {output_path}")
    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displayModeBar": True,
            "responsive": True,
            "scrollZoom": True,
        }
    )
    
    return fig


def parse_args():
    p = argparse.ArgumentParser(description="Create PHEME embeddings dashboard with t-SNE")
    p.add_argument("--data-dir", default="GCN/data", help="Directory containing PHEME data")
    p.add_argument("--output", default="pheme_tsne_dashboard.html", help="Output HTML file")
    p.add_argument("--archs", nargs="+", default=["gcn", "gat", "gatv2"], 
                   choices=["gcn", "gat", "gatv2"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample", type=int, default=10000)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--width", type=int, default=1000)
    p.add_argument("--perplexity", type=int, default=30, 
                   help="t-SNE perplexity parameter (typical range: 5-50)")
    p.add_argument("--n-iter", type=int, default=1000,
                   help="t-SNE iterations (higher values: better quality, slower)")
    p.add_argument("--quiet", action="store_true")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=log_level)
    
    # Make data_dir path absolute if it's not already
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print(f"Current working directory: {os.getcwd()}")
        sys.exit(1)
    
    try:
        # Start timing
        start_time = time.time()
        
        # Store DataFrames for each model
        model_dfs = {}
        
        # Process raw embeddings
        logging.info("Processing raw SBERT embeddings...")
        raw_df = process_model_data(
            data_dir=data_dir,
            arch="raw",
            sample_size=args.sample,
            seed=args.seed,
            is_raw=True
        )
        if not raw_df.empty:
            model_dfs["Raw SBERT"] = raw_df
        
        # Process each architecture
        for arch in args.archs:
            logging.info(f"Processing {arch.upper()} embeddings...")
            arch_df = process_model_data(
                data_dir=data_dir,
                arch=arch,
                sample_size=args.sample,
                seed=args.seed,
                is_raw=False
            )
            if not arch_df.empty:
                model_dfs[arch.upper()] = arch_df
        
        if not model_dfs:
            print("No valid embeddings found for any model.")
            sys.exit(1)
        
        # Create output directory if necessary
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create and save the dashboard
        fig = create_dashboard(
            model_dfs=model_dfs,
            output_path=output_path,
            height=args.height,
            width=args.width
        )
        
        # Report completion
        elapsed_time = time.time() - start_time
        logging.info(f"Dashboard created successfully in {elapsed_time:.1f} seconds")
        logging.info(f"Dashboard saved to: {output_path.resolve()}")
        
        # Open in browser if not in a headless environment
        if not args.quiet and not os.environ.get("DISPLAY") == "":
            try:
                import webbrowser
                webbrowser.open(f"file://{output_path.resolve()}")
                logging.info("Opening dashboard in web browser...")
            except Exception as e:
                logging.warning(f"Failed to open browser: {e}")
                
    except Exception as e:
        print(f"Error creating dashboard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()