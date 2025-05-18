#!/usr/bin/env python3
"""
Animate t-SNE Embeddings Evolution During GNN Training.
"""
import argparse
import json
import logging
from pathlib import Path
import shutil
from typing import List, Dict, Any, Optional 

import imageio.v2 as imageio 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D 
from tqdm import tqdm
import matplotlib.colors as mcolors 
import seaborn as sns # Import seaborn for styling if preferred

# --- Configure logging ---
logging.basicConfig(format="%(asctime)s | %(levelname)-7s | %(message)s", level=logging.INFO)

# --- Try to import cuML for GPU-accelerated t-SNE ---
try:
    from cuml.manifold import TSNE as cuTSNE
    HAS_CUML = True
    logging.info("cuML found. Will use GPU for t-SNE if available and enabled.")
except ImportError:
    HAS_CUML = False
    logging.info("cuML not found. Using scikit-learn (CPU) for t-SNE.")


def run_tsne_on_all_epochs(
    all_epoch_embeddings_raw: List[np.ndarray],
    epoch_numbers_for_logging: List[int], 
    n_components: int = 3, 
    perplexity: int = 30, 
    learning_rate: Any = 'auto',
    n_iter: int = 300, 
    seed: int = 42,
    use_gpu: bool = False
) -> List[np.ndarray]:
    transformed_embeddings_all_epochs = []
    
    base_tsne_params = {
        "learning_rate": learning_rate,
        "random_state": seed,
    }
    try: TSNE(max_iter=n_iter); base_tsne_params["max_iter"] = n_iter
    except TypeError: base_tsne_params["n_iter"] = n_iter

    if use_gpu and HAS_CUML:
        logging.info("Using cuML t-SNE for each epoch.")
        base_tsne_params["method"] = 'barnes_hut' 
    else:
        logging.info("Using scikit-learn t-SNE for each epoch.")
        base_tsne_params["n_jobs"] = -1

    for i, epoch_embeddings in enumerate(tqdm(all_epoch_embeddings_raw, desc="Running t-SNE per epoch")):
        current_epoch_num_log = epoch_numbers_for_logging[i] if i < len(epoch_numbers_for_logging) else f"index {i}"
        n_samples_epoch, n_features_epoch = epoch_embeddings.shape
        
        current_n_components = min(n_components, n_features_epoch) if n_features_epoch > 0 else n_components
        if n_samples_epoch <= current_n_components:
            current_n_components = max(1, n_samples_epoch - 1)
        
        if n_samples_epoch <= 1 or current_n_components <= 0 : 
            logging.warning(f"Epoch {current_epoch_num_log}: Too few samples ({n_samples_epoch}) or effective components ({current_n_components}) for t-SNE. Appending zeros.")
            transformed_embeddings_all_epochs.append(np.zeros((n_samples_epoch, n_components)))
            continue
        
        current_perplexity = perplexity
        if n_samples_epoch <= current_perplexity: 
            current_perplexity = max(1, n_samples_epoch - 1) 
            logging.warning(f"Epoch {current_epoch_num_log}: Perplexity {perplexity} too high for {n_samples_epoch} samples. Adjusted to {current_perplexity}.")

        epoch_specific_tsne_params = base_tsne_params.copy()
        epoch_specific_tsne_params["n_components"] = current_n_components
        epoch_specific_tsne_params["perplexity"] = current_perplexity

        try:
            if use_gpu and HAS_CUML:
                cuml_params = {k: v for k, v in epoch_specific_tsne_params.items() if k not in ['n_jobs', 'init', 'n_iter', 'max_iter']}
                if "max_iter" in base_tsne_params: cuml_params["n_iter"] = base_tsne_params["max_iter"]
                elif "n_iter" in base_tsne_params: cuml_params["n_iter"] = base_tsne_params["n_iter"]
                tsne_model = cuTSNE(**cuml_params)
            else:
                current_init = 'pca'
                if n_features_epoch < current_n_components or n_features_epoch < 2 : 
                    current_init = 'random'
                epoch_specific_tsne_params["init"] = current_init
                tsne_model = TSNE(**epoch_specific_tsne_params)
            
            transformed = tsne_model.fit_transform(epoch_embeddings)
            
            if transformed.shape[1] < n_components:
                 padding = np.zeros((transformed.shape[0], n_components - transformed.shape[1]))
                 transformed = np.hstack((transformed, padding))
            transformed_embeddings_all_epochs.append(transformed)

        except Exception as e:
            logging.error(f"t-SNE failed for epoch {current_epoch_num_log}: {e}. Appending zeros.")
            transformed_embeddings_all_epochs.append(np.zeros((n_samples_epoch, n_components)))
            
    return transformed_embeddings_all_epochs


def plot_epoch_frame(
    embeddings_2d_or_3d: np.ndarray, 
    labels: np.ndarray, 
    epoch_num: int, 
    output_filename: Path,
    title_prefix: str = "",
    is_3d: bool = True,
):
    try:
        plt.style.use('seaborn-v0_8-whitegrid') 
    except OSError:
        try: plt.style.use('seaborn-whitegrid')
        except OSError: logging.warning("Seaborn whitegrid style not found, using 'ggplot'."); plt.style.use('ggplot')

    fig = plt.figure(figsize=(10, 8))
    
    rumor_color = 'orangered'
    non_rumor_color = 'dodgerblue'
    colors = [rumor_color if l == 1 else non_rumor_color for l in labels]

    edgecolor_rgba = (0.0, 0.0, 0.0, 0.25) 

    if is_3d and embeddings_2d_or_3d.shape[1] >= 3: 
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embeddings_2d_or_3d[:, 0], embeddings_2d_or_3d[:, 1], embeddings_2d_or_3d[:, 2], 
                   c=colors, alpha=0.60, s=10, 
                   edgecolors=edgecolor_rgba, 
                   linewidths=0.4) 
        ax.set_zlabel("t-SNE Dim 3", fontsize=12)
        ax.view_init(elev=20., azim=45) 
        all_coords = embeddings_2d_or_3d
        x_min, x_max = all_coords[:,0].min(), all_coords[:,0].max(); x_pad = (x_max - x_min) * 0.05 if (x_max - x_min) > 1e-6 else 1
        y_min, y_max = all_coords[:,1].min(), all_coords[:,1].max(); y_pad = (y_max - y_min) * 0.05 if (y_max - y_min) > 1e-6 else 1
        z_min, z_max = all_coords[:,2].min(), all_coords[:,2].max(); z_pad = (z_max - z_min) * 0.05 if (z_max - z_min) > 1e-6 else 1
        ax.set_xlim([x_min - x_pad, x_max + x_pad]); ax.set_ylim([y_min - y_pad, y_max + y_pad]); ax.set_zlim([z_min - z_pad, z_max + z_pad])

    elif embeddings_2d_or_3d.shape[1] >= 2 : 
        is_3d = False 
        ax = fig.add_subplot(111)
        ax.scatter(embeddings_2d_or_3d[:, 0], embeddings_2d_or_3d[:, 1], 
                   c=colors, alpha=0.60, s=10, 
                   edgecolors=edgecolor_rgba, 
                   linewidths=0.4)
        all_coords = embeddings_2d_or_3d
        x_min, x_max = all_coords[:,0].min(), all_coords[:,0].max(); x_pad = (x_max - x_min) * 0.05 if (x_max - x_min) > 1e-6 else 1
        y_min, y_max = all_coords[:,1].min(), all_coords[:,1].max(); y_pad = (y_max - y_min) * 0.05 if (y_max - y_min) > 1e-6 else 1
        ax.set_xlim([x_min - x_pad, x_max + x_pad]); ax.set_ylim([y_min - y_pad, y_max + y_pad])
    else:
        logging.warning(f"Not enough dimensions ({embeddings_2d_or_3d.shape[1]}) to plot for epoch {epoch_num}. Skipping frame.")
        plt.close(fig); return

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Non-Rumor', markersize=8, markerfacecolor=non_rumor_color, markeredgecolor='grey'),
        plt.Line2D([0], [0], marker='o', color='w', label='Rumor', markersize=8, markerfacecolor=rumor_color, markeredgecolor='grey')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10, frameon=True, facecolor='white', framealpha=0.7) 
    fig.suptitle(f"{title_prefix}\nEpoch: {epoch_num}", fontsize=14, y=0.97) 
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(output_filename, dpi=120) 
    plt.close(fig)


def create_animation(image_folder: Path, output_gif_path: Path, fps: int = 5):
    images = []
    def get_epoch_num_from_frame_filename(path_obj: Path) -> int:
        try: return int(path_obj.stem.split('_')[-1]) 
        except: return -1 
    
    filenames = sorted(image_folder.glob("frame_epoch_*.png"), key=get_epoch_num_from_frame_filename) 
    if not filenames: logging.error(f"No frames found in {image_folder} matching 'frame_epoch_*.png'."); return

    for filename in tqdm(filenames, desc="Creating animation"): images.append(imageio.imread(filename))
    if not images: logging.error(f"No images loaded for animation from {image_folder}."); return

    logging.info(f"Saving animation with {len(images)} frames to {output_gif_path} at {fps} FPS...")
    try: imageio.mimsave(output_gif_path, images, fps=fps, loop=0, subrectangles=True) 
    except Exception as e: logging.error(f"Error saving GIF: {e}. Try 'pip install imageio[ffmpeg]' or 'pip install pillow'.")
    logging.info(f"Animation saved successfully.")


def parse_args_animation():
    p = argparse.ArgumentParser(description="Animate GNN embedding evolution.")
    p.add_argument("--embeddings-dir", type=str, required=True, 
                   help="Directory containing epoch-wise embeddings (e.g., animation_data_gnn/gcn/germanwings-crash/)")
    p.add_argument("--output-gif", type=str, default="embedding_animation.gif",
                   help="Output GIF file name (will be saved inside embeddings-dir).")
    p.add_argument("--frames-subdir", type=str, default="animation_frames_temp", 
                   help="Subdirectory within embeddings-dir to store temporary frame images.")
    p.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity.")
    p.add_argument("--n-iter", type=int, default=250, help="t-SNE iterations (sklearn default is 1000, 250 is min for convergence).")
    p.add_argument("--fps", type=int, default=5, help="Frames per second for the GIF.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for t-SNE.")
    p.add_argument("--use-gpu-tsne", action="store_true", help="Use cuML for GPU accelerated t-SNE.")
    p.add_argument("--plot-3d", action="store_true", help="Generate 3D t-SNE plots (default is 2D if not specified).")
    return p.parse_args()

def main_animation():
    args = parse_args_animation()
    
    # Resolve paths relative to CWD, as this script might be run from anywhere
    # The --embeddings-dir is expected to be relative to CWD or absolute.
    embeddings_path = Path(args.embeddings_dir).resolve()
    if not embeddings_path.exists() or not embeddings_path.is_dir():
        logging.error(f"Embeddings directory not found: {embeddings_path}"); sys.exit(1)

    frames_dir = embeddings_path / args.frames_subdir
    if frames_dir.exists(): shutil.rmtree(frames_dir) 
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_gif = embeddings_path / args.output_gif

    try:
        labels = np.load(embeddings_path / "labels.npy")
        with open(embeddings_path / "info.json", "r") as f: info = json.load(f)
        event_name = info.get("event_name", "Unknown Event")
        arch = info.get("arch", "Unknown Arch")
    except FileNotFoundError: logging.error(f"labels.npy or info.json missing in {embeddings_path}."); sys.exit(1)

    def get_epoch_from_filename_stem(stem_str: str) -> int: 
        parts = stem_str.split('_')
        if len(parts) >= 2 and parts[0] == "epoch" and parts[1].isdigit():
            return int(parts[1])
        logging.warning(f"Could not parse epoch number from stem: {stem_str}. Assigning -1.")
        return -1

    epoch_embedding_files = sorted(
        list(embeddings_path.glob("epoch_*_embeddings.npy")),
        key=lambda x: get_epoch_from_filename_stem(x.stem)
    )
    if not epoch_embedding_files: logging.error(f"No epoch embedding files found in {embeddings_path} (e.g. epoch_000_embeddings.npy)"); sys.exit(1)

    all_epoch_embeddings_raw = []
    epoch_numbers = []
    for f in epoch_embedding_files:
        try:
            epoch_num = get_epoch_from_filename_stem(f.stem)
            if epoch_num != -1:
                all_epoch_embeddings_raw.append(np.load(f))
                epoch_numbers.append(epoch_num)
        except Exception as e: logging.error(f"Error loading embedding file {f}: {e}")

    if not all_epoch_embeddings_raw: logging.error(f"No embeddings loaded from {embeddings_path}"); sys.exit(1)
    logging.info(f"Loaded {len(all_epoch_embeddings_raw)} embedding snapshots for {event_name} ({arch}). Epochs: {epoch_numbers}")

    n_plot_components = 3 if args.plot_3d else 2
    all_epoch_embeddings_tsne = run_tsne_on_all_epochs(
        all_epoch_embeddings_raw, 
        epoch_numbers_for_logging=epoch_numbers, 
        n_components=n_plot_components,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        seed=args.seed,
        use_gpu=args.use_gpu_tsne
    )

    plot_title_prefix = f"t-SNE: {arch.upper()} on {event_name}"
    for i, epoch_num in enumerate(tqdm(epoch_numbers, desc="Generating frames")):
        if i < len(all_epoch_embeddings_tsne) and all_epoch_embeddings_tsne[i].shape[0] > 0 : 
            frame_filename = frames_dir / f"frame_epoch_{epoch_num:04d}.png"
            current_labels = labels 
            if all_epoch_embeddings_tsne[i].shape[0] != labels.shape[0]:
                 logging.warning(f"Epoch {epoch_num}: Mismatch: t-SNE samples ({all_epoch_embeddings_tsne[i].shape[0]}) vs labels ({labels.shape[0]}). Skipping frame.")
                 continue
            plot_epoch_frame(all_epoch_embeddings_tsne[i], current_labels, epoch_num, frame_filename, 
                             title_prefix=plot_title_prefix, is_3d=args.plot_3d)
    create_animation(frames_dir, output_gif, fps=args.fps)

if __name__ == "__main__":
    main_animation()