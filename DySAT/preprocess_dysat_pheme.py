#!/usr/bin/env python3
"""
Preprocess the PHEME dataset for DySAT (Dynamic Self-Attention Network) experiments.
This script extracts temporal information and creates time-based graph snapshots.

Key features (May 2025)
---------------------------
1. **Temporal extraction** - Extracts timestamps from tweet data to create a dynamic 
   graph representation
2. **Time-windowed snapshots** - Divides the data into configurable time windows
3. **Cumulative graph structure** - Each snapshot contains all previous interactions
4. **Tweet embeddings** - Uses BERT embeddings similar to the static GNN pipeline
5. **Temporal node masks** - Tracks which nodes exist at each time step

Directory layout created (relative to --output-dir, default "data_dysat"):

    data_dysat/
    ├── all/                 # complete PHEME across every event
    │   ├── X.npy            # (N, 768) text embeddings
    │   ├── edge_indices/    # directory containing edge indices for each time snapshot
    │   │   ├── t0.npy       # (2, E_0) edges at time 0
    │   │   ├── t1.npy       # (2, E_1) edges at time 1
    │   │   └── ...
    │   ├── node_masks/      # masks indicating which nodes exist at each time step
    │   │   ├── t0.npy       # (N,) boolean mask for nodes at time 0
    │   │   ├── t1.npy       # (N,) boolean mask for nodes at time 1
    │   │   └── ...  
    │   ├── time_info.json   # metadata about time windows and node mapping
    │   └── labels.npy       # (N,) 0 = non‑rumour, 1 = rumour
    ├── <event_1>/           # e.g. charliehebdo/
    │   ├── X.npy
    │   ├── edge_indices/
    │   │   └── ...
    │   ├── node_masks/
    │   │   └── ...
    │   ├── time_info.json
    │   └── labels.npy
    ├── <event_2>/
    │   └── …
    └── …

Usage
-----
$ python preprocess_dysat_pheme.py \
        --data-dir PHEME \
        --output-dir data_dysat \
        --model bert-base-uncased \
        --time-window hours \
        --max-time-steps 10 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


# ---------------------------------------------------------------------------
# Data loading helpers with temporal information
# ---------------------------------------------------------------------------

def parse_twitter_time(time_str: str) -> datetime | None:
    """Parse Twitter's timestamp format into a datetime object. Returns None on failure."""
    if not time_str: return None
    try:
        # Twitter format example: "Wed Aug 27 13:08:45 +0000 2014"
        dt = datetime.strptime(time_str, "%a %b %d %H:%M:%S %z %Y")
        # Ensure timezone-aware datetime, converting to UTC if not already
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        # print(f"Warning: Could not parse timestamp '{time_str}'")
        return None


def load_pheme_temporal(root: Path) -> pd.DataFrame:
    """Walk the PHEME directory structure and return a DataFrame with temporal info.
    
    Columns: event, thread_id, tweet_id, text, label, parent_id, timestamp (UTC)
    """
    records: List[dict] = []
    events = [d for d in root.iterdir() if d.is_dir()]

    for event_dir in tqdm(events, desc="Scanning events", unit="event"):
        event = event_dir.name
        for label_name, label_val in [("rumours", 1), ("non-rumours", 0)]:
            label_dir = event_dir / label_name
            if not label_dir.exists():
                continue

            threads = [t for t in label_dir.iterdir() if t.is_dir()]
            for thread_dir in tqdm(threads, desc=f"Threads in {event}/{label_name}", leave=False, unit="thread"):
                thread_id = thread_dir.name
                
                src_file = thread_dir / "source-tweet" / f"{thread_id}.json"
                if src_file.exists():
                    with src_file.open(encoding="utf-8") as f:
                        src = json.load(f)
                    timestamp = parse_twitter_time(src.get("created_at"))
                    records.append({
                        "event": event, "thread_id": thread_id,
                        "tweet_id": str(src.get("id_str", src.get("id"))), # Ensure ID is string
                        "text": src.get("text", ""), "label": label_val,
                        "parent_id": None, "timestamp": timestamp
                    })

                replies_dir = thread_dir / "reactions"
                if replies_dir.exists():
                    for reply_path in replies_dir.glob("*.json"):
                        with reply_path.open(encoding="utf-8") as f:
                            rep = json.load(f)
                        timestamp = parse_twitter_time(rep.get("created_at"))
                        parent_id_val = rep.get("in_reply_to_status_id_str", rep.get("in_reply_to_status_id"))
                        records.append({
                            "event": event, "thread_id": thread_id,
                            "tweet_id": str(rep.get("id_str", rep.get("id"))), # Ensure ID is string
                            "text": rep.get("text", ""), "label": label_val,
                            "parent_id": str(parent_id_val) if parent_id_val else None, # Ensure parent ID is string
                            "timestamp": timestamp
                        })
    df = pd.DataFrame(records)
    
    # Handle potential issues
    df.drop_duplicates(subset=["tweet_id"], inplace=True) # Remove duplicate tweet IDs if any
    df["text"].fillna("", inplace=True) # Ensure no NaN text

    null_timestamps_before = df["timestamp"].isna().sum()
    df.dropna(subset=["timestamp"], inplace=True) # Critical: remove rows with no timestamp
    null_timestamps_after = null_timestamps_before - df["timestamp"].isna().sum()
    if null_timestamps_after > 0:
         print(f"Warning: Removed {null_timestamps_after} records due to missing timestamps.")
    
    df = df.sort_values(by="timestamp").reset_index(drop=True) # Sort by time globally initially
    return df


# ---------------------------------------------------------------------------
# Temporal graph creation utilities
# ---------------------------------------------------------------------------

def create_time_windows(
    df: pd.DataFrame, 
    window_type: str = "hours", 
    max_windows: int = 10
) -> Tuple[List[Tuple[datetime, datetime]], Dict[str, int]]:
    """Create time windows for graph snapshots based on tweet timestamps in the DataFrame.
    
    Args:
        df: DataFrame with tweet data including 'timestamp' (UTC, non-null).
        window_type: Type of time window ('hours', 'days', 'minutes').
        max_windows: Maximum number of time windows to create.
    
    Returns:
        List of (start_time, end_time) tuples and a dict mapping tweet_ids to window indices.
    """
    if "timestamp" not in df.columns or df["timestamp"].isna().any():
        raise ValueError("DataFrame must contain valid, non-null 'timestamp' information (UTC).")
    if df.empty:
        return [], {}

    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()

    if min_time == max_time: # All tweets at the same time, create one window
        num_windows = 1
        window_size_adjusted = timedelta(seconds=1) # A nominal small duration
    else:
        if window_type == "hours":    window_delta = timedelta(hours=1)
        elif window_type == "days":   window_delta = timedelta(days=1)
        elif window_type == "minutes":window_delta = timedelta(minutes=10)
        else: raise ValueError(f"Unknown window type: {window_type}")
        
        total_duration = max_time - min_time
        # Calculate ideal number of windows based on delta, then cap by max_windows
        ideal_num_windows = max(1, int(np.ceil(total_duration / window_delta)))
        num_windows = min(max_windows, ideal_num_windows)
        
        if num_windows == 1 and total_duration > timedelta(0): # If capped to 1 window, it spans the whole duration
            window_size_adjusted = total_duration
        elif num_windows > 1 :
            window_size_adjusted = total_duration / num_windows
        else: # num_windows is 1 and total_duration is 0
            window_size_adjusted = timedelta(seconds=1) # fallback for single point in time

    time_windows_def = []
    current_start_time = min_time
    for i in range(num_windows):
        current_end_time = current_start_time + window_size_adjusted
        # Ensure the last window includes max_time
        if i == num_windows - 1:
            current_end_time = max(current_end_time, max_time + timedelta(microseconds=1)) # Ensure max_time is included
        time_windows_def.append((current_start_time, current_end_time))
        current_start_time = current_end_time
    
    tweet_to_window_map = {}
    for _, row in df.iterrows():
        timestamp = row["timestamp"]
        tweet_id = str(row["tweet_id"])
        for i, (start, end) in enumerate(time_windows_def):
            if start <= timestamp < end:
                tweet_to_window_map[tweet_id] = i
                break
            elif i == len(time_windows_def) - 1 and timestamp == end: # Include if exactly at end of last window
                 tweet_to_window_map[tweet_id] = i
                 break
        # if tweet_id not in tweet_to_window_map:
        #     print(f"Warning: Tweet {tweet_id} at {timestamp} did not fall into any window. Min: {min_time}, Max: {max_time}, Windows: {time_windows_def}")


    return time_windows_def, tweet_to_window_map


def build_temporal_edge_indices(
    df_subset: pd.DataFrame, 
    tweet_to_window_map: Dict[str, int],
    num_total_windows: int,
    subset_id_to_local_idx: Dict[str, int] # Precomputed mapping for this subset
) -> List[np.ndarray]:
    """Build cumulative edge indices for each time window for the given DataFrame subset."""
    edge_lists_for_windows = [[] for _ in range(num_total_windows)]
    
    for _, row in df_subset.iterrows():
        child_tid_str = str(row["tweet_id"])
        parent_tid_str = str(row["parent_id"]) if pd.notna(row["parent_id"]) else None

        if not parent_tid_str: continue # No parent, no edge

        # Edge exists if child appeared in a window and both parent/child are in our local index map
        if child_tid_str in tweet_to_window_map and \
           child_tid_str in subset_id_to_local_idx and \
           parent_tid_str in subset_id_to_local_idx:
            
            edge_creation_window_idx = tweet_to_window_map[child_tid_str]
            local_child_idx = subset_id_to_local_idx[child_tid_str]
            local_parent_idx = subset_id_to_local_idx[parent_tid_str]
            
            # Add edge to its creation window and all future windows (cumulative)
            for w_idx in range(edge_creation_window_idx, num_total_windows):
                edge_lists_for_windows[w_idx].append((local_parent_idx, local_child_idx)) # (src, dst)
    
    final_edge_indices = []
    for edges_in_window in edge_lists_for_windows:
        if edges_in_window:
            # Remove duplicate edges within the same snapshot
            unique_edges = sorted(list(set(edges_in_window)))
            edge_index_arr = np.asarray(unique_edges, dtype=np.int64).T
        else:
            edge_index_arr = np.empty((2, 0), dtype=np.int64)
        final_edge_indices.append(edge_index_arr)
    
    return final_edge_indices


def create_node_masks(
    df_subset: pd.DataFrame,
    tweet_to_window_map: Dict[str, int],
    num_total_windows: int,
    subset_id_to_local_idx: Dict[str, int] # Precomputed mapping for this subset
) -> List[np.ndarray]:
    """Create boolean masks indicating which nodes (from df_subset) exist in each time step."""
    num_nodes_in_subset = len(df_subset)
    node_activity_masks = [np.zeros(num_nodes_in_subset, dtype=bool) for _ in range(num_total_windows)]
    
    for tweet_id_str, node_creation_window_idx in tweet_to_window_map.items():
        if tweet_id_str in subset_id_to_local_idx: # Ensure tweet is part of the current subset
            local_node_idx = subset_id_to_local_idx[tweet_id_str]
            # Mark node as active from its creation window onwards (cumulative)
            for w_idx in range(node_creation_window_idx, num_total_windows):
                if 0 <= local_node_idx < num_nodes_in_subset: # Boundary check
                     node_activity_masks[w_idx][local_node_idx] = True
    
    return node_activity_masks


# ---------------------------------------------------------------------------
# Embedding & saving utilities
# ---------------------------------------------------------------------------

def embed_texts(
    texts: List[str],
    tokenizer: BertTokenizer,
    model: BertModel,
    device: torch.device,
    batch_size: int = 32, # Added batching
    max_length: int = 128,
) -> np.ndarray:
    """Return BERT CLS‑token embeddings with shape (len(texts), model_hidden_size)."""
    all_embeddings: List[np.ndarray] = []
    model.eval() # Ensure model is in eval mode
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts", unit=f"batch ({batch_size})"):
        batch_texts = texts[i:i+batch_size]
        if not batch_texts: continue

        tokens = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            out = model(**tokens)
        
        # CLS token embeddings (for BERT-like models)
        # For some models, mean pooling might be better, but CLS is common.
        cls_embeddings = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)
        
    if not all_embeddings: return np.array([]).reshape(0, model.config.hidden_size)
    return np.vstack(all_embeddings)


def save_temporal_dataset(
    df_for_saving: pd.DataFrame, # DataFrame subset for this specific save operation
    X_subset: np.ndarray,
    labels_subset: np.ndarray,
    time_windows_list: List[Tuple[datetime, datetime]],
    tweet_id_to_window_idx_map: Dict[str, int], # Filtered for df_for_saving
    output_directory: Path,
):
    """Create *output_directory* and store X, temporal edge indices, node masks, labels and time info."""
    output_directory.mkdir(parents=True, exist_ok=True)
    
    edge_indices_dir = output_directory / "edge_indices"
    edge_indices_dir.mkdir(exist_ok=True)
    node_masks_dir = output_directory / "node_masks"
    node_masks_dir.mkdir(exist_ok=True)
    
    # Create a local mapping {tweet_id_str: local_idx_0_to_N-1} for this specific df_for_saving
    # This ensures edge indices and node masks use 0-based indexing for X_subset and labels_subset
    subset_tweet_id_to_local_idx = {
        str(tid): i for i, tid in enumerate(df_for_saving["tweet_id"])
    }

    num_actual_windows = len(time_windows_list)
    temporal_edge_indices = build_temporal_edge_indices(
        df_for_saving, tweet_id_to_window_idx_map, num_actual_windows, subset_tweet_id_to_local_idx
    )
    temporal_node_masks = create_node_masks(
        df_for_saving, tweet_id_to_window_idx_map, num_actual_windows, subset_tweet_id_to_local_idx
    )
    
    np.save(output_directory / "X.npy", X_subset)
    np.save(output_directory / "labels.npy", labels_subset)
    
    for i in range(num_actual_windows):
        np.save(edge_indices_dir / f"t{i}.npy", temporal_edge_indices[i])
        np.save(node_masks_dir / f"t{i}.npy", temporal_node_masks[i])
    
    # Filter tweet_id_to_window_idx_map to only include tweet_ids present in subset_tweet_id_to_local_idx
    # This ensures time_info.json is consistent with the actual nodes saved.
    final_tweet_to_window_for_json = {
        tid: window for tid, window in tweet_id_to_window_idx_map.items() if tid in subset_tweet_id_to_local_idx
    }

    time_metadata = {
        "num_windows": num_actual_windows,
        "window_times": [
            {"start": start.isoformat(), "end": end.isoformat()}
            for start, end in time_windows_list
        ],
        "tweet_to_window": final_tweet_to_window_for_json # Store the filtered map
    }
    with (output_directory / "time_info.json").open("w", encoding="utf-8") as f:
        json.dump(time_metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PHEME pre‑processor for DySAT")
    p.add_argument("--data-dir", default="PHEME", help="Root directory of raw PHEME dataset")
    p.add_argument("--output-dir", default="data_dysat", help="Where to write processed outputs")
    p.add_argument("--model", default="bert-base-uncased", help="HF model name for sentence embeddings")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Computation device")
    p.add_argument("--time-window", default="hours", choices=["minutes", "hours", "days"], help="Time window granularity")
    p.add_argument("--max-time-steps", type=int, default=10, help="Maximum number of time steps")
    p.add_argument("--embedding-batch-size", type=int, default=32, help="Batch size for BERT embeddings")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_data_root = Path(args.data_dir)
    processed_output_root = Path(args.output_dir)
    processed_output_root.mkdir(exist_ok=True)

    print(f"Loading PHEME dataset with temporal information from {raw_data_root}")
    df_all_events = load_pheme_temporal(raw_data_root)
    
    if df_all_events.empty or "timestamp" not in df_all_events.columns or df_all_events["timestamp"].isna().all():
        print("Error: No valid data with timestamps loaded. Exiting.")
        return

    event_names = sorted(df_all_events["event"].unique())
    min_global_time = df_all_events["timestamp"].min()
    max_global_time = df_all_events["timestamp"].max()
    print(f"Global dataset time range: {min_global_time} to {max_global_time}")

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device))
    print(f"Using device for embeddings: {device}")
    tokenizer = BertTokenizer.from_pretrained(args.model)
    embedding_model = BertModel.from_pretrained(args.model).to(device)

    print("Embedding all unique texts from the dataset...")
    # Ensure tweet_id is string for consistent mapping later
    df_all_events["tweet_id"] = df_all_events["tweet_id"].astype(str) 
    all_texts_list = df_all_events["text"].tolist()
    X_all_embeddings = embed_texts(all_texts_list, tokenizer, embedding_model, device, args.embedding_batch_size)
    all_labels_array = df_all_events["label"].to_numpy(dtype=np.int64)

    # Create global time windows and mapping (can be used for "all" dataset or as reference)
    # print(f"Creating global time windows (granularity: {args.time_window}, max_steps: {args.max_time_steps})")
    # global_time_windows, global_tweet_to_window_map = create_time_windows(
    #     df_all_events, args.time_window, args.max_time_steps
    # )
    # print(f"Created {len(global_time_windows)} global time windows.")

    # print("Saving aggregated 'all' temporal dataset (using global time windows)...")
    # save_temporal_dataset(
    #     df_all_events, X_all_embeddings, all_labels_array,
    #     global_time_windows, global_tweet_to_window_map,
    #     processed_output_root / "all"
    # )
    # Note: The "all" dataset might be very large and its time windows might not be ideal for specific events.
    # It's often better to focus on per-event processing with event-specific time windows.

    print("\nProcessing and saving per-event temporal datasets...")
    for event_name_iter in tqdm(event_names, desc="Events", unit="event"):
        event_mask = df_all_events["event"] == event_name_iter
        df_event_subset = df_all_events[event_mask].copy() # Use .copy() to avoid SettingWithCopyWarning later
        
        if df_event_subset.empty or df_event_subset["timestamp"].isna().all() or len(df_event_subset) < 5: # Min 5 tweets for an event
            print(f"  Skipping event '{event_name_iter}': insufficient data or timestamps.")
            continue

        # Create event-specific time windows
        event_time_windows, event_tweet_id_to_window_idx = create_time_windows(
            df_event_subset, args.time_window, args.max_time_steps
        )
        if not event_time_windows:
            print(f"  Skipping event '{event_name_iter}': failed to create time windows (e.g., all tweets at one instant but not handled).")
            continue
            
        # Get indices from the original df_all_events that correspond to this event_subset
        # This allows us to slice X_all_embeddings and all_labels_array correctly
        event_original_indices = np.where(event_mask)[0]
        X_event_subset = X_all_embeddings[event_original_indices]
        labels_event_subset = all_labels_array[event_original_indices]

        # Filter the event_tweet_id_to_window_idx map to ensure it only contains tweet IDs
        # that are actually present in df_event_subset (and thus in X_event_subset/labels_event_subset)
        valid_event_tweet_ids = set(df_event_subset["tweet_id"].astype(str))
        filtered_event_tweet_to_window = {
            tid: window_idx for tid, window_idx in event_tweet_id_to_window_idx.items()
            if tid in valid_event_tweet_ids
        }
        
        print(f"  Event '{event_name_iter}': {len(df_event_subset)} tweets, {len(event_time_windows)} time windows.")
        save_temporal_dataset(
            df_event_subset, X_event_subset, labels_event_subset,
            event_time_windows, filtered_event_tweet_to_window,
            processed_output_root / event_name_iter
        )

    print("\nPreprocessing complete!")
    # print("Aggregated 'all' dataset saved in:", processed_output_root / "all")
    print("Per‑event datasets saved in respective subfolders under:", processed_output_root)


if __name__ == "__main__":
    main()