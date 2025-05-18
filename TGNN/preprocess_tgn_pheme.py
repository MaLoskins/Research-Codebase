#!/usr/bin/env python3
"""
Preprocess the PHEME dataset for TGN (Temporal Graph Network) or similar
continuous-time dynamic graph models.

This script extracts individual timestamped interaction events (replies).

Key features:
1.  **Event-based structure**: Each reply is an event with a source, destination,
    timestamp, and associated "message" feature (embedding of the replying tweet).
2.  **Continuous Timestamps**: Timestamps are preserved as numerical values.
3.  **Node Features**: BERT embeddings for all tweets.
4.  **Flexibility**: Outputs data in a format suitable for models like TGN.

Directory layout created (relative to --output-dir, default "data_tgn"):

    data_tgn/
    ├── all/                 # complete PHEME across every event
    │   ├── node_features.npy  # (N_nodes, D_feat) BERT embeddings of all tweets
    │   ├── labels.npy         # (N_nodes,) 0 = non‑rumour, 1 = rumour for each tweet (ROOTS ONLY, -1 for replies)
    │   ├── events.csv         # DataFrame: u, i, timestamp, event_idx
    │   ├── edge_features.npy  # (N_events, D_feat) "Message" features for each event
    │   ├── event_labels.npy   # (N_events,) Label for each event (thread label)
    │   └── metadata.json      # Mappings, start time, etc.
    ├── <event_1>/           # e.g. charliehebdo/
    │   ├── node_features.npy
    │   ├── labels.npy
    │   ├── events.csv
    │   ├── edge_features.npy
    │   ├── event_labels.npy
    │   └── metadata.json
    ├── <event_2>/
    │   └── …
    └── …

Usage
-----
$ python preprocess_tgn_pheme.py \
        --data-dir PHEME \
        --output-dir data_tgn \
        --model bert-base-uncased \
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

# --- Existing helper functions from your DySAT script (slightly adapted if needed) ---

def parse_twitter_time(time_str: str) -> datetime | None:
    """Parse Twitter's timestamp format into a datetime object. Returns None on failure."""
    if not time_str: return None
    try:
        dt = datetime.strptime(time_str, "%a %b %d %H:%M:%S %z %Y")
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None

def load_pheme_temporal(root: Path) -> pd.DataFrame:
    """Walk the PHEME directory structure and return a DataFrame with temporal info.
    Columns: event, thread_id, tweet_id, text, label, parent_id, timestamp (UTC)
    """
    records: List[dict] = []
    pheme_events_dirs = [d for d in root.iterdir() if d.is_dir()] # Renamed for clarity

    for event_dir in tqdm(pheme_events_dirs, desc="Scanning PHEME events", unit="event"):
        event_name = event_dir.name # PHEME event category like 'charliehebdo'
        for label_name, label_val in [("rumours", 1), ("non-rumours", 0)]:
            label_dir = event_dir / label_name
            if not label_dir.exists():
                continue

            threads = [t for t in label_dir.iterdir() if t.is_dir()]
            for thread_dir in tqdm(threads, desc=f"Threads in {event_name}/{label_name}", leave=False, unit="thread"):
                thread_id = thread_dir.name
                
                src_json_path_v1 = thread_dir / "source-tweet" / f"{thread_id}.json" 
                src_json_path_v2 = thread_dir / f"source-tweets/{thread_id}.json" 
                src_json_path_v3 = thread_dir / "source-tweet.json"

                src_file_to_load = None
                if src_json_path_v1.exists():
                    src_file_to_load = src_json_path_v1
                elif src_json_path_v2.exists():
                    src_file_to_load = src_json_path_v2
                elif src_json_path_v3.exists():
                    src_file_to_load = src_json_path_v3
                
                if src_file_to_load:
                    with src_file_to_load.open(encoding="utf-8") as f:
                        src = json.load(f)
                    timestamp = parse_twitter_time(src.get("created_at"))
                    records.append({
                        "event_category": event_name, "thread_id": thread_id,
                        "tweet_id": str(src.get("id_str", src.get("id"))),
                        "text": src.get("text", ""), "label": label_val, # This is the thread label
                        "parent_id": None, "timestamp": timestamp,
                        "user_id": str(src.get("user", {}).get("id_str", src.get("user", {}).get("id")))
                    })

                replies_dir = thread_dir / "reactions"
                if replies_dir.exists():
                    for reply_path in replies_dir.glob("*.json"):
                        with reply_path.open(encoding="utf-8") as f:
                            rep = json.load(f)
                        timestamp = parse_twitter_time(rep.get("created_at"))
                        parent_id_val = rep.get("in_reply_to_status_id_str", rep.get("in_reply_to_status_id"))
                        records.append({
                            "event_category": event_name, "thread_id": thread_id,
                            "tweet_id": str(rep.get("id_str", rep.get("id"))),
                            "text": rep.get("text", ""), "label": label_val, # This is the thread label
                            "parent_id": str(parent_id_val) if parent_id_val else None,
                            "timestamp": timestamp,
                            "user_id": str(rep.get("user", {}).get("id_str", rep.get("user", {}).get("id")))
                        })
    df = pd.DataFrame(records)
    
    df.drop_duplicates(subset=["tweet_id"], inplace=True) # Keep first occurrence if duplicate tweet_ids
    df["text"].fillna("", inplace=True)

    null_timestamps_before = df["timestamp"].isna().sum()
    df.dropna(subset=["timestamp"], inplace=True) # Drop rows with no timestamp
    dropped_count = null_timestamps_before - df["timestamp"].isna().sum()
    if dropped_count > 0:
         print(f"Warning: Removed {dropped_count} records due to missing timestamps.")
    
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    return df


def embed_texts(
    texts: List[str],
    tokenizer: BertTokenizer,
    model: BertModel,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 128,
) -> np.ndarray:
    all_embeddings: List[np.ndarray] = []
    model.eval()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts", unit=f"batch ({batch_size})", leave=False):
        batch_texts = texts[i:i+batch_size]
        if not batch_texts: continue

        tokens = tokenizer(
            batch_texts, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            out = model(**tokens)
        cls_embeddings = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)
        
    if not all_embeddings: return np.array([]).reshape(0, model.config.hidden_size)
    return np.vstack(all_embeddings)

# ---------------------------------------------------------------------------
# TGN-specific processing and saving
# ---------------------------------------------------------------------------
def save_tgn_dataset(
    df_subset: pd.DataFrame,          # DataFrame for the current split (e.g., one PHEME event)
    all_X_embeddings: np.ndarray,     # Embeddings for ALL tweets in the original full df
    all_tweet_id_to_idx: Dict[str, int], # Mapping from original tweet_id to index in all_X_embeddings
    output_directory: Path,
):
    """
    Process and save data for a TGN model for a given subset of tweets.
    The subset df_subset is assumed to be sorted by timestamp.
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    if df_subset.empty:
        print(f"Skipping {output_directory.name}, no data in subset.")
        return

    subset_tweet_ids = df_subset["tweet_id"].unique().tolist()
    local_tweet_id_to_idx = {tid: i for i, tid in enumerate(subset_tweet_ids)}
    
    subset_original_indices = [all_tweet_id_to_idx[tid] for tid in subset_tweet_ids if tid in all_tweet_id_to_idx]
    
    if not subset_original_indices:
        print(f"Skipping {output_directory.name}, no valid tweets after mapping for embeddings.")
        return

    node_features_subset = all_X_embeddings[subset_original_indices]
    
    # Fix 1.2: Tweet-level labels duplicated from the thread label.
    # Keep the node labels only on the root tweet. For every other node use “unlabelled” (-1).
    # Create a temporary mapping from tweet_id to its parent_id and original thread label for tweets in this subset
    temp_df_for_labels = df_subset.drop_duplicates(subset=['tweet_id'], keep='first').set_index('tweet_id')
    
    labels_subset_list = []
    for tid in subset_tweet_ids: # Iterate in the order of local_tweet_id_to_idx
        row_data = temp_df_for_labels.loc[tid]
        parent_id_val = row_data["parent_id"]
        thread_label = row_data["label"] # This is the original thread label for the tweet's thread
        
        if pd.isna(parent_id_val) or parent_id_val is None: # It's a root tweet
            labels_subset_list.append(thread_label)
        else: # It's a reply
            labels_subset_list.append(-1) # Unlabelled for loss masking
    labels_subset = np.array(labels_subset_list, dtype=np.int64)

    np.save(output_directory / "node_features.npy", node_features_subset)
    np.save(output_directory / "labels.npy", labels_subset)

    # Prepare event data list: (source_idx, dest_idx, timestamp, edge_feature_array, event_thread_label)
    events_raw_data: List[Tuple[int, int, float, np.ndarray, int]] = []

    min_time_subset = df_subset["timestamp"].min()

    # Iterate through tweets in the subset (already sorted by timestamp from df_full_dataset)
    for _, row in df_subset.iterrows():
        parent_id_str = row["parent_id"]
        current_tweet_id_str = str(row["tweet_id"]) # ensure string
        
        if pd.notna(parent_id_str) and parent_id_str is not None:
            parent_id_str = str(parent_id_str) # ensure string
            
            # Only create an event if both parent and current tweet are in this subset's local map
            if parent_id_str in local_tweet_id_to_idx and current_tweet_id_str in local_tweet_id_to_idx:
                p_local_idx = local_tweet_id_to_idx[parent_id_str]
                c_local_idx = local_tweet_id_to_idx[current_tweet_id_str]
                
                ts = (row["timestamp"] - min_time_subset).total_seconds()
                event_thread_label = row["label"] # Thread label associated with this interaction

                p_embed = node_features_subset[p_local_idx]
                c_embed = node_features_subset[c_local_idx]

                # Fix 1.1 & 1.3: Edges and Edge Features
                # 1. Parent → Child (original interaction direction)
                edge_feat_pc = c_embed - p_embed  # dst_embed – src_embed
                events_raw_data.append((p_local_idx, c_local_idx, ts, edge_feat_pc, event_thread_label))

                # 2. Child → Parent (reverse edge)
                edge_feat_cp = p_embed - c_embed  # dst_embed(parent) – src_embed(child)
                events_raw_data.append((c_local_idx, p_local_idx, ts, edge_feat_cp, event_thread_label))
                
                # 3. Child → Child (self-loop for the replying child node)
                edge_feat_cc = c_embed - c_embed  # zero vector
                events_raw_data.append((c_local_idx, c_local_idx, ts, edge_feat_cc, event_thread_label))

    if not events_raw_data:
        print(f"No reply events constructed for {output_directory.name}. Saving empty event files.")
        # Save empty structures to maintain consistency (use feature dim if available, else default)
        feat_dim = node_features_subset.shape[1] if node_features_subset.ndim > 1 and node_features_subset.shape[0] > 0 else 768
        pd.DataFrame(columns=['u', 'i', 'timestamp', 'event_idx']).to_csv(output_directory / "events.csv", index=False)
        np.save(output_directory / "edge_features.npy", np.empty((0, feat_dim)))
        np.save(output_directory / "event_labels.npy", np.empty((0,), dtype=np.int64))
        num_actual_events = 0
    else:
        # Sort all collected events strictly by timestamp
        events_raw_data.sort(key=lambda x: x[2])

        event_sources = [e[0] for e in events_raw_data]
        event_destinations = [e[1] for e in events_raw_data]
        event_timestamps = [e[2] for e in events_raw_data]
        # Stack edge features into a single numpy array
        final_edge_features = np.array([e[3] for e in events_raw_data])
        final_event_labels = np.array([e[4] for e in events_raw_data], dtype=np.int64)
        num_actual_events = len(event_sources)

        events_df = pd.DataFrame({
            'u': event_sources,
            'i': event_destinations,
            'timestamp': event_timestamps,
            'event_idx': np.arange(num_actual_events) # Sequential event index after sorting
        })

        events_df.to_csv(output_directory / "events.csv", index=False)
        np.save(output_directory / "edge_features.npy", final_edge_features)
        np.save(output_directory / "event_labels.npy", final_event_labels)

    metadata = {
        "num_nodes": node_features_subset.shape[0],
        "num_events": num_actual_events, # Use the count of actual events generated
        "tweet_id_to_local_idx": local_tweet_id_to_idx,
        "subset_start_time_iso": min_time_subset.isoformat(),
        "subset_start_time_unix": min_time_subset.timestamp()
    }
    with (output_directory / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # print(f"  Saved TGN data for {output_directory.name}: {metadata['num_nodes']} nodes, {metadata['num_events']} events.")


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PHEME pre‑processor for TGN-style models")
    p.add_argument("--data-dir", default="PHEME", help="Root directory of raw PHEME dataset")
    p.add_argument("--output-dir", default="data_tgn", help="Where to write processed outputs")
    p.add_argument("--model", default="bert-base-uncased", help="HF model name for sentence embeddings")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Computation device")
    p.add_argument("--embedding-batch-size", type=int, default=32, help="Batch size for BERT embeddings")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    raw_data_root = Path(args.data_dir)
    processed_output_root = Path(args.output_dir)
    processed_output_root.mkdir(exist_ok=True)

    print(f"Loading PHEME dataset with temporal information from {raw_data_root}")
    df_full_dataset = load_pheme_temporal(raw_data_root) 
    
    if df_full_dataset.empty:
        print("Error: No data loaded. Exiting.")
        return
    
    df_full_dataset["tweet_id"] = df_full_dataset["tweet_id"].astype(str)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device))
    print(f"Using device for embeddings: {device}")
    tokenizer = BertTokenizer.from_pretrained(args.model)
    embedding_model = BertModel.from_pretrained(args.model).to(device)

    print("Embedding all unique texts from the entire dataset...")
    # To embed unique texts efficiently and then map back:
    unique_texts_df = df_full_dataset.drop_duplicates(subset=['text'])
    unique_texts_list = unique_texts_df["text"].tolist()
    unique_embeddings = embed_texts(unique_texts_list, tokenizer, embedding_model, device, args.embedding_batch_size)
    
    text_to_embedding_map = {text: emb for text, emb in zip(unique_texts_list, unique_embeddings)}
    
    # Reconstruct X_all_embeddings in the order of df_full_dataset
    # This ensures X_all_embeddings[i] corresponds to df_full_dataset.iloc[i]
    hidden_size = embedding_model.config.hidden_size
    X_all_embeddings_list = []
    for text_content in tqdm(df_full_dataset["text"], desc="Mapping embeddings to full dataset", unit="tweet"):
        X_all_embeddings_list.append(text_to_embedding_map.get(text_content, np.zeros(hidden_size))) # Fallback if text somehow missing
    X_all_embeddings = np.array(X_all_embeddings_list)

    all_tweet_id_to_original_idx = {
        tweet_id: i for i, tweet_id in enumerate(df_full_dataset["tweet_id"])
    }
    
    print("\nProcessing and saving 'all' dataset for TGN...")
    save_tgn_dataset(
        df_full_dataset.copy(), 
        X_all_embeddings,
        all_tweet_id_to_original_idx,
        processed_output_root / "all"
    )

    pheme_event_categories = sorted(df_full_dataset["event_category"].unique())
    print(f"\nProcessing per-PHEME-event datasets ({len(pheme_event_categories)} categories)...")

    for category_name in tqdm(pheme_event_categories, desc="PHEME Event Categories", unit="category"):
        df_category_subset = df_full_dataset[df_full_dataset["event_category"] == category_name].copy()
        
        if df_category_subset.empty or len(df_category_subset) < 1: # Allow single node graphs
            print(f"  Skipping category '{category_name}': insufficient data ({len(df_category_subset)} tweets).")
            continue
        
        save_tgn_dataset(
            df_category_subset,
            X_all_embeddings,
            all_tweet_id_to_original_idx,
            processed_output_root / category_name
        )

    print("\nPreprocessing for TGN complete!")
    print("Aggregated 'all' dataset saved in:", processed_output_root / "all")
    print("Per‑PHEME‑event datasets saved in respective subfolders under:", processed_output_root)

if __name__ == "__main__":
    main()