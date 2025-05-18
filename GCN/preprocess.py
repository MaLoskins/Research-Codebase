#!/usr/bin/env python3
"""
Preprocess the PHEME dataset and generate per‑event as well as full aggregated
outputs that are ready for Geometric Neural Network (GNN) experiments.

Directory layout created (relative to --output-dir, default "data"):

    data/
    ├── all/                # complete PHEME across every event
    │   ├── X.npy           # (N, 768) text embeddings
    │   ├── edge_index.npy  # (2, E)   parent → reply edges
    │   └── labels.npy      # (N,)      0 = non‑rumour, 1 = rumour
    ├── <event_1>/          # e.g. charliehebdo/
    │   ├── X.npy
    │   ├── edge_index.npy
    │   └── labels.npy
    ├── <event_2>/
    │   └── …
    └── …

Usage
-----
$ python preprocess_pheme_structured.py \
        --data-dir PHEME \
        --output-dir data \
        --model bert-base-uncased \
        --device cuda

If a GPU is unavailable, set --device cpu (default picks automatically).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_pheme(root: Path) -> pd.DataFrame:
    """Walk the PHEME directory structure and return a DataFrame.

    Columns: event, thread_id, tweet_id, text, label, parent_id
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

                # Source tweet ------------------------------------------------
                src_file = thread_dir / "source-tweet.json"
                if src_file.exists():
                    with src_file.open(encoding="utf-8") as f:
                        src = json.load(f)
                    records.append(
                        {
                            "event": event,
                            "thread_id": thread_id,
                            "tweet_id": src.get("id_str", src.get("id")),
                            "text": src.get("text", ""),
                            "label": label_val,
                            "parent_id": None,
                        }
                    )

                # Replies -----------------------------------------------------
                replies_dir = thread_dir / "reactions"
                if replies_dir.exists():
                    for reply_path in replies_dir.glob("*.json"):
                        with reply_path.open(encoding="utf-8") as f:
                            rep = json.load(f)
                        records.append(
                            {
                                "event": event,
                                "thread_id": thread_id,
                                "tweet_id": rep.get("id_str", rep.get("id")),
                                "text": rep.get("text", ""),
                                "label": label_val,
                                "parent_id": rep.get("in_reply_to_status_id_str", rep.get("in_reply_to_user_id_str")),
                            }
                        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Embedding & graph utilities
# ---------------------------------------------------------------------------

def embed_texts(
    texts: List[str],
    tokenizer: BertTokenizer,
    model: BertModel,
    device: torch.device,
    max_length: int = 128,
) -> np.ndarray:
    """Return BERT CLS‑token embeddings with shape (len(texts), 768)."""

    embeddings: List[np.ndarray] = []
    for txt in tqdm(texts, desc="Embedding texts", unit="tweet"):
        tokens = tokenizer(
            txt,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            out = model(**tokens)
        cls = out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        embeddings.append(cls)
    return np.vstack(embeddings)


def build_edge_index(df_subset: pd.DataFrame) -> np.ndarray:
    """Parent→reply edges for a DataFrame slice (2, E)."""
    id_to_local = {tid: idx for idx, tid in enumerate(df_subset["tweet_id"])}
    edges: List[Tuple[int, int]] = []
    for _, row in df_subset[df_subset["parent_id"].notna()].iterrows():
        parent_tid, child_tid = row["parent_id"], row["tweet_id"]
        if parent_tid in id_to_local:
            edges.append((id_to_local[parent_tid], id_to_local[child_tid]))
    return np.asarray(edges).T if edges else np.empty((2, 0), dtype=int)


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_dataset(
    df_subset: pd.DataFrame,
    X: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
):
    """Create *out_dir* and store X, edge_index, labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_index = build_edge_index(df_subset)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "edge_index.npy", edge_index)
    np.save(out_dir / "labels.npy", labels)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PHEME pre‑processor with structured outputs")
    p.add_argument("--data-dir", default="PHEME", help="Root directory of raw PHEME dataset")
    p.add_argument("--output-dir", default="data", help="Where to write processed outputs")
    p.add_argument("--model", default="bert-base-uncased", help="HF model name for sentence embeddings")
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device (auto picks cuda if available)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(exist_ok=True)

    # ---------------------------------------------------------------------
    # 1) Load entire corpus once ------------------------------------------
    # ---------------------------------------------------------------------
    df_all = load_pheme(root)
    events = sorted(df_all["event"].unique())

    # ---------------------------------------------------------------------
    # 2) Initialise encoder ------------------------------------------------
    # ---------------------------------------------------------------------
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = BertModel.from_pretrained(args.model).to(device).eval()

    # ---------------------------------------------------------------------
    # 3) Embed ALL tweets only once ---------------------------------------
    # ---------------------------------------------------------------------
    X_all = embed_texts(df_all["text"].tolist(), tokenizer, model, device)
    labels_all = df_all["label"].to_numpy()

    # ---------------------------------------------------------------------
    # 4) Save aggregated dataset ------------------------------------------
    # ---------------------------------------------------------------------
    save_dataset(df_all, X_all, labels_all, out_root / "all")

    # ---------------------------------------------------------------------
    # 5) Save per‑event datasets ------------------------------------------
    # ---------------------------------------------------------------------
    for event in tqdm(events, desc="Writing per‑event datasets", unit="event"):
        mask = df_all["event"] == event
        df_evt = df_all[mask]
        idx = np.where(mask)[0]
        save_dataset(df_evt, X_all[idx], labels_all[idx], out_root / event)

    print("\nProcessing complete!\nSaved aggregates in", out_root / "all")
    print("Per‑event folders:")
    for event in events:
        print("  -", out_root / event)


if __name__ == "__main__":
    main()
