#!/usr/bin/env python3
"""
Static MLP baseline for node classification on PHEME (or similar) data.
Uses pre-computed node features (e.g., BERT embeddings) and labels.
Filters to use only root nodes (labels != -1) for PHEME thread classification.
Saves results in a structured way.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json # For saving hyperparameters

# --- MLP Model ---
class StaticMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, output_dim: int, dropout_rate: float):
        super(StaticMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.act1(self.fc1(x)))
        x = self.dropout2(self.act2(self.fc2(x)))
        x = self.fc3(x)
        return x

# --- Results Saving Function ---
def save_mlp_results(event_name, results_base_dir, args_used, best_val_f1, final_test_metrics):
    event_dir = Path(results_base_dir) / event_name
    event_dir.mkdir(parents=True, exist_ok=True)

    md_content = f"# MLP Baseline Results for Event: {event_name}\n\n"
    md_content += "## Hyperparameters Used:\n"
    for k, v in vars(args_used).items():
        # Filter for relevant HPs to display
        if k not in ["device", "data_path"]: # Exclude non-HP arguments
            md_content += f"- {k}: {v}\n"
    md_content += "\n"

    md_content += "## Performance Metrics:\n"
    if best_val_f1 != -1.0 :
        md_content += f"- Best Validation F1-Score: {best_val_f1:.4f}\n"
    else:
        md_content += "- Validation F1-Score: N/A (no validation set or no improvement)\n"

    if final_test_metrics:
        md_content += "\n### Final Test Metrics (at best validation or last epoch):\n"
        md_content += f"- Test Accuracy: {final_test_metrics.get('accuracy', 0.0):.4f}\n"
        md_content += f"- Test F1-Score: {final_test_metrics.get('f1', 0.0):.4f}\n"
        md_content += f"- Test Precision: {final_test_metrics.get('precision', 0.0):.4f}\n"
        md_content += f"- Test Recall: {final_test_metrics.get('recall', 0.0):.4f}\n"
        if "conf_matrix" in final_test_metrics:
             md_content += f"- Test Confusion Matrix:\n```\n{np.array(final_test_metrics['conf_matrix'])}\n```\n"
    else:
        md_content += "\nTest metrics not available or not evaluated for this run.\n"

    with open(event_dir / "results.md", "w") as f:
        f.write(md_content)
    print(f"MLP Results summary saved to {event_dir / 'results.md'}")

    # Save hyperparameters as JSON for easy parsing if needed later
    with open(event_dir / "hyperparameters.json", "w") as f:
        json.dump(vars(args_used), f, indent=4)
    print(f"Hyperparameters saved to {event_dir / 'hyperparameters.json'}")


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Static MLP Baseline for Node Classification.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to a specific preprocessed TGN event data folder (e.g., data_tgn_fixed/charliehebdo), "
                             "from which node_features.npy and labels.npy will be used.")
    parser.add_argument("--results_base_dir", type=str, default="RESULTS/MLP_Baseline",
                        help="Base directory to save results for this MLP baseline.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"], help="Computation device.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--hidden_dim1", type=int, default=256, help="Dimension of first hidden layer.")
    parser.add_argument("--hidden_dim2", type=int, default=128, help="Dimension of second hidden layer.")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data for testing.")
    parser.add_argument("--val_size", type=float, default=0.15, help="Proportion of original data for validation (taken from training set).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    data_dir = Path(args.data_path)
    event_name = data_dir.name # Extract event name from the data path

    # Create results directory
    Path(args.results_base_dir).mkdir(parents=True, exist_ok=True)

    try:
        X_all_nodes_raw = np.load(data_dir / "node_features.npy")
        y_all_nodes_raw = np.load(data_dir / "labels.npy")
    except FileNotFoundError:
        print(f"Error: node_features.npy or labels.npy not found in {data_dir}")
        # Save a failure message if possible
        save_mlp_results(event_name, args.results_base_dir, args, -1.0, {"error": f"Data files not found in {data_dir}"})
        return

    print(f"Loaded {X_all_nodes_raw.shape[0]} total nodes with {X_all_nodes_raw.shape[1]}-dim features for event: {event_name}.")

    # Filter for root nodes (labels != -1) for PHEME thread classification
    root_node_mask = y_all_nodes_raw != -1
    X_root_nodes = X_all_nodes_raw[root_node_mask]
    y_root_nodes = y_all_nodes_raw[root_node_mask]

    if X_root_nodes.shape[0] == 0:
        print("Error: No root nodes found (all labels are -1 or no nodes). Cannot train MLP baseline.")
        save_mlp_results(event_name, args.results_base_dir, args, -1.0, {"error": "No root nodes found"})
        return

    print(f"Filtered to {X_root_nodes.shape[0]} root nodes for MLP baseline.")

    X_all_nodes_tensor = torch.from_numpy(X_root_nodes).float()
    y_all_nodes_tensor = torch.from_numpy(y_root_nodes).long()

    num_classes = len(torch.unique(y_all_nodes_tensor))
    print(f"Number of unique classes among root nodes: {num_classes}")
    if num_classes < 2:
        print("Error: Need at least 2 classes for classification among root nodes.")
        save_mlp_results(event_name, args.results_base_dir, args, -1.0, {"error": "Less than 2 classes found"})
        return

    # Stratified train-test split
    try:
        train_indices, test_indices = train_test_split(
            np.arange(X_all_nodes_tensor.shape[0]),
            test_size=args.test_size,
            stratify=y_all_nodes_tensor.numpy(), # Stratify by labels
            random_state=args.seed
        )
    except ValueError as e: # Handle cases where stratification is not possible (e.g., too few samples in a class)
        print(f"Warning: Stratification failed for train/test split: {e}. Proceeding without stratification.")
        train_indices, test_indices = train_test_split(
            np.arange(X_all_nodes_tensor.shape[0]),
            test_size=args.test_size,
            random_state=args.seed
        )


    # Stratified train-validation split from the training set
    relative_val_size = args.val_size / (1.0 - args.test_size) if (1.0 - args.test_size) > 0 else 0
    # Ensure relative_val_size is reasonable
    if not (0 < relative_val_size < 1.0):
        relative_val_size = 0.18 # Default to ~15% of original if val_size was 0.15 and test_size 0.2

    val_indices = np.array([], dtype=np.int64)
    if relative_val_size > 0 and len(train_indices) > 1 :
        try:
            train_indices, val_indices = train_test_split(
                train_indices, # Split from the current training set
                test_size=relative_val_size,
                stratify=y_all_nodes_tensor[train_indices].numpy(),
                random_state=args.seed
            )
        except ValueError as e:
            print(f"Warning: Stratification failed for train/validation split: {e}. Proceeding without stratification.")
            train_indices, val_indices = train_test_split(
                train_indices,
                test_size=relative_val_size,
                random_state=args.seed
            )

    X_train, y_train = X_all_nodes_tensor[train_indices], y_all_nodes_tensor[train_indices]
    X_val, y_val = (X_all_nodes_tensor[val_indices], y_all_nodes_tensor[val_indices]) if len(val_indices) > 0 else \
                   (torch.empty(0, X_all_nodes_tensor.shape[1]), torch.empty(0, dtype=torch.long))
    X_test, y_test = X_all_nodes_tensor[test_indices], y_all_nodes_tensor[test_indices]

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Not enough data for train/test splits after filtering. Exiting.")
        save_mlp_results(event_name, args.results_base_dir, args, -1.0, {"error": "Not enough data for train/test splits"})
        return

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if len(X_val) > 0:
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = StaticMLP(
        input_dim=X_all_nodes_tensor.shape[1],
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        output_dim=num_classes,
        dropout_rate=args.dropout_rate
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    class_counts = torch.bincount(y_all_nodes_tensor)
    class_weights = None
    if len(class_counts) == num_classes and class_counts.min() > 0:
        print(f"Class distribution for weighting (from {len(y_all_nodes_tensor)} root nodes): {class_counts.cpu().numpy()}")
        class_weights = (1. / class_counts.float()).to(device)
        print(f"Using class weights (unnormalized): {class_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\n--- Starting Static MLP Training ---")
    best_val_f1 = -1.0
    best_test_metrics_at_best_val = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} Training", leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch}/{args.epochs} - Avg Train Loss: {avg_epoch_loss:.4f}")

        current_val_f1 = -1.0
        if val_loader:
            model.eval()
            all_val_preds, all_val_true = [], []
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader:
                    batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                    outputs_val = model(batch_X_val)
                    preds_val = torch.argmax(outputs_val, dim=1)
                    all_val_preds.append(preds_val.cpu().numpy())
                    all_val_true.append(batch_y_val.cpu().numpy())

            if all_val_preds:
                final_val_preds = np.concatenate(all_val_preds)
                final_val_true = np.concatenate(all_val_true)
                val_acc = accuracy_score(final_val_true, final_val_preds)
                avg_metric_sklearn = 'binary' if num_classes == 2 else 'weighted'
                current_val_f1 = f1_score(final_val_true, final_val_preds, average=avg_metric_sklearn, zero_division=0)
                val_prec = precision_score(final_val_true, final_val_preds, average=avg_metric_sklearn, zero_division=0)
                val_rec = recall_score(final_val_true, final_val_preds, average=avg_metric_sklearn, zero_division=0)
                print(f"  Validation - Acc: {val_acc:.4f}, F1: {current_val_f1:.4f}, P: {val_prec:.4f}, R: {val_rec:.4f}")
            else:
                current_val_f1 = 0
                print("  No validation data to evaluate this epoch.")
        else:
            print("  No validation set. Evaluating on test set at end of each epoch.")


        if val_loader is None or current_val_f1 > best_val_f1:
            if val_loader: best_val_f1 = current_val_f1
            model.eval()
            all_test_preds, all_test_true = [], []
            with torch.no_grad():
                for batch_X_test, batch_y_test in test_loader:
                    batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
                    outputs_test = model(batch_X_test)
                    preds_test = torch.argmax(outputs_test, dim=1)
                    all_test_preds.append(preds_test.cpu().numpy())
                    all_test_true.append(batch_y_test.cpu().numpy())

            if all_test_preds:
                final_test_preds = np.concatenate(all_test_preds)
                final_test_true = np.concatenate(all_test_true)
                avg_metric_sklearn = 'binary' if num_classes == 2 else 'weighted'
                test_acc = accuracy_score(final_test_true, final_test_preds)
                test_f1 = f1_score(final_test_true, final_test_preds, average=avg_metric_sklearn, zero_division=0)
                test_prec = precision_score(final_test_true, final_test_preds, average=avg_metric_sklearn, zero_division=0)
                test_rec = recall_score(final_test_true, final_test_preds, average=avg_metric_sklearn, zero_division=0)
                cm_test = confusion_matrix(final_test_true, final_test_preds, labels=np.arange(num_classes))

                best_test_metrics_at_best_val = {
                    "accuracy": test_acc, "f1": test_f1, "precision": test_prec, "recall": test_rec, "conf_matrix": cm_test.tolist()
                }
                print(f"  {'New best val F1' if val_loader else 'End of epoch'}. Test results:")
                print(f"  Test - Acc: {test_acc:.4f}, F1: {test_f1:.4f}, P: {test_prec:.4f}, R: {test_rec:.4f}")
                print(f"  Test CM:\n{cm_test}")
            else:
                print("  No test data to evaluate this epoch.")


    print("\n--- Static MLP Training Complete ---")
    if val_loader: print(f"Best Validation F1 achieved: {best_val_f1:.4f}")

    # Save results
    save_mlp_results(event_name, args.results_base_dir, args, best_val_f1, best_test_metrics_at_best_val)


if __name__ == "__main__":
    main()