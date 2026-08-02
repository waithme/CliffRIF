#!/usr/bin/env python3
"""Finetune the multi-scale detector on one dataset, then predict official test.

Training rows come from the requested dataset inside the detector JSONL, but
are additionally intersected with the official CSV ``split=train`` SMILES to
prevent test leakage. The official test split is used only for final inference.

Example:
  python finetune_gine_region_detector.py \
    --checkpoint runs/gine_region_detector/best.pt \
    --dataset CHEMBL244_Ki \
    --data_file data/region_detector_mol.jsonl \
    --dataset_dir data/dataset \
    --output_dir runs/region_finetune_CHEMBL244_Ki \
    --epochs 100 --batch_size 64 --lr 1e-4 --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gine_region_detector import GINEMultiScaleRegionDetector, MultiScaleRegionConfig
from train_gine_region_detector import losses, row_to_data
from molecule_features import atom_features, bond_features

def validate_loss_switches(args):
    enabled = {"y": True, "mask": True, "existence": True}
    print(json.dumps({"loss_switches": enabled}))

RDLogger.DisableLog("rdApp.warning")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else None
    except Exception:
        return None


def resolve_device(name):
    available = (name == "cpu" or (name.startswith("cuda") and torch.cuda.is_available())
                 or (name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
    return torch.device(name if available else "cpu")


def read_official_csv(path, smiles_col, y_col, split_col, cliff_col):
    frame = pd.read_csv(path)
    for column in (smiles_col, y_col, split_col):
        if column not in frame.columns:
            raise ValueError(f"Missing column {column}; available={list(frame.columns)}")
    rows = {"train": [], "test": []}
    bad = 0
    for row_idx, row in frame.iterrows():
        split = str(row[split_col]).strip().lower()
        if split not in rows or pd.isna(row[smiles_col]) or pd.isna(row[y_col]):
            continue
        canonical = canonicalize(row[smiles_col])
        if canonical is None:
            bad += 1
            continue
        cliff = float(row[cliff_col]) if cliff_col in frame.columns and not pd.isna(row[cliff_col]) else float("nan")
        rows[split].append({"row_idx": int(row_idx), "smiles": str(row[smiles_col]),
                            "canonical_smiles": canonical, "y": float(row[y_col]), "cliff_mol": cliff})
    return rows, bad


def load_finetune_rows(path, dataset_name, official_train_smiles, task_id, num_layers):
    items, dataset_rows, outside_train, bad = [], 0, 0, 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if str(row.get("dataset")) != dataset_name:
                continue
            dataset_rows += 1
            canonical = row.get("canonical_smiles") or canonicalize(row.get("smiles", ""))
            if canonical not in official_train_smiles:
                outside_train += 1
                continue
            try:
                data = row_to_data(row, task_id, num_layers)
            except Exception:
                data = None
            if data is None:
                bad += 1
                continue
            data.canonical_smiles = canonical
            items.append(data)
    return items, {"jsonl_dataset_rows": dataset_rows, "excluded_nontrain_rows": outside_train,
                   "bad_jsonl_rows": bad}


def make_test_graphs(rows, dataset_name, task_id, num_layers):
    items, bad = [], 0
    for row in rows:
        # Build test graphs directly. Do not call row_to_data here: older
        # versions of the training script discard rows with regions=[], while
        # downstream test molecules intentionally have no ground-truth region.
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None or mol.GetNumAtoms() == 0:
            bad += 1
            continue
        x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
        pairs, attrs = [], []
        for bond in mol.GetBonds():
            i, j, feature = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_features(bond)
            pairs.extend([(i, j), (j, i)])
            attrs.extend([feature, feature])
        edge_index = (torch.tensor(pairs, dtype=torch.long).t().contiguous()
                      if pairs else torch.empty((2, 0), dtype=torch.long))
        edge_attr = (torch.tensor(attrs, dtype=torch.float32)
                     if attrs else torch.empty((0, 12), dtype=torch.float32))
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=torch.tensor([row["y"]], dtype=torch.float32))
        data.task_id = torch.tensor([task_id], dtype=torch.long)
        data.smiles = row["smiles"]
        data.dataset = dataset_name
        data.row_idx = row["row_idx"]
        data.cliff_mol = torch.tensor([row["cliff_mol"]], dtype=torch.float32)
        items.append(data)
    return items, bad


def train_epoch(model, loader, optimizer, device, args):
    model.train()
    totals = {key: 0.0 for key in ("loss", "y", "mask", "existence")}
    count = 0
    for batch in loader:
        batch = batch.to(device)
        values = losses(model(batch), batch, args)
        optimizer.zero_grad(set_to_none=True)
        values["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        size = int(batch.num_graphs)
        count += size
        for key in totals:
            totals[key] += float(values[key].detach()) * size
    return {key: value / max(1, count) for key, value in totals.items()}


@torch.no_grad()
def evaluate_y(model, loader, device, predictions=False, y_mean=0.0, y_std=1.0):
    model.eval()
    y_true, y_pred, cliffs, output_rows = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred_standardized = model(batch)["y_pred"].detach().cpu().view(-1).numpy()
        pred = pred_standardized * float(y_std) + float(y_mean)
        true = batch.y.detach().cpu().view(-1).numpy()
        cliff = batch.cliff_mol.detach().cpu().view(-1).numpy() if hasattr(batch, "cliff_mol") else np.full(len(true), np.nan)
        row_idx = batch.row_idx.detach().cpu().view(-1).numpy() if hasattr(batch, "row_idx") else np.full(len(true), -1)
        smiles = list(getattr(batch, "smiles", [""] * len(true)))
        y_true.extend(true.tolist()); y_pred.extend(pred.tolist()); cliffs.extend(cliff.tolist())
        if predictions:
            for idx in range(len(true)):
                output_rows.append({"row_idx": int(row_idx[idx]), "smiles": smiles[idx],
                                    "y_true": float(true[idx]), "y_pred": float(pred[idx]),
                                    "cliff_mol": float(cliff[idx]), "abs_error": float(abs(pred[idx] - true[idx]))})
    true, pred, cliff = np.asarray(y_true), np.asarray(y_pred), np.asarray(cliffs)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2))) if len(true) else float("nan")
    mae = float(np.mean(np.abs(pred - true))) if len(true) else float("nan")
    cliff_mask = np.isfinite(cliff) & (cliff >= 0.5)
    cliff_rmse = float(np.sqrt(np.mean((pred[cliff_mask] - true[cliff_mask]) ** 2))) if cliff_mask.any() else float("nan")
    return {"rmse": rmse, "mae": mae, "cliff_rmse": cliff_rmse}, output_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset-specific finetuning and official-test inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reset_y_head", action="store_true",
                        help="Replace the pretraining task outputs with one new downstream y output")
    parser.add_argument("--dataset", required=True, help="Example: CHEMBL244_Ki")
    parser.add_argument("--data_file", default="data/region_detector_mol.jsonl")
    parser.add_argument("--val_data_file", required=True,
                        help="Pre-generated validation JSONL; it is never re-split")
    parser.add_argument("--dataset_dir", default="data/dataset")
    parser.add_argument("--csv", default=None, help="Optional explicit official CSV path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--smiles_col", default="smiles")
    parser.add_argument("--y_col", default="y [pEC50/pKi]")
    parser.add_argument("--split_col", default="split")
    parser.add_argument("--cliff_col", default="cliff_mol")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--lambda_y", type=float, default=1.0)
    parser.add_argument("--lambda_mask", type=float, default=0.5)
    parser.add_argument("--lambda_exist", type=float, default=0.5)
    parser.add_argument("--single_cut_region_weight", type=float, default=1.0)
    parser.add_argument("--mcs_region_weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_loss_switches(args)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else Path(args.dataset_dir) / f"{args.dataset}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Official dataset CSV not found: {csv_path}")
    official, bad_csv = read_official_csv(csv_path, args.smiles_col, args.y_col, args.split_col, args.cliff_col)
    if not official["train"] or not official["test"]:
        raise ValueError(f"Official train/test must be non-empty: {len(official['train'])}/{len(official['test'])}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pretrained_task_to_index = checkpoint.get("task_to_index", {})
    pretrained_cfg = MultiScaleRegionConfig(**checkpoint["model_config"])
    if args.reset_y_head:
        task_to_index = {args.dataset: 0}
        task_id = 0
        cfg = MultiScaleRegionConfig(
            atom_in_dim=pretrained_cfg.atom_in_dim,
            bond_in_dim=pretrained_cfg.bond_in_dim,
            num_tasks=1,
            hidden_dim=pretrained_cfg.hidden_dim,
            num_layers=pretrained_cfg.num_layers,
            dropout=pretrained_cfg.dropout,
        )
    else:
        task_to_index = pretrained_task_to_index
        if args.dataset not in task_to_index:
            raise ValueError(
                f"Dataset {args.dataset!r} is absent from checkpoint task mapping. "
                "For a Papyrus region-only checkpoint, add --reset_y_head."
            )
        task_id = int(task_to_index[args.dataset])
        cfg = pretrained_cfg
    official_train_smiles = {row["canonical_smiles"] for row in official["train"]}
    train_items, train_load_stats = load_finetune_rows(
        args.data_file, args.dataset, official_train_smiles, task_id, cfg.num_layers
    )
    val_items, val_load_stats = load_finetune_rows(
        args.val_data_file, args.dataset, official_train_smiles, task_id, cfg.num_layers
    )
    if not train_items or not val_items:
        raise ValueError(
            f"Train/validation JSONL must both contain {args.dataset}: "
            f"{len(train_items)}/{len(val_items)}"
        )
    train_smiles = {item.canonical_smiles for item in train_items}
    val_smiles = {item.canonical_smiles for item in val_items}
    overlap = train_smiles & val_smiles
    if overlap:
        raise ValueError(
            f"Train/validation molecule overlap for {args.dataset}: {len(overlap)} canonical SMILES"
        )
    y_mean, y_std, y_stats_source = 0.0, 1.0, "unstandardized"
    test_items, bad_test = make_test_graphs(official["test"], args.dataset, task_id, cfg.num_layers)
    if not test_items:
        raise ValueError(
            f"No valid official test molecules were constructed for {args.dataset}; "
            f"official_test={len(official['test'])}, bad_test={bad_test}"
        )
    train_generator = torch.Generator(); train_generator.manual_seed(args.seed)
    train_loader = DataLoader(train_items, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=train_generator)
    val_loader = DataLoader(val_items, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_items, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = resolve_device(args.device)
    model = GINEMultiScaleRegionDetector(cfg).to(device)
    if args.reset_y_head:
        transferable_state = {
            key: value for key, value in checkpoint["model_state_dict"].items()
            if not key.startswith("y_head.")
        }
        missing, unexpected = model.load_state_dict(transferable_state, strict=False)
        non_y_missing = [key for key in missing if not key.startswith("y_head.")]
        if non_y_missing or unexpected:
            raise RuntimeError(
                f"Unexpected checkpoint mismatch: missing={non_y_missing}, unexpected={unexpected}"
            )
        print(json.dumps({"reset_y_head": True, "transferred_parameters": len(transferable_state),
                          "new_downstream_tasks": 1}))
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_rmse, best_epoch, history, stale_epochs = float("inf"), -1, [], 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, args)
        val_metrics, _ = evaluate_y(model, val_loader, device, y_mean=y_mean, y_std=y_std)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row); print(json.dumps(row))
        if val_metrics["rmse"] < best_rmse:
            best_rmse, best_epoch = val_metrics["rmse"], epoch
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "model_config": asdict(cfg),
                        "task_to_index": task_to_index, "epoch": epoch, "best_val_rmse": best_rmse,
                        "task_y_stats": {args.dataset: {"mean": y_mean, "std": y_std,
                                                        "source": y_stats_source}},
                        "dataset": args.dataset, "args": vars(args)}, output_dir / "best.pt")
        else:
            stale_epochs += 1
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    best = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    test_metrics, predictions = evaluate_y(model, test_loader, device, predictions=True,
                                           y_mean=y_mean, y_std=y_std)
    prediction_frame = pd.DataFrame(predictions)
    if prediction_frame.empty:
        raise RuntimeError(f"Test loader produced no predictions although it contains {len(test_items)} molecules")
    prediction_frame.sort_values("row_idx").to_csv(output_dir / "test_predictions.csv", index=False)
    final = {"dataset": args.dataset, "checkpoint": str(args.checkpoint), "best_epoch": best_epoch,
             "best_val_rmse": best_rmse, **{f"test_{key}": value for key, value in test_metrics.items()},
             "n_train": len(train_items), "n_val": len(val_items),
             "n_official_test": len(test_items), "bad_csv_rows": bad_csv,
             "bad_test_rows": bad_test,
             "train_load_stats": train_load_stats, "val_load_stats": val_load_stats}
    final["y_standardization"] = {"mean": y_mean, "std": y_std, "source": y_stats_source}
    (output_dir / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
