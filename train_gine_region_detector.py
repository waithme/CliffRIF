#!/usr/bin/env python3
"""Train and evaluate the multi-scale GINE region detector.

The ACNet combined file contains multiple targets, so y is learned with one
output per target while all targets share the GINE and region detector.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gine_region_detector import (
    GINEMultiScaleRegionDetector,
    MultiScaleRegionConfig,
    dice_loss,
)
from molecule_features import atom_features, bond_features, safe_mol_from_smiles

def validate_loss_switches(args):
    enabled = {"y": True, "mask": True, "existence": True}
    print(json.dumps({"loss_switches": enabled}))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_rows(path, dataset_filter=None, max_records=0):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if dataset_filter and str(row.get("dataset")) != dataset_filter:
                continue
            rows.append(row)
            if max_records and len(rows) >= max_records:
                break
    return rows


def row_to_data(row, task_id, num_layers):
    smiles = str(row.get("smiles") or row.get("canonical_smiles") or "")
    mol = safe_mol_from_smiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    pairs, attrs = [], []
    for bond in mol.GetBonds():
        i, j, feature = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_features(bond)
        pairs.extend([(i, j), (j, i)])
        attrs.extend([feature, feature])
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32) if attrs else torch.empty((0, 12), dtype=torch.float32)
    n = mol.GetNumAtoms()
    # Single-scale ablation: only the final GINE layer predicts a region mask.
    num_region_scales = 1
    mask = torch.zeros((n, num_region_scales), dtype=torch.float32)
    strongest_magnitude = torch.full((1, num_region_scales), -1.0, dtype=torch.float32)
    region_target_valid = torch.zeros((1, num_region_scales), dtype=torch.float32)
    # Used only to weight mask supervision: 0=none, 1=single-cut, 2=MCS.
    region_method = torch.zeros((1, num_region_scales), dtype=torch.long)
    for region in row.get("regions", []) or []:
        atoms = sorted(set(int(atom) for atom in region.get("atom_indices", []) if 0 <= int(atom) < n))
        if not atoms:
            continue
        signed = float(region.get("signed_delta_y", 0.0))
        magnitude = float(region.get("magnitude", abs(signed)))
        if signed == 0:
            continue
        for scale in range(num_region_scales):
            for atom in atoms:
                mask[atom, scale] = 1.0
        methods = region.get("localization_methods") or [region.get("localization_method", "single_cut_mmp")]
        method_code = 1 if "single_cut_mmp" in methods else (2 if "mcs_fallback" in methods else 1)
        for scale in range(num_region_scales):
            # Match the Full model's supervision weight: the strongest
            # localized region determines the per-scale localization method.
            if magnitude > strongest_magnitude[0, scale]:
                strongest_magnitude[0, scale] = magnitude
                region_target_valid[0, scale] = 1.0
                region_method[0, scale] = method_code
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.mask_target = mask
    data.region_target_valid = region_target_valid
    data.region_method = region_method
    # New official-label files explicitly provide cliff_existence_label. Region
    # availability is a different concept: an official cliff can legitimately
    # have no localizable single-cut MMP. Older files lack this explicit field,
    # so retain their former mask-derived behavior for backward compatibility.
    existence = row.get("cliff_existence_label", float(mask.sum() > 0))
    data.has_region = torch.tensor([float(bool(existence))], dtype=torch.float32)
    # An official cliff with no localized region has unknown mask labels, not
    # an all-negative mask. Explicit-label files can therefore skip mask loss.
    explicit_existence = "cliff_existence_label" in row
    mask_valid = not (explicit_existence and bool(existence) and mask.sum() == 0)
    data.mask_supervision_valid = torch.tensor([float(mask_valid)], dtype=torch.float32)
    data.y = torch.tensor([float(row["y"])], dtype=torch.float32)
    # Avoid the substring "index": PyG automatically offsets index-like
    # attributes while batching.
    data.task_id = torch.tensor([task_id], dtype=torch.long)
    data.smiles = smiles
    data.dataset = str(row.get("dataset", "unknown"))
    return data


class DetectorDataset(Dataset):
    def __init__(self, path, num_layers, dataset_filter=None, max_records=0,
                 task_to_index=None):
        rows = load_rows(path, dataset_filter, max_records)
        task_names = sorted({str(row.get("dataset", "unknown")) for row in rows})
        if task_to_index is None:
            self.task_to_index = {task: idx for idx, task in enumerate(task_names)}
        else:
            unknown_tasks = sorted(set(task_names).difference(task_to_index))
            if unknown_tasks:
                raise ValueError(f"Validation contains tasks absent from train: {unknown_tasks}")
            self.task_to_index = dict(task_to_index)
        self.items, self.bad, self.no_region = [], 0, 0
        for row in rows:
            task = str(row.get("dataset", "unknown"))
            try:
                data = row_to_data(row, self.task_to_index[task], num_layers)
            except Exception:
                data = None
            if data is None:
                self.bad += 1
            else:
                if float(data.has_region.item()) == 0.0:
                    self.no_region += 1
                self.items.append(data)
        if not self.items:
            raise ValueError("No valid molecules")
        self.atom_in_dim = int(self.items[0].x.size(-1))
        self.bond_in_dim = int(self.items[0].edge_attr.size(-1))

    def __len__(self): return len(self.items)
    def __getitem__(self, index): return self.items[index]


def losses(output, batch, args):
    mask_total = output["y_pred"].new_tensor(0.0)
    method_weights = torch.ones_like(batch.region_method, dtype=torch.float32)
    method_weights = torch.where(
        batch.region_method == 1,
        method_weights.new_tensor(args.single_cut_region_weight),
        method_weights,
    )
    method_weights = torch.where(
        batch.region_method == 2,
        method_weights.new_tensor(args.mcs_region_weight),
        method_weights,
    )
    for layer in range(len(output["mask_logits"])):
        target = batch.mask_target[:, layer].float()
        graph_valid = batch.mask_supervision_valid.view(-1) > 0.5
        node_valid = graph_valid[batch.batch.long()]
        if node_valid.any():
            logits_valid = output["mask_logits"][layer][node_valid]
            target_valid = target[node_valid]
            graph_layer_weight = method_weights[:, layer]
            node_weight = graph_layer_weight[batch.batch.long()][node_valid]
            raw_focal = F.binary_cross_entropy_with_logits(logits_valid, target_valid, reduction="none")
            probability = torch.sigmoid(logits_valid)
            pt = probability * target_valid + (1.0 - probability) * (1.0 - target_valid)
            alpha_t = 0.75 * target_valid + 0.25 * (1.0 - target_valid)
            mask_loss = (node_weight * alpha_t * (1.0 - pt).pow(2.0) * raw_focal).mean()
        else:
            mask_loss = output["y_pred"].new_tensor(0.0)
        # Dice is meaningful only when the current batch/layer contains a
        # positive region. For an all-negative batch, focal BCE supplies the
        # correct pressure toward zero without adding a near-constant Dice 1.
        if node_valid.any() and (target[node_valid] > 0.5).any():
            positive_graph = (batch.region_target_valid[:, layer] > 0.5)
            dice_weight = method_weights[:, layer][positive_graph].mean() if positive_graph.any() else 1.0
            mask_loss = mask_loss + dice_weight * dice_loss(logits_valid, target_valid)
        mask_total = mask_total + mask_loss
    layers = len(output["mask_logits"])
    mask_total = mask_total / layers
    y_target = batch.y_target if hasattr(batch, "y_target") else batch.y
    y_loss = F.mse_loss(output["y_pred"], y_target.view(-1).float())
    existence_target = batch.has_region.view(-1).float()
    existence_loss = F.binary_cross_entropy_with_logits(
        output["region_existence_logit"], existence_target
    )
    total = (
        args.lambda_y * y_loss
        + args.lambda_mask * mask_total
        + args.lambda_exist * existence_loss
    )
    return {"loss": total, "y": y_loss, "mask": mask_total, "existence": existence_loss}


def run_epoch(model, loader, device, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ["loss", "y", "mask", "existence"]}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            values = losses(model(batch), batch, args)
            if training:
                optimizer.zero_grad(set_to_none=True)
                values["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            size = int(batch.num_graphs)
            count += size
            for key in totals:
                totals[key] += float(values[key].detach()) * size
    return {key: value / max(count, 1) for key, value in totals.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-scale dense GINE region detector")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional region-pretrained checkpoint; y_head is reset for current tasks")
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--val_data_file", required=True,
                        help="Pre-generated validation JSONL; it is never re-split")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=300)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_y", type=float, default=1.0)
    parser.add_argument("--lambda_mask", type=float, default=0.5)
    parser.add_argument("--lambda_exist", type=float, default=0.5)
    parser.add_argument("--single_cut_region_weight", type=float, default=1.0,
                        help="Loss weight for single-cut MMP region supervision")
    parser.add_argument("--mcs_region_weight", type=float, default=0.5,
                        help="Loss weight for lower-confidence MCS region supervision")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_loss_switches(args)
    set_seed(args.seed)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    train_set = DetectorDataset(args.data_file, args.num_layers, args.dataset, args.max_records)
    val_set = DetectorDataset(args.val_data_file, args.num_layers, args.dataset,
                              args.max_records, task_to_index=train_set.task_to_index)
    task_y_stats = {
        task_name: {"mean": 0.0, "std": 1.0, "source": "unstandardized",
                    "n_train": sum(int(item.task_id.item()) == task_index
                                   for item in train_set.items)}
        for task_name, task_index in train_set.task_to_index.items()
    }
    train_generator = torch.Generator(); train_generator.manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=train_generator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    initialization_checkpoint = None
    if args.checkpoint:
        initialization_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        pretrained_cfg = MultiScaleRegionConfig(**initialization_checkpoint["model_config"])
        if pretrained_cfg.atom_in_dim != train_set.atom_in_dim or pretrained_cfg.bond_in_dim != train_set.bond_in_dim:
            raise ValueError(
                f"Feature dimensions differ: checkpoint={pretrained_cfg.atom_in_dim}/{pretrained_cfg.bond_in_dim}, "
                f"data={train_set.atom_in_dim}/{train_set.bond_in_dim}"
            )
        cfg = MultiScaleRegionConfig(
            train_set.atom_in_dim, train_set.bond_in_dim, len(train_set.task_to_index),
            pretrained_cfg.hidden_dim, pretrained_cfg.num_layers, pretrained_cfg.dropout,
        )
    else:
        cfg = MultiScaleRegionConfig(train_set.atom_in_dim, train_set.bond_in_dim, len(train_set.task_to_index),
                                     args.hidden_dim, args.num_layers, args.dropout)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = GINEMultiScaleRegionDetector(cfg).to(device)
    if initialization_checkpoint is not None:
        transferable_state = {
            key: value for key, value in initialization_checkpoint["model_state_dict"].items()
            if not key.startswith("y_head.")
        }
        missing, unexpected = model.load_state_dict(transferable_state, strict=False)
        non_y_missing = [key for key in missing if not key.startswith("y_head.")]
        if non_y_missing or unexpected:
            raise RuntimeError(
                f"Unexpected checkpoint mismatch: missing={non_y_missing}, unexpected={unexpected}"
            )
        print(json.dumps({
            "initialized_from": str(args.checkpoint),
            "transferred_parameters": len(transferable_state),
            "reset_y_head_num_tasks": len(train_set.task_to_index),
        }))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, history, stale_epochs = float("inf"), [], 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, args, optimizer)
        val_metrics = run_epoch(model, val_loader, device, args)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row); print(json.dumps(row))
        checkpoint = {"model_state_dict": model.state_dict(), "model_config": asdict(cfg),
                      "task_to_index": train_set.task_to_index, "task_y_stats": task_y_stats,
                      "epoch": epoch, "args": vars(args)}
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["y"] < best:
            best = val_metrics["y"]
            stale_epochs = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            stale_epochs += 1
        with (output_dir / "history.json").open("w") as handle: json.dump(history, handle, indent=2)
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best validation y MSE={best:.6f}")
            break
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    with (output_dir / "meta.json").open("w") as handle:
        json.dump({"best_val_y_mse": best, "num_train_molecules": len(train_set),
                   "num_val_molecules": len(val_set), "num_tasks": len(train_set.task_to_index),
                   "train_no_region_molecules": train_set.no_region,
                   "val_no_region_molecules": val_set.no_region,
                   "train_bad_records": train_set.bad, "val_bad_records": val_set.bad,
                   "task_y_stats": task_y_stats}, handle, indent=2)


if __name__ == "__main__":
    main()
