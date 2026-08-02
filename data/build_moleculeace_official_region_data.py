#!/usr/bin/env python3
"""Build MoleculeACE supervision using the official cliff definition.

Design rules
------------
1. ``cliff_mol`` is copied from the official CSV and is never inferred from a
   region or from the label of a reference molecule.
2. Reference candidates follow MoleculeACE: potency difference > 10-fold
   (equivalent to ``abs(delta_y) > 1`` for pKi/pEC50) and at least one of
   Morgan-Tanimoto, generic-scaffold-Tanimoto, or normalized SMILES
   Levenshtein similarity >= 0.9.
3. Only molecules whose official ``cliff_mol == 1`` receive region labels.
   A non-cliff reference remains a negative molecule even when it explains a
   cliff molecule's region.
4. Region atoms are localized first with the single-cut MMP procedure defined
   in this file and then, if needed, with a conservative MCS fallback. An
   official cliff still not localizable is retained with ``regions=[]`` and
   ``cliff_mol=1``.
5. All official test molecules are removed immediately after reading each CSV,
   before fragmentation, train/validation splitting, or region localization.
6. The remaining official training molecules are deterministically divided
   into final train (subtrain) and validation subsets before region
   localization. Train regions use subtrain references only; validation regions
   also use subtrain references only.
7. Test molecules are never fragmented, localized, used as references, or
   written by this script.

Example
-------
python build_moleculeace_official_region_data.py \
  --input_dir data/moleculeace \
  --output data/moleculeace_official_train_regions.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
from Levenshtein import distance as levenshtein_distance
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.warning")

# Similar to the default RDKit MMPA cut pattern: non-ring, non-multiple,
# carbon-side cut.
CUT_BOND_SMARTS = '[#6+0;!$(*=,#[!#6])]!@!=!#[*]'


def safe_mol_from_smiles(smiles: str, sanitize: bool = True) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        if mol is not None:
            return mol
    except Exception:
        pass
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=(
                    Chem.SanitizeFlags.SANITIZE_ALL
                    ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                ),
            )
        except Exception:
            pass
        return mol
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = safe_mol_from_smiles(smiles, sanitize=True)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def mol_to_smiles_no_sanitize(mol: Chem.Mol) -> Optional[str]:
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        try:
            return Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)
        except Exception:
            return None


def heavy_atom_count_in_original_atoms(
    mol: Chem.Mol, atom_ids: Sequence[int]
) -> int:
    return sum(
        1 for atom_id in atom_ids
        if mol.GetAtomWithIdx(int(atom_id)).GetAtomicNum() > 1
    )


def candidate_cut_bond_indices(mol: Chem.Mol) -> List[int]:
    """Return MMP-like cuttable bond indices with a conservative fallback."""
    bond_ids = set()
    pattern = Chem.MolFromSmarts(CUT_BOND_SMARTS)
    if pattern is not None:
        for atom_a, atom_b in mol.GetSubstructMatches(pattern):
            bond = mol.GetBondBetweenAtoms(int(atom_a), int(atom_b))
            if bond is not None:
                bond_ids.add(bond.GetIdx())
    if bond_ids:
        return sorted(bond_ids)

    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        atom_a = bond.GetBeginAtom()
        atom_b = bond.GetEndAtom()
        if atom_a.GetAtomicNum() <= 1 or atom_b.GetAtomicNum() <= 1:
            continue
        bond_ids.add(bond.GetIdx())
    return sorted(bond_ids)


def choose_center_and_radius(
    mol: Chem.Mol, sub_atoms: Sequence[int], max_radius: int
) -> Tuple[int, int]:
    """Choose a stable fragment center and its capped graph radius."""
    sub_atoms = [int(atom) for atom in sub_atoms]
    if not sub_atoms:
        return -1, 0
    if len(sub_atoms) == 1:
        return sub_atoms[0], 1
    try:
        distances = Chem.GetDistanceMatrix(mol)
        scored = []
        for atom in sub_atoms:
            atom_distances = [int(distances[atom, other]) for other in sub_atoms]
            scored.append((
                max(atom_distances),
                sum(atom_distances) / len(atom_distances),
                atom,
            ))
        scored.sort()
        center = int(scored[0][2])
        real_radius = max(int(distances[center, atom]) for atom in sub_atoms)
        return center, max(1, min(int(real_radius), int(max_radius)))
    except Exception:
        return sub_atoms[0], 1


def fragment_molecule_single_cut_with_indices(
    smiles: str,
    min_core_atoms: int = 3,
    max_sub_heavy_atoms: Optional[int] = 20,
    max_radius: int = 3,
) -> List[dict]:
    """Create single-cut core/sub records while preserving original indices."""
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return []
    mol = safe_mol_from_smiles(canonical, sanitize=True)
    if mol is None:
        return []

    num_original_atoms = mol.GetNumAtoms()
    output = []
    seen = set()
    for bond_idx in candidate_cut_bond_indices(mol):
        bond = mol.GetBondWithIdx(int(bond_idx))
        atom_a = int(bond.GetBeginAtomIdx())
        atom_b = int(bond.GetEndAtomIdx())
        try:
            fragmented = Chem.FragmentOnBonds(
                mol,
                [int(bond_idx)],
                addDummies=True,
                dummyLabels=[(1, 1)],
            )
            fragment_atom_tuples = Chem.GetMolFrags(
                fragmented, asMols=False, sanitizeFrags=False
            )
            fragment_mols = Chem.GetMolFrags(
                fragmented, asMols=True, sanitizeFrags=False
            )
        except Exception:
            continue
        if len(fragment_atom_tuples) != 2 or len(fragment_mols) != 2:
            continue

        sides = []
        for fragment_index, atom_tuple in enumerate(fragment_atom_tuples):
            original_atoms = sorted(
                int(atom) for atom in atom_tuple
                if int(atom) < num_original_atoms
            )
            if not original_atoms:
                continue
            fragment_smiles = mol_to_smiles_no_sanitize(
                fragment_mols[fragment_index]
            )
            if not fragment_smiles:
                continue
            sides.append({
                "atoms": original_atoms,
                "smi": fragment_smiles,
                "heavy": heavy_atom_count_in_original_atoms(
                    mol, original_atoms
                ),
            })
        if len(sides) != 2:
            continue

        if (sides[0]["heavy"], sides[0]["smi"]) >= (
            sides[1]["heavy"], sides[1]["smi"]
        ):
            core_side, sub_side = sides[0], sides[1]
        else:
            core_side, sub_side = sides[1], sides[0]
        if core_side["heavy"] < min_core_atoms:
            continue
        if (
            max_sub_heavy_atoms is not None
            and sub_side["heavy"] > max_sub_heavy_atoms
        ):
            continue
        if core_side["smi"] == sub_side["smi"]:
            continue

        core_atoms = sorted(core_side["atoms"])
        sub_atoms = sorted(sub_side["atoms"])
        if atom_a in core_atoms and atom_b in sub_atoms:
            connection_atom_core, connection_atom_sub = atom_a, atom_b
        elif atom_b in core_atoms and atom_a in sub_atoms:
            connection_atom_core, connection_atom_sub = atom_b, atom_a
        else:
            continue

        center_atom, radius = choose_center_and_radius(
            mol, sub_atoms, max_radius=max_radius
        )
        if center_atom < 0:
            continue
        key = (
            core_side["smi"],
            sub_side["smi"],
            tuple(sub_atoms),
            int(bond_idx),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append({
            "core": core_side["smi"],
            "sub": sub_side["smi"],
            "core_atoms": core_atoms,
            "sub_atoms": sub_atoms,
            "connection_bond_idx": int(bond_idx),
            "connection_bond": [atom_a, atom_b],
            "connection_atom_core": int(connection_atom_core),
            "connection_atom_sub": int(connection_atom_sub),
            "center_atom": int(center_atom),
            "radius": int(radius),
            "localization_valid": True,
        })
    return sorted(
        output,
        key=lambda item: (
            item["core"], item["sub"], item["connection_bond_idx"]
        ),
    )


def canonicalize(value):
    mol = Chem.MolFromSmiles(str(value)) if value is not None else None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else None


def finite_float(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def generic_scaffold_fp(mol, radius, nbits):
    try:
        scaffold = MakeScaffoldGeneric(mol)
    except Exception:
        scaffold = GetScaffoldForMol(mol)
    return AllChem.GetMorganFingerprintAsBitVect(scaffold, radius=radius, nBits=nbits)


def normalized_levenshtein_similarity(a, b):
    denominator = max(len(a), len(b))
    return 1.0 if denominator == 0 else 1.0 - levenshtein_distance(a, b) / denominator


def official_similarity(a, b, threshold):
    """Return whether a pair passes any official similarity and its details."""
    tani = DataStructs.TanimotoSimilarity(a["fp"], b["fp"])
    scaffold = DataStructs.TanimotoSimilarity(a["scaffold_fp"], b["scaffold_fp"])
    lev = normalized_levenshtein_similarity(a["smiles"], b["smiles"])
    methods = []
    if tani >= threshold:
        methods.append("morgan_tanimoto")
    if scaffold >= threshold:
        methods.append("scaffold_tanimoto")
    if lev >= threshold:
        methods.append("smiles_levenshtein")
    return bool(methods), {
        "similarity_methods": methods,
        "morgan_tanimoto": tani,
        "scaffold_tanimoto": scaffold,
        "smiles_levenshtein": lev,
        "max_similarity": max(tani, scaffold, lev),
    }


def read_dataset(path, args):
    frame = pd.read_csv(path)
    required = [args.smiles_col, args.y_col, args.split_col, args.cliff_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; available={list(frame.columns)}")

    rows, invalid = [], 0
    for row_index, raw in frame.iterrows():
        smiles = canonicalize(raw[args.smiles_col])
        y = finite_float(raw[args.y_col])
        split = str(raw[args.split_col]).strip().lower()
        if smiles is None or y is None or split not in {"train", "test"}:
            invalid += 1
            continue
        mol = Chem.MolFromSmiles(smiles)
        rows.append({
            "dataset": path.stem,
            "row_idx": int(row_index),
            "smiles": smiles,
            "canonical_smiles": smiles,
            "y": y,
            "split": split,
            "cliff_mol": int(bool(raw[args.cliff_col])),
            "mol": mol,
            "fp": AllChem.GetMorganFingerprintAsBitVect(mol, radius=args.fp_radius, nBits=args.fp_bits),
            "scaffold_fp": generic_scaffold_fp(mol, args.fp_radius, args.fp_bits),
        })
    return rows, invalid


def fragment_rows(rows, args):
    cache = {}
    for row in rows:
        smiles = row["smiles"]
        if smiles not in cache:
            cache[smiles] = fragment_molecule_single_cut_with_indices(
                smiles,
                min_core_atoms=args.min_core_atoms,
                max_sub_heavy_atoms=args.max_sub_heavy_atoms,
                max_radius=args.max_radius,
            )
        row["fragments"] = cache[smiles]


def split_official_train(rows, val_ratio, split_seed):
    """Deterministically split official train rows, stratifying when possible."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if len(rows) < 2:
        raise ValueError("At least two official train rows are required for train/val splitting")

    labels = [int(row["cliff_mol"]) for row in rows]
    label_counts = {label: labels.count(label) for label in set(labels)}
    stratify = labels if len(label_counts) > 1 and min(label_counts.values()) >= 2 else None
    try:
        train_rows, val_rows = train_test_split(
            rows,
            test_size=val_ratio,
            random_state=split_seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        # Very small tasks may not have enough rows in train and validation to
        # represent every class. Keep the split deterministic without stratification.
        train_rows, val_rows = train_test_split(
            rows,
            test_size=val_ratio,
            random_state=split_seed,
            shuffle=True,
            stratify=None,
        )
    # Preserve source-file order in the generated artifacts.
    train_rows = sorted(train_rows, key=lambda row: row["row_idx"])
    val_rows = sorted(val_rows, key=lambda row: row["row_idx"])
    return train_rows, val_rows


def single_cut_regions(current, reference):
    """Locate the current molecule's changed side for every shared MMP core."""
    ref_by_core = defaultdict(list)
    for fragment in reference["fragments"]:
        ref_by_core[fragment["core"]].append(fragment)

    found = []
    for fragment in current["fragments"]:
        for ref_fragment in ref_by_core.get(fragment["core"], []):
            if fragment["sub"] == ref_fragment["sub"]:
                continue
            found.append(fragment)
            break
    # A molecule can expose the same atom set through equivalent cuts.
    unique = {}
    for fragment in found:
        key = tuple(sorted(int(atom) for atom in fragment["sub_atoms"]))
        unique.setdefault(key, fragment)
    return [{
        "atom_indices": list(key),
        "connection_bond_indices": [int(fragment["connection_bond_idx"])],
        "localization_method": "single_cut_mmp",
        "localization_confidence": 1.0,
    } for key, fragment in unique.items()]


def count_region_components(mol, atoms):
    remaining = set(atoms)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            atom = mol.GetAtomWithIdx(stack.pop())
            for neighbor in atom.GetNeighbors():
                idx = neighbor.GetIdx()
                if idx in remaining:
                    remaining.remove(idx)
                    stack.append(idx)
    return components


def mcs_region(current, reference, args):
    """Conservative fallback: changed atoms plus the mapped interface atoms."""
    mol, ref_mol = current["mol"], reference["mol"]
    result = rdFMCS.FindMCS(
        [mol, ref_mol],
        timeout=args.mcs_timeout_seconds,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
    )
    if result.canceled or result.numAtoms < args.mcs_min_atoms:
        return None
    query = Chem.MolFromSmarts(result.smartsString)
    if query is None:
        return None
    current_match = mol.GetSubstructMatch(query)
    reference_match = ref_mol.GetSubstructMatch(query)
    if not current_match or not reference_match:
        return None

    mcs_fraction = result.numAtoms / max(1, min(mol.GetNumAtoms(), ref_mol.GetNumAtoms()))
    if mcs_fraction < args.mcs_min_core_fraction:
        return None

    current_core = set(current_match)
    reference_core = set(reference_match)
    region = set(range(mol.GetNumAtoms())) - current_core

    # Add interface atoms on the current side. This also creates a meaningful
    # target when the structural change is a deletion relative to the reference.
    query_to_current = {query_idx: atom_idx for query_idx, atom_idx in enumerate(current_match)}
    reference_to_query = {atom_idx: query_idx for query_idx, atom_idx in enumerate(reference_match)}
    for atom_idx in list(region):
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            if neighbor.GetIdx() in current_core:
                region.add(neighbor.GetIdx())
    for ref_atom_idx in reference_core:
        if any(neighbor.GetIdx() not in reference_core
               for neighbor in ref_mol.GetAtomWithIdx(ref_atom_idx).GetNeighbors()):
            region.add(query_to_current[reference_to_query[ref_atom_idx]])

    if not region or len(region) > args.mcs_max_region_atoms:
        return None
    region_fraction = len(region) / mol.GetNumAtoms()
    if region_fraction > args.mcs_max_region_fraction:
        return None
    if count_region_components(mol, region) > args.mcs_max_components:
        return None

    crossing_bonds = []
    for bond in mol.GetBonds():
        inside_a = bond.GetBeginAtomIdx() in region
        inside_b = bond.GetEndAtomIdx() in region
        if inside_a != inside_b:
            crossing_bonds.append(bond.GetIdx())
    confidence = max(0.4, min(0.8, mcs_fraction * (1.0 - 0.5 * region_fraction)))
    return {
        "atom_indices": sorted(region),
        "connection_bond_indices": sorted(crossing_bonds),
        "localization_method": "mcs_fallback",
        "localization_confidence": confidence,
        "mcs_atoms": int(result.numAtoms),
        "mcs_core_fraction": mcs_fraction,
        "region_fraction": region_fraction,
    }


def collect_effects(rows, allowed_references, args):
    """Collect region effects only for official cliff anchors."""
    effects = defaultdict(lambda: defaultdict(list))
    stats = defaultdict(int)
    for current in rows:
        if current["cliff_mol"] != 1:
            continue
        stats["official_cliff_molecules"] += 1
        qualifying = []
        for reference in allowed_references(current):
            if reference is current or reference["smiles"] == current["smiles"]:
                continue
            delta = current["y"] - reference["y"]
            # Official MoleculeACE uses a strict potency_fold > 10 comparison.
            if abs(delta) <= args.min_abs_delta_y:
                continue
            similar, similarity_info = official_similarity(current, reference, args.similarity_threshold)
            if not similar:
                continue
            qualifying.append((similarity_info["max_similarity"], abs(delta), reference, delta, similarity_info))

        if qualifying:
            stats["cliffs_with_official_reference"] += 1
        qualifying.sort(key=lambda item: (item[0], item[1]), reverse=True)
        localized_reference_count = 0
        molecule_methods = set()
        localized_reference_ids = set()

        def add_regions(reference, signed_delta, similarity_info, regions):
            nonlocal localized_reference_count
            stats["localized_official_pairs"] += 1
            localized_reference_count += 1
            localized_reference_ids.add(reference["row_idx"])
            for region in regions:
                method = region["localization_method"]
                molecule_methods.add(method)
                stats[f"localized_pairs_{method}"] += 1
                atom_key = tuple(region["atom_indices"])
                effects[current["row_idx"]][atom_key].append({
                    "signed_delta_y": signed_delta,
                    "reference_smiles": reference["smiles"],
                    "reference_row_idx": reference["row_idx"],
                    "reference_split": reference["split"],
                    "connection_bond_indices": region["connection_bond_indices"],
                    "localization_method": method,
                    "localization_confidence": region["localization_confidence"],
                    "mcs_core_fraction": region.get("mcs_core_fraction"),
                    **similarity_info,
                })

        # Pass 1: exhaust all official candidates with the precise single-cut
        # method. The cap is applied only after successful localization.
        if "single_cut" in args.localization_methods:
            for _, _, reference, signed_delta, similarity_info in qualifying:
                if (args.max_references_per_molecule > 0 and
                        localized_reference_count >= args.max_references_per_molecule):
                    break
                regions = single_cut_regions(current, reference)
                if regions:
                    add_regions(reference, signed_delta, similarity_info, regions)

        # Pass 2: MCS is deliberately used only when single-cut found nothing.
        # This preserves the most interpretable labels and bounds runtime.
        if (localized_reference_count == 0 and "mcs" in args.localization_methods):
            mcs_attempts = 0
            for _, _, reference, signed_delta, similarity_info in qualifying:
                if args.max_mcs_attempts_per_molecule > 0 and mcs_attempts >= args.max_mcs_attempts_per_molecule:
                    break
                mcs_attempts += 1
                stats["mcs_pairs_attempted"] += 1
                region = mcs_region(current, reference, args)
                if region is None:
                    continue
                add_regions(reference, signed_delta, similarity_info, [region])
                # One conservative fallback label is enough for an otherwise
                # unlocalized molecule and avoids mixing several uncertain MCSs.
                break

        stats["official_pairs_without_localized_region"] += max(
            0, len(qualifying) - len(localized_reference_ids)
        )
        if localized_reference_count:
            stats["cliffs_with_localized_region"] += 1
            for method in molecule_methods:
                stats[f"molecules_localized_by_{method}"] += 1
    return effects, dict(stats)


def output_row(row, observations_by_atoms, max_regions):
    regions = []
    for atoms, observations in observations_by_atoms.items():
        signed_values = [item["signed_delta_y"] for item in observations]
        mean_signed = sum(signed_values) / len(signed_values)
        if mean_signed == 0:
            continue
        direction = 1 if mean_signed > 0 else -1
        agreement = sum((value > 0) == (direction > 0) for value in signed_values) / len(signed_values)
        regions.append({
            "atom_indices": list(atoms),
            "connection_bond_indices": sorted({
                bond for item in observations for bond in item["connection_bond_indices"]
            }),
            "signed_delta_y": mean_signed,
            "direction": direction,
            "direction_label": int(direction > 0),
            "magnitude": sum(abs(value) for value in signed_values) / len(signed_values),
            "direction_agreement": agreement,
            "localization_methods": sorted({item["localization_method"] for item in observations}),
            "localization_confidence": sum(item["localization_confidence"] for item in observations) / len(observations),
            "count": len(observations),
            "references": [{
                "smiles": item["reference_smiles"],
                "row_idx": item["reference_row_idx"],
                "split": item["reference_split"],
                "signed_delta_y": item["signed_delta_y"],
                "similarity_methods": item["similarity_methods"],
                "max_similarity": item["max_similarity"],
                "localization_method": item["localization_method"],
                "localization_confidence": item["localization_confidence"],
            } for item in observations],
        })
    regions.sort(key=lambda item: item["magnitude"] * item["direction_agreement"], reverse=True)
    if max_regions > 0:
        regions = regions[:max_regions]
    return {
        "dataset": row["dataset"],
        "row_idx": row["row_idx"],
        "smiles": row["smiles"],
        "canonical_smiles": row["canonical_smiles"],
        "y": row["y"],
        "split": row["split"],
        "cliff_mol": row["cliff_mol"],
        # Existence supervision is the immutable official label. has_region is
        # kept as a separate localization-availability diagnostic.
        "cliff_existence_label": row["cliff_mol"],
        "regions": regions,
        "num_regions": len(regions),
        "has_region": int(bool(regions)),
        "max_region_magnitude": max((item["magnitude"] for item in regions), default=0.0),
    }


def write_jsonl(path, rows, effects, max_regions):
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            item = output_row(row, effects.get(row["row_idx"], {}), max_regions)
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            counts["molecules"] += 1
            counts["official_cliff_molecules"] += item["cliff_mol"]
            counts["molecules_with_region"] += item["has_region"]
            counts["regions"] += item["num_regions"]
            if item["cliff_mol"] and not item["has_region"]:
                counts["official_cliffs_without_region"] += 1
    return dict(counts)


def parse_args():
    parser = argparse.ArgumentParser(description="Build leakage-safe official MoleculeACE region supervision")
    parser.add_argument("--input_dir", required=True, help="Directory containing the 30 MoleculeACE CSV files")
    parser.add_argument("--output", required=True, help="TRAIN-only JSONL used by the model")
    parser.add_argument("--val_output", default=None,
                        help="Validation JSONL; default derived from --output")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of each official train split reserved for validation")
    parser.add_argument("--val_split_seed", type=int, default=0,
                        help="Fixed seed for the reproducible train/validation split")
    parser.add_argument("--smiles_col", default="smiles")
    parser.add_argument("--y_col", default="y [pEC50/pKi]")
    parser.add_argument("--split_col", default="split")
    parser.add_argument("--cliff_col", default="cliff_mol")
    parser.add_argument("--similarity_threshold", type=float, default=0.9)
    parser.add_argument("--min_abs_delta_y", type=float, default=1.0,
                        help="Strict lower bound; 1.0 corresponds to MoleculeACE potency_fold > 10")
    parser.add_argument("--fp_radius", type=int, default=2)
    parser.add_argument("--fp_bits", type=int, default=1024)
    parser.add_argument("--min_core_atoms", type=int, default=3)
    parser.add_argument("--max_sub_heavy_atoms", type=int, default=30)
    parser.add_argument("--max_radius", type=int, default=3)
    parser.add_argument("--max_references_per_molecule", type=int, default=20)
    parser.add_argument("--max_regions_per_mol", type=int, default=16)
    parser.add_argument("--localization_methods", default="single_cut,mcs",
                        help="Comma-separated ordered methods: single_cut,mcs")
    parser.add_argument("--mcs_timeout_seconds", type=int, default=1)
    parser.add_argument("--max_mcs_attempts_per_molecule", type=int, default=1,
                        help="MCS is expensive; raise to 3-5 when runtime permits")
    parser.add_argument("--mcs_min_atoms", type=int, default=3)
    parser.add_argument("--mcs_min_core_fraction", type=float, default=0.60)
    parser.add_argument("--mcs_max_region_atoms", type=int, default=20)
    parser.add_argument("--mcs_max_region_fraction", type=float, default=0.50)
    parser.add_argument("--mcs_max_components", type=int, default=2)
    parser.add_argument("--progress_every_dataset", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    args.localization_methods = [item.strip() for item in args.localization_methods.split(",") if item.strip()]
    unknown_methods = set(args.localization_methods) - {"single_cut", "mcs"}
    if unknown_methods:
        raise ValueError(f"Unknown localization methods: {sorted(unknown_methods)}")
    input_dir = Path(args.input_dir)
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    output = Path(args.output)
    val_output = (Path(args.val_output) if args.val_output else
                  output.with_name(output.stem + "_val.jsonl"))
    train_items, val_items, per_dataset = [], [], {}

    for dataset_index, csv_path in enumerate(csv_files, 1):
        rows, invalid = read_dataset(csv_path, args)
        input_valid_rows = len(rows)
        removed_test_rows = sum(row["split"] == "test" for row in rows)

        # Remove the complete official test partition before any expensive or
        # label-dependent processing. From this point onward, fragmentation,
        # splitting, reference selection, and localization can only observe
        # official training molecules.
        official_train = [row for row in rows if row["split"] == "train"]
        if len(official_train) + removed_test_rows != input_valid_rows:
            raise AssertionError(f"Unexpected split value survived parsing in {csv_path}")
        if any(row["split"] == "test" for row in official_train):
            raise AssertionError(f"Official test row survived filtering in {csv_path}")

        fragment_rows(official_train, args)
        train, val = split_official_train(
            official_train, args.val_ratio, args.val_split_seed
        )

        # Split before localization: neither train labels nor their aggregates
        # can observe validation molecules or validation activities.
        train_effects, train_stats = collect_effects(train, lambda _: train, args)
        # Validation labels simulate inference against the information available
        # after training, so validation anchors may use final-train references only.
        val_effects, val_stats = collect_effects(val, lambda _: train, args)
        train_items.extend((row, train_effects.get(row["row_idx"], {})) for row in train)
        val_items.extend((dict(row, split="val"), val_effects.get(row["row_idx"], {})) for row in val)
        per_dataset[csv_path.stem] = {
            "input_valid_rows": input_valid_rows, "invalid_rows": invalid,
            "removed_test_rows_before_processing": removed_test_rows,
            "official_train_rows": len(official_train),
            "train_rows": len(train), "val_rows": len(val),
            "train_region_search": train_stats,
            "val_region_search": val_stats,
        }
        if args.progress_every_dataset and dataset_index % args.progress_every_dataset == 0:
            print(f"Processed {dataset_index}/{len(csv_files)}: {csv_path.stem}; "
                  f"subtrain={len(train):,}, val={len(val):,}, "
                  f"removed_test={removed_test_rows:,}", flush=True)

    # row_idx is only unique inside a dataset, so write directly from paired
    # (row, effects) tuples rather than combining effect dictionaries.
    def write_items(path, items):
        path.parent.mkdir(parents=True, exist_ok=True)
        counts = defaultdict(int)
        with path.open("w", encoding="utf-8") as handle:
            for row, effect in items:
                item = output_row(row, effect, args.max_regions_per_mol)
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                counts["molecules"] += 1
                counts["official_cliff_molecules"] += item["cliff_mol"]
                counts["molecules_with_region"] += item["has_region"]
                counts["regions"] += item["num_regions"]
                counts["official_cliffs_without_region"] += int(item["cliff_mol"] == 1 and not item["has_region"])
        return dict(counts)

    train_counts = write_items(output, train_items)
    val_counts = write_items(val_output, val_items)
    summary = {
        "input_dir": str(input_dir), "datasets": len(csv_files),
        "train_output": str(output), "val_output": str(val_output),
        "validation_split": {
            "ratio": args.val_ratio,
            "seed": args.val_split_seed,
            "stratification": "official cliff_mol within each dataset when possible",
        },
        "official_definition": {
            "similarity": f"any of Morgan/scaffold/SMILES-Levenshtein >= {args.similarity_threshold}",
            "potency": f"abs(delta_y) > {args.min_abs_delta_y}",
        },
        "leakage_policy": (
            "remove all official test rows before fragmentation and splitting; split remaining "
            "official train before region localization; subtrain and validation regions use "
            "subtrain references only; test rows are never processed or generated"
        ),
        "region_policy": f"{args.localization_methods}, official cliff anchors only",
        "train": train_counts, "validation": val_counts,
        "per_dataset": per_dataset,
    }
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
