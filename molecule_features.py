#!/usr/bin/env python3
"""Shared RDKit molecule, atom, and bond feature utilities for 2207version."""

from __future__ import annotations

from typing import List, Sequence

from rdkit import Chem


def safe_mol_from_smiles(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
    except Exception:
        return None


def one_hot(value, choices: Sequence) -> List[float]:
    return [float(value == choice) for choice in choices] + [float(value not in choices)]


def atom_features(atom: Chem.Atom) -> List[float]:
    features = []
    features += one_hot(atom.GetAtomicNum(), [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53])
    features += one_hot(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
    features += one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    features += one_hot(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ])
    features += [
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.GetTotalNumHs()),
        float(atom.GetMass() * 0.01),
    ]
    return features


def bond_features(bond: Chem.Bond) -> List[float]:
    features = one_hot(bond.GetBondType(), [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ])
    features += [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    features += one_hot(bond.GetStereo(), [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ])
    return features
