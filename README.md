# CliffRIF

CliffRIF is a **region-informed auxiliary-learning framework** for activity-cliff-aware
molecular property prediction. It converts structural
differences obtained from matched molecular pairs (MMPs) and a conservative maximum
common substructure (MCS) fallback into atom-level region supervision. During training,
the model jointly learns molecular activity, cliff-sensitive atom regions, and
molecule-level cliff existence. During inference, it requires only one molecular graph.

## Motivation

[![Activity-cliff motivation](docs/assets/activity_cliff_motivation.png)](figures/cliff.pdf)

Highly similar molecules can exhibit large differences in potency. A conventional
graph-level predictor may place these molecules close together in representation space,
whereas pair-dependent models often require a reference molecule at inference. CliffRIF
uses molecular pairs only to construct training supervision and preserves graph-only,
single-molecule inference.

The image above is a preview of the [activity-cliff motivation figure](figures/cliff.pdf).

## Framework

[![CliffRIF pipeline](docs/assets/cliffrif_pipeline.png)](figures/cliffmodel.pdf)

The pipeline has three stages:

1. **Data construction:** official MoleculeACE test rows are removed before any matching
   or splitting. The remaining official training partition is split into final training
   and validation subsets. MMP localization and a conservative MCS fallback provide
   cliff-sensitive atom-region labels.
2. **Auxiliary-guided training:** a shared three-layer GINE encoder is optimized using
   property regression, final-layer atom-region localization, and cliff-existence losses.
   The property head receives only the globally pooled molecular representation.
3. **Single-molecule inference:** region annotations and paired reference molecules are
   not required. A test molecule is passed through the GINE encoder and property head.

The image above is a preview of the full [CliffRIF model and data pipeline](figures/cliffmodel.pdf).

## Repository Layout

```text
data_generation/   Leakage-safe MoleculeACE region-label construction
final_model/       CliffRIF model, pretraining, fine-tuning, and multi-seed runner
ablation/          Ablation variants and result summarization
results/           Benchmark and ablation result tables
visual/            Region-prediction visualization utilities
writing/           Manuscript figures
```

## Requirements

The main dependencies are:

- Python 3.10 or later
- PyTorch
- PyTorch Geometric
- RDKit
- NumPy and pandas
- scikit-learn
- python-Levenshtein

Install packages using the method appropriate for your CUDA and PyTorch versions. A
minimal CPU-oriented installation is:

```bash
python -m pip install torch torch-geometric rdkit numpy pandas scikit-learn python-Levenshtein
```

## Data Preparation

Place the 30 official MoleculeACE CSV files in `data/moleculeace/`. Each CSV must retain
the official split and cliff-label columns.

Generate disjoint training and validation region files from the repository root:

```bash
python data_generation/build_moleculeace_official_region_datav2.py \
  --input_dir data/moleculeace \
  --output data/train_regions.jsonl \
  --val_output data/val_regions.jsonl \
  --val_ratio 0.1 \
  --val_split_seed 0
```

The preprocessing script removes all official test rows before fragmentation,
train-validation splitting, reference search, and region localization. Training anchors
are matched only against final-training references; validation anchors are also matched
only against final-training references.

## Run the Full Experiment

The following command runs all five seeds (`0 1 2 3 4`) on every MoleculeACE dataset:

```bash
python final_model/run_multi_ablation.py final \
  --train_file data/train_regions.jsonl \
  --val_file data/val_regions.jsonl \
  --dataset_dir data/moleculeace \
  --output_root results/final_model \
  --device cuda \
  --overwrite
```

To inspect all generated commands without starting training:

```bash
python final_model/run_multi_ablation.py final \
  --train_file data/train_regions.jsonl \
  --val_file data/val_regions.jsonl \
  --dataset_dir data/moleculeace \
  --output_root results/final_model \
  --device cuda \
  --dry_run
```

To run selected seeds or datasets, pass them explicitly:

```bash
python final_model/run_multi_ablation.py final \
  --seeds 0 1 2 3 4 \
  --datasets CHEMBL244_Ki CHEMBL204_Ki \
  --train_file data/train_regions.jsonl \
  --val_file data/val_regions.jsonl \
  --dataset_dir data/moleculeace \
  --output_root results/final_model \
  --device cuda
```

The runner stores checkpoints and per-dataset fine-tuning artifacts in a temporary
workspace. After each successful seed, it keeps the cleaned summary at:

```text
results/final_model/final/seed_<seed>/summary.csv
```

Temporary checkpoints, including `.pt` files, are removed after the corresponding seed
finishes successfully.

## Run Pretraining and Fine-Tuning Separately

Pretrain the shared model:

```bash
python final_model/train_gine_region_detector.py \
  --data_file data/train_regions.jsonl \
  --val_data_file data/val_regions.jsonl \
  --output_dir checkpoints/cliffrif_pretrain \
  --seed 0 \
  --device cuda
```

Fine-tune and evaluate one target:

```bash
python final_model/finetune_gine_region_detector.py \
  --checkpoint checkpoints/cliffrif_pretrain/best.pt \
  --dataset CHEMBL244_Ki \
  --data_file data/train_regions.jsonl \
  --val_data_file data/val_regions.jsonl \
  --dataset_dir data/moleculeace \
  --output_dir results/CHEMBL244_Ki_seed0 \
  --seed 0 \
  --device cuda
```

See [`final_model/cmd.md`](final_model/cmd.md) for commands with the complete set of
training hyperparameters.

## Results

On 30 MoleculeACE targets, the reported five-seed experiment achieved the best overall
test RMSE on 24 targets and the best cliff-subset RMSE on 18 targets, with average ranks
of 1.30 and 1.73, respectively. In the controlled ablation, the full model improved
cliff-subset RMSE over activity-only training on 21 of 30 targets, with a mean relative
improvement of 2.96%.

These results should be interpreted within the reported protocol: region annotations are
computational supervision proxies rather than experimentally validated causal sites, and
the multi-target pretraining setting contains cross-target molecular overlap.

## Method Summary

For final-layer atom states `H`, CliffRIF computes a soft region representation using
atom probabilities from the region head. The existence head receives the concatenated
region and global graph representations, while the property head uses only global
mean/max pooling. The training objective is:

```text
L = lambda_y * L_y + lambda_mask * L_mask + lambda_exist * L_exist
```

This separation lets the region and existence tasks regularize the shared encoder without
making property prediction depend on region labels or reference molecules at inference.

## Citation

If you use this repository, please cite the accompanying manuscript:

```bibtex
@article{jin2026cliffrif,
  title   = {CliffRIF: Region-Informed Auxiliary Learning for Activity-Cliff-Aware
             Molecular Property Prediction},
  author  = {Jin, Xiaobo and Nguyen, Bach Hoai and Nguyen, Binh P.},
  year    = {2026}
}
```
