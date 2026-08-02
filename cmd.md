# Single-Scale Region Ablation

## Experiment Definition

This experiment uses the latest V3 model as the baseline and replaces multi-layer
region modeling with final-layer, single-scale region modeling:

- The backbone remains a three-layer GINE;
- Only the third-layer GINE representation, `H3`, is connected to the atom-region mask head;
- Soft region pooling is applied to `H3` once, using only the third-layer mask probabilities;
- The region existence head uses `concat(graph_h, region_h3)`;
- The y head still uses only final-layer whole-molecule mean/max pooling;
- The mask, existence, and y losses and their weights remain unchanged;
- The handling and weighting of single-cut and MCS regions remain unchanged.

The computation is:

```text
p3          = sigmoid(mask_head(H3))
region_h3   = sum_i(p3_i H3_i) / sum_i(p3_i)
graph_h     = concat(mean_pool(H3), max_pool(H3))
exist_logit = existence_head(concat(graph_h, region_fusion(region_h3)))
y_pred      = y_head(graph_h)
```

The only difference from V3 is that V3 predicts regions at layers 1, 2, and 3 and
fuses the three scales, whereas this variant predicts and aggregates regions only
at layer 3. This experiment measures the contribution of multi-scale region
supervision and multi-layer region fusion.

Run all commands below from the project root directory.

## 1. Pretraining

```bash
python final_model/train_gine_region_detector.py \
  --data_file data/moleculeace_official_train_regions_v3.jsonl \
  --val_data_file data/moleculeace_official_val_regions_v3.jsonl \
  --output_dir ablation_checkpoints/single_scale_v3_pretrain \
  --hidden_dim 300 \
  --num_layers 3 \
  --dropout 0.1 \
  --epochs 100 \
  --batch_size 128 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --lambda_y 1.0 \
  --lambda_mask 0.5 \
  --lambda_exist 0.5 \
  --patience 15 \
  --single_cut_region_weight 1.0 \
  --mcs_region_weight 0.5 \
  --seed 0 \
  --device cuda
```

## 2. Fine-Tuning and Testing on One Dataset

```bash
python final_model/finetune_gine_region_detector.py \
  --checkpoint ablation_checkpoints/single_scale_v3_pretrain/best.pt \
  --dataset CHEMBL244_Ki \
  --data_file data/moleculeace_official_train_regions_v3.jsonl \
  --val_data_file data/moleculeace_official_val_regions_v3.jsonl \
  --dataset_dir data/moleculeace \
  --output_dir ablation_results/single_scale_v3_CHEMBL244_Ki \
  --epochs 100 \
  --batch_size 128 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --lambda_y 1.0 \
  --lambda_mask 0.5 \
  --lambda_exist 0.5 \
  --patience 15 \
  --single_cut_region_weight 1.0 \
  --mcs_region_weight 0.5 \
  --seed 0 \
  --device cuda
```

## 3. Fine-Tuning and Testing on All Datasets

```bash
python final_model/finetune_all_gine_region_detector.py \
  --checkpoint ablation_checkpoints/single_scale_v3_pretrain/best.pt \
  --data_file data/moleculeace_official_train_regions_v3.jsonl \
  --val_data_file data/moleculeace_official_val_regions_v3.jsonl \
  --dataset_dir data/moleculeace \
  --output_dir ablation_results/single_scale_v3_all \
  --epochs 100 \
  --batch_size 128 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --lambda_y 1.0 \
  --lambda_mask 0.5 \
  --lambda_exist 0.5 \
  --patience 15 \
  --single_cut_region_weight 1.0 \
  --mcs_region_weight 0.5 \
  --seed 0 \
  --device cuda \
  --overwrite
```

The summary results are saved to:

```text
ablation_results/single_scale_v3_all/summary_results.csv
```
