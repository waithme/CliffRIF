# CliffRIF

Region-informed auxiliary learning for activity-cliff-aware molecular property
prediction. CliffRIF uses a three-layer GINE encoder with global **mean pooling**
for property regression. Region and existence tasks supply auxiliary supervision;
single-molecule inference does not require a reference pair or region annotation.

## Audited reproducibility package — 2026-10-26

**Start with [REPRODUCIBILITY.md](REPRODUCIBILITY.md).** Download
[cliffrif_reproducibility_20261026.zip](cliffrif_reproducibility_20261026.zip),
verify [SHA256SUMS](SHA256SUMS), and extract it. The package contains the canonical
model code, exact frozen annotations and splits, configuration, environment
specification, integrity checks, and seed-level results. It also contains code,
fixed inputs, and results for the supplementary experiments.

The root-level Python files and `data/` directory predate this audited package.
They are retained for history. Use the files **inside the package** for the
revised workflow; legacy defaults and generators are not the canonical entrypoint.
The package uses equal region weights and mean pooling, and preserves checkpoints
and predictions by default. No existing baseline code or results were changed.

## Experimental scope

The frozen development inputs contain 30,403 training and 3,416 validation rows
across 30 MoleculeACE targets. Training and validation canonical identities are
disjoint; official test rows are excluded from construction. The included split
manifest records every assignment. Official source CSVs are obtained separately
from the MoleculeACE commit given in the package README.

For the updated matched GINE3 Only comparison, historical Full has lower
five-seed mean overall RMSE on 18/30 targets and lower cliff RMSE on 18/30 targets.
Effects vary by dataset. Results do not require, or claim, improvement on every
dataset. The distributed tables retain each target and seed rather than only
reporting a cross-target average.

The supplementary threshold study contains seven settings × five model seeds ×
30 targets, evaluated only on validation data. Six earlier non-nominal seed-0
runs were reused and 29 seed runs were added. The original thresholds remain
unchanged. Other distributed supplementary results cover matched GINE3 Only,
fixed random regions, scratch property/full training, skipping unsupported
auxiliary labels, and same-model localization/Input×Gradient diagnostics.

Computational region labels are supervision proxies, not verified causal sites.
The historical Full score records are retained, but its complete original
per-run invocation and checkpoints were not recovered; this package documents
an audited workflow and does not claim to reconstruct every historical argument.
See the package README for experiment-specific limits, source provenance,
MoleculeACE attribution, and the distinction between validation and test results.

Code is distributed under the repository MIT license; MoleculeACE attribution
and its license are included in the package.
