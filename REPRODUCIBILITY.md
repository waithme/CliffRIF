# CliffRIF reproducibility snapshot (2026-10-26)

Download `cliffrif_reproducibility_20261026.zip` from this repository revision.
The archive expands to `cliffrif_reproducibility/` and includes all canonical
and supplementary files with their original directory hierarchy. The source
archive supplied automatically by GitHub also includes this file.

```bash
shasum -a 256 -c SHA256SUMS
unzip cliffrif_reproducibility_20261026.zip
cd cliffrif_reproducibility
python -B tools/verify_release.py
```

The verification uses the Python standard library. It checks every distributed
file hash, canonical train/validation identity separation, fixed threshold input
rows and coverage, and completeness/finiteness of the published result tables.
Environment installation and actual training are separate; no training is started
by verification or supplementary launcher dry-runs.

## Contents

- `model/`: canonical equal-region, mean-pooling model and training files; a
  launcher that keeps checkpoints, predictions and command/environment records.
- `data/`: exact original v2 JSONL inputs, row-level split manifest, generator,
  coverage/statistics and label/reference audits.
- `configuration.json`, `requirements-core.txt`: fixed scientific configuration
  and recorded core dependencies. Follow the README for RDKit/CUDA installation.
- `original_source_manifest.json`: original source hashes, including the archived
  historical launcher. The portable launcher has its own hash in the release manifest.
- `results/`: five per-target seed tables each for historical Full, matched
  GINE3 Only and the random-region control.
- `supplementary/`: unchanged scientific core snapshots for the matched controls,
  portable launcher, exact fixed random/threshold annotations, scratch and
  unsupported-label scores, all 35 threshold seed tables (1,050 validation runs),
  per-target comparisons, and localization diagnostic code/results for the two
  models included in SI.
- `release_manifest.json`: SHA-256 and byte size of every packaged file except
  the manifest itself. An identical copy is provided beside this document as
  `REPRODUCIBILITY_MANIFEST.json` for inspection without extracting the archive.
- `LICENSE`, `MOLECULEACE_LICENSE`: code/data attribution notices.

## Reproduce

Read the extracted root README for the canonical five-seed workflow and
`supplementary/README.md` for each supplementary experiment. Obtain the official
CSV source at MoleculeACE commit
`7e6de0bd2968c56589c580f2a397f01c531ede26`:
https://github.com/molML/MoleculeACE/tree/7e6de0bd2968c56589c580f2a397f01c531ede26

Use explicit input and **new output** paths. The supplementary runner defaults
to dry-run and needs `--execute` to train. No external baseline is run. Threshold
modes force validation-only execution; original thresholds are retained.

## Scope and provenance

The fixed development partition and each annotation setting are shared by the
five model seeds; seed SD does not measure split or annotation-generation
variability. Frozen JSONL hashes, not a new timeout-limited MCS reconstruction,
define the distributed inputs. The nominal reference of the threshold study is
an independent validation-only follow-up, not a replacement historical Full.

Original Full checkpoints/per-molecule predictions, external baseline code and
results, and new-run checkpoints are not distributed in this snapshot. The
included historical Full scores and its candidate source do not recover the
entire old execution chain. The new workflows retain outputs when run afresh.
The localization comparison is an oracle-size validation diagnostic against
computed references, with the same model for mask and attribution; it is not
an independent chemical validation or the original Full checkpoint's result.

The package does not include manuscript sources, correspondence, credentials,
or personal/server paths. Existing repository code/data remain intact.
