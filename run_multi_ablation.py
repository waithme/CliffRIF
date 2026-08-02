#!/usr/bin/env python3
"""Run the final model over multiple random seeds.

For every requested seed, all pretraining checkpoints and per-dataset
fine-tuning artifacts are written to a temporary directory. After a successful
seed run, only a cleaned ``summary.csv`` is copied to:

    multi_seed_results/final/seed_<seed>/summary.csv

The temporary directory, including every generated ``.pt`` file, is then
removed. By default the script runs seeds 0, 1, 2, 3, and 4.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCRIPT_PATH = Path(__file__).resolve()
ABLATION_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = ABLATION_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Default pre-generated train/validation data used by both pretraining and
# downstream fine-tuning. Command-line --train_file/--val_file override these.
FINETUNE_TRAIN_FILE = (
    WORKSPACE_ROOT / "data" / "finetune_dataset" / "train_regions.jsonl"
)
FINETUNE_VAL_FILE = (
    WORKSPACE_ROOT / "data" / "finetune_dataset" / "val_regions.jsonl"
)
MOLECULEACE_TEST_DIR = WORKSPACE_ROOT / "data" / "moleculeace"


@dataclass(frozen=True)
class Variant:
    directory: str
    pretrain: bool
    finetune: bool
    existence_loss: bool
    mask_loss: bool = True
    existence_pos_weight: bool = False


VARIANTS = {"final": Variant(".", True, True, True)}
CANONICAL_VARIANTS = ("final",)
OUTPUT_NAMES = {".": "final"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "variant",
        nargs="?",
        default="final",
        choices=sorted(VARIANTS),
        help=(
            "Model variant. Available names: "
            + ", ".join(CANONICAL_VARIANTS)
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds to run sequentially.",
    )
    parser.add_argument(
        "--train_file",
        type=Path,
        default=FINETUNE_TRAIN_FILE,
        help="MoleculeACE subtrain region JSONL used for downstream fine-tuning.",
    )
    parser.add_argument(
        "--val_file",
        type=Path,
        default=FINETUNE_VAL_FILE,
        help="MoleculeACE validation region JSONL used for model selection.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=MOLECULEACE_TEST_DIR,
        help="Directory containing the official MoleculeACE CSV files.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=ABLATION_DIR / "multi_seed_results",
        help="Permanent output root. Only final summary files are stored here.",
    )
    parser.add_argument(
        "--tmp_dir",
        type=Path,
        default=None,
        help=(
            "Optional parent directory for temporary checkpoints and "
            "fine-tuning artifacts. The system temporary directory is used "
            "when omitted."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset stems; default runs every CSV in dataset_dir.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=300)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_y", type=float, default=1.0)
    parser.add_argument("--lambda_mask", type=float, default=0.5)
    parser.add_argument("--lambda_exist", type=float, default=0.5)
    parser.add_argument("--existence_pos_weight_min", type=float, default=0.5)
    parser.add_argument("--existence_pos_weight_max", type=float, default=10.0)
    parser.add_argument("--single_cut_region_weight", type=float, default=1.0)
    parser.add_argument("--mcs_region_weight", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun a seed even when its final summary.csv already exists.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue with later seeds if one seed fails.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands and paths without training or creating outputs.",
    )
    return parser.parse_args()


def option(name: str, value: object) -> list[str]:
    return [f"--{name}", str(value)]


def common_loss_options(args: argparse.Namespace, variant: Variant) -> list[str]:
    result = [*option("lambda_y", args.lambda_y)]
    if variant.mask_loss:
        result.extend(option("lambda_mask", args.lambda_mask))
    if variant.existence_loss:
        result.extend(option("lambda_exist", args.lambda_exist))
    if variant.existence_pos_weight:
        result.extend(option(
            "existence_pos_weight_min", args.existence_pos_weight_min
        ))
        result.extend(option(
            "existence_pos_weight_max", args.existence_pos_weight_max
        ))
    if variant.mask_loss:
        result.extend([
            *option(
                "single_cut_region_weight", args.single_cut_region_weight
            ),
            *option("mcs_region_weight", args.mcs_region_weight),
        ])
    return result


def train_command(
    args: argparse.Namespace,
    variant: Variant,
    seed: int,
    checkpoint_dir: Path,
    pretrain_train_file: Path,
    pretrain_val_file: Path,
) -> list[str]:
    train_script = ABLATION_DIR / variant.directory / (
        "train_gine_region_detector.py"
    )
    return [
        args.python,
        str(train_script),
        *option("data_file", pretrain_train_file.resolve()),
        *option("val_data_file", pretrain_val_file.resolve()),
        *option("output_dir", checkpoint_dir),
        *option("hidden_dim", args.hidden_dim),
        *option("num_layers", args.num_layers),
        *option("dropout", args.dropout),
        *option("epochs", args.epochs),
        *option("batch_size", args.batch_size),
        *option("lr", args.pretrain_lr),
        *option("weight_decay", args.weight_decay),
        *common_loss_options(args, variant),
        *option("grad_clip", args.grad_clip),
        *option("patience", args.patience),
        *option("seed", seed),
        *option("num_workers", args.num_workers),
        *option("device", args.device),
    ]


def batch_command(
    args: argparse.Namespace,
    variant: Variant,
    seed: int,
    checkpoint: Path | None,
    finetune_dir: Path,
) -> list[str]:
    batch_script = ABLATION_DIR / variant.directory / (
        "finetune_all_gine_region_detector.py"
    )
    command = [
        args.python,
        str(batch_script),
    ]
    if checkpoint is not None:
        command.extend(option("checkpoint", checkpoint))
    if variant.finetune:
        command.extend(option("data_file", args.train_file.resolve()))
        command.extend(option("val_data_file", args.val_file.resolve()))
    command.extend(
        [
            *option("dataset_dir", args.dataset_dir.resolve()),
            *option("output_dir", finetune_dir),
        ]
    )
    if args.datasets:
        command.extend(["--datasets", *args.datasets])

    if variant.directory == "without_finetune":
        command.extend(
            [
                *option("batch_size", args.batch_size),
                *option("num_workers", args.num_workers),
                *option("seed", seed),
                *option("device", args.device),
                "--overwrite",
                "--stop_on_error",
            ]
        )
        return command

    if variant.directory == "no_pretrain_ablation":
        command.extend(
            [
                *option("hidden_dim", args.hidden_dim),
                *option("num_layers", args.num_layers),
                *option("dropout", args.dropout),
            ]
        )

    command.extend(
        [
            *option("epochs", args.epochs),
            *option("batch_size", args.batch_size),
            *option("lr", args.finetune_lr),
            *option("weight_decay", args.weight_decay),
            *common_loss_options(args, variant),
            *option("grad_clip", args.grad_clip),
            *option("patience", args.patience),
            *option("seed", seed),
            *option("num_workers", args.num_workers),
            *option("device", args.device),
            "--overwrite",
            "--stop_on_error",
        ]
    )
    return command


def print_command(stage: str, command: Sequence[str]) -> None:
    print(f"\n[{stage}]\n{shlex.join(command)}", flush=True)


def run_command(stage: str, command: Sequence[str]) -> None:
    print_command(stage, command)
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{stage} failed with return code {completed.returncode}"
        )


def validate_summary(
    source: Path,
    dataset_dir: Path,
    requested_datasets: Sequence[str] | None,
) -> tuple[list[str], list[dict[str, str]]]:
    if not source.is_file():
        raise FileNotFoundError(f"Summary was not produced: {source}")
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    if not rows:
        raise ValueError(f"Summary contains no dataset rows: {source}")

    expected = (
        set(requested_datasets)
        if requested_datasets
        else {path.stem for path in dataset_dir.glob("*.csv")}
    )
    observed = {row.get("dataset", "") for row in rows}
    if observed != expected:
        raise ValueError(
            "Summary dataset mismatch: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    failed = [
        row.get("dataset", "<unknown>")
        for row in rows
        if row.get("status") not in {"success", "skipped_existing"}
    ]
    if failed:
        raise RuntimeError(f"Failed datasets in summary: {failed}")
    return fields, rows


def save_clean_summary(
    source: Path,
    destination: Path,
    dataset_dir: Path,
    requested_datasets: Sequence[str] | None,
) -> None:
    fields, rows = validate_summary(
        source, dataset_dir, requested_datasets
    )
    # Temporary paths become invalid after cleanup and are deliberately
    # excluded from the permanent summary.
    remove_fields = {
        "output_dir",
        "log_file",
        "checkpoint",
    }
    output_fields = [field for field in fields if field not in remove_fields]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = destination.with_suffix(".csv.tmp")
    with temporary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in output_fields})
    temporary_output.replace(destination)


def validate_inputs(args: argparse.Namespace, variant: Variant) -> None:
    variant_dir = (ABLATION_DIR / variant.directory).resolve()
    required_scripts = [
        variant_dir / "finetune_all_gine_region_detector.py"
    ]
    if variant.pretrain:
        required_scripts.append(
            variant_dir / "train_gine_region_detector.py"
        )
    missing_scripts = [
        str(path) for path in required_scripts if not path.is_file()
    ]
    if missing_scripts:
        raise FileNotFoundError(
            f"Missing variant scripts: {missing_scripts}"
        )
    if args.dry_run:
        return
    required_data_files = [
        args.train_file,
        args.val_file,
    ]
    missing_data_files = [
        str(path) for path in required_data_files if not path.is_file()
    ]
    if missing_data_files:
        raise FileNotFoundError(
            f"Required fixed data files not found: {missing_data_files}"
        )
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(
            f"MoleculeACE test directory not found: {args.dataset_dir}"
        )
    if not list(args.dataset_dir.glob("*.csv")):
        raise FileNotFoundError(
            f"No CSV datasets found in: {args.dataset_dir}"
        )
    if args.tmp_dir is not None and not args.dry_run:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)


def dry_run(args: argparse.Namespace, variant: Variant) -> None:
    output_name = OUTPUT_NAMES[variant.directory]
    for seed in args.seeds:
        work_dir = Path("/tmp") / f"multi_ablation_{output_name}_seed{seed}_XXXX"
        checkpoint_dir = work_dir / "pretrain"
        pretrain_train_file = args.train_file.resolve()
        pretrain_val_file = args.val_file.resolve()
        finetune_dir = work_dir / "downstream"
        checkpoint = checkpoint_dir / "best.pt" if variant.pretrain else None
        final_summary = (
            args.output_root.resolve()
            / output_name
            / f"seed_{seed}"
            / "summary.csv"
        )
        print(f"\n=== seed {seed} ===")
        print(f"temporary workspace: {work_dir}")
        if variant.pretrain:
            print_command(
                "pretrain",
                train_command(
                    args,
                    variant,
                    seed,
                    checkpoint_dir,
                    pretrain_train_file,
                    pretrain_val_file,
                ),
            )
        print_command(
            "downstream",
            batch_command(
                args, variant, seed, checkpoint, finetune_dir
            ),
        )
        print(f"final summary: {final_summary}")
        print("temporary workspace: deleted after the seed")


def run_seed(
    args: argparse.Namespace,
    variant: Variant,
    output_name: str,
    seed: int,
) -> Path:
    final_summary = (
        args.output_root.resolve()
        / output_name
        / f"seed_{seed}"
        / "summary.csv"
    )
    if final_summary.exists() and not args.overwrite:
        print(
            f"[seed {seed}] skipped: {final_summary} already exists "
            "(use --overwrite to rerun)",
            flush=True,
        )
        return final_summary

    temp_parent = args.tmp_dir.resolve() if args.tmp_dir else None
    work_dir = Path(
        tempfile.mkdtemp(
            prefix=f"multi_ablation_{output_name}_seed{seed}_",
            dir=temp_parent,
        )
    ).resolve()
    checkpoint_dir = work_dir / "pretrain"
    pretrain_train_file = args.train_file.resolve()
    pretrain_val_file = args.val_file.resolve()
    finetune_dir = work_dir / "downstream"
    checkpoint = checkpoint_dir / "best.pt" if variant.pretrain else None
    try:
        print(
            f"\n=== {output_name}: seed {seed} ===\n"
            f"temporary workspace: {work_dir}",
            flush=True,
        )
        if variant.pretrain:
            print(
                "using the supplied pre-generated train/validation files "
                "for pretraining and downstream fine-tuning",
                flush=True,
            )
            run_command(
                f"seed {seed} pretrain",
                train_command(
                    args,
                    variant,
                    seed,
                    checkpoint_dir,
                    pretrain_train_file,
                    pretrain_val_file,
                ),
            )
            if checkpoint is None or not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Pretraining best.pt was not produced: {checkpoint}"
                )

        run_command(
            f"seed {seed} downstream",
            batch_command(
                args, variant, seed, checkpoint, finetune_dir
            ),
        )
        source_summary = finetune_dir / "summary_results.csv"
        save_clean_summary(
            source_summary,
            final_summary,
            args.dataset_dir.resolve(),
            args.datasets,
        )
        print(
            f"[seed {seed}] saved final result: {final_summary}",
            flush=True,
        )
        return final_summary
    finally:
        # The path is created by tempfile.mkdtemp with this exact prefix.
        # Refuse cleanup if that invariant is ever violated.
        expected_prefix = f"multi_ablation_{output_name}_seed{seed}_"
        if work_dir.name.startswith(expected_prefix) and work_dir.is_dir():
            shutil.rmtree(work_dir)
            print(
                f"[seed {seed}] removed temporary workspace and all .pt files: "
                f"{work_dir}",
                flush=True,
            )
        elif work_dir.exists():
            raise RuntimeError(
                f"Refusing to remove unexpected temporary path: {work_dir}"
            )


def main() -> None:
    args = parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError(f"Duplicate seeds are not allowed: {args.seeds}")
    variant = VARIANTS[args.variant]
    validate_inputs(args, variant)

    if args.dry_run:
        dry_run(args, variant)
        return

    output_name = OUTPUT_NAMES[variant.directory]
    failures: list[tuple[int, str]] = []
    completed: list[Path] = []
    for seed in args.seeds:
        try:
            completed.append(
                run_seed(args, variant, output_name, seed)
            )
        except Exception as exc:
            failures.append((seed, str(exc)))
            print(f"[seed {seed}] FAILED: {exc}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                break

    print("\n=== run summary ===")
    print(f"variant: {output_name}")
    print(f"completed seeds: {len(completed)}/{len(args.seeds)}")
    for path in completed:
        print(f"  {path}")
    if failures:
        for seed, message in failures:
            print(f"  failed seed {seed}: {message}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
