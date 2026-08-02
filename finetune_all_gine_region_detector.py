#!/usr/bin/env python3
"""Run the single-dataset finetuning script for every CSV in a folder.

Example:
  python 2207version/finetune_all_gine_region_detector.py \
    --checkpoint runs/gine_region_detector/best.pt \
    --data_file data/office/moleculeace_official_train_regions.jsonl \
    --dataset_dir data/dataset \
    --output_dir runs/gine_region_finetune_all \
    --reset_y_head --device cuda

Unknown arguments are passed directly to the single-dataset script, so loss
switches and hyperparameters can be supplied without duplicating them here.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().with_name("finetune_gine_region_detector.py")


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--val_data_file", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Optional dataset stems; default: every matching CSV")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    return parser.parse_known_args()


def csv_safe(metrics):
    return {
        key: json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, (dict, list)) else value
        for key, value in metrics.items()
    }


def write_summary(path, rows):
    preferred = ["dataset", "status", "returncode", "best_epoch", "best_val_rmse",
                 "test_rmse", "test_mae", "test_cliff_rmse", "n_train", "n_val",
                 "n_official_test", "output_dir", "log_file", "error"]
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    fields = [key for key in preferred if key in keys]
    fields += sorted(keys.difference(fields))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args, forwarded = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary_results.csv"
    csv_paths = sorted(dataset_dir.glob(args.pattern))

    if args.datasets:
        requested = set(args.datasets)
        csv_paths = [path for path in csv_paths if path.stem in requested]
        missing = sorted(requested.difference(path.stem for path in csv_paths))
        if missing:
            raise FileNotFoundError(f"Requested datasets not found: {missing}")
    if not csv_paths:
        raise FileNotFoundError(f"No files matched {args.pattern!r} in {dataset_dir}")

    rows = []
    for number, csv_path in enumerate(csv_paths, 1):
        dataset = csv_path.stem
        run_dir = output_root / dataset
        metrics_path = run_dir / "final_metrics.json"
        log_path = run_dir / "run.log"
        run_dir.mkdir(parents=True, exist_ok=True)

        if metrics_path.is_file() and not args.overwrite:
            metrics = csv_safe(json.loads(metrics_path.read_text(encoding="utf-8")))
            rows.append({"status": "skipped_existing", "returncode": 0,
                         "output_dir": str(run_dir), "log_file": str(log_path), **metrics})
            write_summary(summary_path, rows)
            print(f"[{number}/{len(csv_paths)}] {dataset}: skipped (result exists)", flush=True)
            continue

        command = [args.python, str(SCRIPT_PATH),
                   "--checkpoint", str(Path(args.checkpoint).resolve()),
                   "--data_file", str(Path(args.data_file).resolve()),
                   "--val_data_file", str(Path(args.val_data_file).resolve()),
                   "--dataset_dir", str(dataset_dir), "--csv", str(csv_path.resolve()),
                   "--dataset", dataset, "--output_dir", str(run_dir), *forwarded]
        print(f"[{number}/{len(csv_paths)}] {dataset}: running", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write("COMMAND: " + " ".join(command) + "\n\n")
            log.flush()
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                print(f"[{dataset}] {line}", end="", flush=True)
                log.write(line)
            returncode = process.wait()

        if returncode == 0 and metrics_path.is_file():
            metrics = csv_safe(json.loads(metrics_path.read_text(encoding="utf-8")))
            row = {"status": "success", "returncode": 0,
                   "output_dir": str(run_dir), "log_file": str(log_path), **metrics}
        else:
            row = {"dataset": dataset, "status": "failed", "returncode": returncode,
                   "output_dir": str(run_dir), "log_file": str(log_path),
                   "error": "See run.log; final_metrics.json was not produced."}
        rows.append(row)
        write_summary(summary_path, rows)
        print(f"[{number}/{len(csv_paths)}] {dataset}: {row['status']}", flush=True)
        if returncode != 0 and args.stop_on_error:
            break

    failures = sum(row["status"] == "failed" for row in rows)
    print(json.dumps({"datasets": len(rows), "failed": failures,
                      "summary": str(summary_path)}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
