"""Evaluate SWE-bench patches from experiment JSONL files.

Converts our JSONL format to SWE-bench predictions format,
then runs the SWE-bench evaluation harness (requires Docker).

Usage:
    python eval_swebench.py \
        --results-dir ../results/swebench_v1 \
        --output-dir ../results/swebench_v1/eval \
        --max-workers 4

The script:
1. Reads all JSONL files in results-dir
2. Extracts patches (stored in 'answer' field)
3. Writes predictions.json in SWE-bench format
4. Runs swebench evaluation via Docker
5. Prints pass/fail results and saves summary
"""

import argparse
import json
import os
import sys
from pathlib import Path


def extract_predictions(results_dir):
    """Extract predictions from experiment JSONL files.

    Returns dict: {topology_key: {instance_id: prediction_dict}}
    """
    results_dir = Path(results_dir)
    all_preds = {}

    for fname in sorted(results_dir.glob("*.jsonl")):
        topo_key = fname.stem  # e.g. Qwen_Qwen3.5-9B_sas_k50
        preds = {}

        with open(fname) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "answer" not in r:
                    continue

                instance_id = r.get("task_id", r.get("instance_id", ""))
                patch = r.get("answer", "")

                if not instance_id or not patch:
                    continue

                # Only include records with actual diffs
                if "diff --git" not in patch:
                    continue

                preds[instance_id] = {
                    "instance_id": instance_id,
                    "model_name_or_path": r.get("model", "unknown"),
                    "model_patch": patch,
                }

        if preds:
            all_preds[topo_key] = preds
            print(f"  {fname.name}: {len(preds)} predictions with patches")
        else:
            print(f"  {fname.name}: no valid patches")

    return all_preds


def write_predictions(preds, output_path):
    """Write predictions in SWE-bench format (list of dicts)."""
    pred_list = list(preds.values())
    with open(output_path, "w") as f:
        json.dump(pred_list, f, indent=2)
    return len(pred_list)


def run_evaluation(predictions_path, output_dir, max_workers=4, dataset="princeton-nlp/SWE-bench_Lite"):
    """Run SWE-bench evaluation harness."""
    from swebench.harness.run_evaluation import main as run_eval

    run_id = Path(predictions_path).stem
    args_list = [
        "--predictions_path", str(predictions_path),
        "--swe_bench_tasks", dataset,
        "--log_dir", str(Path(output_dir) / "logs"),
        "--testbed", str(Path(output_dir) / "testbed"),
        "--log_suffix", run_id,
        "--verbose",
    ]
    if max_workers:
        args_list += ["--num_processes", str(max_workers)]

    print(f"\nRunning SWE-bench evaluation: {run_id}")
    print(f"  Predictions: {predictions_path}")
    print(f"  Output: {output_dir}")
    print(f"  Workers: {max_workers}")

    try:
        run_eval(args_list)
    except SystemExit:
        pass


def parse_results(output_dir, run_id):
    """Parse SWE-bench evaluation results."""
    log_dir = Path(output_dir) / "logs"

    results = {"pass": [], "fail": [], "error": []}
    if not log_dir.exists():
        print(f"No logs found at {log_dir}")
        return results

    for instance_dir in log_dir.iterdir():
        if not instance_dir.is_dir():
            continue
        instance_id = instance_dir.name

        report_file = instance_dir / f"report_{run_id}.json"
        if report_file.exists():
            with open(report_file) as f:
                report = json.load(f)
            resolved = report.get(instance_id, {}).get("resolved", False)
            if resolved:
                results["pass"].append(instance_id)
            else:
                results["fail"].append(instance_id)
        else:
            results["error"].append(instance_id)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SWE-bench patches")
    parser.add_argument("--results-dir", required=True,
                        help="Directory with experiment JSONL files")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for evaluation output (default: results-dir/eval)")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--skip-eval", action="store_true",
                        help="Only extract predictions, don't run Docker evaluation")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Extracting predictions ===")
    all_preds = extract_predictions(results_dir)

    if not all_preds:
        print("No predictions found!")
        sys.exit(1)

    # Write predictions per topology and run evaluation
    summary = {}
    for topo_key, preds in all_preds.items():
        pred_file = output_dir / f"predictions_{topo_key}.json"
        n = write_predictions(preds, pred_file)
        print(f"\nWrote {n} predictions to {pred_file}")

        if args.skip_eval:
            summary[topo_key] = {"n_predictions": n, "status": "skipped"}
            continue

        # Run evaluation
        topo_output = output_dir / topo_key
        topo_output.mkdir(exist_ok=True)
        run_evaluation(str(pred_file), str(topo_output),
                       max_workers=args.max_workers, dataset=args.dataset)

        # Parse results
        results = parse_results(str(topo_output), f"predictions_{topo_key}")
        n_pass = len(results["pass"])
        n_fail = len(results["fail"])
        n_error = len(results["error"])
        n_total = n_pass + n_fail + n_error

        summary[topo_key] = {
            "n_predictions": n,
            "n_evaluated": n_total,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_error": n_error,
            "pass_rate": n_pass / n_total if n_total > 0 else 0,
            "passed": results["pass"],
            "failed": results["fail"],
            "errored": results["error"],
        }

        print(f"\n  {topo_key}: {n_pass}/{n_total} passed ({n_pass/n_total:.1%})"
              if n_total > 0 else f"\n  {topo_key}: no results")

    # Save summary
    summary_file = output_dir / "eval_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Summary saved to {summary_file} ===")

    # Print final table
    print(f"\n{'Topology':<50} {'Preds':>5} {'Pass':>5} {'Fail':>5} {'Err':>5} {'Rate':>7}")
    print("-" * 80)
    for topo_key, s in sorted(summary.items()):
        if "n_pass" in s:
            print(f"{topo_key:<50} {s['n_predictions']:>5} {s['n_pass']:>5} "
                  f"{s['n_fail']:>5} {s['n_error']:>5} {s['pass_rate']:>6.1%}")
        else:
            print(f"{topo_key:<50} {s['n_predictions']:>5}  (eval {s['status']})")


if __name__ == "__main__":
    main()
