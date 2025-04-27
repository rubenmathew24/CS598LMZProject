#!/usr/bin/env python3
import argparse
import subprocess
import json
import sys
from pathlib import Path

def run_rerank(dataset, patch_folders, total_samples, results_folder):
    cmd = [
        sys.executable, "agentless/repair/rerank.py",
        "--dataset", dataset,
        "--patch_folder", patch_folders,
        "--num_samples", str(total_samples),
        "--deduplicate",
        "--regression",
        "--reproduction",
        "--output_file", str(results_folder / "all_preds.jsonl")
    ]
    print("Running rerank:", " ".join(cmd))
    subprocess.check_call(cmd)

def load_top_patch(results_folder, instance_id):
    out = results_folder / "all_preds.jsonl"
    with open(out) as f:
        for line in f:
            data = json.loads(line)
            if data["instance_id"] == instance_id:
                return data["model_patch"]
    raise RuntimeError(f"No patch for {instance_id} in {out}")

def apply_patch(repo_dir, patch_text):
    print("Applying patch to", repo_dir)
    p = subprocess.Popen(["patch", "-p1"], cwd=repo_dir, stdin=subprocess.PIPE)
    stdout, stderr = p.communicate(patch_text.encode())
    if p.returncode != 0:
        print(stdout.decode(), stderr.decode(), file=sys.stderr)
        raise RuntimeError("Patch command failed")

def run_regression_tests(dataset, tests_file, predictions_file, run_id):
    cmd = [
        sys.executable, "agentless/test/run_regression_tests.py",
        "--dataset", dataset,
        "--regression_tests", tests_file,
        "--predictions_path", predictions_file,
        "--run_id", run_id,
        "--num_workers", "4"
    ]
    print("Running regression tests:", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",       required=True)
    parser.add_argument("--results_folder",required=True)
    parser.add_argument("--repo_dir",      required=True)
    parser.add_argument("--patch_folders", required=True,
                        help="comma‑sep list of repair_sample_X folders")
    parser.add_argument("--total_samples", type=int, default=40,
                        help="total patches (= #runs × #samples per run)")
    parser.add_argument("--instance_id",   required=True)
    parser.add_argument("--tests_file",    default="results/swe-bench-lite/select_regression/output.jsonl",
                        help="final regression tests JSONL")
    args = parser.parse_args()

    results_folder = Path(args.results_folder)
    repo_dir       = Path(args.repo_dir)

    # 1) pick the top patch
    run_rerank(args.dataset, args.patch_folders, args.total_samples, results_folder)

    # 2) load & apply it
    top_patch = load_top_patch(results_folder, args.instance_id)
    apply_patch(repo_dir, top_patch)

    # 3) run regression tests on that single-patch repo
    try:
        run_regression_tests(args.dataset,
                             args.tests_file,
                             str(results_folder / "all_preds.jsonl"),
                             "check_top")
        print("✅ Top‐voted patch passed all regression tests; exiting.")
        sys.exit(0)
    except subprocess.CalledProcessError:
        print("❌ Top‐voted patch failed tests; falling back to iterative repair.")

    cmd = [
        sys.executable, "iterative_repair.py",
        "--patch_folders", args.patch_folders,
        "--num_samples", str(args.total_samples // len(args.patch_folders.split(","))),
        "--instance_id", args.instance_id,
        "--k", "5",
        "--regression",
        "--reproduction"
    ]
    print("Invoking iterative repair:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("✨ Iterative repair finished; inspect its output above.")

if __name__=="__main__":
    main()
