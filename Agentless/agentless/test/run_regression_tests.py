import argparse
import json
import os

import jsonlines
from datasets import load_dataset
from swebench.harness.constants import (
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    PASS_TO_PASS,
    TestStatus,
)
from swebench.harness.grading import get_eval_tests_report, get_logs_eval

from agentless.test.run_tests import run_tests


def rewrite_report(instance_id, input_folder_path, regression_tests):
    log_path = f"{input_folder_path}/test/{instance_id}/test_output.txt"
    with open(log_path, "r") as log_fp:
        eval_sm, found = get_logs_eval(log_path, log_fp)

    eval_ref = {
        KEY_INSTANCE_ID: instance_id,
        FAIL_TO_PASS: [],
        PASS_TO_PASS: regression_tests[instance_id],
    }

    report = get_eval_tests_report(eval_sm, eval_ref)

    return report["PASS_TO_PASS"]["failure"]


def save_passing_tests(output_jsonl_path, input_folder_path, dataset):
    ds = load_dataset(dataset)

    with jsonlines.open(output_jsonl_path, mode="w") as writer:
        for entry in ds["test"]:
            instance_id = entry["instance_id"]

            log_dir = os.path.join(input_folder_path, "test", instance_id)
            log_path = os.path.join(log_dir, "test_output.txt")

            # skip missing log files
            if not os.path.isfile(log_path):
                print(f"Skipping {instance_id}: no {log_path}")
                continue

            # parse the evaluation log
            with open(log_path, "r", encoding="utf-8") as log_fp:
                try:
                    eval_sm, found = get_logs_eval(log_path, log_fp)
                except Exception as e:
                    print(f"Error parsing logs for {instance_id}: {e}")
                    eval_sm, found = {}, False

            successful_test = []
            for test_name, status in eval_sm.items():
                if status == TestStatus.PASSED.value:
                    successful_test.append(test_name)

            if not successful_test:
                print(f"{instance_id} didn't get any passing tests")

            result_entry = {
                "instance_id": instance_id,
                "tests_passing_in_original_repo": successful_test,
            }
            writer.write(result_entry)


def run_regression_for_each_instance(args, lines, run_id):
    instance_ids = [line["instance_id"] for line in lines]
    patches = [line["model_patch"] for line in lines]

    instance_to_plausible = run_tests(
        instance_ids,
        patches,
        args.num_workers,
        run_id,
        args.regression_tests,
        args.instance_ids,
        args.timeout,
        apply_model_patch=True,
        dataset_name=args.dataset,
    )
    return instance_to_plausible


def check_if_all_instances_pass(instance_to_plausible):
    all_passed = True
    not_passing_instances = []
    for key, value in instance_to_plausible.items():
        if not value:
            all_passed = False
            not_passing_instances.append(key)

    if all_passed:
        print("All the chosen regression tests pass in the base repository")
    else:
        print(
            f"One or more of the regression tests for instances {not_passing_instances} do not pass in the original repository"
        )
        print(len(not_passing_instances))


def _run_regression(args):
    if args.predictions_path == "gold":
        ds = load_dataset(args.dataset)
        instance_ids = ds["test"]["instance_id"]
        patches = ds["test"]["patch"]

        instance_to_plausible = run_tests(
            instance_ids,
            patches,
            args.num_workers,
            args.run_id,
            args.regression_tests,
            args.instance_ids,
            args.timeout,
            apply_model_patch=True,
            dataset_name=args.dataset,
        )
        check_if_all_instances_pass(instance_to_plausible)
    elif args.predictions_path:
        assert args.predictions_path.endswith("_processed.jsonl")
        with open(args.predictions_path, "r") as file:
            data_lines = [json.loads(line) for line in file]

        if not args.load:
            run_regression_for_each_instance(args, data_lines, args.run_id)

        regression_dict = {}
        instance_test_dict = {}

        with open(args.regression_tests, "r") as file:
            for line in file:
                json_obj = json.loads(line.strip())
                instance_id = json_obj["instance_id"]
                test = json_obj["tests_passing_in_original_repo"]
                instance_test_dict[instance_id] = test

        for data in data_lines:
            instance_id = data["instance_id"]
            report_path = os.path.join("logs", "run_evaluation", args.run_id, "test", instance_id, "report.json")
            if os.path.isfile(report_path):
                regression_dict[instance_id] = rewrite_report(
                    instance_id,
                    os.path.join("logs", "run_evaluation", args.run_id),
                    instance_test_dict,
                )
            else:
                regression_dict[instance_id] = []

        updated_data_lines = []
        for data in data_lines:
            instance_id = data["instance_id"]
            if instance_id in regression_dict:
                data["regression"] = regression_dict[instance_id]
            updated_data_lines.append(data)

        out_path = args.predictions_path.replace("processed.jsonl", "regression_test_results.jsonl")
        with open(out_path, "w") as file:
            for data in updated_data_lines:
                file.write(json.dumps(data) + "\n")

    else:
        ds = load_dataset(args.dataset)
        instance_ids = (
            ds["test"]["instance_id"]
            if args.instance_ids is None
            else args.instance_ids
        )

        patches = [
            {"instance_id": instance_id, "patch": "", "normalized_patch": ""}
            for instance_id in instance_ids
        ]

        instance_to_plausible = run_tests(
            instance_ids,
            patches,
            args.num_workers,
            args.run_id,
            args.regression_tests,
            args.instance_ids,
            args.timeout,
            apply_model_patch=False,
            dataset_name=args.dataset,
        )

        if args.regression_tests:
            check_if_all_instances_pass(instance_to_plausible)
        else:
            save_passing_tests(
                args.output_file,
                os.path.join("logs", "run_evaluation", args.run_id),
                args.dataset,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument(
        "--predictions_path",
        type=str,
        help="Patch file with normalized patches",
    )
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--regression_tests", type=str)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument(
        "--timeout", type=int, default=1200, help="Timeout for running tests in seconds"
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated), if not provided, all instances will be run",
    )
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        choices=["princeton-nlp/SWE-bench_Lite", "princeton-nlp/SWE-bench_Verified"],
    )

    args = parser.parse_args()

    assert not (
        args.predictions_path and args.output_file
    ), "An output file is only required when selecting regression tests"

    _run_regression(args)


if __name__ == "__main__":
    main()
