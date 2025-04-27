#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import Counter

# agentless’s helpers:
from agentless.util.model import make_model
from agentless.util.postprocess_data import normalize_patch
from agentless.util.utils import load_jsonl

# reuse the rerank’s globals & loaders:
execution_results = {}
def _load_results(patch_folders, num_samples, do_regression, do_repro):
    global execution_results
    roots = [Path(f) for f in patch_folders.split(",")]
    # first normalize all patches into output_i_normalized.jsonl
    for root in roots:
        for i in range(num_samples):
            proc = root / f"output_{i}_processed.jsonl"
            norm  = root / f"output_{i}_normalized.jsonl"
            if norm.exists(): continue
            patches = load_jsonl(proc)
            for d in patches:
                d["normalized_patch"] = normalize_patch(
                    d["instance_id"], d["model_patch"],
                    d.get("original_file_content", []),
                    d.get("new_file_content", []),
                    d.get("edited_files", []),
                )
            with open(norm, "w") as w:
                for d in patches:
                    w.write(json.dumps(d)+"\n")

    # then load them plus test results into execution_results
    for root in roots:
        for i in range(num_samples):
            patches = load_jsonl(root / f"output_{i}_normalized.jsonl")
            reg_rs = load_jsonl(root / f"output_{i}_regression_test_results.jsonl") if do_regression else []
            repro = load_jsonl(root / f"output_{i}_reproduction_test_results.jsonl") if do_repro else []

            for p in patches:
                iid = p["instance_id"]
                execution_results.setdefault(iid, []).append({
                    "normalized_patch":      p["normalized_patch"].strip(),
                    "patch":                 p["model_patch"],
                    "regression_test_result": (
                        len([r for r in reg_rs if r["instance_id"]==iid and r.get("regression")]))
                        if do_regression else 0,
                    "reproduction_test_result": (
                        [r for r in repro if r["instance_id"]==iid][0].get("reproduction", False))
                        if do_repro else True
                })

def top_k_patches_with_results(iid, k):
    entries = execution_results[iid]
    freq = Counter(e["normalized_patch"] for e in entries if e["normalized_patch"])
    def score(e):
        return (
            0 if e["reproduction_test_result"] else 1,
            e["regression_test_result"],
            -freq[e["normalized_patch"]]
        )
    return sorted(entries, key=score)[:k]

def refine_with_llm(iid, k):
    best = top_k_patches_with_results(iid, k)
    parts = ["We tried these candidate patches (with regression failures / repro result):\n"]
    for idx,e in enumerate(best,1):
        parts.append(f"--- Patch #{idx} ---\n{e['patch']}\n"
                     f"regression failures: {e['regression_test_result']}\n"
                     f"repro success?       : {e['reproduction_test_result']}\n")
    parts.append(
        "\nPlease propose 5 *new* patches that address these failures. "
        "For each patch, output a SEARCH/REPLACE diff in the Agentless format.\n"
    )
    prompt = "\n".join(parts)

    model = make_model(model="gpt-4o-2024-05-13", temperature=0.8, max_tokens=1024)
    resp  = model.codegen(prompt, num_samples=1)[0]["response"]
    return resp

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patch_folders", required=True,
                   help="Comma‑sep list of repair_sample_X folders")
    p.add_argument("--num_samples",  type=int, default=10)
    p.add_argument("--instance_id",  required=True)
    p.add_argument("--k",            type=int, default=5)
    p.add_argument("--regression",   action="store_true")
    p.add_argument("--reproduction", action="store_true")
    args = p.parse_args()

    # 1) load existing results
    _load_results(args.patch_folders, args.num_samples,
                  args.regression, args.reproduction)

    # 2) ask the LLM to refine
    refined = refine_with_llm(args.instance_id, args.k)
    print("=== LLM’s refined patches ===\n")
    print(refined)
