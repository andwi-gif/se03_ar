# =============================================================================
# analysis/compute_metrics.py
#
# Reads execution and repair results and computes all benchmark metrics:
#   - Pass@1 and Pass@3 with Wilson 95% confidence intervals
#   - Version Fidelity Rate    (diagonal of drift matrix)
#   - Forward Breaking Rate    (off-diagonal)
#   - Repair Success Rate      (fraction of broken snippets fixed by repair)
#   - Error Category Breakdown (what types of errors cause failures)
#   - Drift Matrix             (full 3x3 table per model)
#
# Pass@k definitions (following Chen et al. 2021):
#   Pass@1 = mean pass rate across individual samples
#             (expected probability that a single attempt passes)
#   Pass@3 = fraction of problems where at least 1 of 3 samples passes
#
# Confidence intervals use the Wilson score interval, which is well-behaved
# at extreme proportions (near 0 or 1) unlike the normal approximation.
#
# Usage:
#   python analysis/compute_metrics.py
#   python analysis/compute_metrics.py --model kimi_k2
#
# Output:
#   data/analysis/metrics_summary.json   -- full metrics dict
#   Printed drift matrices and summary tables to stdout
# =============================================================================

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ALL_MODELS, QISKIT_VERSIONS, EXECUTION_RESULTS_DIR, REPAIRS_DIR, ANALYSIS_DIR, K_SAMPLES

VERSION_ORDER = list(QISKIT_VERSIONS.keys())   # ["v0", "v1", "v2"]

# Z-score for 95% confidence interval
Z_95 = 1.96


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_results(model_label: str) -> tuple:
    exec_path   = EXECUTION_RESULTS_DIR / f"{model_label}_execution.json"
    repair_path = REPAIRS_DIR / f"{model_label}_repair.json"

    if not exec_path.exists():
        return None, None

    with open(exec_path) as f:
        execution = json.load(f)

    repair = []
    if repair_path.exists():
        with open(repair_path) as f:
            repair = json.load(f)

    return execution, repair


def is_pass(result: dict) -> bool:
    return result.get("status") == "pass"


def is_fail(result: dict) -> bool:
    return result.get("status") in ("hard_fail", "soft_fail")


def wilson_ci(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.
    Returns (lower, upper) bounds.
    Handles total = 0 gracefully by returning (0, 0).

    Preferred over normal approximation because it is well-behaved
    at p near 0 or 1, which is common in code generation benchmarks.
    Reference: Wilson (1927), also used in HumanEval follow-up analyses.
    """
    if total == 0:
        return (0.0, 0.0)
    p    = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = (z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def pass_at_1(sample_results: list[bool]) -> float:
    """
    Pass@1 = mean pass rate across individual samples.
    Each sample is an independent Bernoulli trial.
    """
    if not sample_results:
        return 0.0
    return sum(sample_results) / len(sample_results)


def pass_at_k(sample_results: list[bool]) -> bool:
    """
    Pass@k = True if at least one sample passes.
    """
    return any(sample_results)


# -----------------------------------------------------------------------------
# Metric computations
# -----------------------------------------------------------------------------

def compute_drift_matrix(execution_results: list) -> dict:
    """
    Build a 3x3 matrix of Pass@1 and Pass@3 rates with Wilson CIs.
    Rows = prompted version, Columns = execution environment.

    For each (problem, prompted_version, exec_version) cell:
      - Collect all sample results (pass/fail booleans)
      - Compute Pass@1 (mean of individual samples) and Pass@3 (any passes)
      - Aggregate across problems and compute Wilson CI on the proportions
    """
    # Group by (problem_id, prompted_version) -> exec_version -> list of bool
    groups = defaultdict(lambda: defaultdict(list))
    for entry in execution_results:
        key = (entry["id"], entry["prompted_version"])
        for version, result in entry["exec_results"].items():
            if result.get("status") != "skipped":
                groups[key][version].append(is_pass(result))

    # Aggregate per matrix cell
    # Each cell accumulates: pass@1 numerator/denominator, pass@3 numerator/denominator
    cell_data = {
        pv: {ev: {"p1_successes": 0, "p1_total": 0,
                  "p3_successes": 0, "p3_total": 0}
             for ev in VERSION_ORDER}
        for pv in VERSION_ORDER
    }

    for (problem_id, prompted_v), version_results in groups.items():
        for exec_v in VERSION_ORDER:
            samples = version_results.get(exec_v, [])
            if not samples:
                continue
            cell = cell_data[prompted_v][exec_v]
            # Pass@1: each sample contributes independently
            cell["p1_successes"] += sum(samples)
            cell["p1_total"]     += len(samples)
            # Pass@3: problem-level binary (did any sample pass?)
            cell["p3_successes"] += int(pass_at_k(samples))
            cell["p3_total"]     += 1

    # Compute rates and CIs
    matrix = {}
    for pv in VERSION_ORDER:
        matrix[pv] = {}
        for ev in VERSION_ORDER:
            d    = cell_data[pv][ev]
            p1   = d["p1_successes"] / d["p1_total"] if d["p1_total"] else None
            p3   = d["p3_successes"] / d["p3_total"] if d["p3_total"] else None
            p1_ci = wilson_ci(d["p1_successes"], d["p1_total"])
            p3_ci = wilson_ci(d["p3_successes"], d["p3_total"])
            matrix[pv][ev] = {
                "pass_at_1":      p1,
                "pass_at_1_ci":   p1_ci,
                "pass_at_3":      p3,
                "pass_at_3_ci":   p3_ci,
                "p1_successes":   d["p1_successes"],
                "p1_total":       d["p1_total"],
                "p3_successes":   d["p3_successes"],
                "p3_total":       d["p3_total"],
            }

    return matrix


def compute_error_taxonomy(execution_results: list) -> dict:
    """
    Count error types across all failures, broken down by:
      - overall counts
      - per prompted version (to see if v0-prompted code fails differently than v2-prompted)
    """
    overall    = defaultdict(int)
    by_version = defaultdict(lambda: defaultdict(int))

    for entry in execution_results:
        prompted_v = entry["prompted_version"]
        for exec_v, result in entry["exec_results"].items():
            if is_fail(result):
                error_type = result.get("error_type") or "Unknown"
                overall[error_type] += 1
                by_version[prompted_v][error_type] += 1

    return {
        "overall":    dict(sorted(overall.items(), key=lambda x: -x[1])),
        "by_prompted_version": {v: dict(d) for v, d in by_version.items()},
    }


def compute_repair_rate(repair_results: list) -> dict:
    """
    Compute repair success rate overall and broken down by:
      - repair target version
      - original error type
    All rates include Wilson 95% CIs.
    """
    if not repair_results:
        return {}

    total   = len(repair_results)
    success = sum(1 for r in repair_results if is_pass(r.get("repair_exec_result", {})))

    by_target = defaultdict(lambda: {"success": 0, "total": 0})
    by_error  = defaultdict(lambda: {"success": 0, "total": 0})

    for r in repair_results:
        target = r.get("repair_target_version", "unknown")
        error  = r.get("original_error_type", "Unknown")
        passed = is_pass(r.get("repair_exec_result", {}))

        by_target[target]["total"]  += 1
        by_error[error]["total"]    += 1
        if passed:
            by_target[target]["success"] += 1
            by_error[error]["success"]   += 1

    def rate_dict(d):
        return {
            k: {
                **v,
                "rate": v["success"] / v["total"] if v["total"] else None,
                "ci":   wilson_ci(v["success"], v["total"]),
            }
            for k, v in d.items()
        }

    overall_ci = wilson_ci(success, total)
    return {
        "overall":   {"success": success, "total": total,
                      "rate": success / total if total else None,
                      "ci":   overall_ci},
        "by_target": rate_dict(by_target),
        "by_error":  rate_dict(by_error),
    }


# -----------------------------------------------------------------------------
# Pretty printing
# -----------------------------------------------------------------------------

def fmt_rate(rate, ci=None) -> str:
    if rate is None:
        return "N/A"
    s = f"{rate*100:.1f}%"
    if ci:
        s += f" [{ci[0]*100:.1f},{ci[1]*100:.1f}]"
    return s


def print_drift_matrix(model_label: str, matrix: dict):
    print(f"\n  Drift Matrix -- {model_label}  (95% Wilson CI in brackets)")
    for metric, label in [("pass_at_1", "Pass@1"), ("pass_at_3", "Pass@3")]:
        ci_key = f"{metric}_ci"
        print(f"\n  {label}")
        header = "    Prompted \\ Run in".ljust(26) + "".join(v.ljust(28) for v in VERSION_ORDER)
        print(f"  {header}")
        print("  " + "-" * (24 + 28 * len(VERSION_ORDER)))
        for pv in VERSION_ORDER:
            row = f"    {pv.ljust(22)}"
            for ev in VERSION_ORDER:
                cell = matrix[pv][ev]
                row += fmt_rate(cell[metric], cell[ci_key]).ljust(28)
            print(row)


def print_error_taxonomy(model_label: str, taxonomy: dict):
    print(f"\n  Error Taxonomy -- {model_label}")
    overall = taxonomy.get("overall", {})
    total   = sum(overall.values())
    for error_type, count in overall.items():
        pct = count / total * 100 if total else 0
        bar = "#" * int(pct / 2)
        print(f"    {error_type:<30} {count:4d}  ({pct:5.1f}%)  {bar}")

    print(f"\n  By prompted version:")
    for pv, counts in taxonomy.get("by_prompted_version", {}).items():
        pv_total = sum(counts.values())
        items    = ", ".join(f"{e}: {c}" for e, c in counts.items())
        print(f"    {pv} ({pv_total} failures): {items}")


def print_repair_summary(model_label: str, repair_stats: dict):
    if not repair_stats:
        print(f"\n  Repair results not available for {model_label}")
        return
    overall = repair_stats.get("overall", {})
    print(f"\n  Repair Summary -- {model_label}  (95% Wilson CI in brackets)")
    print(f"    Overall:  {overall.get('success')}/{overall.get('total')}  "
          f"({fmt_rate(overall.get('rate'), overall.get('ci'))})")
    print(f"    By target version:")
    for target, stats in repair_stats.get("by_target", {}).items():
        print(f"      {target}: {stats['success']}/{stats['total']}  "
              f"({fmt_rate(stats['rate'], stats['ci'])})")
    print(f"    By error type:")
    for etype, stats in repair_stats.get("by_error", {}).items():
        print(f"      {etype:<30} {stats['success']}/{stats['total']}  "
              f"({fmt_rate(stats['rate'], stats['ci'])})")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute drift metrics from execution results")
    parser.add_argument("--model", help="Only analyze this model label")
    args = parser.parse_args()

    models_to_run = [m for m in ALL_MODELS if not args.model or m["label"] == args.model]
    all_metrics   = {}

    for model_info in models_to_run:
        label = model_info["label"]
        execution, repair = load_results(label)

        if execution is None:
            print(f"[SKIP] No results found for '{label}'")
            continue

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        drift_matrix   = compute_drift_matrix(execution)
        error_taxonomy = compute_error_taxonomy(execution)
        repair_stats   = compute_repair_rate(repair)

        print_drift_matrix(label, drift_matrix)
        print_error_taxonomy(label, error_taxonomy)
        print_repair_summary(label, repair_stats)

        all_metrics[label] = {
            "drift_matrix":   drift_matrix,
            "error_taxonomy": error_taxonomy,
            "repair_stats":   repair_stats,
        }

    # Save full metrics to JSON for further use (plots, tables, etc.)
    summary_path = ANALYSIS_DIR / "metrics_summary.json"
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[SAVED] Full metrics -> {summary_path}")


if __name__ == "__main__":
    main()
