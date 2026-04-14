# =============================================================================
# experiments/run_execution.py
#
# Takes model output files from data/generations/ and executes each code snippet
# inside the appropriate Conda environment using `conda run`.
#
# Each snippet is executed as:
#   [generated code]  +  [test harness that calls the expected function]
#
# This means a snippet that defines the right function but uses a deprecated
# API will fail at the test_call line with the actual runtime error, giving
# a much more meaningful signal than a bare import check.
#
# For each snippet we test TWO things:
#   1. Version Fidelity:  run the v0-prompted code in the v0 environment, etc.
#   2. Forward Drift:     also run that same code in all other environments.
#
# Usage:
#   python experiments/run_execution.py
#   python experiments/run_execution.py --model kimi_k2    # single model only
#
# Output file (one per model):
#   data/execution/{model_label}_execution.json
#   [
#     {
#       "id":               "problem_001",
#       "prompted_version": "v0",
#       "sample_index":     0,
#       "exec_results": {
#         "v0": { "status": "pass",      "error_type": null,               "message": "" },
#         "v1": { "status": "hard_fail", "error_type": "ImportError",      "message": "..." },
#         "v2": { "status": "soft_fail", "error_type": "DeprecationWarning","message": "..." }
#       }
#     },
#     ...
#   ]
# =============================================================================

import argparse
import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ALL_MODELS, QISKIT_VERSIONS,
    ANCHORED_FILE, GENERATIONS_DIR, EXECUTION_RESULTS_DIR,
    EXECUTION_TIMEOUT,
)


# -----------------------------------------------------------------------------
# Runner template
#
# Structure of the executed file:
#
#   import warnings
#   warnings.filterwarnings("error", category=DeprecationWarning)
#
#   try:
#       <generated code -- indented>
#
#       # Test harness
#       <test_call -- indented>
#
#   except DeprecationWarning ...
#   except ImportError ...
#   ...
#
# Turning DeprecationWarnings into errors means soft failures are caught
# at the same level as hard failures, keeping classification clean.
# -----------------------------------------------------------------------------

RUNNER_TEMPLATE = textwrap.dedent("""\
import warnings

# Treat DeprecationWarnings as errors so we can detect and classify them
warnings.filterwarnings("error", category=DeprecationWarning)

try:
{indented_code}

    # ---- Test harness: call the expected function ----
{indented_test_call}

except DeprecationWarning as e:
    print(f"SOFT_FAIL:DeprecationWarning:{{e}}")
    exit(2)
except ImportError as e:
    print(f"HARD_FAIL:ImportError:{{e}}")
    exit(1)
except ModuleNotFoundError as e:
    print(f"HARD_FAIL:ModuleNotFoundError:{{e}}")
    exit(1)
except AttributeError as e:
    print(f"HARD_FAIL:AttributeError:{{e}}")
    exit(1)
except AssertionError as e:
    # test_call assertion failed -- function ran but returned wrong type/value
    print(f"HARD_FAIL:AssertionError:{{e}}")
    exit(1)
except Exception as e:
    print(f"HARD_FAIL:{{type(e).__name__}}:{{e}}")
    exit(1)

print("PASS")
exit(0)
""")

# Fallback test call used when a problem has no test_call defined.
# It tries to call the function with no arguments, which will at minimum
# confirm that the function exists and the imports work.
FALLBACK_TEST_TEMPLATE = "assert callable({entry_point}), 'Function {entry_point} was not defined'"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def build_runner(generated_code: str, test_call: str, entry_point: str) -> str:
    """
    Combine generated code and test harness into the full runner script.

    The generated code is indented by 4 spaces (placed inside the try block).
    The test_call is also indented by 4 spaces (appended inside the same try block,
    after the generated code).

    If no test_call is provided, we fall back to a callable() check.
    """
    if not test_call or not test_call.strip():
        test_call = FALLBACK_TEST_TEMPLATE.format(entry_point=entry_point or "unknown")

    indented_code      = textwrap.indent(generated_code.strip(), "    ")
    indented_test_call = textwrap.indent(test_call.strip(), "    ")

    return RUNNER_TEMPLATE.format(
        indented_code      = indented_code,
        indented_test_call = indented_test_call,
    )


def classify_output(stdout: str, stderr: str, returncode: int) -> dict:
    """
    Parse the runner output and classify the execution result.
    Returns a dict with keys: status, error_type, message.
    """
    stdout = stdout.strip()

    if returncode == 0 and "PASS" in stdout:
        return {"status": "pass", "error_type": None, "message": ""}

    # Scan for our structured output lines (first match wins)
    for line in stdout.splitlines():
        if line.startswith("SOFT_FAIL:"):
            parts = line.split(":", 2)
            return {
                "status":     "soft_fail",
                "error_type": parts[1] if len(parts) > 1 else "Unknown",
                "message":    parts[2] if len(parts) > 2 else "",
            }
        if line.startswith("HARD_FAIL:"):
            parts = line.split(":", 2)
            return {
                "status":     "hard_fail",
                "error_type": parts[1] if len(parts) > 1 else "Unknown",
                "message":    parts[2] if len(parts) > 2 else "",
            }

    # Fallback: non-zero exit with no structured output
    error_msg = (stderr or stdout)[:300]
    return {
        "status":     "hard_fail",
        "error_type": "UnknownError",
        "message":    error_msg,
    }


def run_snippet_in_env(code: str, test_call: str, entry_point: str, conda_env: str) -> dict:
    """
    Write the generated code + test harness to a temp file and execute it
    inside the given Conda environment.
    Returns a classification dict.
    """
    runner_code = build_runner(code, test_call, entry_point)

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
        tmp.write(runner_code)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", conda_env, "python", tmp_path],
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT,
        )
        return classify_output(proc.stdout, proc.stderr, proc.returncode)
    except subprocess.TimeoutExpired:
        return {
            "status":     "hard_fail",
            "error_type": "TimeoutError",
            "message":    "Execution timed out",
        }
    except FileNotFoundError:
        return {
            "status":     "hard_fail",
            "error_type": "EnvironmentError",
            "message":    f"conda env '{conda_env}' not found. Did you run setup_envs.sh?",
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# Per-model execution
# -----------------------------------------------------------------------------

def load_test_call_lookup(anchored_file: Path) -> dict:
    """
    Build a lookup from problem id -> {entry_point, test_call}.
    We only need one version's entry since entry_point and test_call
    are the same across all versions of the same problem.
    """
    if not anchored_file.exists():
        return {}
    with open(anchored_file) as f:
        anchored = json.load(f)
    lookup = {}
    for entry in anchored:
        if entry["id"] not in lookup:
            lookup[entry["id"]] = {
                "entry_point": entry.get("entry_point", ""),
                "test_call":   entry.get("test_call", ""),
            }
    return lookup


def run_execution_for_model(model_label: str, test_lookup: dict):
    """
    Load a model's generation outputs and run every snippet in every environment.
    """
    input_path  = GENERATIONS_DIR / f"{model_label}.json"
    output_path = EXECUTION_RESULTS_DIR / f"{model_label}_execution.json"

    if not input_path.exists():
        print(f"[SKIP] No generation output found for '{model_label}' at {input_path}")
        return

    with open(input_path) as f:
        generations = json.load(f)

    # Resume support: skip entries already saved
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        done_keys = {(r["id"], r["prompted_version"], r["sample_index"]) for r in results}
    else:
        results   = []
        done_keys = set()

    total = sum(len(g["samples"]) for g in generations)
    done  = 0

    for entry in generations:
        problem_id       = entry["id"]
        prompted_version = entry["version"]

        # Look up the test harness for this problem
        meta        = test_lookup.get(problem_id, {})
        entry_point = meta.get("entry_point", "")
        test_call   = meta.get("test_call", "")

        if not test_call:
            print(f"[WARN] No test_call for problem '{problem_id}'. Using fallback.")

        for sample_idx, code in enumerate(entry["samples"]):
            done += 1
            key = (problem_id, prompted_version, sample_idx)

            if key in done_keys:
                continue

            if code is None:
                results.append({
                    "id":               problem_id,
                    "prompted_version": prompted_version,
                    "sample_index":     sample_idx,
                    "exec_results": {
                        v: {"status": "skipped", "error_type": None, "message": "Generation failed"}
                        for v in QISKIT_VERSIONS
                    },
                })
                continue

            print(f"[{model_label}] {done}/{total}  "
                  f"problem={problem_id}  prompted={prompted_version}  sample={sample_idx}")

            exec_results = {}
            for version_key, version_cfg in QISKIT_VERSIONS.items():
                conda_env = version_cfg["conda_env"]
                result    = run_snippet_in_env(code, test_call, entry_point, conda_env)
                exec_results[version_key] = result
                status_str = result["status"].upper().ljust(10)
                print(f"  run in {conda_env}: {status_str}  {result.get('error_type') or ''}")

            results.append({
                "id":               problem_id,
                "prompted_version": prompted_version,
                "sample_index":     sample_idx,
                "exec_results":     exec_results,
            })

            # Save after every snippet so crashes don't lose progress
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    print(f"[DONE] {model_label} execution -> {output_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Execute generated code in Conda environments")
    parser.add_argument("--model", help="Run only this model label (e.g. kimi_k2, codex_gpt_5_4_medium, claude_sonnet_46)")
    args = parser.parse_args()

    EXECUTION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load test harness info once, shared across all models
    test_lookup = load_test_call_lookup(ANCHORED_FILE)
    if not test_lookup:
        print(f"[WARN] Could not load test harness data from {ANCHORED_FILE}.")
        print("       Run prepare_prompts.py first, or execution will use fallback checks only.")

    models_to_run = [m for m in ALL_MODELS if not args.model or m["label"] == args.model]
    if not models_to_run:
        print(f"[ERROR] No model found with label '{args.model}'")
        sys.exit(1)

    for model_info in models_to_run:
        print(f"\n{'='*60}")
        print(f" Executing outputs for: {model_info['label']}")
        print(f"{'='*60}")
        run_execution_for_model(model_info["label"], test_lookup)


if __name__ == "__main__":
    main()
