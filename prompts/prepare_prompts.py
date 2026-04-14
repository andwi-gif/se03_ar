# =============================================================================
# prompts/prepare_prompts.py
#
# Loads raw problem descriptions and produces version-anchored prompts
# for each configured Qiskit version.
#
# Input:
#   data/problems/raw_problems.json   -- list of problem dicts (see format below)
#
# Output:
#   data/problems/prompts_anchored.json   -- version-anchored prompts (all versions)
#
# Raw problem format:
#   [
#     {
#       "id": "problem_001",
#       "description": "Create a quantum circuit for the given integer n_qubits and
#                       return it. You must implement this using a function named
#                       create_quantum_circuit with the following argument: n_qubits.",
#       "entry_point": "create_quantum_circuit",
#       "test_call":   "result = create_quantum_circuit(3)\nassert result is not None"
#     },
#     ...
#   ]
#
# Fields:
#   entry_point  -- the exact function name the model is expected to define.
#                   Used in the prompt footer to reinforce the requirement.
#   test_call    -- a small snippet that calls the function and asserts basic
#                   correctness. This is appended to the generated code at
#                   execution time (see run_execution.py).
#                   Rules for writing test_call:
#                     - Call the function with simple, representative arguments
#                     - Use only: assert, isinstance, is not None, basic comparisons
#                     - Do NOT import anything -- the generated code handles imports
#                     - Do NOT verify quantum correctness, just that it runs and
#                       returns something sensible
#                   Example:
#                     "result = create_quantum_circuit(3)\nassert result is not None"
# =============================================================================

import json
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROBLEMS_DIR, ANCHORED_FILE, QISKIT_VERSIONS


def make_anchored_prompt(description: str, entry_point: str, version_key: str) -> str:
    """
    Prepend the version anchor to the original problem description.
    The footer reminds the model of the required function name and output format.
    """
    prefix = QISKIT_VERSIONS[version_key]["anchor_prefix"]
    footer = (
        f"\n\nYou must define a function named `{entry_point}` as described above. "
        "Return only a complete, runnable Python code block with all necessary imports. "
        "Do not include explanations outside the code. "
        "Use only standard library imports and the Qiskit SDK."
    )
    return f"{prefix}{description}{footer}"


def validate_problems(raw_problems: list) -> list[str]:
    """
    Check that all required fields are present.
    Returns a list of warning strings (empty if all is fine).
    """
    warnings = []
    required_fields = ["id", "description", "entry_point", "test_call"]
    for i, p in enumerate(raw_problems):
        for field in required_fields:
            if field not in p or not p[field]:
                warnings.append(f"  Problem {i} (id={p.get('id', '?')}): missing or empty '{field}'")
    return warnings


def main():
    raw_path = PROBLEMS_DIR / "raw_problems.json"

    if not raw_path.exists():
        print(f"[ERROR] Raw problems file not found: {raw_path}")
        print("        Please place your raw problems there.")
        print("        See the format comment at the top of this file.")
        sys.exit(1)

    with open(raw_path) as f:
        raw_problems = json.load(f)

    print(f"[INFO] Loaded {len(raw_problems)} raw problems.")

    # Validate fields before doing anything
    warnings = validate_problems(raw_problems)
    if warnings:
        print("[WARN] Some problems are missing fields:")
        for w in warnings:
            print(w)
        print("       These problems will still be processed but execution may fail.")

    # --- Build version-anchored prompts ---
    # One entry per (problem, version) combination.
    # entry_point and test_call are carried through so run_execution.py
    # can access them without needing to re-load raw_problems.json.
    anchored = []
    for problem in raw_problems:
        for version_key in QISKIT_VERSIONS:
            anchored.append({
                "id":          problem["id"],
                "version":     version_key,
                "entry_point": problem.get("entry_point", ""),
                "test_call":   problem.get("test_call", ""),
                "prompt":      make_anchored_prompt(
                                   problem["description"],
                                   problem.get("entry_point", ""),
                                   version_key,
                               ),
            })

    with open(ANCHORED_FILE, "w") as f:
        json.dump(anchored, f, indent=2)
    print(f"[OK] Saved anchored prompts  -> {ANCHORED_FILE}")
    print(f"     Total prompt variants: {len(anchored)} "
          f"({len(raw_problems)} problems x {len(QISKIT_VERSIONS)} versions)")


if __name__ == "__main__":
    main()
