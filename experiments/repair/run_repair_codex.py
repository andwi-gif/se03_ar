# =============================================================================
# experiments/repair/run_repair_codex.py
#
# For each Codex-generated snippet that fails forward drift (passes its target
# version but fails a newer one), this script:
#   1. Loads the relevant migration notes from data/migration_notes/
#   2. Constructs a repair prompt with the broken code + error + migration notes
#   3. Queries the Codex CLI for a repaired version
#   4. Re-executes the repaired code in the target environment
#   5. Saves the repair results
#
# Usage:
#   python experiments/repair/run_repair_codex.py
#   python experiments/repair/run_repair_codex.py --model codex_gpt_5_4_medium
#   python experiments/repair/run_repair_codex.py --target-version v2
#
# Output:
#   data/repairs/{model_label}_repair.json
# =============================================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    ANCHORED_FILE, CODEX_MODELS, MIGRATION_DIR,
    GENERATIONS_DIR, QISKIT_VERSIONS, EXECUTION_RESULTS_DIR, REPAIRS_DIR,
)
from lib.families.common import run_repair_family, select_models
from lib.prompting import REPAIR_SYSTEM_PROMPT
from lib.providers.codex import call_codex, check_codex_installed

# Reuse the execution helper from run_execution.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_execution import load_test_call_lookup, run_snippet_in_env


# -----------------------------------------------------------------------------
# Codex CLI settings
# -----------------------------------------------------------------------------

CALL_DELAY = 1.5


def main():
    parser = argparse.ArgumentParser(description="Run documentation-augmented repair prompts via Codex CLI")
    parser.add_argument("--model", help="Run only this model label")
    parser.add_argument("--target-version", help="Only repair drift toward this version (e.g. v2)")
    args = parser.parse_args()

    if not check_codex_installed():
        print("[ERROR] Codex CLI is not installed or not on PATH.")
        print("        Run: npm install -g @openai/codex")
        print("        Then authenticate with OPENAI_API_KEY or run: codex --login")
        sys.exit(1)

    models_to_run = select_models(CODEX_MODELS, args.model, "CODEX_MODELS")
    run_repair_family(
        models=models_to_run,
        generations_dir=GENERATIONS_DIR,
        execution_results_dir=EXECUTION_RESULTS_DIR,
        repairs_dir=REPAIRS_DIR,
        anchored_file=ANCHORED_FILE,
        migration_dir=MIGRATION_DIR,
        qiskit_versions=QISKIT_VERSIONS,
        load_test_call_lookup=load_test_call_lookup,
        run_snippet_in_env=run_snippet_in_env,
        make_query_model=lambda model_info, dry_run: (
            lambda active_model_info, prompt: call_codex(
                prompt,
                active_model_info,
                REPAIR_SYSTEM_PROMPT,
                "codex-repair-",
            )
        ),
        call_delay=CALL_DELAY,
        target_version_filter=args.target_version,
        missing_generation_hint="Run experiments/generation/run_generation_codex.py first.",
        before_model=lambda model_info: print(
            "[INFO] Using Codex CLI with "
            f"model: {model_info['id']} "
            f"(reasoning: {model_info['reasoning_effort']})"
        ),
    )


if __name__ == "__main__":
    main()
