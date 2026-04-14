# =============================================================================
# experiments/repair/run_repair_anthropic.py
#
# For each Anthropic-generated snippet that fails forward drift (passes its
# target version but fails a newer one), this script:
#   1. Loads the relevant migration notes from data/migration_notes/
#   2. Constructs a repair prompt with the broken code + error + migration notes
#   3. Queries the Anthropic API for a repaired version
#   4. Re-executes the repaired code in the target environment
#   5. Saves the repair results
#
# Usage:
#   python experiments/repair/run_repair_anthropic.py
#   python experiments/repair/run_repair_anthropic.py --model claude_sonnet_46
#   python experiments/repair/run_repair_anthropic.py --target-version v2
#
# Output:
#   data/repairs/{model_label}_repair.json
# =============================================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, ANTHROPIC_MODELS, MAX_TOKENS,
    ANCHORED_FILE, MIGRATION_DIR, GENERATIONS_DIR, QISKIT_VERSIONS, EXECUTION_RESULTS_DIR, REPAIRS_DIR, TEMPERATURE,
)
from lib.families.common import run_repair_family, select_models
from lib.prompting import REPAIR_SYSTEM_PROMPT
from lib.providers.anthropic_openai_compat import create_chat_completion

# Reuse the execution helper from run_execution.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_execution import load_test_call_lookup, run_snippet_in_env

def query_model(model_info: dict, prompt: str) -> str:
    """Send a single repair prompt to the Anthropic API and return the response text."""
    return create_chat_completion(
        api_key=ANTHROPIC_API_KEY,
        base_url=ANTHROPIC_BASE_URL,
        model=model_info["id"],
        messages=[
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        extra_body=model_info.get("extra_body"),
    )


def make_query_model(model_info: dict, dry_run: bool):
    return lambda active_model_info, prompt: query_model(active_model_info, prompt)


def main():
    parser = argparse.ArgumentParser(description="Run documentation-augmented repair prompts via Anthropic OpenAI-compatible API")
    parser.add_argument("--model", help="Run only this model label")
    parser.add_argument("--target-version", help="Only repair drift toward this version (e.g. v2)")
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY:
        print("[ERROR] ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    models_to_run = select_models(ANTHROPIC_MODELS, args.model, "ANTHROPIC_MODELS")
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
        make_query_model=make_query_model,
        call_delay=0.5,
        target_version_filter=args.target_version,
        missing_generation_hint="Run experiments/generation/run_generation_anthropic.py first.",
    )


if __name__ == "__main__":
    main()
