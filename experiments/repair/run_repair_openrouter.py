# =============================================================================
# experiments/repair/run_repair_openrouter.py
#
# For each OpenRouter-generated snippet that fails forward drift, this script:
#   1. Loads the relevant migration notes from data/migration_notes/
#   2. Constructs a repair prompt with the broken code + error + migration notes
#   3. Queries OpenRouter for a repaired version
#   4. Re-executes the repaired code in the target environment
#   5. Saves the repair results
#
# Usage:
#   python experiments/repair/run_repair_openrouter.py
#   python experiments/repair/run_repair_openrouter.py --model openrouter_claude_sonnet_45
#   python experiments/repair/run_repair_openrouter.py --target-version v2
# =============================================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    ANCHORED_FILE, MAX_TOKENS, MIGRATION_DIR, OPENROUTER_API_KEY, OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL, OPENROUTER_MODELS, OPENROUTER_SITE_URL, GENERATIONS_DIR,
    QISKIT_VERSIONS, EXECUTION_RESULTS_DIR, REPAIRS_DIR, TEMPERATURE,
)
from lib.families.common import run_repair_family, select_models
from lib.prompting import REPAIR_SYSTEM_PROMPT
from lib.providers.openrouter import create_chat_completion

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_execution import load_test_call_lookup, run_snippet_in_env

def query_model(model_id: str, prompt: str) -> str:
    """Send a single repair prompt to OpenRouter and return the response text."""
    return create_chat_completion(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=model_id,
        messages=[
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        site_url=OPENROUTER_SITE_URL,
        app_name=OPENROUTER_APP_NAME,
    )


def main():
    parser = argparse.ArgumentParser(description="Run documentation-augmented repair prompts via OpenRouter")
    parser.add_argument("--model", help="Run only this model label")
    parser.add_argument("--target-version", help="Only repair drift toward this version (e.g. v2)")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("[ERROR] OPENROUTER_API_KEY is not set.")
        print("        Export it: export OPENROUTER_API_KEY='your_key_here'")
        sys.exit(1)

    models_to_run = select_models(OPENROUTER_MODELS, args.model, "OPENROUTER_MODELS")
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
            lambda active_model_info, prompt: query_model(active_model_info["id"], prompt)
        ),
        call_delay=0.5,
        target_version_filter=args.target_version,
        missing_generation_hint="Run experiments/generation/run_generation_openrouter.py first.",
    )


if __name__ == "__main__":
    main()
