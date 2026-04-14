# =============================================================================
# experiments/repair/run_repair_openai.py
#
# For each OpenAI-generated snippet that fails forward drift, this script
# constructs a repair prompt, queries the OpenAI API, re-executes the repaired
# code, and saves the repair results.
# =============================================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    ANCHORED_FILE,
    EXECUTION_RESULTS_DIR,
    GENERATIONS_DIR,
    MAX_TOKENS,
    MIGRATION_DIR,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODELS,
    QISKIT_VERSIONS,
    REPAIRS_DIR,
    TEMPERATURE,
)
from lib.families.common import run_repair_family, select_models
from lib.families.openai import (
    make_query_model,
    require_openai_api_key,
    resolve_static_credentials,
)
from lib.prompting import REPAIR_SYSTEM_PROMPT

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_execution import load_test_call_lookup, run_snippet_in_env


def main():
    parser = argparse.ArgumentParser(description="Run documentation-augmented repair prompts via the OpenAI API")
    parser.add_argument("--model", help="Run only this model label")
    parser.add_argument("--target-version", help="Only repair drift toward this version (e.g. v2)")
    args = parser.parse_args()

    require_openai_api_key(OPENAI_API_KEY, dry_run=False)
    models_to_run = select_models(OPENAI_MODELS, args.model, "OPENAI_MODELS")
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
        make_query_model=make_query_model(
            system_prompt=REPAIR_SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            resolve_credentials=lambda model_info: resolve_static_credentials(
                model_info,
                OPENAI_API_KEY,
                OPENAI_BASE_URL,
                "https://api.openai.com/v1",
            ),
        ),
        call_delay=0.5,
        target_version_filter=args.target_version,
        missing_generation_hint="Run experiments/generation/run_generation_openai.py first.",
    )


if __name__ == "__main__":
    main()
