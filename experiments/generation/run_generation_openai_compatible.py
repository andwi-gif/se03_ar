# =============================================================================
# experiments/generation/run_generation_openai_compatible.py
#
# Generates quantum code samples using an OpenAI-compatible chat completions API.
# Produces output in exactly the same JSON format as the other generation
# scripts so the rest of the pipeline (execution, repair, metrics) works
# unchanged.
# =============================================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    ANCHORED_FILE,
    GENERATIONS_DIR,
    K_SAMPLES,
    MAX_TOKENS,
    OPENAI_COMPATIBLE_API_KEY,
    OPENAI_COMPATIBLE_BASE_URL,
    OPENAI_COMPATIBLE_MODELS,
    TEMPERATURE,
)
from lib.families.common import load_generation_prompts, run_generation_family, select_models
from lib.families.openai import (
    make_query_model,
    require_openai_compatible_credentials,
    resolve_openai_compatible_credentials,
)
from lib.prompting import GENERATION_SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser(description="Run LLM generation via an OpenAI-compatible API")
    parser.add_argument(
        "--model",
        help="Run only this model label (must match a label in OPENAI_COMPATIBLE_MODELS).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    args = parser.parse_args()

    prompts = load_generation_prompts(ANCHORED_FILE)
    models_to_run = select_models(
        OPENAI_COMPATIBLE_MODELS,
        args.model,
        "OPENAI_COMPATIBLE_MODELS",
    )

    for model_info in models_to_run:
        api_key, base_url = resolve_openai_compatible_credentials(
            model_info,
            OPENAI_COMPATIBLE_API_KEY,
            OPENAI_COMPATIBLE_BASE_URL,
        )
        require_openai_compatible_credentials(model_info, api_key, base_url, args.dry_run)

    run_generation_family(
        models=models_to_run,
        prompts=prompts,
        generations_dir=GENERATIONS_DIR,
        sample_count=K_SAMPLES,
        call_delay=0.5,
        dry_run=args.dry_run,
        make_query_model=make_query_model(
            system_prompt=GENERATION_SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            resolve_credentials=lambda model_info: resolve_openai_compatible_credentials(
                model_info,
                OPENAI_COMPATIBLE_API_KEY,
                OPENAI_COMPATIBLE_BASE_URL,
            ),
        ),
        describe_model=_describe_model,
    )


def _describe_model(model_info: dict) -> str:
    extra_body = model_info.get("extra_body")
    if extra_body:
        return f"{model_info['id']}, extra_body={extra_body}"
    return model_info["id"]


if __name__ == "__main__":
    main()
