# =============================================================================
# experiments/generation/run_generation_openrouter.py
#
# Generates quantum code samples using the OpenRouter chat completions API.
# Produces output in exactly the same JSON format as the other generation
# scripts so the rest of the pipeline (execution, repair, metrics) works
# unchanged.
#
# Setup:
#   export OPENROUTER_API_KEY="your_key_here"
#
# Usage:
#   python experiments/generation/run_generation_openrouter.py
#   python experiments/generation/run_generation_openrouter.py --model openrouter_claude_sonnet_45
#   python experiments/generation/run_generation_openrouter.py --dry-run
# =============================================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    ANCHORED_FILE, K_SAMPLES, MAX_TOKENS, OPENROUTER_API_KEY, OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL, OPENROUTER_MODELS, OPENROUTER_SITE_URL, GENERATIONS_DIR,
    TEMPERATURE,
)
from lib.families.common import load_generation_prompts, run_generation_family, select_models
from lib.prompting import GENERATION_SYSTEM_PROMPT
from lib.providers.openrouter import create_chat_completion

def query_model(model_id: str, prompt: str) -> str:
    """Send a single prompt to OpenRouter and return the response text."""
    return create_chat_completion(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=model_id,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        site_url=OPENROUTER_SITE_URL,
        app_name=OPENROUTER_APP_NAME,
    )


def main():
    parser = argparse.ArgumentParser(description="Run LLM generation via OpenRouter")
    parser.add_argument(
        "--model",
        help="Run only this model label (must match a label in OPENROUTER_MODELS).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY and not args.dry_run:
        print("[ERROR] OPENROUTER_API_KEY is not set.")
        print("        Export it: export OPENROUTER_API_KEY='your_key_here'")
        sys.exit(1)

    prompts = load_generation_prompts(ANCHORED_FILE)
    models_to_run = select_models(OPENROUTER_MODELS, args.model, "OPENROUTER_MODELS")
    run_generation_family(
        models=models_to_run,
        prompts=prompts,
        generations_dir=GENERATIONS_DIR,
        sample_count=K_SAMPLES,
        call_delay=0.5,
        dry_run=args.dry_run,
        make_query_model=lambda model_info, dry_run: (
            lambda active_model_info, prompt: query_model(active_model_info["id"], prompt)
        ),
    )


if __name__ == "__main__":
    main()
