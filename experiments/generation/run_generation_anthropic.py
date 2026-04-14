# =============================================================================
# experiments/generation/run_generation_anthropic.py
#
# Generates quantum code samples using the Anthropic API (Claude models).
# Produces output in exactly the same JSON format as the other generation
# scripts so the rest of the pipeline (execution, repair,
# metrics) works unchanged.
#
# Setup (do this once before running):
#   export ANTHROPIC_API_KEY="your_key_here"
#
# Usage:
#   python experiments/generation/run_generation_anthropic.py
#   python experiments/generation/run_generation_anthropic.py --model claude_sonnet_46
#   python experiments/generation/run_generation_anthropic.py --dry-run
#
# Output:
#   data/generations/{model_label}.json   (same format as Groq/Codex outputs)
# =============================================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, ANTHROPIC_MODELS,
    ANCHORED_FILE, GENERATIONS_DIR,
    TEMPERATURE, MAX_TOKENS, K_SAMPLES,
)
from lib.families.common import load_generation_prompts, run_generation_family, select_models
from lib.prompting import GENERATION_SYSTEM_PROMPT
from lib.providers.anthropic_openai_compat import create_chat_completion


# -----------------------------------------------------------------------------
# Prompt settings
# -----------------------------------------------------------------------------

def query_model(model_info: dict, prompt: str) -> str:
    """Send a single prompt to the Anthropic API and return the response text."""
    return create_chat_completion(
        api_key=ANTHROPIC_API_KEY,
        base_url=ANTHROPIC_BASE_URL,
        model=model_info["id"],
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        extra_body=model_info.get("extra_body"),
    )


def make_query_model(model_info: dict, dry_run: bool):
    return lambda active_model_info, prompt: query_model(active_model_info, prompt)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM generation via the Anthropic OpenAI-compatible API"
    )
    parser.add_argument(
        "--model",
        help="Run only this model label (e.g. claude_sonnet_46). "
             "Must match a label in ANTHROPIC_MODELS in config.py."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts without calling the API"
    )
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY and not args.dry_run:
        print("[ERROR] ANTHROPIC_API_KEY is not set.")
        print("        Export it: export ANTHROPIC_API_KEY='your_key_here'")
        sys.exit(1)

    prompts = load_generation_prompts(ANCHORED_FILE)
    models_to_run = select_models(ANTHROPIC_MODELS, args.model, "ANTHROPIC_MODELS")
    run_generation_family(
        models=models_to_run,
        prompts=prompts,
        generations_dir=GENERATIONS_DIR,
        sample_count=K_SAMPLES,
        call_delay=0.5,
        dry_run=args.dry_run,
        make_query_model=make_query_model,
    )


if __name__ == "__main__":
    main()
