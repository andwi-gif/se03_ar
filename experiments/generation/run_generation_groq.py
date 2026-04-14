# =============================================================================
# experiments/generation/run_generation_groq.py
#
# Queries the Groq API for each (prompt, model) combination K_SAMPLES times.
# Saves raw model outputs to data/generations/.
#
# Usage:
#   python experiments/generation/run_generation_groq.py
#   python experiments/generation/run_generation_groq.py --model kimi_k2
#   python experiments/generation/run_generation_groq.py --dry-run
#
# Output file format (one file per model):
#   data/generations/{model_label}.json
#   [
#     {
#       "id":      "problem_001",
#       "version": "v0",
#       "prompt":  "Using Qiskit v0.43, ...",
#       "samples": ["<code attempt 1>", "<code attempt 2>", "<code attempt 3>"]
#     },
#     ...
#   ]
# =============================================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODELS,
    ANCHORED_FILE, GENERATIONS_DIR,
    TEMPERATURE, MAX_TOKENS, K_SAMPLES,
)
from lib.families.common import load_generation_prompts, run_generation_family, select_models
from lib.prompting import GENERATION_SYSTEM_PROMPT
from lib.providers.groq import create_chat_completion


def query_model(model_info: dict, prompt: str) -> str:
    """Send a single prompt to the Groq API and return the response text."""
    return create_chat_completion(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
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


def main():
    parser = argparse.ArgumentParser(description="Run LLM generation via Groq API")
    parser.add_argument("--model",   help="Run only this model label (e.g. kimi_k2)")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    args = parser.parse_args()

    if not GROQ_API_KEY and not args.dry_run:
        print("[ERROR] GROQ_API_KEY is not set. Export it or use --dry-run.")
        sys.exit(1)

    prompts = load_generation_prompts(ANCHORED_FILE)
    models_to_run = select_models(GROQ_MODELS, args.model, "GROQ_MODELS")
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
