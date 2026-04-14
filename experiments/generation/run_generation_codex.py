# =============================================================================
# experiments/generation/run_generation_codex.py
#
# Generates quantum code samples using the Codex CLI tool (`codex` command)
# instead of the Groq API. Produces output in exactly the same JSON format
# as other generation scripts so the rest of the pipeline (execution, repair,
# metrics) works unchanged.
#
# The Codex CLI is called via `codex exec` from an isolated temporary
# directory in a read-only sandbox so it stays non-interactive, does not see
# the project tree, and cannot modify the repository while generating answers.
#
# Setup (do this once before running):
#   npm install -g @openai/codex
#   export OPENAI_API_KEY="your_key_here"   # or run: codex --login
#   codex --version   # verify installation
#
# Usage:
#   python experiments/generation/run_generation_codex.py
#   python experiments/generation/run_generation_codex.py --model codex_gpt_5_4_medium
#   python experiments/generation/run_generation_codex.py --dry-run
#
# Output:
#   data/generations/{model_label}.json   (same format as other model outputs)
# =============================================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    ANCHORED_FILE, GENERATIONS_DIR,
    K_SAMPLES, CODEX_MODELS,
)
from lib.families.common import load_generation_prompts, run_generation_family, select_models
from lib.prompting import GENERATION_SYSTEM_PROMPT
from lib.providers.codex import call_codex, check_codex_installed

# -----------------------------------------------------------------------------
# Codex CLI settings
# -----------------------------------------------------------------------------

CALL_DELAY = 1.5


def main():
    parser = argparse.ArgumentParser(
        description="Run code generation using the Codex CLI"
    )
    parser.add_argument(
        "--model",
        help="Run only this model label (e.g. codex_gpt_5_4_medium)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts without calling the CLI"
    )
    args = parser.parse_args()

    # Pre-flight checks
    if not args.dry_run:
        if not check_codex_installed():
            print("[ERROR] Codex CLI is not installed or not on PATH.")
            print("        Run: npm install -g @openai/codex")
            print("        Then authenticate with OPENAI_API_KEY or run: codex --login")
            sys.exit(1)

    prompts = load_generation_prompts(ANCHORED_FILE)
    models_to_run = select_models(CODEX_MODELS, args.model, "CODEX_MODELS")
    run_generation_family(
        models=models_to_run,
        prompts=prompts,
        generations_dir=GENERATIONS_DIR,
        sample_count=K_SAMPLES,
        call_delay=CALL_DELAY,
        dry_run=args.dry_run,
        make_query_model=lambda model_info, dry_run: (
            lambda active_model_info, prompt: call_codex(
                prompt,
                active_model_info,
                GENERATION_SYSTEM_PROMPT,
                "codex-gen-",
            )
        ),
        describe_model=lambda model_info: f"{model_info['id']}, reasoning={model_info['reasoning_effort']}",
        before_model=lambda model_info, dry_run: (
            print(
                "[INFO] Using Codex CLI with "
                f"model: {model_info['id']} "
                f"(reasoning: {model_info['reasoning_effort']})"
            ) if not dry_run else None
        ),
    )


if __name__ == "__main__":
    main()
