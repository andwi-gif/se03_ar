"""Shared model-family registry helpers for top-level runners."""

import sys

from config import (
    ANTHROPIC_MODELS,
    CODEX_MODELS,
    GROQ_MODELS,
    OPENAI_MODELS,
    OPENAI_COMPATIBLE_MODELS,
    OPENROUTER_MODELS,
)


FAMILY_MODELS = {
    "groq": GROQ_MODELS,
    "codex": CODEX_MODELS,
    "openai": OPENAI_MODELS,
    "anthropic": ANTHROPIC_MODELS,
    "openrouter": OPENROUTER_MODELS,
    "openai_compatible": OPENAI_COMPATIBLE_MODELS,
}


def determine_families(model_label: str | None) -> list[str]:
    if not model_label:
        families = [family for family, models in FAMILY_MODELS.items() if models]
        if not families:
            print("[ERROR] No models are configured in config.py.")
            sys.exit(1)
        return families

    for family, models in FAMILY_MODELS.items():
        if any(model["label"] == model_label for model in models):
            return [family]

    print(f"[ERROR] No model found with label '{model_label}'")
    sys.exit(1)
