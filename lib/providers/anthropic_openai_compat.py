"""Anthropic OpenAI-compat request helpers."""

from lib.providers.openai_compatible import create_chat_completion as create_openai_compatible_chat_completion


def create_chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    extra_body: dict | None = None,
) -> str:
    """Call Anthropic through its OpenAI-compatible chat completions API."""
    return create_openai_compatible_chat_completion(
        api_key=api_key,
        base_url=base_url,
        default_base_url="https://api.anthropic.com/v1",
        model=model,
        messages=messages,
        temperature=max(0.0, min(temperature, 1.0)),
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
