"""Groq-specific request helpers."""

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
    """Call Groq through its OpenAI-compatible chat completions API."""
    return create_openai_compatible_chat_completion(
        api_key=api_key,
        base_url=base_url,
        default_base_url="https://api.groq.com/openai/v1",
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
