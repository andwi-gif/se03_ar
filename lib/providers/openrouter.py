from lib.providers.openai_compatible import create_chat_completion as create_openai_compatible_chat_completion


def create_chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    site_url: str = "",
    app_name: str = "",
) -> str:
    """Call OpenRouter's OpenAI-compatible chat completions API."""
    return create_openai_compatible_chat_completion(
        api_key=api_key,
        base_url=base_url,
        default_base_url="https://openrouter.ai/api/v1",
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers={
            **({"HTTP-Referer": site_url} if site_url else {}),
            **({"X-Title": app_name} if app_name else {}),
        },
    )
