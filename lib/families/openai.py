"""OpenAI-family-specific helpers."""

import os
import sys

from lib.providers.openai_compatible import create_chat_completion


def query_chat_model(
    *,
    model_info: dict,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    base_url: str,
    extra_body: dict | None = None,
    extra_headers: dict[str, str] | None = None,
    default_base_url: str = "",
) -> str:
    return create_chat_completion(
        api_key=api_key,
        base_url=base_url,
        default_base_url=default_base_url,
        model=model_info["id"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
        extra_headers=extra_headers,
    )


def resolve_static_credentials(
    model_info: dict,
    api_key: str,
    base_url: str,
    default_base_url: str = "",
) -> tuple[str, str, str]:
    return api_key, base_url, default_base_url


def resolve_openai_compatible_credentials(
    model_info: dict,
    fallback_api_key: str,
    fallback_base_url: str,
) -> tuple[str, str]:
    api_key = model_info.get("api_key") or os.environ.get(model_info.get("api_key_env", ""), "")
    if not api_key:
        api_key = fallback_api_key

    base_url = model_info.get("base_url") or os.environ.get(model_info.get("base_url_env", ""), "")
    if not base_url:
        base_url = fallback_base_url

    return api_key, base_url


def make_query_model(
    *,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    resolve_credentials,
    extra_body_config_key: str | None = "extra_body",
):
    return lambda model_info, dry_run: (
        lambda active_model_info, prompt: _query_with_resolved_credentials(
            model_info=active_model_info,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            resolve_credentials=resolve_credentials,
            extra_body_config_key=extra_body_config_key,
        )
    )


def _query_with_resolved_credentials(
    *,
    model_info: dict,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    resolve_credentials,
    extra_body_config_key: str | None,
) -> str:
    credentials = resolve_credentials(model_info)
    if len(credentials) == 2:
        api_key, base_url = credentials
        default_base_url = ""
    elif len(credentials) == 3:
        api_key, base_url, default_base_url = credentials
    else:
        raise ValueError(
            "resolve_credentials must return (api_key, base_url) or "
            "(api_key, base_url, default_base_url)."
        )
    extra_body = None
    extra_headers = None
    if extra_body_config_key:
        extra_body = model_info.get(extra_body_config_key)
        if extra_body is not None and not isinstance(extra_body, dict):
            raise ValueError(
                f"model_info['{extra_body_config_key}'] must be a dict when provided."
            )
    extra_headers = model_info.get("extra_headers")
    if extra_headers is not None and not isinstance(extra_headers, dict):
        raise ValueError("model_info['extra_headers'] must be a dict when provided.")
    return query_chat_model(
        model_info=model_info,
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        extra_body=extra_body,
        extra_headers=extra_headers,
        default_base_url=default_base_url,
    )


def require_openai_api_key(api_key: str, dry_run: bool) -> None:
    if not api_key and not dry_run:
        print("[ERROR] OPENAI_API_KEY is not set.")
        print("        Export it: export OPENAI_API_KEY='your_key_here'")
        sys.exit(1)


def require_openai_compatible_credentials(model_info: dict, api_key: str, base_url: str, dry_run: bool) -> None:
    if not dry_run and not api_key:
        print(f"[ERROR] Missing API key for model '{model_info['label']}'.")
        print("        Set model_info['api_key'], model_info['api_key_env'], or OPENAI_COMPATIBLE_API_KEY.")
        sys.exit(1)
    if not dry_run and not base_url:
        print(f"[ERROR] Missing base URL for model '{model_info['label']}'.")
        print("        Set model_info['base_url'], model_info['base_url_env'], or OPENAI_COMPATIBLE_BASE_URL.")
        sys.exit(1)
