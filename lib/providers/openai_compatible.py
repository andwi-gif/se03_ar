import json
import urllib.error
import urllib.request

from lib.experiments import build_rate_limit_error


def _normalize_base_url(base_url: str, default_base_url: str = "") -> str:
    text = (base_url or "").strip().rstrip("/")
    return text or default_base_url


def create_chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    extra_body: dict | None = None,
    default_base_url: str = "",
    extra_headers: dict[str, str] | None = None,
) -> str:
    """Call an OpenAI-compatible chat completions API and return text content."""
    normalized_base_url = _normalize_base_url(base_url, default_base_url)
    if not normalized_base_url:
        raise RuntimeError("Missing base URL for OpenAI-compatible provider.")

    url = f"{normalized_base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        payload.update(extra_body)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Some providers sit behind bot protection and reject urllib's default
        # Python-urllib/x.y user agent even when curl works with the same key.
        "User-Agent": "curl/8.5.0",
        **(extra_headers or {}),
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace").strip()

        if exc.code == 429:
            raise build_rate_limit_error(headers=exc.headers, details=details) from exc

        raise RuntimeError(f"HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"Malformed OpenAI-compatible response: {body}")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        if text_parts:
            return "".join(text_parts)

    raise RuntimeError(f"Missing message content in OpenAI-compatible response: {body}")
