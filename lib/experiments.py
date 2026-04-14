"""Shared experiment helpers used by generation and repair scripts."""

import json
import re
from pathlib import Path


class RateLimitError(Exception):
    """Raised when API returns 429 Too Many Requests.
    
    Carries the required delay time extracted from Retry-After header.
    """
    def __init__(self, delay_seconds: int, details: str = ""):
        self.delay_seconds = delay_seconds
        self.details = details
        super().__init__(f"Rate limited, retry after {delay_seconds} seconds: {details}")


DEFAULT_RATE_LIMIT_DELAY_SECONDS = 30


def parse_retry_after_seconds(value, default_delay: int = DEFAULT_RATE_LIMIT_DELAY_SECONDS) -> int:
    """Parse a Retry-After header value into seconds, falling back safely."""
    if value is None:
        return default_delay

    if isinstance(value, (int, float)):
        delay = int(value)
        return delay if delay > 0 else default_delay

    text = str(value).strip()
    if not text:
        return default_delay

    try:
        delay = int(float(text))
    except ValueError:
        return default_delay
    return delay if delay > 0 else default_delay


def extract_retry_after_seconds(
    headers,
    default_delay: int = DEFAULT_RATE_LIMIT_DELAY_SECONDS,
) -> int:
    """Read Retry-After from a headers-like object, returning a safe default."""
    if headers is None:
        return default_delay

    if hasattr(headers, "get"):
        retry_after = headers.get("Retry-After")
        if retry_after is None:
            retry_after = headers.get("retry-after")
        return parse_retry_after_seconds(retry_after, default_delay)

    return default_delay


def build_rate_limit_error(
    *,
    headers=None,
    details: str = "",
    default_delay: int = DEFAULT_RATE_LIMIT_DELAY_SECONDS,
) -> RateLimitError:
    """Create the shared RateLimitError from provider-specific response data."""
    return RateLimitError(
        delay_seconds=extract_retry_after_seconds(headers, default_delay),
        details=details,
    )


def load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def extract_code(raw: str) -> str:
    """
    Return code-only content from a model response.
    If a fenced block is present anywhere, use its contents.
    Otherwise return stripped raw text.
    """
    text = raw.strip()
    fenced = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
        return "\n".join(inner).strip()
    return text


def load_migration_notes(from_version: str, to_version: str, migration_dir: Path, qiskit_versions: dict) -> str:
    """
    Load relevant migration notes for a version transition, concatenating
    intermediate steps when no direct file is provided.
    """
    version_order = list(qiskit_versions.keys())
    from_idx = version_order.index(from_version)
    to_idx = version_order.index(to_version)

    notes = []
    for i in range(from_idx, to_idx):
        step_from = version_order[i]
        step_to = version_order[i + 1]
        notes_file = migration_dir / f"{step_from}_to_{step_to}.txt"
        if notes_file.exists():
            notes.append(f"--- Migration: {step_from} to {step_to} ---\n{notes_file.read_text().strip()}")
        else:
            print(f"[WARN] Migration notes not found: {notes_file}  (repair quality may be reduced)")

    return "\n\n".join(notes) if notes else "(No migration notes available)"


def find_drift_failures(execution_results: list, qiskit_versions: dict) -> list:
    """
    Return execution entries that passed in their prompted version and failed
    in at least one newer version.
    """
    candidates = []
    version_order = list(qiskit_versions.keys())

    for entry in execution_results:
        prompted_v = entry["prompted_version"]
        exec_r = entry["exec_results"]

        own_result = exec_r.get(prompted_v, {})
        if own_result.get("status") != "pass":
            continue

        prompted_idx = version_order.index(prompted_v)
        for newer_v in version_order[prompted_idx + 1:]:
            newer_result = exec_r.get(newer_v, {})
            if newer_result.get("status") in ("hard_fail", "soft_fail"):
                candidates.append({
                    **entry,
                    "repair_target_version": newer_v,
                    "error_type": newer_result.get("error_type"),
                    "error_message": newer_result.get("message"),
                })

    return candidates


def build_code_lookup(generations: list) -> dict:
    """Build a lookup of generated samples keyed by problem/version/sample index."""
    code_lookup = {}
    for entry in generations:
        for idx, code in enumerate(entry["samples"]):
            code_lookup[(entry["id"], entry["version"], idx)] = code
    return code_lookup
