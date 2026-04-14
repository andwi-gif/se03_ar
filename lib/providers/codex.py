"""Codex CLI helpers shared by generation and repair scripts."""

import subprocess
import tempfile
from lib.experiments import RateLimitError


CODEX_TIMEOUT = 120


def check_codex_installed() -> bool:
    """Return True if the `codex` CLI is available on PATH."""
    try:
        result = subprocess.run(
            ["codex", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def call_codex(prompt: str, model_info: dict, instructions: str, temp_prefix: str) -> str | None:
    """Call the Codex CLI and return the last assistant message text."""
    full_prompt = (
        f"{instructions}\n\n"
        "Task:\n"
        f"{prompt}"
    )

    try:
        with tempfile.TemporaryDirectory(prefix=temp_prefix) as workspace_dir:
            with tempfile.NamedTemporaryFile("r+", suffix=".txt") as output_file:
                cmd = [
                    "codex",
                    "exec",
                    "--model", model_info["id"],
                    "--config", f'model_reasoning_effort="{model_info["reasoning_effort"]}"',
                    "--sandbox", "read-only",
                    "--skip-git-repo-check",
                    "--cd", workspace_dir,
                    "--color", "never",
                    "--ephemeral",
                    "--output-last-message", output_file.name,
                    "-",
                ]

                result = subprocess.run(
                    cmd,
                    input=full_prompt,
                    capture_output=True,
                    text=True,
                    timeout=CODEX_TIMEOUT,
                )
                if result.returncode != 0:
                    stderr = result.stderr.strip()
                    print(f"  [WARN] Codex CLI exited with code {result.returncode}: {stderr[:400]}")

                    # Check for rate limit errors in stderr
                    stderr_lower = stderr.lower()
                    if "rate limit" in stderr_lower or "429" in stderr_lower or "too many requests" in stderr_lower or "quota exceeded" in stderr_lower:
                        # Try to extract delay information from the error message
                        delay_seconds = 30  # default delay
                        # Look for patterns like "retry after X seconds" or "wait X seconds"
                        import re
                        retry_match = re.search(r'(retry after|wait) (\d+) (seconds?|minutes?|hours?)', stderr_lower)
                        if retry_match:
                            delay_value = int(retry_match.group(2))
                            time_unit = retry_match.group(3)
                            if 'minute' in time_unit:
                                delay_seconds = delay_value * 60
                            elif 'hour' in time_unit:
                                delay_seconds = delay_value * 3600
                            else:
                                delay_seconds = delay_value

                        raise RateLimitError(delay_seconds=delay_seconds, details=stderr[:200])

                    return None

                output_file.seek(0)
                return output_file.read()
    except subprocess.TimeoutExpired:
        print(f"  [WARN] Codex CLI timed out after {CODEX_TIMEOUT}s")
        return None
    except FileNotFoundError:
        print("  [ERROR] `codex` command not found. Did you run `npm install -g @openai/codex`?")
        return None
