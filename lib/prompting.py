"""Shared prompt templates for generation and repair workflows."""

GENERATION_SYSTEM_PROMPT = (
    "You are an expert quantum computing engineer. "
    "When asked to write quantum code, return only a complete, runnable Python "
    "code block using the exact SDK version specified. "
    "Do not include markdown fences or explanations outside the code."
)

REPAIR_SYSTEM_PROMPT = (
    "You are an expert quantum computing engineer. "
    "You repair Python code across Qiskit SDK versions using migration notes. "
    "Return only the complete, corrected Python code. "
    "Do not include markdown fences or explanations outside the code."
)


def build_repair_prompt(
    original_code: str,
    prompted_version: str,
    target_version: str,
    error_type: str,
    error_message: str,
    migration_notes: str,
) -> str:
    return f"""\
The following Python code was written for Qiskit {prompted_version} but fails \
when run under Qiskit {target_version}.

Error type:    {error_type}
Error message: {error_message}

--- Broken Code ---
{original_code}

--- Relevant Migration Notes ({prompted_version} -> {target_version}) ---
{migration_notes}

--- Task ---
Rewrite the code above to be fully compatible with Qiskit {target_version}.
Fix all deprecated or removed APIs based on the migration notes.
Return only the complete, corrected Python code with no explanation outside the code.
"""
