"""Shared control-flow runners for generation and repair experiments."""

import time
from pathlib import Path

from lib.experiments import (
    RateLimitError,
    build_code_lookup,
    extract_code,
    find_drift_failures,
    load_json,
    load_migration_notes,
    save_json,
)
from lib.prompting import build_repair_prompt


def _request_interval_seconds(model_info: dict, default_delay: float) -> float:
    """
    Return the enforced spacing between requests for a model.
    If requests_per_minute is configured, it overrides the default fixed delay.
    """
    rpm = model_info.get("requests_per_minute")
    if rpm is not None:
        try:
            rpm_value = float(rpm)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid requests_per_minute for model '{model_info.get('label', 'unknown')}': {rpm}"
            )
        if rpm_value <= 0:
            raise ValueError(
                f"requests_per_minute must be > 0 for model '{model_info.get('label', 'unknown')}'"
            )
        return 60.0 / rpm_value
    return default_delay


def _wait_for_request_slot(next_request_time: float) -> None:
    """Sleep until the next request slot if a throttle delay is still active."""
    remaining = next_request_time - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def _load_resumable_results(path: Path, label: str, artifact_name: str) -> tuple[list, set]:
    """
    Load an incremental results file for resume.
    If the file exists but is empty, treat it as an interrupted write and start
    fresh instead of crashing the whole run.
    """
    if not path.exists():
        return [], set()

    if path.stat().st_size == 0:
        print(
            f"[WARN] {label}: found empty {artifact_name} file at {path}; "
            "starting fresh."
        )
        return [], set()

    results = load_json(path)
    done_ids = {(r["id"], r["version"]) for r in results}
    return results, done_ids


def run_generation_for_model(
    *,
    model_info: dict,
    prompts: list,
    out_path,
    sample_count: int,
    call_delay: float,
    query_model,
    dry_run: bool = False,
) -> None:
    """
    Run generation for a single model across all prompts and save incremental
    output in the benchmark's standard JSON format.
    """
    label = model_info["label"]

    if out_path.exists():
        results, done_ids = _load_resumable_results(out_path, label, "generation results")
        print(f"[RESUME] {label}: {len(done_ids)} entries already done, skipping.")
    else:
        results = []
        done_ids = set()

    total = len(prompts)
    request_interval = _request_interval_seconds(model_info, call_delay)
    next_request_time = 0.0

    for i, entry in enumerate(prompts):
        key = (entry["id"], entry["version"])
        if key in done_ids:
            continue

        print(f"[{label}] {i+1}/{total}  problem={entry['id']}  version={entry['version']}")

        if dry_run:
            print(f"  PROMPT (first 120 chars): {entry['prompt'][:120]}")
            continue

        samples = []
        for k in range(sample_count):
            try:
                _wait_for_request_slot(next_request_time)
                raw = query_model(model_info, entry["prompt"])
                next_request_time = time.monotonic() + request_interval
                code = extract_code(raw)
                samples.append(code)
                print(f"  sample {k+1}/{sample_count} OK ({len(code)} chars)")
            except RateLimitError as exc:
                print(f"  sample {k+1}/{sample_count} RATE LIMITED: {exc}")
                print(f"  [STOP] Rate limit hit, stopping generation for {label}")
                return
            except Exception as exc:
                next_request_time = time.monotonic() + request_interval
                print(f"  sample {k+1}/{sample_count} ERROR: {exc}")
                samples.append(None)

        results.append({
            "id": entry["id"],
            "version": entry["version"],
            "prompt": entry["prompt"],
            "samples": samples,
        })
        save_json(out_path, results)

    if not dry_run:
        print(f"[DONE] {label} -> {out_path}  ({len(results)} entries)")


def run_repair_for_model(
    *,
    model_info: dict,
    generations_dir,
    execution_results_dir,
    repairs_dir,
    anchored_file,
    migration_dir,
    qiskit_versions: dict,
    load_test_call_lookup,
    run_snippet_in_env,
    query_model,
    call_delay: float,
    target_version_filter: str | None = None,
    missing_generation_hint: str | None = None,
) -> None:
    """
    Run repair generation for drift failures for a single model and save
    incremental results in the benchmark's standard repair JSON format.
    """
    label = model_info["label"]
    exec_path = execution_results_dir / f"{label}_execution.json"
    gen_path = generations_dir / f"{label}.json"
    repair_path = repairs_dir / f"{label}_repair.json"

    if not exec_path.exists():
        print(f"[SKIP] No execution results for '{label}'. Run run_execution.py first.")
        return
    if not gen_path.exists():
        hint = f" {missing_generation_hint}" if missing_generation_hint else ""
        print(f"[SKIP] No generation outputs for '{label}'.{hint}")
        return

    execution_results = load_json(exec_path)
    generations = load_json(gen_path)
    test_lookup = load_test_call_lookup(anchored_file)
    code_lookup = build_code_lookup(generations)

    candidates = find_drift_failures(execution_results, qiskit_versions)
    if target_version_filter:
        candidates = [c for c in candidates if c["repair_target_version"] == target_version_filter]

    print(f"[INFO] {label}: {len(candidates)} drift failures to repair.")

    if repair_path.exists():
        repair_results = load_json(repair_path)
        done_keys = {
            (r["id"], r["prompted_version"], r["sample_index"], r["repair_target_version"])
            for r in repair_results
        }
    else:
        repair_results = []
        done_keys = set()

    request_interval = _request_interval_seconds(model_info, call_delay)
    next_request_time = 0.0

    for i, candidate in enumerate(candidates):
        key = (
            candidate["id"],
            candidate["prompted_version"],
            candidate["sample_index"],
            candidate["repair_target_version"],
        )
        if key in done_keys:
            continue

        original_code = code_lookup.get(
            (candidate["id"], candidate["prompted_version"], candidate["sample_index"])
        )
        if not original_code:
            print(f"[WARN] Could not find original code for {key}, skipping.")
            continue

        target_v = candidate["repair_target_version"]
        prompted_v = candidate["prompted_version"]
        migration = load_migration_notes(prompted_v, target_v, migration_dir, qiskit_versions)

        repair_prompt = build_repair_prompt(
            original_code=original_code,
            prompted_version=prompted_v,
            target_version=target_v,
            error_type=candidate["error_type"],
            error_message=candidate["error_message"],
            migration_notes=migration,
        )

        print(
            f"[{label}] {i+1}/{len(candidates)}  "
            f"problem={candidate['id']}  {prompted_v}->{target_v}  sample={candidate['sample_index']}"
        )

        try:
            _wait_for_request_slot(next_request_time)
            raw = query_model(model_info, repair_prompt)
            next_request_time = time.monotonic() + request_interval
            repaired_code = extract_code(raw)
        except RateLimitError as exc:
            print(f"  [RATE LIMITED] Repair API call failed: {exc}")
            print(f"  [STOP] Rate limit hit, stopping repair for {label}")
            return
        except Exception as exc:
            next_request_time = time.monotonic() + request_interval
            print(f"  [ERROR] Repair API call failed: {exc}")
            repaired_code = None

        if repaired_code:
            conda_env = qiskit_versions[target_v]["conda_env"]
            meta = test_lookup.get(candidate["id"], {})
            test_call = meta.get("test_call", "")
            entry_point = meta.get("entry_point", "")
            exec_result = run_snippet_in_env(repaired_code, test_call, entry_point, conda_env)
            print(f"  repair exec in {conda_env}: {exec_result['status'].upper()}")
        else:
            exec_result = {"status": "skipped", "error_type": None, "message": "Repair generation failed"}

        repair_results.append({
            "id": candidate["id"],
            "prompted_version": prompted_v,
            "sample_index": candidate["sample_index"],
            "repair_target_version": target_v,
            "original_error_type": candidate["error_type"],
            "repaired_code": repaired_code,
            "repair_exec_result": exec_result,
        })
        save_json(repair_path, repair_results)

    print(f"[DONE] {label} repair -> {repair_path}")
