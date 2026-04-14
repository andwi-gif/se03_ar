"""Shared family-runner helpers."""

import sys

from lib.experiments import load_json
from lib.runners import run_generation_for_model, run_repair_for_model


def load_generation_prompts(anchored_file):
    if not anchored_file.exists():
        print(f"[ERROR] Anchored prompts not found: {anchored_file}")
        print("        Run: python prompts/prepare_prompts.py")
        sys.exit(1)

    prompts = load_json(anchored_file)
    print(f"[INFO] Loaded {len(prompts)} anchored prompts.")
    return prompts


def select_models(models: list[dict], requested_label: str | None, config_name: str) -> list[dict]:
    if not models:
        print(f"[ERROR] {config_name} is empty in config.py.")
        print("        Add at least one model entry to run this script.")
        sys.exit(1)

    selected = [m for m in models if not requested_label or m["label"] == requested_label]
    if not selected:
        print(f"[ERROR] No model found with label '{requested_label}' in {config_name}.")
        sys.exit(1)
    return selected


def print_generation_header(model_info: dict, describe_model=None) -> None:
    descriptor = describe_model(model_info) if describe_model else model_info["id"]
    print(f"\n{'='*60}")
    print(f" Model: {model_info['label']}  ({descriptor})")
    print(f"{'='*60}")


def print_repair_header(model_info: dict) -> None:
    print(f"\n{'='*60}")
    print(f" Repair pass for: {model_info['label']}")
    print(f"{'='*60}")


def run_generation_family(
    *,
    models: list[dict],
    prompts: list,
    generations_dir,
    sample_count: int,
    call_delay: float,
    dry_run: bool,
    make_query_model,
    describe_model=None,
    before_model=None,
) -> None:
    generations_dir.mkdir(parents=True, exist_ok=True)

    for model_info in models:
        if before_model:
            before_model(model_info, dry_run)
        print_generation_header(model_info, describe_model=describe_model)
        run_generation_for_model(
            model_info=model_info,
            prompts=prompts,
            out_path=generations_dir / f"{model_info['label']}.json",
            sample_count=sample_count,
            call_delay=call_delay,
            dry_run=dry_run,
            query_model=make_query_model(model_info, dry_run),
        )


def run_repair_family(
    *,
    models: list[dict],
    generations_dir,
    execution_results_dir,
    repairs_dir,
    anchored_file,
    migration_dir,
    qiskit_versions: dict,
    load_test_call_lookup,
    run_snippet_in_env,
    make_query_model,
    call_delay: float,
    target_version_filter: str | None,
    missing_generation_hint: str,
    before_model=None,
) -> None:
    for model_info in models:
        if before_model:
            before_model(model_info)
        print_repair_header(model_info)
        run_repair_for_model(
            model_info=model_info,
            generations_dir=generations_dir,
            execution_results_dir=execution_results_dir,
            repairs_dir=repairs_dir,
            anchored_file=anchored_file,
            migration_dir=migration_dir,
            qiskit_versions=qiskit_versions,
            load_test_call_lookup=load_test_call_lookup,
            run_snippet_in_env=run_snippet_in_env,
            query_model=make_query_model(model_info, False),
            call_delay=call_delay,
            target_version_filter=target_version_filter,
            missing_generation_hint=missing_generation_hint,
        )
