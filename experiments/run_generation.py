import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.model_families import determine_families


GENERATION_RUNNERS = {
    "groq": Path(__file__).resolve().parent / "generation" / "run_generation_groq.py",
    "codex": Path(__file__).resolve().parent / "generation" / "run_generation_codex.py",
    "openai": Path(__file__).resolve().parent / "generation" / "run_generation_openai.py",
    "anthropic": Path(__file__).resolve().parent / "generation" / "run_generation_anthropic.py",
    "openrouter": Path(__file__).resolve().parent / "generation" / "run_generation_openrouter.py",
    "openai_compatible": Path(__file__).resolve().parent / "generation" / "run_generation_openai_compatible.py",
}

def run_family(family: str, args: argparse.Namespace) -> None:
    cmd = [sys.executable, str(GENERATION_RUNNERS[family])]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.dry_run:
        cmd.append("--dry-run")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run generation across all configured model families"
    )
    parser.add_argument("--model", help="Run only this model label")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling APIs")
    args = parser.parse_args()

    for family in determine_families(args.model):
        print(f"\n{'='*60}")
        print(f" Generation family: {family}")
        print(f"{'='*60}", flush=True)
        run_family(family, args)


if __name__ == "__main__":
    main()
