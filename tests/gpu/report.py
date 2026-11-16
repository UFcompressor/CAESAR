"""Publish a saved cluster result to the exact tested commit using authenticated gh."""

import argparse
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--repo", required=True, help="OWNER/REPO")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    sha, mode, backend = result["sha"], result["mode"], result["backend"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Result must contain a full commit SHA")
    if mode not in ("correctness", "performance") or backend not in ("nvidia", "amd"):
        raise ValueError("Invalid result mode/backend")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
        raise ValueError("Expected repository OWNER/REPO")
    passed = result["passed"] is True
    if passed and (
        not result["tests"] or not all(t["passed"] is True for t in result["tests"])
    ):
        raise ValueError("Successful result must contain passing tests")
    if passed and mode == "performance":
        from run_suite import PERFORMANCE_BYTES, PERFORMANCE_SHAPE, performance_summary

        if result["shape"] != PERFORMANCE_SHAPE or result["error_bound"] != 0.1:
            raise ValueError("Unexpected performance workload")
        samples = {}
        for operation in ("compression", "decompression"):
            samples[operation] = []
            for item in result["performance"][operation]["runs"]:
                seconds = item["seconds"]
                if seconds <= 0:
                    raise ValueError("Invalid duration")
                samples[operation].append(
                    {
                        "seconds": seconds,
                        "gb_per_second": PERFORMANCE_BYTES / seconds / 1e9,
                    }
                )
        if not all(v["passed"] for v in performance_summary(samples).values()):
            raise ValueError("Performance results do not meet the thresholds")
    payload = {
        "state": "success" if passed else "failure",
        "context": f"gpu-tests/{backend}/{mode}",
        "description": f"Slurm GPU {mode} {'passed' if passed else 'failed'} (job {result.get('slurm_job_id')})"[
            :140
        ],
    }
    print(json.dumps({"sha": sha, **payload}, indent=2))
    if not args.dry_run:
        subprocess.run(
            [
                "gh",
                "api",
                "--method",
                "POST",
                f"repos/{args.repo}/statuses/{sha}",
                "--input",
                "-",
            ],
            input=json.dumps(payload),
            text=True,
            check=True,
        )


if __name__ == "__main__":
    main()
