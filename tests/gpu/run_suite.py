"""Manual cluster GPU checks. Results are reported to GitHub separately."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

PERFORMANCE_SHAPE = "256,256,256,256"
PERFORMANCE_BYTES = 256**4 * 4


def metric(log, operation, expected_bytes):
    matches = re.findall(
        rf"^CAESAR_BENCHMARK {operation} (\S+) (\d+)$", log, re.MULTILINE
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one {operation} timing record")
    seconds, size = float(matches[0][0]), int(matches[0][1])
    if not math.isfinite(seconds) or seconds <= 0 or size != expected_bytes:
        raise ValueError(f"Invalid {operation} timing or byte count")
    return {"seconds": seconds, "gb_per_second": size / seconds / 1e9}


def performance_summary(samples):
    summary = {}
    for operation, minimum in (("compression", 1.3), ("decompression", 3.5)):
        values = samples[operation]
        if len(values) != 5:
            raise ValueError(f"Expected five {operation} runs")
        best = max(item["gb_per_second"] for item in values)
        summary[operation] = {
            "runs": values,
            "best_gb_per_second": best,
            "minimum_gb_per_second": minimum,
            "passed": best >= minimum,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("performance", "correctness"))
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    build = args.build.resolve()
    sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    work = build / "tests" / "gpu-results" / f"{sha[:12]}-{args.mode}-{stamp}"
    work.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "sha": sha,
        "mode": args.mode,
        "passed": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "work_directory": str(work),
        "tests": [],
    }
    env = dict(os.environ, CAESAR_REQUIRE_GPU="1")

    def run(name, command):
        with (work / f"{name}.log").open("w") as log:
            process = subprocess.Popen(
                [str(arg) for arg in command],
                cwd=work,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
        output = (work / f"{name}.log").read_text()
        # Some legacy GAE tests print failures without a nonzero exit status.
        printed_failure = re.search(
            r"\bFAIL(?:ED)?\b|failed with exception", output, re.IGNORECASE
        )
        result["tests"].append(
            {
                "name": name,
                "exit_code": code,
                "passed": code == 0 and not printed_failure,
            }
        )
        if code or printed_failure:
            raise RuntimeError(f"{name} failed; see {work / (name + '.log')}")
        return output

    def executable(name):
        path = build / "tests" / name
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"Required GPU test was not built: {path}")
        result.setdefault("binary_sha256", {})[name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        return path

    try:
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        )
        if dirty.strip():
            raise RuntimeError(
                "Commit tracked changes before running a reportable GPU suite"
            )
        if sha != env.get("CAESAR_GPU_SHA", sha):
            raise RuntimeError(
                "Checkout changed during the build; rebuild the intended commit"
            )
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm GPU required; CPU fallback is forbidden")
        backend = "amd" if torch.version.hip else "nvidia"
        result["backend"] = backend
        result["environment"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "hip": torch.version.hip,
            "gpu": torch.cuda.get_device_name(0),
            "model_directory": env["CAESAR_MODEL_DIR"],
        }
        data = Path(env["CAESAR_GPU_DATA"]).resolve(strict=True)
        result["dataset"] = {"path": str(data), "bytes": data.stat().st_size}
        cli = build / "CAESAR" / "caesar"
        model_info = run("model-info", [cli, "--model-info"])
        expected_device = r"(?:cuda|rocm)" if backend == "amd" else "cuda"
        if not re.search(rf"Device\s*:\s*{expected_device}\b", model_info):
            raise RuntimeError("Exported model is not configured for this GPU backend")
        if args.mode == "performance":
            if data.stat().st_size < PERFORMANCE_BYTES:
                raise RuntimeError(
                    f"Expected at least {PERFORMANCE_BYTES} bytes (256^4 float32), "
                    f"got {data.stat().st_size}; input will not be resized or overwritten"
                )
            env.update(
                CAESAR_BENCH_DATA=str(data), CAESAR_BENCH_SHAPE=PERFORMANCE_SHAPE
            )
            result.update(
                shape=PERFORMANCE_SHAPE, error_bound=0.1, units="decimal GB/s"
            )
            samples = {"compression": [], "decompression": []}
            compressor, decompressor = executable("test_caesarC"), executable(
                "test_caesarD"
            )
            for iteration in range(1, 6):
                for operation, command in (
                    ("compression", [compressor, "0.1"]),
                    ("decompression", [decompressor]),
                ):
                    output = run(f"{operation}-{iteration}", command)
                    samples[operation].append(
                        metric(output, operation, PERFORMANCE_BYTES)
                    )
            result["performance"] = performance_summary(samples)
            result["passed"] = all(
                item["passed"] for item in result["performance"].values()
            )
        else:
            # Use a small prefix for correctness, preserving the full performance dataset.
            small_bytes = 20 * 256 * 256 * 4
            with data.open("rb") as source:
                small = source.read(small_bytes)
            if len(small) != small_bytes:
                raise RuntimeError("Dataset is too short for correctness round trips")
            (work / "TCf48.bin.f32").write_bytes(small)
            env.update(CAESAR_BENCH_DATA=str(work / "TCf48.bin.f32"))
            env.pop("CAESAR_BENCH_SHAPE", None)
            names = [
                "test_model_metadata",
                "test_model_cache",
                "test_padding",
                "test_serialization",
                "test_rans_kernels",
                "test_runGaeCuda",
                "test_LBRC",
                "test_NGLR",
                "test_caesarCD",
                "test_caesarC",
                "test_caesarD",
            ]
            if backend == "nvidia":
                names.append("test_rans_cuda")
                # nvCOMP is optional at build time; explicitly record its absence.
                if (build / "tests" / "test_nvcomp").exists():
                    names.append("test_nvcomp")
                else:
                    result["skipped"] = {
                        "test_nvcomp": "optional nvCOMP target not built"
                    }
            else:
                result["skipped"] = {
                    "test_rans_cuda": "NVIDIA-only",
                    "test_nvcomp": "NVIDIA-only",
                }
            for name in names:
                run(name, [executable(name)])
            run("nglr-gpu-workload", [executable("test_NGLR"), "--gpu-workload"])
            env.update(
                CAESAR_TEST_EXE=str(cli),
                CAESAR_TEST_DATA=str(work / "TCf48.bin.f32"),
                CAESAR_TEST_SHAPE="1,1,20,256,256",
            )
            run("cli-round-trips", ["bash", root / "tests" / "test_CLI.sh"])
            result["passed"] = True
    except Exception as error:
        result["error"] = str(error)
        print(f"ERROR: {error}", file=sys.stderr)
    finally:
        report = work / "result.json"
        report.write_text(json.dumps(result, indent=2) + "\n")
        print(f"GPU RESULT: {report}")
        print("Report from a network-connected node using:")
        print(f"python3 {root / 'tests/gpu/report.py'} {report} --repo OWNER/REPO")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
