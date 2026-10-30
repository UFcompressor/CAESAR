"""Export one verified checkpoint into a complete CAESAR installation."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from model_registry import (
    ARCHITECTURE, DEPTH_ARCHITECTURES, COMPONENTS, atomic_json,
    read_selection, sha256, validate_model, write_installation,
)


def compile_installation(device, selection, output):
    selection = Path(selection).resolve()
    model, _ = read_selection(selection)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # A second installer must never interleave publication of a different model.
    lock = output.with_name(output.name + ".lock")
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            f"Installation is locked: {lock}. If an installer crashed, remove this directory before retrying."
        ) from error
    try:
        with tempfile.TemporaryDirectory(
            dir=output.parent, prefix=".caesar-export-"
        ) as temporary:
            stage = Path(temporary) / "exported_model"
            stage.mkdir()
            env = dict(
                os.environ,
                CAESAR_MODEL_SELECTION=str(selection),
                CAESAR_EXPORT_DIR=str(stage),
            )
            # Freeze selection for all three subprocesses, even if another download occurs.
            from model_registry import atomic_json

            frozen = Path(temporary) / "checkpoint"
            frozen.mkdir()
            _, checkpoint = read_selection(selection)
            shutil.copyfile(checkpoint, frozen / model["filename"])
            atomic_json(
                frozen / "selected_model.json", {"schema_version": 1, "model": model}
            )
            env["CAESAR_MODEL_SELECTION"] = str(frozen / "selected_model.json")
            for component in COMPONENTS:
                script = Path(__file__).with_name(
                    component.replace("caesar_", "CAESAR_", 1) + ".py"
                )
                subprocess.run(
                    [sys.executable, str(script), device], env=env, check=True
                )
            write_installation(stage, model, device)
            backup = Path(temporary) / "previous"
            if output.exists():
                output.rename(backup)
            try:
                stage.rename(output)
            except BaseException:
                if backup.exists():
                    backup.rename(output)
                raise
        print(f"Installed {model['id']} for {device} in {output}")
    finally:
        lock.rmdir()


def compile_local_checkpoint(device, checkpoint, architecture, output):
    """Export experiment weights directly, without a catalog or registration."""
    checkpoint = Path(checkpoint).resolve()
    digest = sha256(checkpoint)
    model = validate_model(dict(
        name=checkpoint.stem, filename=checkpoint.name, sha256=digest,
        id=f"local:{architecture}@sha256:{digest}", registration_id=0,
        architecture=architecture, min_dims=3, max_dims=5,
        description="Local experiment checkpoint", url="",
    ))
    with tempfile.TemporaryDirectory(prefix="caesar-local-") as temporary:
        root = Path(temporary)
        shutil.copyfile(checkpoint, root / model["filename"])
        selection = root / "selected_model.json"
        atomic_json(selection, {"schema_version": 1, "model": model})
        compile_installation(device, selection, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", choices=("cpu", "cuda", "mps", "xpu"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--selection", default="pretrained/selected_model.json")
    source.add_argument("--checkpoint", type=Path, help="Export local weights without UFL registration")
    parser.add_argument("--architecture", choices=[ARCHITECTURE, *DEPTH_ARCHITECTURES])
    parser.add_argument("--output", default="exported_model")
    args = parser.parse_args()
    if args.checkpoint:
        if not args.architecture:
            parser.error("--checkpoint requires --architecture matching the training code")
        compile_local_checkpoint(args.device, args.checkpoint, args.architecture, args.output)
    else:
        if args.architecture:
            parser.error("--architecture is only used with --checkpoint")
        compile_installation(args.device, args.selection, args.output)


if __name__ == "__main__":
    main()
