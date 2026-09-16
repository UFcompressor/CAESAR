"""Export one verified checkpoint into a complete CAESAR installation."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from model_registry import COMPONENTS, read_selection, write_installation


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", choices=("cpu", "cuda", "mps", "xpu"))
    parser.add_argument("--selection", default="pretrained/selected_model.json")
    parser.add_argument("--output", default="exported_model")
    args = parser.parse_args()
    compile_installation(args.device, args.selection, args.output)


if __name__ == "__main__":
    main()
