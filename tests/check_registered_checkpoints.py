"""Strict-load every registered checkpoint into each of the three export models."""

import argparse
import ast
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import model_registry as registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", help="Local UFL_MODELS checkout; otherwise download"
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        checkpoints = []
        for entry in registry.load_catalog()["models"]:
            path = registry.download(
                entry, Path(temporary) / entry["name"], args.source_dir
            )
            raw = torch.load(path, map_location="cpu", weights_only=True)
            normalized = {}
            for name, tensor in raw.items():
                while name.startswith(("module.", "_orig_mod.")):
                    name = name.split(".", 1)[1]
                if name in normalized:
                    raise ValueError("Duplicate normalized state_dict key")
                normalized[name] = tensor
            checkpoints.append((entry, normalized))
        for component in registry.COMPONENTS:
            path = ROOT / (component.replace("caesar_", "CAESAR_", 1) + ".py")
            tree = ast.parse(path.read_text(), filename=str(path))
            # Evaluate the real exporter definitions without executing its compile CLI.
            definitions = [
                node
                for node in tree.body
                if isinstance(
                    node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)
                )
            ]
            namespace = {"__name__": "checkpoint_validation"}
            exec(
                compile(
                    ast.Module(body=definitions, type_ignores=[]), str(path), "exec"
                ),
                namespace,
            )
            model = namespace["CompressorMix"](
                dim=16,
                dim_mults=[1, 2, 3, 4],
                reverse_dim_mults=[4, 3, 2],
                hyper_dims_mults=[4, 4, 4],
                channels=1,
                out_channels=1,
                d3=True,
                sr_dim=16,
                device="cpu",
            )
            for entry, state in checkpoints:
                model.load_state_dict(state, strict=True)
                print(
                    f"{component}: {entry['name']} ({entry['id']}) strict load passed"
                )


if __name__ == "__main__":
    main()
