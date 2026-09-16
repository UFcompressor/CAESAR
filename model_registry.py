"""Checkpoint catalog, verified downloads, and installation metadata (stdlib only)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import urlopen

DEFAULT_CATALOG = Path(__file__).with_name("model_catalog.json")
ARCHITECTURE = "caesar-bcrn-v1"
COMPONENTS = ("caesar_compressor", "caesar_hyper_decompressor", "caesar_decompressor")
TABLES = tuple(
    f"{kind}_{table}.bin"
    for kind in ("vbr", "gs")
    for table in ("quantized_cdf", "cdf_length", "offset")
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_model(model):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model["name"]):
        raise ValueError("Invalid model name")
    if not re.fullmatch(r"[0-9a-f]{64}", model["sha256"]):
        raise ValueError("Invalid checkpoint SHA-256")
    if type(model.get("registration_id")) is not int or model["registration_id"] <= 0:
        raise ValueError("A positive UFL registration number is required")
    if model["id"] != f'ufl:{model["registration_id"]}@sha256:{model["sha256"]}':
        raise ValueError(
            "Model identity does not match registration number and checkpoint hash"
        )
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.pt", model["filename"]):
        raise ValueError("Invalid checkpoint filename")
    if not (
        type(model["min_dims"]) is int
        and type(model["max_dims"]) is int
        and 2 <= model["min_dims"] <= model["max_dims"] <= 5
    ):
        raise ValueError("Supported dimensions must be within 2D–5D")
    if not model.get("description") or model.get("architecture") != ARCHITECTURE:
        raise ValueError("Missing description or unsupported model architecture")
    if not model["url"].startswith("https://"):
        raise ValueError("Checkpoint URL must use HTTPS")
    return model


def load_catalog(path=DEFAULT_CATALOG):
    path = str(path)
    if path.startswith("https://"):
        with urlopen(path, timeout=60) as response:
            catalog = json.load(response)
    else:
        catalog = json.loads(Path(path).read_text())
    if catalog["schema_version"] != 1:
        raise ValueError("Unsupported model catalog schema")
    names, identities, numbers, hashes = set(), set(), set(), set()
    for model in catalog["models"]:
        validate_model(model)
        if (
            model["name"] in names
            or model["id"] in identities
            or model["registration_id"] in numbers
            or model["sha256"] in hashes
        ):
            raise ValueError(
                "Duplicate model name, registration number, checkpoint hash, or identity"
            )
        names.add(model["name"])
        identities.add(model["id"])
        numbers.add(model["registration_id"])
        hashes.add(model["sha256"])
    if catalog["default_model"] not in names:
        raise ValueError("Default model is not registered")
    return catalog


def select_model(catalog, requested=None):
    requested = requested or catalog["default_model"]
    for model in catalog["models"]:
        if requested in (model["name"], model["id"]):
            return model
    raise ValueError(f"Required model {requested!r} is not registered in this catalog")


def verify_checkpoint(path, model):
    actual = sha256(path)
    if actual != model["sha256"]:
        raise ValueError(
            f"Required model {model['id']}; checkpoint {path} has SHA-256 {actual}"
        )


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".metadata-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def download(model, destination, source_dir=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / model["filename"]
    if target.exists() and sha256(target) == model["sha256"]:
        pass
    else:
        fd, temporary = tempfile.mkstemp(dir=destination, prefix=".checkpoint-")
        try:
            with os.fdopen(fd, "wb") as output:
                source = (
                    open(Path(source_dir) / model["filename"], "rb")
                    if source_dir
                    else urlopen(model["url"], timeout=60)
                )
                with source:
                    shutil.copyfileobj(source, output)
            verify_checkpoint(temporary, model)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    atomic_json(
        destination / "selected_model.json", {"schema_version": 1, "model": model}
    )
    return target


def read_selection(path):
    path = Path(path).resolve()
    selected = json.loads(path.read_text())
    if selected["schema_version"] != 1:
        raise ValueError("Unsupported selection schema")
    model = validate_model(selected["model"])
    # A selection file is not a registration: independently consult our catalog.
    registered = select_model(load_catalog(), model["id"])
    if model != registered:
        differences = {
            key: (repr(model.get(key)), repr(registered.get(key)))
            for key in sorted(set(model) | set(registered))
            if model.get(key) != registered.get(key)
        }
        raise ValueError(
            "Selection does not match its registered UFL catalog entry: "
            + repr(differences)
        )
    checkpoint = path.parent / model["filename"]
    verify_checkpoint(checkpoint, model)
    return model, checkpoint


def export_context():
    """All exporters must receive one verified selection and a staging directory."""
    selection = os.environ.get("CAESAR_MODEL_SELECTION")
    output = os.environ.get("CAESAR_EXPORT_DIR")
    if not selection or not output:
        raise RuntimeError(
            "Run compile_model.py <device> to export a complete installation"
        )
    model, checkpoint = read_selection(selection)
    return model, checkpoint, Path(output)


def write_installation(output, model, device):
    output = Path(output)
    for filename in (*[f"{c}.pt2" for c in COMPONENTS], *TABLES):
        if not (output / filename).is_file() or (output / filename).stat().st_size == 0:
            raise ValueError(f"Incomplete model installation: {filename}")
    fields = dict(
        schema_version=1,
        registration_id=model["registration_id"],
        model_name=model["name"],
        model_id=model["id"],
        checkpoint_sha256=model["sha256"],
        checkpoint_file=model["filename"],
        architecture=model["architecture"],
        min_dims=model["min_dims"],
        max_dims=model["max_dims"],
        device=device,
    )
    (output / "model_metadata.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in fields.items())
    )


def validate_installation(directory):
    directory = Path(directory)
    fields = {}
    for line in (directory / "model_metadata.txt").read_text().splitlines():
        key, separator, value = line.partition("=")
        if not separator or not value or key in fields:
            raise ValueError("Malformed installation metadata")
        fields[key] = value
    model = select_model(load_catalog(), fields.get("model_id", "unregistered"))
    expected = dict(
        schema_version="1",
        registration_id=str(model["registration_id"]),
        model_name=model["name"],
        model_id=model["id"],
        checkpoint_sha256=model["sha256"],
        checkpoint_file=model["filename"],
        architecture=model["architecture"],
        min_dims=str(model["min_dims"]),
        max_dims=str(model["max_dims"]),
        device=fields.get("device"),
    )
    if fields != expected or fields.get("device") not in ("cpu", "cuda", "mps", "xpu"):
        raise ValueError("Installation metadata does not match a registered UFL model")
    for filename in (*[f"{c}.pt2" for c in COMPONENTS], *TABLES):
        if (
            not (directory / filename).is_file()
            or (directory / filename).stat().st_size == 0
        ):
            raise ValueError(f"Incomplete model installation: {filename}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", nargs="?", help="Registered name or exact ufl:number@sha256:hash"
    )
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--validate-installation", metavar="DIRECTORY")
    parser.add_argument("--output", default="pretrained")
    parser.add_argument(
        "--source-dir", help="Read checkpoints from a local UFL_MODELS checkout"
    )
    args = parser.parse_args()
    if args.validate_installation:
        model = validate_installation(args.validate_installation)
        print(f"Registered installation: {model['id']}")
        return
    catalog = load_catalog(args.catalog)
    if args.list:
        for model in catalog["models"]:
            print(
                f"{model.get('display_name', model['name'])} [{model['name']}] ({model['min_dims']}D–{model['max_dims']}D): {model['description']}\n  {model['id']}"
            )
        return
    model = select_model(catalog, args.model)
    target = download(model, args.output, args.source_dir)
    print(f"Verified {model['id']}\nSaved checkpoint: {target}")


if __name__ == "__main__":
    main()
