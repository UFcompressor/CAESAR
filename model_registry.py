"""Checkpoint catalog, verified downloads, and installation metadata (stdlib only)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.request import urlopen

DEFAULT_CATALOG = (
    "https://raw.githubusercontent.com/UFcompressor/UFL_MODELS/main/model_catalog.json"
)

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


LOCAL_WARNING = (
    "WARNING: THIS IS NOT AN OFFICIALLY SUPPORTED MODEL AND CAN BE UNSAFE. "
    "REGISTER THIS MODEL ON A LOGIN NODE OR ANOTHER NETWORKED MACHINE "
    "USING register_model.py BEFORE OFFICIAL USE."
)


def warn_local():
    print(LOCAL_WARNING, file=sys.stderr)
    print(
        "HOW TO SUBMIT TO UFL (ON A NETWORKED LOGIN NODE):\n"
        "  gh auth login\n"
        "  python3 register_model.py add --name MY_MODEL --display-name 'My model' "
        "--description 'Training data, intended use, and provenance' "
        "--checkpoint /path/to/model.pt --min-dims 3 --max-dims 5\n"
        "Replace the example values with your checkpoint details. "
        "See docs/model_registration.md for setup. Nothing is submitted by bypass.",
        file=sys.stderr,
    )


def validate_model(model, allow_local=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model["name"]):
        raise ValueError("Invalid model name")
    if not re.fullmatch(r"[0-9a-f]{64}", model["sha256"]):
        raise ValueError("Invalid checkpoint SHA-256")
    local = (
        allow_local
        and type(model.get("registration_id")) is int
        and model["registration_id"] == 0
    )
    if not local and (
        type(model.get("registration_id")) is not int or model["registration_id"] <= 0
    ):
        raise ValueError("A positive UFL registration number is required")
    identity = (
        f'local@sha256:{model["sha256"]}'
        if local
        else f'ufl:{model["registration_id"]}@sha256:{model["sha256"]}'
    )
    if model["id"] != identity:
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
    if not local and not model["url"].startswith("https://"):
        raise ValueError("Checkpoint URL must use HTTPS")
    return model


def load_catalog(path=None, timeout=60):
    if path is None:
        path = os.environ.get("CAESAR_MODEL_CATALOG", DEFAULT_CATALOG)
    path = str(path)
    if path.startswith("https://"):
        with urlopen(path, timeout=timeout) as response:
            catalog = json.load(response)
    else:
        catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_catalog(catalog)


def validate_catalog(catalog):
    """Validate either an upstream catalog or a saved download snapshot."""
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
    for model in catalog.get("retired_models", []):
        if requested in (model["name"], model["id"]):
            raise ValueError(
                f"Required model {requested!r} has been removed and is no longer available"
            )
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
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def download(model, destination, source_dir=None, catalog=None):
    catalog = validate_catalog(catalog) if catalog is not None else load_catalog()
    if select_model(catalog, model["id"]) != model:
        raise ValueError("Selection does not match its registered UFL catalog entry")
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
        destination / "selected_model.json",
        {"schema_version": 1, "model": model, "catalog": catalog},
    )
    return target


def read_selection(path, bypass=False):
    path = Path(path).resolve()
    selected = json.loads(path.read_text(encoding="utf-8"))
    if selected["schema_version"] != 1:
        raise ValueError("Unsupported selection schema")
    model = validate_model(selected["model"], allow_local=bypass)
    checkpoint = path.parent / model["filename"]
    verify_checkpoint(checkpoint, model)
    if bypass:
        if model["registration_id"] > 0:
            try:
                # Bypass does not revoke a valid saved UFL registration.
                return read_selection(path)
            except (ValueError, KeyError, OSError):
                pass
        # Models without a matching saved registration remain explicitly local.
        model = dict(model, registration_id=0, id=f'local@sha256:{model["sha256"]}')
        warn_local()
        return model, checkpoint
    # Downloads carry their validated registration snapshot onto offline nodes.
    registered = select_model(selection_catalog(selected), model["id"])
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
    return model, checkpoint


def selection_catalog(selected):
    if "catalog" in selected:
        return validate_catalog(selected["catalog"])
    local = os.environ.get("CAESAR_MODEL_CATALOG")
    if local and not local.startswith("https://"):
        return load_catalog(local)
    # Legacy selections already contain the complete model record and hash.
    # This snapshot preserves that record; it does not query upstream approval.
    if "model" not in selected:
        raise ValueError(
            "Missing local model catalog snapshot. Recompile with compile_model.py."
        )
    model = validate_model(selected["model"])
    return {"schema_version": 1, "default_model": model["name"], "models": [model]}


def export_context():
    """Resolve a staged installation or a direct, individual exporter invocation."""
    selection = os.environ.get("CAESAR_MODEL_SELECTION")
    output = os.environ.get("CAESAR_EXPORT_DIR")
    bypass = os.environ.get("CAESAR_MODEL_BYPASS") == "1"
    direct = not selection or not output
    if direct:
        parser = argparse.ArgumentParser(
            description="Compile this model component only; use compile_model.py for a complete installation"
        )
        parser.add_argument("device", choices=("cpu", "cuda", "mps", "xpu"))
        parser.add_argument("mode", nargs="?", choices=("bypass",))
        parser.add_argument("--bypass", action="store_true")
        parser.add_argument("--selection", default="pretrained/selected_model.json")
        parser.add_argument("--output", default="exported_model")
        args = parser.parse_args()
        selection, output = args.selection, args.output
        bypass = args.bypass or args.mode == "bypass"
        print(
            "COMPILING ONE COMPONENT ONLY. USE compile_model.py FOR A COMPLETE INSTALLATION.",
            file=sys.stderr,
        )
    model, checkpoint = read_selection(selection, bypass=bypass)
    if direct:
        check_current_registration(model)
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
        "".join(f"{key}={value}\n" for key, value in fields.items()),
        encoding="utf-8",
        newline="\n",
    )


def validate_installation(directory):
    directory = Path(directory)
    fields = {}
    for line in (
        (directory / "model_metadata.txt").read_text(encoding="utf-8").splitlines()
    ):
        key, separator, value = line.partition("=")
        if not separator or not value or key in fields:
            raise ValueError("Malformed installation metadata")
        fields[key] = value
    local_selection = directory / "local_model_selection.json"
    if local_selection.is_file():
        model = validate_model(
            json.loads(local_selection.read_text())["model"], allow_local=True
        )
        warn_local()
    else:
        snapshot = directory / "model_catalog.json"
        catalog = (
            load_catalog(snapshot) if snapshot.is_file() else selection_catalog({})
        )
        model = select_model(catalog, fields.get("model_id", "unregistered"))
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
        raise ValueError("Installation metadata does not match the selected model")
    for filename in (*[f"{c}.pt2" for c in COMPONENTS], *TABLES):
        if (
            not (directory / filename).is_file()
            or (directory / filename).stat().st_size == 0
        ):
            raise ValueError(f"Incomplete model installation: {filename}")
    return model


def check_registration(selection, catalog_path=None):
    """Check current UFL membership explicitly; never alter a local selection."""
    path = Path(selection).resolve()
    selected = json.loads(path.read_text(encoding="utf-8"))
    if selected["schema_version"] != 1:
        raise ValueError("Unsupported selection schema")
    model = validate_model(selected["model"], allow_local=True)
    verify_checkpoint(path.parent / model["filename"], model)
    return check_current_registration(model, catalog_path)


def check_current_registration(model, catalog_path=None):
    """Advisory check only: network failures must never prevent compilation."""
    try:
        # Short timeout for compute nodes with no outbound network access.
        catalog = load_catalog(catalog_path or DEFAULT_CATALOG, timeout=3)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(
            f"WARNING: COULD NOT CHECK THE UFL DATABASE: {error}. "
            "THIS NODE MAY HAVE NO INTERNET ACCESS. REGISTRATION IS UNCONFIRMED; "
            "CONTINUING WITH THE SAVED LOCAL SELECTION. "
            "RETRY --check-registration ON A NETWORKED LOGIN NODE.",
            file=sys.stderr,
        )
        return 2
    matches = [
        entry for entry in catalog["models"] if entry["sha256"] == model["sha256"]
    ]
    if matches:
        registered = matches[0]
        print(
            f"Checkpoint is registered in UFL: {registered['name']} ({registered['id']})"
        )
        if registered["id"] != model["id"]:
            print(
                "Your selection uses a different identity. To select the UFL registration, run: "
                f"python3 model_registry.py {registered['name']}"
            )
        return 0
    retired = any(
        entry.get("sha256") == model["sha256"]
        for entry in catalog.get("retired_models", [])
    )
    print(
        (
            "WARNING: THIS CHECKPOINT IS RETIRED FROM UFL."
            if retired
            else "WARNING: THIS CHECKPOINT IS NOT IN THE UFL DATABASE. SUBMIT IT TO UFL."
        ),
        file=sys.stderr,
    )
    warn_local()
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", nargs="?", help="Registered name or exact ufl:number@sha256:hash"
    )
    parser.add_argument(
        "--catalog",
        help="Catalog path or HTTPS URL (default: UFL_MODELS; override with CAESAR_MODEL_CATALOG)",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--check-registration",
        nargs="?",
        const="pretrained/selected_model.json",
        metavar="SELECTION",
        help="Check the selected checkpoint against the current UFL database; does not compile or change files",
    )
    parser.add_argument(
        "--local",
        metavar="CHECKPOINT",
        help="Select a local checkpoint without a registry merge (requires --bypass)",
    )
    parser.add_argument("--bypass", action="store_true")
    parser.add_argument("--min-dims", type=int, default=2)
    parser.add_argument("--max-dims", type=int, default=5)
    parser.add_argument("--validate-installation", metavar="DIRECTORY")
    parser.add_argument("--output", default="pretrained")
    parser.add_argument(
        "--source-dir", help="Read checkpoints from a local UFL_MODELS checkout"
    )
    args = parser.parse_args()
    if args.check_registration:
        raise SystemExit(check_registration(args.check_registration, args.catalog))
    if args.local:
        if not args.bypass:
            parser.error("--local requires --bypass")
        source = Path(args.local).resolve()
        digest = sha256(source)
        model = dict(
            name=source.stem,
            filename=source.name,
            registration_id=0,
            id=f"local@sha256:{digest}",
            sha256=digest,
            architecture=ARCHITECTURE,
            description="Local unregistered checkpoint",
            min_dims=args.min_dims,
            max_dims=args.max_dims,
        )
        validate_model(model, allow_local=True)
        destination = Path(args.output)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / source.name
        if target.resolve() != source:
            shutil.copyfile(source, target)
        verify_checkpoint(target, model)
        atomic_json(
            destination / "selected_model.json", {"schema_version": 1, "model": model}
        )
        warn_local()
        print(f"Selected local checkpoint: {target}")
        return
    if args.bypass:
        parser.error("--bypass requires --local")
    if args.catalog:
        os.environ["CAESAR_MODEL_CATALOG"] = args.catalog
    if args.validate_installation:
        model = validate_installation(args.validate_installation)
        print(f"Validated installation: {model['id']}")
        return
    catalog = load_catalog(args.catalog)
    if args.list:
        headers = ("Reg.", "Name", "Display name", "Dimensions", "Default")
        rows = [
            (
                str(model["registration_id"]),
                model["name"],
                model.get("display_name", model["name"]),
                f"{model['min_dims']}D–{model['max_dims']}D",
                "*" if model["name"] == catalog["default_model"] else "",
            )
            for model in catalog["models"]
        ]
        widths = [
            max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))
        ]

        def table_row(row):
            return "  ".join(
                value.ljust(width) for value, width in zip(row, widths)
            ).rstrip()

        print("Available CAESAR models")
        print("=======================")
        print(table_row(headers))
        print(table_row(tuple("-" * width for width in widths)))
        for row in rows:
            print(table_row(row))
        print("\n* Catalog default (the executable uses its compiled installation).")
        for model in catalog["models"]:
            print(f"\n{model['name']}")
            print(f"  Description : {model['description']}")
            print(f"  Identity    : {model['id']}")
        retired = catalog.get("retired_models", [])
        if retired:
            print("\nRemoved CAESAR models — registration history")
            for model in sorted(retired, key=lambda item: item["registration_id"]):
                print(
                    f"  {model['registration_id']}  {model['name']}"
                    f"  — Removed ({model['filename']})"
                )
        return
    model = select_model(catalog, args.model)
    target = download(model, args.output, args.source_dir, catalog=catalog)
    print(f"Verified {model['id']}\nSaved checkpoint: {target}")


if __name__ == "__main__":
    main()
