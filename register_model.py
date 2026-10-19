"""Submit model requests, or process trusted GitHub issue events for UFL_MODELS.

No checkpoint code is loaded by the registration service. Publication validates
metadata and checkpoint bytes; CAESAR's compiler checks network compatibility.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.request import urlopen

from model_registry import validate_model
from registry_catalog import render_readme

DATABASE = "UFcompressor/UFL_MODELS"
LOGIN = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?"
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024


def validate_catalog(catalog):
    if type(catalog.get("schema_version")) is not int or catalog["schema_version"] != 1:
        raise ValueError("Unsupported catalog schema")
    seen = {
        key: set() for key in ("name", "registration_id", "id", "sha256", "filename")
    }
    for model in catalog["models"] + catalog.get("retired_models", []):
        validate_model(model)
        if not re.fullmatch(LOGIN, model.get("owner", "")):
            raise ValueError("Every registration requires a GitHub owner")
        if type(model.get("owner_id")) is not int or model["owner_id"] <= 0:
            raise ValueError("Every registration requires a stable GitHub owner ID")
        for key, values in seen.items():
            if model[key] in values:
                raise ValueError(f"Duplicate {key}")
            values.add(model[key])
    if catalog["default_model"] not in {m["name"] for m in catalog["models"]}:
        raise ValueError("Default must be an active model")


def checkpoint_url(request, actor, admin):
    repository = request["repository"]
    revision = request["revision"]
    filename = request["filename"]
    if not re.fullmatch(LOGIN + r"/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid checkpoint repository")
    if not admin and repository.split("/")[0].casefold() != actor.casefold():
        raise ValueError("Checkpoint repository must belong to the submitter")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Checkpoint revision must be a full 40-character commit SHA")
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.pt", filename):
        raise ValueError("Checkpoint must be a .pt file at the repository root")
    return f"https://raw.githubusercontent.com/{repository}/{revision}/{filename}"


def verify_download(url, expected):
    digest = hashlib.sha256()
    total = 0
    with urlopen(url, timeout=60) as response:
        if response.geturl() != url:
            raise ValueError("Checkpoint redirects are not accepted")
        for block in iter(lambda: response.read(1024 * 1024), b""):
            total += len(block)
            if total > MAX_CHECKPOINT_BYTES:
                raise ValueError("Checkpoint exceeds the 512 MiB registration limit")
            digest.update(block)
    if not total or digest.hexdigest() != expected:
        raise ValueError("Empty checkpoint or SHA-256 mismatch")


def apply_request(
    catalog, request, actor, policy, issue_number, verify=None, *, actor_id
):
    """Pure catalog transition except for the injected checkpoint verifier.

    actor MUST come from the trusted issue event, never from request JSON.
    """
    validate_catalog(catalog)
    if not re.fullmatch(LOGIN, actor):
        raise ValueError("Invalid authenticated GitHub login")
    if type(actor_id) is not int or actor_id <= 0:
        raise ValueError("Invalid authenticated GitHub user ID")
    if policy.get("schema_version") != 1:
        raise ValueError("Unsupported administrator policy schema")
    admins = policy["admins"]
    if not admins or any(
        type(value.get("id")) is not int or value["id"] <= 0 for value in admins
    ):
        raise ValueError("Invalid administrator policy")
    admin = actor_id in {value["id"] for value in admins}
    if type(issue_number) is not int or issue_number <= 0:
        raise ValueError("Invalid issue number")
    result = copy.deepcopy(catalog)
    # Re-running a successful workflow is harmless, even after model retirement.
    for model in result["models"] + result.get("retired_models", []):
        if (
            model.get("registration_issue") == issue_number
            or model.get("retirement_issue") == issue_number
        ):
            return result
    action = request.get("action")
    common = {"schema_version", "action", "name"}
    if request.get("schema_version") != 1:
        raise ValueError("Unsupported request schema")
    if action == "register":
        allowed = common | {
            "display_name",
            "description",
            "repository",
            "revision",
            "filename",
            "sha256",
            "min_dims",
            "max_dims",
        }
        if set(request) != allowed:
            raise ValueError(
                "Registration has missing or unexpected fields (owner is assigned automatically)"
            )
        number = (
            max(
                (
                    m["registration_id"]
                    for m in result["models"] + result.get("retired_models", [])
                ),
                default=0,
            )
            + 1
        )
        model = {
            key: request[key]
            for key in (
                "name",
                "display_name",
                "description",
                "filename",
                "sha256",
                "min_dims",
                "max_dims",
            )
        }
        for field, limit in (("display_name", 120), ("description", 4000)):
            if (
                not isinstance(model[field], str)
                or not model[field].strip()
                or len(model[field]) > limit
            ):
                raise ValueError(f"Invalid {field}")
        model.update(
            registration_id=number,
            id=f"ufl:{number}@sha256:{model['sha256']}",
            architecture="caesar-bcrn-v1",
            owner=actor,
            owner_id=actor_id,
            registration_issue=issue_number,
            url=checkpoint_url(request, actor, admin),
        )
        result["models"].append(model)
        validate_catalog(result)
        (verify or verify_download)(model["url"], model["sha256"])
    elif action == "retire":
        if set(request) - (common | {"default_model"}) or not common <= set(request):
            raise ValueError("Retirement has missing or unexpected fields")
        model = next(
            (m for m in result["models"] if m["name"] == request["name"]), None
        )
        if model is None:
            raise ValueError("No active model with that name")
        if not admin and model["owner_id"] != actor_id:
            raise ValueError("Only the model owner or an administrator may retire it")
        replacement = request.get("default_model")
        if replacement is not None and not admin:
            raise ValueError("Only an administrator may change the default model")
        if result["default_model"] == model["name"] and not replacement:
            raise ValueError(
                "Retiring the default requires an administrator and a replacement default"
            )
        if replacement is not None:
            result["default_model"] = replacement
        result["models"].remove(model)
        model["retirement_issue"] = issue_number
        result.setdefault("retired_models", []).append(model)
        validate_catalog(result)
    else:
        raise ValueError("Unknown request action")
    return result


def process_event(root, event):
    if event.get("action") != "opened" or event["repository"]["full_name"] != DATABASE:
        raise ValueError("Expected an opened issue in UFcompressor/UFL_MODELS")
    issue = event["issue"]
    if "pull_request" in issue or issue["user"].get("type") != "User":
        raise ValueError("Requests must be issues opened by a GitHub user")
    request = json.loads(issue["body"])
    if not isinstance(request, dict):
        raise ValueError("Issue body must be a JSON object")
    path = root / "model_catalog.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    policy = json.loads((root / "registry_policy.json").read_text(encoding="utf-8"))
    updated = apply_request(
        catalog,
        request,
        issue["user"]["login"],
        policy,
        issue["number"],
        actor_id=issue["user"]["id"],
    )
    readme = root / "README.md"
    text = render_readme(updated, readme.read_text(encoding="utf-8"))
    path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    readme.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="Submit a checkpoint registration issue")
    add.add_argument("--name", required=True)
    add.add_argument("--display-name", required=True)
    add.add_argument("--description", required=True)
    add.add_argument(
        "--repository",
        required=True,
        help="Public OWNER/REPO containing the checkpoint",
    )
    add.add_argument("--revision", required=True)
    add.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Local copy used to compute SHA-256",
    )
    add.add_argument("--min-dims", type=int, default=3)
    add.add_argument("--max-dims", type=int, default=5)
    retire = commands.add_parser("remove", help="Request retirement of your model")
    retire.add_argument("name")
    retire.add_argument("--default-model", help="Administrator only")
    for command in (add, retire):
        command.add_argument(
            "--dry-run",
            action="store_true",
            help="Print request without opening an issue",
        )
    process = commands.add_parser(
        "process-event",
        help="Trusted GitHub Actions entrypoint; not a local authorization boundary",
    )
    process.add_argument("--database", type=Path, required=True)
    process.add_argument(
        "--event",
        type=Path,
        default=Path(os.environ.get("GITHUB_EVENT_PATH", "event.json")),
    )
    check = commands.add_parser(
        "check", help="Validate catalog ownership and generated README"
    )
    check.add_argument("--database", type=Path, required=True)
    sync = commands.add_parser(
        "sync",
        help="Update CAESAR's catalog snapshot from a reviewed UFL_MODELS checkout",
    )
    sync.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "process-event":
        process_event(args.database, json.loads(args.event.read_text(encoding="utf-8")))
        return
    if args.command in ("check", "sync"):
        catalog = json.loads(
            (args.database / "model_catalog.json").read_text(encoding="utf-8")
        )
        validate_catalog(catalog)
        text = (args.database / "README.md").read_text(encoding="utf-8")
        if render_readme(catalog, text) != text:
            raise ValueError("README model list is stale")
        if args.command == "sync":
            Path(__file__).with_name("model_catalog.json").write_text(
                json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
            )
            print(
                "Updated CAESAR catalog snapshot; review and commit model_catalog.json"
            )
            return
        print("Catalog and README validated")
        return
    if args.command == "add":
        digest = hashlib.sha256()
        with args.checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        request = dict(
            schema_version=1,
            action="register",
            name=args.name,
            display_name=args.display_name,
            description=args.description,
            repository=args.repository,
            revision=args.revision,
            filename=args.checkpoint.name,
            sha256=digest.hexdigest(),
            min_dims=args.min_dims,
            max_dims=args.max_dims,
        )
        checkpoint_url(request, args.repository.split("/")[0], False)
        validate_model(
            dict(
                request,
                id="ufl:1@sha256:" + request["sha256"],
                registration_id=1,
                architecture="caesar-bcrn-v1",
                url=checkpoint_url(request, args.repository.split("/")[0], False),
            )
        )
    else:
        request = dict(schema_version=1, action="retire", name=args.name)
        if args.default_model:
            request["default_model"] = args.default_model
    body = json.dumps(request, indent=2)
    if args.dry_run:
        print(body)
    else:
        # Structured argv and stdin: request contents are never shell commands.
        subprocess.run(
            [
                "gh",
                "issue",
                "create",
                "--repo",
                DATABASE,
                "--title",
                f"[model-registry] {request['action']}: {args.name}",
                "--body-file",
                "-",
            ],
            input=body,
            text=True,
            check=True,
        )


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(f"Error: {error}") from error
