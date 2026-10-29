"""Submit model registrations to UFL_MODELS as pull requests, or check its catalog.

`add` and `remove` clone UFL_MODELS, commit the change on a branch, push it to the
submitter's fork, and open a pull request. UFL_MODELS's workflow checks the pull
request with its test/check_registration.py and merges it. This command imports
that same module from the clone to prepare and pre-check the change, so local
and workflow checks agree. Trusted automation, not this command, enforces access.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from model_registry import validate_model
from registry_catalog import render_readme

DATABASE = "UFcompressor/UFL_MODELS"
LOGIN = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?"


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


def run(*args, cwd=None):
    # Structured argv: request contents are never shell commands.
    return subprocess.run(
        args, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_checker(clone):
    """Import UFL_MODELS's test/check_registration.py from a clone."""
    spec = importlib.util.spec_from_file_location(
        "check_registration", Path(clone) / "test" / "check_registration.py"
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    return checker


def prepare(clone, args, login, user_id, database):
    """Commit the request on a new branch of a UFL_MODELS clone.

    A registration takes two commits: the checkpoint, then the catalog entry whose
    URL names the first commit. The workflow merges without squashing, so that
    commit becomes part of UFL_MODELS and the URL stays valid.
    """
    checker = load_checker(clone)
    identity = [
        "-c",
        f"user.name={login}",
        "-c",
        f"user.email={user_id}+{login}@users.noreply.github.com",
    ]

    def commit(message, *paths):
        run("git", "add", "--", *paths, cwd=clone)
        run("git", *identity, "commit", "--quiet", "-m", message, cwd=clone)
        return run("git", "rev-parse", "HEAD", cwd=clone)

    action = "register" if args.command == "add" else "retire"
    branch = f"registry/{action}-{args.name}"
    run("git", "switch", "--quiet", "-c", branch, cwd=clone)
    catalog_path, readme = clone / "model_catalog.json", clone / "README.md"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if action == "register":
        filename = args.checkpoint.name
        if (clone / filename).exists():
            raise ValueError(f"{filename} already exists in {database}; rename it")
        shutil.copyfile(args.checkpoint, clone / filename)
        added = commit(f"Add {filename}", filename)
        updated = checker.register(
            catalog,
            name=args.name,
            display_name=args.display_name,
            description=args.description,
            filename=filename,
            sha256=hashlib.sha256((clone / filename).read_bytes()).hexdigest(),
            min_dims=args.min_dims,
            max_dims=args.max_dims,
            owner=login,
            owner_id=user_id,
            url=f"https://raw.githubusercontent.com/{database}/{added}/{filename}",
        )
    else:
        updated = checker.retire(catalog, args.name, args.default_model)
    catalog_path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    readme.write_text(
        checker.render_readme(updated, readme.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    commit(f"{action.capitalize()} {args.name}", "model_catalog.json", "README.md")
    return branch, f"[model-registry] {action}: {args.name}"


def submit(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        raise ValueError("Model names may only use letters, digits, '_' and '-'")
    login, user_id = run("gh", "api", "user", "--jq", ".login, .id").split()
    user_id = int(user_id)
    with tempfile.TemporaryDirectory() as directory:
        clone = Path(directory) / "UFL_MODELS"
        run(
            "git",
            "clone",
            "--quiet",
            "--branch",
            args.base,
            f"https://github.com/{args.database}.git",
            str(clone),
        )
        branch, title = prepare(clone, args, login, user_id, args.database)
        try:
            summary = load_checker(clone).check_pull_request(
                clone, f"origin/{args.base}", "HEAD", login, user_id, args.database
            )
            print(f"Local check passed: {summary}")
        except ValueError as error:
            if not args.skip_local_check:
                raise ValueError(f"local check failed, nothing submitted: {error}")
            summary = f"skipped; it failed with: {error}"
            print(f"Local check failed; submitting anyway: {error}")
        print(
            run(
                "git",
                "log",
                "--stat",
                "--format=%h %s",
                f"origin/{args.base}..",
                cwd=clone,
            )
        )
        if args.dry_run:
            print("Dry run: nothing was pushed")
            return
        # Users who can push to the database use a branch there; others use a fork.
        if (
            run("gh", "api", f"repos/{args.database}", "--jq", ".permissions.push")
            == "true"
        ):
            target = args.database
        else:
            target = run(
                "gh",
                "api",
                "--method",
                "POST",
                f"repos/{args.database}/forks",
                "--jq",
                ".full_name",
            )
        for attempt in range(10):  # a new fork takes a few seconds to appear
            try:
                run(
                    "git",
                    "-c",
                    "credential.helper=",
                    "-c",
                    "credential.helper=!gh auth git-credential",
                    "push",
                    "--quiet",
                    "--force",
                    f"https://github.com/{target}.git",
                    f"HEAD:refs/heads/{branch}",
                    cwd=clone,
                )
                break
            except subprocess.CalledProcessError:
                if attempt == 9:
                    raise
                time.sleep(5)
    existing = run(
        "gh",
        "pr",
        "list",
        "--repo",
        args.database,
        "--head",
        branch,
        "--author",
        login,
        "--state",
        "open",
        "--json",
        "url",
        "--jq",
        ".[0].url",
    )
    if existing:
        print(f"Updated pull request {existing}; the workflow checks it again")
        return
    print(
        run(
            "gh",
            "pr",
            "create",
            "--repo",
            args.database,
            "--base",
            args.base,
            "--head",
            f"{target.split('/')[0]}:{branch}",
            "--title",
            title,
            "--body",
            "Opened by CAESAR `register_model.py`. The UFL_MODELS workflow checks "
            f"this request and merges it.\n\nLocal check: {summary}",
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser(
        "add", help="Open a pull request registering a checkpoint"
    )
    add.add_argument("--name", required=True)
    add.add_argument("--display-name", required=True)
    add.add_argument("--description", required=True)
    add.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="state_dict .pt file; its file name is kept in UFL_MODELS",
    )
    add.add_argument("--min-dims", type=int, default=3)
    add.add_argument("--max-dims", type=int, default=5)
    retire = commands.add_parser(
        "remove", help="Open a pull request retiring your model"
    )
    retire.add_argument("name")
    retire.add_argument("--default-model", help="Administrator only")
    for command in (add, retire):
        command.add_argument(
            "--dry-run",
            action="store_true",
            help="Prepare and check the change locally without pushing",
        )
        command.add_argument(
            "--skip-local-check",
            action="store_true",
            help="Submit even if the local check fails (to test the workflow)",
        )
        command.add_argument(
            "--base", default="main", help="UFL_MODELS branch to target"
        )
        command.add_argument(
            "--database", default=DATABASE, help="UFL_MODELS repository (for testing)"
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
    submit(args)


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
        detail = getattr(error, "stderr", None)
        raise SystemExit(f"Error: {error}" + (f"\n{detail.strip()}" if detail else ""))
