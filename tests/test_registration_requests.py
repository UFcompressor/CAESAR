"""Registration pull requests from register_model.py, and catalog check/sync.

The add/remove tests need a UFL_MODELS checkout with test/check_registration.py,
at ../UFL_MODELS or $UFL_MODELS_DIR. GitHub is faked; git runs for real.
"""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import register_model as service
from registry_catalog import render_readme

UFL_MODELS = Path(
    os.environ.get(
        "UFL_MODELS_DIR", Path(service.__file__).resolve().parent.parent / "UFL_MODELS"
    )
)
HAS_DATABASE = (UFL_MODELS / "test" / "check_registration.py").is_file()


def variant(path):
    """A distinct valid checkpoint: one float of an existing one changed."""
    source, out = zipfile.ZipFile(path), io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as target:
        for info in source.infolist():
            content = source.read(info)
            if info.filename.endswith("/data/0"):
                content = b"\x00\x00\x80\x3f" + content[4:]
            target.writestr(info, content)
    return out.getvalue()


class FakeGitHub:
    """Answers gh calls and redirects clone/push; everything else runs for real."""

    def __init__(self, can_push=False, existing=""):
        self.can_push, self.existing, self.calls = can_push, existing, []
        self.real_run = service.run

    def __call__(self, *args, cwd=None):
        self.calls.append(args)
        if args[0] == "gh":
            if args[1:3] == ("api", "user"):
                return "Alice\n101"
            if args[-1] == ".permissions.push":
                return str(self.can_push).lower()
            if args[-3:] == (f"repos/{service.DATABASE}/forks", "--jq", ".full_name"):
                return "Alice/UFL_MODELS"
            if args[1:3] == ("pr", "list"):
                return self.existing
            if args[1:3] == ("pr", "create"):
                return "https://github.com/UFcompressor/UFL_MODELS/pull/7"
            raise AssertionError(f"unexpected gh call {args}")
        if args[:2] == ("git", "clone"):
            args = (*args[:-2], str(UFL_MODELS), args[-1])
        if "push" in args:
            self.pushed = self.real_run("git", "rev-parse", "HEAD", cwd=cwd)
            return ""
        return self.real_run(*args, cwd=cwd)

    def named(self, *prefix):
        return [call for call in self.calls if call[: len(prefix)] == prefix]


@unittest.skipUnless(HAS_DATABASE, "needs a UFL_MODELS checkout with registration checks")
class PullRequestClientTests(unittest.TestCase):
    def setUp(self):
        self.base = subprocess.run(
            ["git", "-C", str(UFL_MODELS), "rev-parse", "--abbrev-ref", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        catalog = json.loads((UFL_MODELS / "model_catalog.json").read_text())
        default = next(
            m for m in catalog["models"] if m["name"] == catalog["default_model"]
        )
        self.directory = tempfile.TemporaryDirectory()
        self.checkpoint = Path(self.directory.name) / "alice_model.pt"
        self.checkpoint.write_bytes(variant(UFL_MODELS / default["filename"]))

    def tearDown(self):
        self.directory.cleanup()

    def main(self, *argv, github=None):
        github = github or FakeGitHub()
        with patch.object(sys, "argv", ["register_model.py", *argv, "--base", self.base]), \
                patch.object(service, "run", github), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            service.main()
        return github, output.getvalue()

    def add(self, *extra, github=None):
        return self.main(
            "add", "--name", "alice_model", "--display-name", "Alice's model",
            "--description", "Test fine-tune", "--checkpoint", str(self.checkpoint),
            *extra, github=github,
        )

    def test_add_opens_pull_request_from_the_submitters_fork(self):
        github, output = self.add()
        self.assertIn("Local check passed: register alice_model as ufl:6@", output)
        self.assertEqual(len(github.named("gh", "api", "--method", "POST")), 1)
        (push,) = [call for call in github.calls if "push" in call]
        self.assertIn("https://github.com/Alice/UFL_MODELS.git", push)
        self.assertIn("HEAD:refs/heads/registry/register-alice_model", push)
        (create,) = github.named("gh", "pr", "create")
        self.assertEqual(
            create[create.index("--head") + 1], "Alice:registry/register-alice_model"
        )
        self.assertEqual(
            create[create.index("--title") + 1], "[model-registry] register: alice_model"
        )
        self.assertEqual(create[create.index("--base") + 1], self.base)
        self.assertIn("pull/7", output)

    def test_dry_run_pushes_nothing(self):
        github, output = self.add("--dry-run")
        self.assertIn("Dry run", output)
        self.assertIn("alice_model.pt", output)
        self.assertFalse([c for c in github.calls if "push" in c or c[1:2] == ("pr",)])

    def test_maintainers_use_a_branch_in_the_database(self):
        github, _ = self.add(github=FakeGitHub(can_push=True))
        self.assertFalse(github.named("gh", "api", "--method", "POST"))
        (push,) = [call for call in github.calls if "push" in call]
        self.assertIn(f"https://github.com/{service.DATABASE}.git", push)

    def test_rerun_updates_the_open_pull_request(self):
        github, output = self.add(github=FakeGitHub(existing="https://x/pull/7"))
        self.assertFalse(github.named("gh", "pr", "create"))
        self.assertIn("Updated pull request https://x/pull/7", output)

    def test_local_check_stops_requests_the_workflow_would_reject(self):
        with self.assertRaisesRegex(ValueError, "local check failed.*Only the model owner"):
            self.main("remove", "eelsM1")
        github, output = self.main("remove", "eelsM1", "--skip-local-check")
        self.assertIn("submitting anyway", output)
        (create,) = github.named("gh", "pr", "create")
        self.assertEqual(
            create[create.index("--title") + 1], "[model-registry] retire: eelsM1"
        )

    def test_names_cannot_inject_git_or_shell_syntax(self):
        with self.assertRaisesRegex(ValueError, "letters, digits"):
            self.main("remove", "x; rm -rf ~")


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads(
            Path(service.__file__).with_name("model_catalog.json").read_text()
        )
        for model in self.catalog["models"]:
            model["owner"] = "E53Klasky"
            model["owner_id"] = 173562962

    def test_check_detects_stale_readme(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model_catalog.json").write_text(json.dumps(self.catalog))
            (root / "README.md").write_text(render_readme(self.catalog, "# Models\n"))
            command = [sys.executable, service.__file__, "check", "--database", directory]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            (root / "README.md").write_text("# Models\n")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertIn("stale", result.stderr)

    def test_catalog_requires_owner_ids(self):
        del self.catalog["models"][0]["owner_id"]
        with self.assertRaisesRegex(ValueError, "owner ID"):
            service.validate_catalog(self.catalog)

    def test_readme_escapes_user_markup(self):
        self.catalog["models"][0]["description"] = "<script>bad</script>|\ntext"
        text = render_readme(self.catalog, "# Models\n")
        self.assertNotIn("<script>", text)
        self.assertIn("&#124;", text)
        self.assertIn("E53Klasky", text)


if __name__ == "__main__":
    unittest.main()
