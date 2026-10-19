"""Ownership, publication validation, and issue event trust boundaries."""

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import register_model as service
from registry_catalog import render_readme


class RegistrationRequestsTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads(
            Path(service.__file__).with_name("model_catalog.json").read_text()
        )
        for model in self.catalog["models"]:
            model["owner"] = "E53Klasky"
            model["owner_id"] = 173562962
        self.policy = {
            "schema_version": 1,
            "admins": [{"login": "E53Klasky", "id": 173562962}],
        }
        self.request = dict(
            schema_version=1,
            action="register",
            name="new_model",
            display_name="New model",
            description="Scientific training checkpoint",
            repository="Alice/checkpoints",
            revision="a" * 40,
            filename="new_model.pt",
            sha256="b" * 64,
            min_dims=3,
            max_dims=5,
        )
        self.verify = Mock()

    def apply(
        self, request=None, actor="Alice", number=10, catalog=None, actor_id=None
    ):
        return service.apply_request(
            catalog or self.catalog,
            request or self.request,
            actor,
            self.policy,
            number,
            self.verify,
            actor_id=actor_id
            or {"alice": 101, "bob": 102, "e53klasky": 173562962}[actor.casefold()],
        )

    def retire(self, name="new_model", **extra):
        return dict(schema_version=1, action="retire", name=name, **extra)

    def test_registration_assigns_owner_number_and_verifies_bytes(self):
        before = copy.deepcopy(self.catalog)
        result = self.apply()
        model = result["models"][-1]
        self.assertEqual(model["owner"], "Alice")
        self.assertEqual(model["registration_id"], 5)
        self.assertEqual(model["registration_issue"], 10)
        self.verify.assert_called_once_with(model["url"], "b" * 64)
        self.assertEqual(self.catalog, before)

    def test_cannot_spoof_owner_or_registration_id(self):
        for key, value in (
            ("owner", "E53Klasky"),
            ("registration_id", 100),
            ("admins", ["Alice"]),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "unexpected"
            ):
                self.apply(dict(self.request, **{key: value}))

    def test_owner_can_retire_and_other_user_cannot(self):
        catalog = self.apply()
        with self.assertRaisesRegex(ValueError, "owner"):
            self.apply(self.retire(), actor="Bob", number=11, catalog=catalog)
        result = self.apply(self.retire(), actor="alice", number=11, catalog=catalog)
        self.assertEqual(result["retired_models"][0]["id"], catalog["models"][-1]["id"])
        self.assertEqual(len(result["models"]), 4)

    def test_admin_can_add_from_organization_and_retire_any_model(self):
        catalog = self.apply(
            dict(self.request, repository="UFcompressor/checkpoints"), actor="e53klasky"
        )
        self.assertEqual(catalog["models"][-1]["owner"], "e53klasky")
        result = self.apply(self.retire("caesar_v1"), actor="E53Klasky", number=11)
        self.assertEqual(result["retired_models"][0]["name"], "caesar_v1")

    def test_login_reuse_cannot_take_over_model_and_rename_keeps_ownership(self):
        catalog = self.apply()
        with self.assertRaisesRegex(ValueError, "owner"):
            self.apply(
                self.retire(), actor="Alice", actor_id=999, number=11, catalog=catalog
            )
        result = self.apply(
            self.retire(),
            actor="RenamedAlice",
            actor_id=101,
            number=11,
            catalog=catalog,
        )
        self.assertEqual(result["retired_models"][0]["owner_id"], 101)
        with self.assertRaisesRegex(ValueError, "owner"):
            self.apply(self.retire("caesar_v1"), actor="E53Klasky", actor_id=999)

    def test_default_changes_require_admin_and_valid_replacement(self):
        for actor, extra in (
            ("Alice", {"default_model": "caesar_v1"}),
            ("E53Klasky", {}),
            ("E53Klasky", {"default_model": "missing"}),
        ):
            with self.subTest(actor=actor, extra=extra), self.assertRaises(ValueError):
                self.apply(self.retire("caesar_v2", **extra), actor=actor)
        result = self.apply(
            self.retire("caesar_v2", default_model="caesar_v1"), actor="E53Klasky"
        )
        self.assertEqual(result["default_model"], "caesar_v1")

    def test_retired_numbers_and_names_stay_reserved(self):
        catalog = self.apply()
        retired = self.apply(self.retire(), number=11, catalog=catalog)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.apply(number=12, catalog=retired)
        changed = dict(
            self.request, name="next_model", filename="next_model.pt", sha256="c" * 64
        )
        result = self.apply(changed, number=12, catalog=retired)
        self.assertEqual(result["models"][-1]["registration_id"], 6)

    def test_successful_issue_retry_is_idempotent(self):
        catalog = self.apply()
        self.assertEqual(self.apply(catalog=catalog), catalog)
        retired = self.apply(self.retire(), number=11, catalog=catalog)
        self.assertEqual(self.apply(self.retire(), number=11, catalog=retired), retired)
        self.assertEqual(self.apply(catalog=retired), retired)

    def test_untrusted_urls_and_revisions_rejected_before_download(self):
        for changes in (
            {"repository": "Bob/checkpoints"},
            {"repository": "Alice/../evil"},
            {"revision": "main"},
            {"filename": "../model.pt"},
            {"sha256": "incorrect"},
            {"min_dims": 6},
            {"action": "delete-all"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.apply(dict(self.request, **changes))
        self.verify.assert_not_called()

    def test_hash_failure_does_not_mutate_catalog(self):
        before = copy.deepcopy(self.catalog)
        self.verify.side_effect = ValueError("SHA-256 mismatch")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.apply()
        self.assertEqual(self.catalog, before)

    def test_download_is_bounded_and_hash_checked(self):
        url = "https://raw.githubusercontent.com/Alice/models/" + "a" * 40 + "/model.pt"

        def response(payload, destination=url):
            stream = io.BytesIO(payload)
            stream.geturl = lambda: destination
            return stream

        with patch.object(service, "urlopen", return_value=response(b"data")):
            service.verify_download(url, service.hashlib.sha256(b"data").hexdigest())
        for payload, destination, limit in (
            (b"data", url, 2),
            (b"wrong", url, 100),
            (b"", url, 100),
            (b"data", "https://elsewhere", 100),
        ):
            with self.subTest(payload=payload, destination=destination):
                with patch.object(
                    service, "urlopen", return_value=response(payload, destination)
                ), patch.object(service, "MAX_CHECKPOINT_BYTES", limit):
                    with self.assertRaises(ValueError):
                        service.verify_download(url, "b" * 64)

    def test_trusted_event_uses_issue_author_not_sender(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model_catalog.json").write_text(json.dumps(self.catalog))
            (root / "registry_policy.json").write_text(json.dumps(self.policy))
            (root / "README.md").write_text("# Registry\n")
            event = dict(
                action="opened",
                repository=dict(full_name=service.DATABASE),
                sender=dict(login="E53Klasky"),
                issue=dict(
                    number=10,
                    user=dict(login="Bob", id=102, type="User"),
                    body=json.dumps(self.retire("caesar_v1")),
                ),
            )
            with self.assertRaisesRegex(ValueError, "owner"):
                service.process_event(root, event)
            self.assertEqual(
                json.loads((root / "model_catalog.json").read_text()), self.catalog
            )
            event["issue"]["user"]["login"] = "E53Klasky"
            event["issue"]["user"]["id"] = 173562962
            service.process_event(root, event)
            updated = json.loads((root / "model_catalog.json").read_text())
            self.assertEqual(updated["retired_models"][0]["name"], "caesar_v1")
            text = (root / "README.md").read_text()
            self.assertEqual(text, render_readme(updated, text))

    def test_cli_remove_dry_run_does_not_contact_github(self):
        result = subprocess.run(
            [sys.executable, service.__file__, "remove", "my_model", "--dry-run"],
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(json.loads(result.stdout), self.retire("my_model"))

    def test_registration_event_verifies_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model_catalog.json"
            path.write_text(json.dumps(self.catalog))
            (root / "registry_policy.json").write_text(json.dumps(self.policy))
            (root / "README.md").write_text("# Registry\n")
            event = dict(
                action="opened",
                repository=dict(full_name=service.DATABASE),
                issue=dict(
                    number=10,
                    user=dict(login="Alice", id=101, type="User"),
                    body=json.dumps(self.request),
                ),
            )
            with patch.object(
                service, "verify_download", side_effect=ValueError("bad hash")
            ):
                with self.assertRaisesRegex(ValueError, "bad hash"):
                    service.process_event(root, event)
            self.assertEqual(json.loads(path.read_text()), self.catalog)
            self.assertEqual((root / "README.md").read_text(), "# Registry\n")
            with patch.object(service, "verify_download") as verify:
                service.process_event(root, event)
            verify.assert_called_once()
            model = json.loads(path.read_text())["models"][-1]
            self.assertEqual((model["owner"], model["owner_id"]), ("Alice", 101))
            self.assertIn("Alice", (root / "README.md").read_text())

    def test_wrong_repository_and_bot_events_are_rejected(self):
        event = dict(action="opened", repository=dict(full_name="Alice/UFL_MODELS"))
        with self.assertRaisesRegex(ValueError, "UFcompressor"):
            service.process_event(Path("/unused"), event)
        event["repository"]["full_name"] = service.DATABASE
        event["issue"] = dict(user=dict(type="Bot"))
        with self.assertRaisesRegex(ValueError, "GitHub user"):
            service.process_event(Path("/unused"), event)

    def test_retirement_does_not_allow_metadata_replacement(self):
        catalog = self.apply()
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.apply(dict(self.retire(), owner="Bob"), number=11, catalog=catalog)

    def test_submission_passes_json_through_stdin_without_a_shell(self):
        with patch.object(
            sys, "argv", ["register_model.py", "remove", "my_model"]
        ), patch.object(service.subprocess, "run") as run:
            service.main()
        args, kwargs = run.call_args
        self.assertEqual(args[0][0:3], ["gh", "issue", "create"])
        self.assertNotIn("shell", kwargs)
        self.assertEqual(json.loads(kwargs["input"]), self.retire("my_model"))

    def test_readme_escapes_user_markup(self):
        self.catalog["models"][0]["description"] = "<script>bad</script>|\ntext"
        text = render_readme(self.catalog, "# Models\n")
        self.assertNotIn("<script>", text)
        self.assertIn("&#124;", text)
        self.assertIn("E53Klasky", text)


if __name__ == "__main__":
    unittest.main()
