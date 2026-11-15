"""Registry invariants and failure behavior, independent of torch/model inference."""

import copy
import io
import os
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from catalog_fixture import make_catalog

import compile_model
import model_registry as registry


class ModelRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = make_catalog()
        self.upstream = json.dumps(self.catalog).encode()
        opener = patch.object(
            registry, "urlopen", side_effect=lambda *a, **kw: io.BytesIO(self.upstream)
        )
        self.urlopen = opener.start()
        self.addCleanup(opener.stop)
        self.model = copy.deepcopy(self.catalog["models"][0])
        self.source = self.root / "source"
        self.source.mkdir()
        checkpoint = self.source / self.model["filename"]
        checkpoint.write_bytes(b"registered checkpoint fixture")
        self.model["sha256"] = registry.sha256(checkpoint)
        self.model["id"] = (
            f"ufl:{self.model['registration_id']}@sha256:{self.model['sha256']}"
        )
        self.catalog["models"][0] = self.model
        self.upstream = json.dumps(self.catalog).encode()

    def catalog_file(self):
        path = self.root / "catalog.json"
        path.write_text(json.dumps(self.catalog))
        return path

    def test_all_names_and_foundation_versions(self):
        catalog = registry.load_catalog()
        self.assertEqual(len(catalog["models"]), 2)
        self.assertEqual(registry.select_model(catalog)["name"], "caesar_v2")
        self.assertEqual(
            registry.select_model(catalog, "caesar_v1")["filename"], "caesar_v.pt"
        )
        self.assertEqual(
            registry.select_model(catalog, "caesar_v2")["filename"],
            "model_bs64_ep100k.pt",
        )

    def test_default_catalog_comes_from_ufl_models(self):
        with patch.dict(os.environ, {}, clear=True):
            registry.load_catalog()
        self.urlopen.assert_called_once_with(
            "https://raw.githubusercontent.com/UFcompressor/UFL_MODELS/main/model_catalog.json",
            timeout=60,
        )

    def test_catalog_environment_override(self):
        with patch.dict(os.environ, {"CAESAR_MODEL_CATALOG": str(self.catalog_file())}):
            self.assertEqual(registry.load_catalog(), self.catalog)
        self.urlopen.assert_not_called()

    def test_retired_model_is_unavailable_and_labeled(self):
        retired = self.catalog["models"].pop(0)
        self.catalog["retired_models"] = [retired]
        catalog = registry.load_catalog(self.catalog_file())
        for requested in (retired["name"], retired["id"]):
            with self.assertRaisesRegex(
                ValueError, "removed and is no longer available"
            ):
                registry.select_model(catalog, requested)
        output = io.StringIO()
        with patch("sys.argv", ["model_registry.py", "--list"]), patch.object(
            registry, "load_catalog", return_value=catalog
        ), patch("sys.stdout", output):
            registry.main()
        active, history = output.getvalue().split("Removed CAESAR models")
        self.assertNotIn(retired["name"], active)
        self.assertIn(retired["name"], history)
        self.assertIn(f"Removed ({retired['filename']})", history)

    def test_duplicate_registration_rejected(self):
        self.catalog["models"][1]["registration_id"] = self.model["registration_id"]
        m = self.catalog["models"][1]
        m["id"] = f"ufl:{m['registration_id']}@sha256:{m['sha256']}"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            registry.load_catalog(self.catalog_file())

    def test_duplicate_checkpoint_rejected(self):
        m = self.catalog["models"][1]
        m["sha256"] = self.model["sha256"]
        m["id"] = f"ufl:{m['registration_id']}@sha256:{m['sha256']}"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            registry.load_catalog(self.catalog_file())

    def test_unknown_and_malformed_identity(self):
        with self.assertRaisesRegex(ValueError, "not registered"):
            registry.select_model(self.catalog, "missing")
        self.model["id"] = "ufl:99@sha256:" + self.model["sha256"]
        with self.assertRaisesRegex(ValueError, "identity"):
            registry.validate_model(self.model)

    def test_verified_download_and_cached_download(self):
        output = self.root / "pretrained"
        target = registry.download(self.model, output, self.source)
        with patch.object(
            registry, "urlopen", side_effect=AssertionError("Should use local cache")
        ):
            registry.download(self.model, output, catalog=self.catalog)
            selected, _ = registry.read_selection(output / "selected_model.json")
            self.assertEqual(selected, self.model)
        with patch.object(registry, "load_catalog", return_value=self.catalog):
            selected, path = registry.read_selection(output / "selected_model.json")
        self.assertEqual(selected, self.model)
        # Temporary directories can use aliases (macOS /var or Windows 8.3 names).
        self.assertEqual(path, target.resolve())

    def test_failed_download_preserves_selection_and_checkpoint(self):
        output = self.root / "pretrained"
        target = registry.download(self.model, output, self.source)
        selection = (output / "selected_model.json").read_bytes()
        target.write_bytes(b"existing wrong checkpoint")
        (self.source / self.model["filename"]).write_bytes(
            b"corrupt incoming checkpoint"
        )
        with self.assertRaisesRegex(ValueError, "Required model"):
            registry.download(self.model, output, self.source)
        self.assertEqual(target.read_bytes(), b"existing wrong checkpoint")
        self.assertEqual((output / "selected_model.json").read_bytes(), selection)
        self.assertFalse(list(output.glob(".checkpoint-*")))

    def test_download_adds_snapshot_to_legacy_cached_checkpoint(self):
        output = self.root / "pretrained"
        target = registry.download(self.model, output, self.source)
        registry.atomic_json(
            output / "selected_model.json", {"schema_version": 1, "model": self.model}
        )
        with patch.object(registry, "urlopen", side_effect=AssertionError("cached")):
            registry.download(self.model, output, catalog=self.catalog)
            selected, checkpoint = registry.read_selection(
                output / "selected_model.json"
            )
        self.assertEqual(selected, self.model)
        self.assertEqual(checkpoint.read_bytes(), target.read_bytes())

    def test_snapshot_survives_upstream_removal_and_still_checks_hash(self):
        output = self.root / "pretrained"
        target = registry.download(self.model, output, self.source)
        self.catalog["models"].remove(self.model)
        self.catalog["retired_models"] = [self.model]
        self.upstream = json.dumps(self.catalog).encode()
        with patch.object(registry, "urlopen", side_effect=AssertionError("offline")):
            self.assertEqual(
                registry.read_selection(output / "selected_model.json")[0], self.model
            )
            target.write_bytes(b"corrupt checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint .* has SHA-256"):
                registry.read_selection(output / "selected_model.json")

    def test_legacy_selection_accepts_explicit_local_catalog(self):
        registry.download(self.model, self.root / "pretrained", self.source)
        path = self.root / "pretrained/selected_model.json"
        registry.atomic_json(path, {"schema_version": 1, "model": self.model})
        with patch.dict(os.environ, {"CAESAR_MODEL_CATALOG": str(self.catalog_file())}):
            with patch.object(
                registry, "urlopen", side_effect=AssertionError("offline")
            ):
                self.assertEqual(registry.read_selection(path)[0], self.model)

    def test_legacy_selection_compiles_offline_without_catalog(self):
        registry.download(self.model, self.root / "pretrained", self.source)
        selection = self.root / "pretrained/selected_model.json"
        data = json.loads(selection.read_text())
        del data["catalog"]
        selection.write_text(json.dumps(data))
        with patch.dict(os.environ, {}, clear=True), patch.object(
            registry, "urlopen", side_effect=AssertionError("offline")
        ):
            selected, checkpoint = registry.read_selection(selection)
            self.assertEqual(selected, self.model)
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                registry.read_selection(selection)

    def test_altered_registration_metadata_rejected(self):
        altered = copy.deepcopy(self.model)
        altered["min_dims"] = 4
        registry.download(self.model, self.root / "pretrained", self.source)
        selection = self.root / "pretrained/selected_model.json"
        data = json.loads(selection.read_text())
        data["model"] = altered
        selection.write_text(json.dumps(data))
        with patch.object(registry, "load_catalog", return_value=self.catalog):
            with self.assertRaisesRegex(ValueError, "does not match"):
                registry.read_selection(self.root / "pretrained/selected_model.json")

    def test_local_selection_requires_bypass_and_still_checks_hash(self):
        selection = self.root / "pretrained/selected_model.json"
        with patch(
            "sys.argv",
            [
                "model_registry.py",
                "--local",
                str(self.source / self.model["filename"]),
                "--bypass",
                "--output",
                str(selection.parent),
            ],
        ):
            registry.main()
        with self.assertRaisesRegex(ValueError, "positive UFL registration"):
            registry.read_selection(selection)
        model, checkpoint = registry.read_selection(selection, bypass=True)
        self.assertEqual(model["id"], "local@sha256:" + self.model["sha256"])
        checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            registry.read_selection(selection, bypass=True)

    def test_bypass_skips_catalog_but_rejects_incompatible_architecture(self):
        registry.download(self.model, self.root / "pretrained", self.source)
        selection = self.root / "pretrained/selected_model.json"
        data = json.loads(selection.read_text())
        data["catalog"] = {"schema_version": 999}
        registry.atomic_json(selection, data)
        warning = io.StringIO()
        with patch("sys.stderr", warning):
            model, _ = registry.read_selection(selection, bypass=True)
        self.assertEqual(model["registration_id"], 0)
        self.assertIn("NOT AN OFFICIALLY SUPPORTED MODEL", warning.getvalue())
        self.assertIn("register_model.py add", warning.getvalue())
        self.assertIn("--checkpoint", warning.getvalue())
        data["model"]["architecture"] = "different-network"
        registry.atomic_json(selection, data)
        with self.assertRaisesRegex(ValueError, "unsupported model architecture"):
            registry.read_selection(selection, bypass=True)

    def test_registered_bypass_preserves_identity_without_unsafe_warning(self):
        registry.download(self.model, self.root / "pretrained", self.source)
        selection = self.root / "pretrained/selected_model.json"
        for legacy in (False, True):
            if legacy:
                registry.atomic_json(
                    selection, {"schema_version": 1, "model": self.model}
                )
            with patch("sys.stderr", io.StringIO()) as warning:
                model, _ = registry.read_selection(selection, bypass=True)
            self.assertEqual(model, self.model)
            self.assertNotIn("UNSAFE", warning.getvalue())

    def test_individual_export_context_all_devices(self):
        registry.download(self.model, self.root / "pretrained", self.source)
        selection = self.root / "pretrained/selected_model.json"
        for device in ("cpu", "cuda", "mps", "xpu"):
            with self.subTest(device=device), patch.dict(
                os.environ, {}, clear=True
            ), patch(
                "sys.argv",
                [
                    "CAESAR_compressor.py",
                    device,
                    "bypass",
                    "--selection",
                    str(selection),
                ],
            ), patch.object(
                compile_model.subprocess, "run"
            ) as run:
                model, checkpoint, output = registry.export_context()
                self.assertEqual(model, self.model)
                self.assertEqual(checkpoint.name, self.model["filename"])
                self.assertEqual(output, Path("exported_model"))
                run.assert_not_called()

    def test_bypass_compile_freezes_selection_and_marks_installation_local(self):
        registry.download(self.model, self.root / "pretrained", self.source)
        selection = self.root / "pretrained/selected_model.json"
        local = dict(
            self.model, registration_id=0, id="local@sha256:" + self.model["sha256"]
        )
        registry.atomic_json(selection, {"schema_version": 1, "model": local})
        output = self.root / "exported_model"
        calls = []

        def export(command, env, **kwargs):
            self.assertEqual(env["CAESAR_MODEL_BYPASS"], "1")
            model, _ = registry.read_selection(
                env["CAESAR_MODEL_SELECTION"], bypass=True
            )
            self.assertEqual(model["registration_id"], 0)
            calls.append(command)
            stage = Path(env["CAESAR_EXPORT_DIR"])
            for filename in (
                *[f"{c}.pt2" for c in registry.COMPONENTS],
                *registry.TABLES,
            ):
                (stage / filename).write_bytes(b"artifact")

        with patch.object(
            compile_model.subprocess, "run", side_effect=export
        ), patch.object(
            compile_model, "compilation_device_name", return_value="CPU: test"
        ), patch.object(
            registry, "urlopen", side_effect=OSError("offline")
        ):
            compile_model.compile_installation("cpu", selection, output, bypass=True)
        self.assertEqual(len(calls), 3)
        self.assertFalse((output / "model_catalog.json").exists())
        self.assertEqual(registry.validate_installation(output)["registration_id"], 0)

    def test_advisory_registration_check_handles_network_timeout(self):
        with patch.object(
            registry, "urlopen", side_effect=TimeoutError("no network")
        ) as query, patch("sys.stderr", io.StringIO()) as warning:
            self.assertEqual(registry.check_current_registration(self.model), 2)
        query.assert_called_once_with(registry.DEFAULT_CATALOG, timeout=3)
        self.assertIn("REGISTRATION IS UNCONFIRMED", warning.getvalue())
        self.assertIn("CONTINUING WITH THE SAVED LOCAL SELECTION", warning.getvalue())
        self.assertNotIn("NOT IN THE UFL DATABASE", warning.getvalue())

    def test_explicit_database_check_present_missing_and_unavailable(self):
        registry.download(self.model, self.root / "pretrained", self.source)
        selection = self.root / "pretrained/selected_model.json"
        before = selection.read_bytes()
        with patch("sys.stdout", io.StringIO()):
            self.assertEqual(registry.check_registration(selection), 0)
        # A saved registration remains compilable after current DB removal.
        self.catalog["models"].remove(self.model)
        self.upstream = json.dumps(self.catalog).encode()
        warning = io.StringIO()
        with patch("sys.stderr", warning):
            self.assertEqual(registry.check_registration(selection), 1)
        self.assertIn("SUBMIT IT TO UFL", warning.getvalue())
        self.assertEqual(registry.read_selection(selection)[0], self.model)
        with patch.object(registry, "urlopen", side_effect=OSError("offline")), patch(
            "sys.stderr", io.StringIO()
        ) as warning:
            self.assertEqual(registry.check_registration(selection), 2)
            self.assertIn("COULD NOT CHECK", warning.getvalue())
        self.assertEqual(selection.read_bytes(), before)

    def test_failed_compile_preserves_installation(self):
        output = self.root / "exported_model"
        output.mkdir()
        (output / "model_metadata.txt").write_text("previous installation")
        registry.download(self.model, self.root / "pretrained", self.source)
        with patch.object(
            registry, "load_catalog", return_value=self.catalog
        ), patch.object(
            compile_model.subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, "export"),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                compile_model.compile_installation(
                    "cpu", self.root / "pretrained/selected_model.json", output
                )
        self.assertEqual(
            (output / "model_metadata.txt").read_text(), "previous installation"
        )
        self.assertFalse(output.with_name("exported_model.lock").exists())

    def test_mps_records_apple_chip_name(self):
        with patch.dict("sys.modules", {"torch": object()}), patch.object(
            compile_model.subprocess, "check_output", return_value="Apple M4 Max\n"
        ) as query:
            self.assertEqual(
                compile_model.compilation_device_name("mps"), "MPS: Apple M4 Max"
            )
        query.assert_called_once_with(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        )

    def test_compile_publishes_recorded_device(self):
        registry.download(self.model, self.root / "pretrained", self.source)
        output = self.root / "exported_model"

        def export(*args, env, **kwargs):
            selected, _ = registry.read_selection(env["CAESAR_MODEL_SELECTION"])
            self.assertEqual(selected, self.model)
            stage = Path(env["CAESAR_EXPORT_DIR"])
            for filename in (
                *[f"{c}.pt2" for c in registry.COMPONENTS],
                *registry.TABLES,
            ):
                (stage / filename).write_bytes(b"artifact")

        with patch.object(
            registry, "load_catalog", return_value=self.catalog
        ), patch.object(
            compile_model.subprocess, "run", side_effect=export
        ), patch.object(
            compile_model, "compilation_device_name", return_value="GPU: NVIDIA B200"
        ), patch.object(
            registry,
            "urlopen",
            side_effect=OSError("Compute node has no network"),
        ):
            compile_model.compile_installation(
                "cuda", self.root / "pretrained/selected_model.json", output
            )
        self.assertEqual(
            (output / "model_device.txt").read_text(), "GPU: NVIDIA B200\n"
        )
        with patch.object(registry, "load_catalog", return_value=self.catalog):
            self.assertEqual(
                registry.validate_installation(output)["id"], self.model["id"]
            )

    def test_complete_installation_only(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            registry.write_installation(self.root, self.model, "cpu")
        for filename in (*[f"{c}.pt2" for c in registry.COMPONENTS], *registry.TABLES):
            (self.root / filename).write_bytes(b"artifact")
        registry.write_installation(self.root, self.model, "cpu")
        text = (self.root / "model_metadata.txt").read_text()
        self.assertIn("model_id=" + self.model["id"], text)
        self.assertIn("device=cpu", text)
        self.assertNotIn("description=", text)

    def test_installation_validation_rejects_unregistered_metadata(self):
        for filename in (*[f"{c}.pt2" for c in registry.COMPONENTS], *registry.TABLES):
            (self.root / filename).write_bytes(b"artifact")
        registry.write_installation(self.root, self.model, "cpu")
        registry.atomic_json(self.root / "model_catalog.json", self.catalog)
        with patch.object(registry, "load_catalog", return_value=self.catalog):
            self.assertEqual(registry.validate_installation(self.root), self.model)
        with patch.object(registry, "urlopen", side_effect=AssertionError("offline")):
            self.assertEqual(registry.validate_installation(self.root), self.model)
        path = self.root / "model_metadata.txt"
        path.write_text(path.read_text().replace("min_dims=3", "min_dims=4"))
        with patch.object(registry, "load_catalog", return_value=self.catalog):
            with self.assertRaisesRegex(ValueError, "does not match"):
                registry.validate_installation(self.root)


if __name__ == "__main__":
    unittest.main()
