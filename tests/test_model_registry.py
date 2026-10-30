"""Registry invariants and failure behavior, independent of torch/model inference."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import compile_model
import model_registry as registry


class ModelRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = registry.load_catalog()
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

    def catalog_file(self):
        path = self.root / "catalog.json"
        path.write_text(json.dumps(self.catalog))
        return path

    def test_all_names_and_foundation_versions(self):
        catalog = registry.load_catalog()
        self.assertEqual(len(catalog["models"]), 4)
        self.assertEqual(registry.select_model(catalog)["name"], "caesar_v2")
        self.assertEqual(
            registry.select_model(catalog, "caesar_v1")["filename"], "caesar_v.pt"
        )
        self.assertEqual(
            registry.select_model(catalog, "caesar_v2")["filename"],
            "model_bs64_ep100k.pt",
        )

    def test_architecture_dimensions_and_unknown_rejection(self):
        cases = {
            registry.ARCHITECTURE: (8, 2, 16, 4),
            "caesar-depth32-t16-v1": (16, 8, 32, 8),
            "caesar-depth16-t16-v1": (16, 4, 16, 4),
            "caesar-depth8-t16-v1": (16, 2, 8, 2),
        }
        for architecture, dimensions in cases.items():
            with self.subTest(architecture=architecture):
                model = dict(self.model, architecture=architecture)
                registry.validate_model(model)
                encoder, decoder, *actual = registry.export_architecture(model)
                self.assertEqual(tuple(actual), dimensions)
                self.assertEqual(encoder[-1], 4)
                self.assertEqual(len(decoder), len(encoder) - 1)
        model = dict(self.model, architecture="unverified-custom-strides")
        with self.assertRaisesRegex(ValueError, "architecture"):
            registry.validate_model(model)
        with self.assertRaisesRegex(ValueError, "Unsupported architecture"):
            registry.export_architecture(model)

    def test_local_export_admin_authentication(self):
        with patch.object(registry.subprocess, "check_output", return_value='{"id": 173562962}'):
            registry.require_local_export_admin()
        for response in ('{"id": 101}', '{"id": "173562962"}', '{}', 'invalid json'):
            with self.subTest(response=response), patch.object(
                registry.subprocess, "check_output", return_value=response
            ), self.assertRaises(PermissionError):
                registry.require_local_export_admin()
        with patch.object(registry.subprocess, "check_output", side_effect=FileNotFoundError), self.assertRaises(PermissionError):
            registry.require_local_export_admin()

    def test_local_selection_rejects_non_admin_before_export(self):
        model = dict(self.model, registration_id=0)
        model["id"] = f'local:{model["architecture"]}@sha256:{model["sha256"]}'
        selection = self.root / "selected_model.json"
        registry.atomic_json(selection, {"schema_version": 1, "model": model})
        with patch.object(registry, "require_local_export_admin", side_effect=PermissionError("admin-only")):
            with self.assertRaisesRegex(PermissionError, "admin-only"):
                registry.read_selection(selection)

    def test_local_export_needs_no_catalog_and_keeps_its_own_identity(self):
        checkpoint = self.source / "experiment.pt"
        checkpoint.write_bytes(b"local checkpoint fixture")
        output = self.root / "local-installation"
        architecture = "caesar-depth32-t16-v1"

        def export_component(argv, env, check):
            model, frozen = registry.read_selection(env["CAESAR_MODEL_SELECTION"])
            self.assertEqual(frozen.read_bytes(), checkpoint.read_bytes())
            self.assertEqual(model["architecture"], architecture)
            for name in (*[f"{c}.pt2" for c in registry.COMPONENTS], *registry.TABLES):
                (Path(env["CAESAR_EXPORT_DIR"]) / name).write_bytes(b"artifact")

        with patch.object(registry, "require_local_export_admin"), patch.object(registry, "load_catalog", side_effect=AssertionError("No registration needed")), patch.object(
            compile_model.subprocess, "run", side_effect=export_component
        ):
            compile_model.compile_local_checkpoint("cpu", checkpoint, architecture, output)
            model = registry.validate_installation(output)
        self.assertEqual(model["registration_id"], 0)
        self.assertEqual(model["id"], f"local:{architecture}@sha256:{registry.sha256(checkpoint)}")
        metadata = output / "model_metadata.txt"
        metadata.write_text(metadata.read_text().replace(architecture, "unknown-architecture"))
        with self.assertRaisesRegex(ValueError, "architecture"):
            registry.validate_installation(output)

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
            registry.download(self.model, output)
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

    def test_selection_alone_does_not_register_checkpoint(self):
        registry.download(self.model, self.root / "pretrained", self.source)
        with self.assertRaisesRegex(ValueError, "not registered"):
            registry.read_selection(self.root / "pretrained/selected_model.json")

    def test_altered_registration_metadata_rejected(self):
        altered = copy.deepcopy(self.model)
        altered["min_dims"] = 4
        registry.download(altered, self.root / "pretrained", self.source)
        with patch.object(registry, "load_catalog", return_value=self.catalog):
            with self.assertRaisesRegex(ValueError, "does not match"):
                registry.read_selection(self.root / "pretrained/selected_model.json")

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

    def test_cmake_installation_validation_rejects_unregistered_metadata(self):
        for filename in (*[f"{c}.pt2" for c in registry.COMPONENTS], *registry.TABLES):
            (self.root / filename).write_bytes(b"artifact")
        registry.write_installation(self.root, self.model, "cpu")
        with patch.object(registry, "load_catalog", return_value=self.catalog):
            self.assertEqual(registry.validate_installation(self.root), self.model)
        with self.assertRaisesRegex(ValueError, "not registered"):
            registry.validate_installation(self.root)
        path = self.root / "model_metadata.txt"
        path.write_text(path.read_text().replace("min_dims=3", "min_dims=4"))
        with patch.object(registry, "load_catalog", return_value=self.catalog):
            with self.assertRaisesRegex(ValueError, "does not match"):
                registry.validate_installation(self.root)


if __name__ == "__main__":
    unittest.main()
