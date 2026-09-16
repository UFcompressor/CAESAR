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
