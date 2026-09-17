# Things to know about the codebase

This page is a map for contributors. It describes where work belongs and which
parts need to stay in agreement. It intentionally avoids documenting the
compression API and the compressed-buffer metadata layout while those are
changing.

## Repository map

| Location | Responsibility |
| --- | --- |
| `CAESAR/` | C++17 library, CLI, data preparation, compression, and decompression. |
| `CAESAR/dataset/` | Blocking, normalization, filtering, and dataset preparation. |
| `CAESAR/models/` | Model execution, entropy coding, GAE and LBRC correction, model loading, and runtime helpers. |
| `pyCAESAR/` | Python model definitions, data loading, and training. |
| Root Python scripts | Model registry, checkpoint selection, and AOTI export. |
| `tests/` | Python and C++ checks. C++ test targets are registered in `tests/CMakeLists.txt`. |
| `examples/` | Small integration examples. |
| `docs/models.md` | Current model installation and identity details. Review it when changing model artifacts. |
| `docs/TODO.txt` | Known limitations and planned work. |

## How the pieces fit together

The Python export path builds three AOTI `.pt2` packages from one selected
checkpoint. The C++ runtime loads those packages and six matching `vbr_*` and
`gs_*` `.bin` probability tables. The tables are part of the model installation,
and entropy decoding depends on using the matching set. Model selection and
artifact validation are handled by `model_registry.py`, `compile_model.py`,
`model_catalog.json`, and the C++ model loading code.

Input preparation and restoration live in the C++ dataset and data utility
code. The compressor produces learned latents, uses the entropy coder, and may
apply correction paths such as GAE or LBRC. The decompressor reverses these
steps. See the source in `CAESAR/models/` for the current call sequence; the
configuration and serialized format are under active change.

The runtime keeps model metadata once per process and caches model runners and
probability tables by model identity per thread. A compressor or decompressor
belongs to the thread that created it. Model installation changes require a
new process so that cached runners and tables stay consistent.

## Boundaries to check when changing code

- **Model artifacts:** Keep the three exported packages and all six matching
  probability tables together. A checkpoint's registration and hash identify
  the model; a device choice does not change that identity.
- **Cross-device decoding:** GPU-compressed output can currently decode as
  garbled data on CPU. Latent indexes are not saved, and the compressed output
  does not carry the required `.bin` tables. `docs/TODO.txt` records the planned
  portability work and the possible index bit packing optimization.
- **Numerical validation:** A requested error bound is a target to measure.
  Filtering and possibly padding can make actual error exceed it. Check the
  reconstructed data when changing these paths.
- **Platform support:** Local CPU tests do not establish GPU or cross-platform
  behavior. Apple GPU double-precision data is unsupported in this path.
- **ADIOS integration:** Preserve the information needed to identify the model
  and decode a buffer. The serialization contract is being revised; avoid
  relying on an undocumented layout.

## Development checks

Use the dependency and build instructions in the root `README.md`. For a local
CPU build, run the affected Python tests and C++ binaries under `build/tests/`.
Some model tests require a compiled installation in `exported_model/`; GPU
tests require appropriate hardware and dependencies. Report which environments
were actually tested.

Format only touched files with `dev/python_format.sh` or
`dev/C++_format.sh`. Keep new checks in `tests/` and register new C++ test
targets in `tests/CMakeLists.txt`. Create a task branch for repository work and
leave publishing to the maintainer.
