# Contributing to CAESAR

Contributions to code, tests, documentation, examples, and bug reports are welcome.
For a substantial feature or design change, open an issue first to discuss the
problem and proposed approach.

## Getting started

1. Fork the repository if you do not have write access, then clone your fork.
2. Create a task branch with a descriptive name, such as `fix/model-validation`.
3. Follow the platform-specific dependency, model compilation, and build steps
   in [README.md](README.md). Use a Python virtual environment.
4. Read the [codebase guide](docs/codebase.md) to find the relevant components
   and understand their dependencies.

The C++ code requires C++17 (GCC 9 or newer when using GCC), and the top-level
build requires CMake 3.18 or newer. Compile a model before building paths that
need model execution. Start with a CPU build unless your change requires a GPU.

## Formatting

Use **Black for Python** and **clang-format for C/C++**, through the scripts in
`dev/`. clang-format is the formatter; Clang itself is a compiler.
Install the versions used by the formatting CI job in your virtual environment:

```bash
python -m pip install black==26.1.0 clang-format==14.0.6
```

Run the scripts from the repository root, passing the files you changed:

```bash
bash dev/python_format.sh model_registry.py tests/test_model_registry.py
bash dev/C++_format.sh CAESAR/models/serialization.h tests/test_serialization.cpp
```

Both scripts also accept directories and multiple paths. With no arguments,
they format the current directory recursively. Format only touched files to
keep unrelated formatting changes out of your pull request. In particular,
avoid recursively formatting build directories or virtual environments.

To check those example files without modifying them:

```bash
black --check --diff model_registry.py tests/test_model_registry.py
clang-format --dry-run --Werror CAESAR/models/serialization.h tests/test_serialization.cpp
```

## Testing changes

Run checks relevant to your change and include their results in your pull
request. Add a regression test for a bug fix when practical, and cover new
behavior with tests. Keep tests in `tests/` and register new C++ executable
targets in [tests/CMakeLists.txt](tests/CMakeLists.txt).

For registry changes, run from the repository root:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 tests/test_model_registry.py
python3 -m unittest discover -s tests -p test_registration_requests.py -v
```

Registration-request tests need a checkout of `UFcompressor/UFL_MODELS`, located
through `UFL_MODELS_DIR`; some checks skip when it is unavailable. Report skips.

For C++ changes, configure with `-DBUILD_TESTS=ON` using the README's build
instructions, rebuild, and run the affected executables directly. For example:

```bash
cmake --build build --parallel 2
./build/tests/test_serialization
./build/tests/test_rans_kernels
./build/tests/test_model_metadata
```

These examples do not execute a compiled model. Integration tests require
additional setup: a matching compiled model, generated input data, and sometimes
a particular working directory. The
[cross-platform workflow](.github/workflows/cross-platform-test.yml) contains
the commands for model, compression/decompression, and CLI tests, including
platform-specific library paths. C++ tests currently run as executables rather
than through CTest.

For numerical or performance changes, report the input shape, model identity,
device, error bound, and relevant before/after results. Measure reconstruction
error when changing compression, correction, normalization, or padding. A
successful round trip alone does not establish numerical accuracy. State which
CPU/GPU and operating-system combinations you actually tested.

## Code and documentation expectations

- Keep each pull request focused on one problem; separate unrelated cleanup.
- Follow the surrounding code's conventions and explain non-obvious decisions.
- Preserve agreement between compression and decompression, model metadata,
  exported packages, and probability tables. See [model installation and
  identity](docs/models.md).
- Describe compatibility implications when changing public APIs, serialized
  formats, model loading, or device behavior. Include appropriate tests.
- Update documentation and examples when user-visible behavior changes.
- Keep generated model exports, checkpoints, datasets, build output, virtual
  environments, and credentials out of commits. Use small synthetic test data
  or a reproducible data-generation script where possible.

## Reporting bugs

Open a GitHub issue with a short description, expected and actual behavior,
exact reproduction commands, and relevant error output. Include your operating
system, compiler, Python/PyTorch versions, build options, and device. For model
or decoding problems, include the model identity and input shape. Share a small
reproducer or synthetic data rather than private datasets or credentials.

## Submitting a pull request

Use a descriptive title and explain the problem, resulting behavior, and any
linked issue. List the checks you ran, their results, and any checks you could
not run. Call out compatibility changes or remaining limitations.

Before requesting review, inspect your diff for unrelated changes and generated
files, format touched code, and run the relevant tests. Check the GitHub Actions
results and address failures. Respond to review feedback and leave merging and
release publishing to the maintainers.
