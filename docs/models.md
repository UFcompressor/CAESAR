# Models and installation

The UFL_MODELS repository owns `model_catalog.json`. CAESAR ships a synchronized
snapshot so a release can download immutable checkpoint URLs and compile offline.
All four currently registered checkpoints support 3D–5D inputs. CAESAR v1
(registration 1) and CAESAR v2 (registration 2) are foundation models; v2 is the
newer optimized release and remains the default. eelsM1 (registration 3) is a
domain fine-tune trained and tested on one EELS dataset. microscopy (registration
4) is a domain-specific model trained and tested across four microscopy datasets.

## Download and compile

```sh
python3 model_registry.py --list
python3 model_registry.py caesar_v2
python3 compile_model.py cpu
```

The shell wrapper `compile_model.sh` invokes the compilation command.
The Python entrypoints also work on Windows. Use the same Python environment that
contains torch and CAESAR's dependencies. Choose `cpu`, `cuda` (including ROCm),
`mps`, or `xpu`; the backend must support AOTI compilation on your platform.

For a local checkpoint repository:

```sh
python3 model_registry.py caesar_v2 --source-dir ../UFL_MODELS
python3 compile_model.py cpu
```

Download accepts an exact `ufl:<number>@sha256:<hash>` identity as well as a
registered name. `--output` selects the download directory; pass its
`selected_model.json` to the compiler with `--selection`. The compiler's
`--output` selects the exported installation directory. Standard CMake packaging
uses `exported_model/`. `CAESAR_MODEL_DIR` selects a compiled installation at runtime.

Downloads preserve actual checkpoint filenames, verify SHA-256, and publish a
selection only after verification. Compilation independently checks the bundled
UFL catalog; a hand-written selection does not register a model. CMake also
validates the compiled installation against that catalog before configuring. All three
exporters use a frozen copy of the same verified checkpoint. A staged export is
published only when all three packages and six entropy tables exist. A failed
export preserves the old installation. Do not replace an installation while a
process is using it. An interrupted installer may leave `exported_model.lock`;
remove that directory only after confirming the installer is no longer running.
Individual exporter scripts are internal to `compile_model.py`.

## Registration and identity

Identity is `ufl:<registration_id>@sha256:<checkpoint_sha256>`. Registration numbers,
names, hashes, and full identities must be unique. Never reuse a number or replace
a registered checkpoint's bytes. A new checkpoint gets a new registration, even
when it is another version of the same model family. Names and descriptions help
users choose; the registration plus hash is what compressed data must record.
The device is installation-specific and is not part of checkpoint identity.

Registration commands now live in CAESAR; UFL_MODELS remains the authoritative
catalog database. Run `python3 register_model.py add --help` to submit a checkpoint
or `python3 register_model.py remove MODEL_NAME` to request retirement. The trusted
UFL_MODELS workflow derives ownership from the authenticated GitHub issue author's
stable account ID, validates requests, and commits accepted changes automatically.
Users may retire only their own models; administrator E53klasky may retire any
model. Retirement preserves the full record and reserves its identity.

See [Automated model registration](model_registration.md) for commands, validation,
one-time GitHub setup, admin rules, and catalog snapshot synchronization. Local
scripts do not enforce repository permissions; restrict direct catalog and workflow
writes to trusted administrators and automation. CAESAR compilation continues to
validate the bundled catalog snapshot and strictly load compatible state_dicts.

## Local metadata and cache

`model_metadata.txt` records schema version, registration number, model name and
identity, checkpoint hash and original filename, architecture, supported ranks,
and compiled device. Runtime derives the artifact directory from that file's
location, allowing installation relocation. The training checkpoint need not be
deployed beside the compiled packages. Descriptions and download URLs remain in
the catalog. Old `model_name.txt`/`model_device.txt` installations must be re-exported;
there is no name-only or implicit CPU fallback.

Metadata initializes once per process using C++ static initialization. Each thread
has its own identity-keyed cache of lazily loaded AOTI runners and probability
tables; MPI ranks have independent process memory. Use a separate Compressor or
Decompressor object in each thread. Calling one from a different thread fails
explicitly. Thread isolation costs extra model memory but avoids sharing a runner
whose concurrent invocation is not guaranteed safe. Cached runners use shared
ownership so `clear()` cannot invalidate handles held by existing objects. Windows retains the existing
AOTI teardown workaround (cached runners are not destroyed at thread teardown).

One compiled installation is selected per process. Set `CAESAR_MODEL_DIR` before
first use; restart the process to switch installations. Missing files in an
explicit directory fail rather than falling back to another installation. All
artifacts are loaded from the metadata's directory, never combined across paths.
The runtime device must match the compilation target and be available.

### Runtime cache lifetime and clearing

`ModelCache` is an in-memory runtime cache, separate from downloaded checkpoints
in `pretrained/` and compiled artifacts in `exported_model/`. It does not cache
input tensors, compressed results, or reconstructed fields. Each thread lazily
loads and reuses its own compressor, hyper-decompressor, decoder, and six
probability tables. AOTI package-loader construction is serialized across threads;
that does not make a compressor/decompressor object callable from another thread.

Reuse objects on their creating thread for repeated fields. The first operation
that needs a package/table pays its load cost; warm runs reuse it. Each additional
worker thread or MPI rank can add model/table memory. There is no automatic LRU,
capacity setting, or cross-thread/rank model sharing. Compressor/decompressor
objects also hold copies of probability tables and shared runner handles.

Advanced callers can clear the cache of the **calling thread**:

```cpp
#include <caesar/models/model_cache.h>

// First destroy compressor/decompressor objects in this thread if the aim is
// to release their model handles as well.
ModelCache::instance().clear();
```

`clear()` resets the cached three runner handles and six probability tables.
Existing objects retain their runner handles and table copies and remain usable;
subsequent cache getters reload artifacts. Clearing does not reset process-wide
model metadata, select another model/device, delete files, or clear another
thread's cache. It does not guarantee an immediate drop in GPU/process memory,
since live handles and backend allocators may retain allocations. On Windows,
the thread-local registry is intentionally retained at teardown to avoid unsafe
AOTI destruction; explicit clearing still resets its entries' loaded resources.

## ADIOS integration contract (next step)

Compression must store `get_model_id()` in the ADIOS buffer. Decompression reads
that identity and calls `require_model(required_id)` before decoding, or supplies
it to `Decompressor(required_id)`. A mismatch or missing installation
throws an error naming the required model. ADIOS should surface that error.
`get_model_metadata().require_dims(rank)` checks the supported original input rank.

This change does not modify the ADIOS operator or existing compressed-buffer
serialization. Existing name-only files cannot establish an exact checkpoint
identity; their compatibility policy belongs to that later integration.

## Input preparation

Padding and conversion to the model's 5D input are internal CAESAR responsibilities.
The 3D–5D catalog range describes user input ranks handled through CAESAR's internal
preparation; it is not an assertion that the exported network accepts arbitrary
shapes directly. Deferred removal of `PaddingInfo.H/W/was_padded` belongs to the
serialization cleanup, not model installation.

## Verification

```sh
python3 -m unittest discover -s tests -p test_model_registry.py -v
python3 tests/check_registered_checkpoints.py --source-dir ../UFL_MODELS
./build/tests/test_model_metadata
CAESAR_MODEL_DIR="$PWD/exported_model" ./build/tests/test_model_cache
```

The checkpoint check strictly loads all four registrations into each of the three
export models. The CPU installation and `test_caesarCD` exercise actual compiled
inference. Metadata tests cover missing/mismatched identities and malformed
manifests; registry tests cover duplicates, unregistered selections, hash failures,
and preserving the previous installation when compilation fails. Cache tests
cover reuse, separate thread runners, and retaining live handles across `clear()`.
