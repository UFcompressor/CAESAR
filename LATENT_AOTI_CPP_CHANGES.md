# Trained depth variants: exact export and C++ edits

The shared depth-architecture registry, Python exporter, and C++ dimension edits
below are now applied in this checkout. No new checkpoint has been added to the
catalog. The existing registered models retain their identities and defaults.
The custom [64,4,32,32] checkpoint still requires its exact training architecture
and checkpoint file before registration, AOTI export, and runtime validation.
No AOTI export or round-trip test of that custom checkpoint has been performed.

## Direct local testing (no registration)

The current branch also accepts a local checkpoint directly:

```bash
python3 compile_model.py cuda \
  --checkpoint snapshots/xcompact3d-latet-size-64-4-32-32/Xcompact3d_latent_32_32.pt \
  --architecture caesar-depth32-t16-v1 \
  --output exported_experiments/xcompact3d-depth32
python3 model_registry.py --validate-installation exported_experiments/xcompact3d-depth32
cmake --build build -j2
export CAESAR_MODEL_DIR="$PWD/exported_experiments/xcompact3d-depth32"
```

This is an explicit trial of the documented depth32 architecture, which produces
[64,8,32,32], not proof that it matches the training run. A [64,4,32,32] stride
experiment still needs its exact training implementation. Strict loading catches
tensor mismatches, but cannot establish the training strides.

Local exports are admin-only: `gh api user` must authenticate GitHub account ID
173562962 (E53klasky), matching the current UFL administrator policy. Run
`gh auth login` with that account first. This is a local CLI permission check;
users who edit their own source can change it. Published artifact loading does
not require GitHub authentication.

Local exports bypass UFL, checkpoint-size comparisons, and catalog selection.
They retain strict weight loading and identify themselves as
`local:<architecture>@sha256:<checkpoint hash>` (registration_id=0).
All components and probability tables are staged before publishing the output.
The C++ metadata reader and Python installation validator accept this identity.
The checkpoint is never registered, uploaded, or added to the public catalog.
Rebuild C++ before using a local export; use `--n-frame 16` for the depth variants.

## 0. Resolve the shape before exporting

The COMPLETE-STAGE experiment you pasted produces:

| New architecture name | dim_mults | reverse_dim_mults | Input T | Latent C,K,L,L | Q |
|---|---|---|---|---|---|
| caesar-depth32-t16-v1 | [1,2,4] | [4,2] | 16 | 64,8,32,32 | 8 |
| caesar-depth16-t16-v1 | [1,2,3,4] | [4,3,2] | 16 | 64,4,16,16 | 4 |
| caesar-depth8-t16-v1 | [1,2,3,4,4] | [4,4,3,2] | 16 | 64,2,8,8 | 2 |

These are proposed architecture identifiers for this implementation, not existing
registry entries. They describe architecture; each checkpoint still gets its own
registered model ID and SHA-256.

**None produces [64,4,32,32].** If that is your actual trained latent, do not
apply the depth32 exporter changes blindly. Obtain the actual training model
construction and stride/padding changes. State-dict shapes do not describe all
stride settings. A checkpoint can load successfully with incorrect strides.
The C++ values for that actual shape would be T=16,K=4,L=32,Q=8, assuming the
same spatial hyperprior, but the Python architecture must reproduce its training.

The rest of this guide implements the complete-stage variants above. For a
confirmed custom [64,4,32,32] model, add a distinct architecture identifier and
its exact training implementation rather than claiming it is depth32-t16-v1.

## 1. model_registry.py: accept explicit depth architecture identifiers

Keep the existing ARCHITECTURE constant. Add immediately below it:

```python
DEPTH_ARCHITECTURES = {
    "caesar-depth32-t16-v1": ([1, 2, 4], [4, 2], 16, 8, 32, 8),
    "caesar-depth16-t16-v1": ([1, 2, 3, 4], [4, 3, 2], 16, 4, 16, 4),
    "caesar-depth8-t16-v1": ([1, 2, 3, 4, 4], [4, 4, 3, 2], 16, 2, 8, 2),
}


def export_architecture(model):
    architecture = model["architecture"]
    if architecture == ARCHITECTURE:
        return [1, 2, 3, 4], [4, 3, 2], 8, 2, 16, 4
    try:
        return DEPTH_ARCHITECTURES[architecture]
    except KeyError as error:
        raise ValueError(f"Unsupported architecture: {architecture}") from error
```

In validate_model(), replace:

```python
    if not model.get("description") or model.get("architecture") != ARCHITECTURE:
```

with:

```python
    if not model.get("description") or model.get("architecture") not in (
        ARCHITECTURE, *DEPTH_ARCHITECTURES
    ):
```

The architecture identifier now determines all dimensions. Existing installation
metadata already writes and validates model['architecture']; keep that behavior.
No compile_model.py changes are required for the registered-model route here.
Its frozen selection and checkpoint copy will preserve this architecture value.

## 2. model_catalog.json: add your registered checkpoint

After registering the checkpoint, append a model entry to the models array.
Use the real registration number, exact hash, filename, and immutable HTTPS URL.
Do not copy an existing foundation model's identity. Example TEMPLATE only:

```json
{
  "name": "hurricane_depth32",
  "id": "ufl:REPLACE_NUMBER@sha256:REPLACE_HASH",
  "sha256": "REPLACE_HASH",
  "description": "Hurricane scratch-trained complete-stage spatial-32 model, T=16 K=8",
  "filename": "hurricane_depth32.pt",
  "url": "https://REPLACE_WITH_IMMUTABLE_CHECKPOINT_URL/hurricane_depth32.pt",
  "min_dims": 3,
  "max_dims": 5,
  "architecture": "caesar-depth32-t16-v1"
  ,"registration_id": 999999
}
```

Replace 999999 too. Preserve any additional owner/display fields supplied by your
registration. Set min_dims/max_dims to the model's actual supported input ranks.
The placeholders above deliberately do not pass validation.

Compute the checkpoint hash with:

```bash
sha256sum snapshots/YOUR_ACTUAL_CHECKPOINT.pt
```

Do not set the existing models' architecture to the new value. No change to the
default_model field is necessary when selecting the new model explicitly.

## 3. All three Python exporters: select architecture from the verified model

Files:

- CAESAR_compressor.py
- CAESAR_hyper_decompressor.py
- CAESAR_decompressor.py

In EACH file change the first import to:

```python
from model_registry import export_context, export_architecture
```

Immediately after the existing bottom-level line:

```python
selected_model, checkpoint_path, export_dir = export_context()
```

add:

```python
dim_mults, reverse_dim_mults, input_frames, latent_frames, latent_size, hyper_size = (
    export_architecture(selected_model)
)
```

In the actual bottom-level `model = CompressorMix(...)` call replace ONLY:

```python
    dim_mults=[1, 2, 3, 4],
    reverse_dim_mults=[4, 3, 2],
```

with:

```python
    dim_mults=dim_mults,
    reverse_dim_mults=reverse_dim_mults,
```

Keep dim=16, hyper_dims_mults=[4,4,4], channels=1, out_channels=1,
d3=True, sr_dim=16, and the existing strict checkpoint loading code.
Do not change every constructor default in the file.

In EACH file, inside ResnetCompressor.__init__, immediately BEFORE its existing
self.build_network(), insert:

```python
        self.n_3d_stages = len(self.in_out) - 2
```

In EACH file, inside ResnetCompressor.build_network(), DECODER LOOP ONLY:

```python
# OLD
            d3 = self.d3 if ind < 2 else False
# NEW
            d3 = self.d3 if ind < self.n_3d_stages else False
```

In EACH file, decode() ONLY:

```python
# OLD
            if i == 2:
# NEW
            if i == self.n_3d_stages:
```

Keep encode()'s `if i == 2` and the encoder's `ind >= 2` unchanged.

In CAESAR_compressor.py and CAESAR_hyper_decompressor.py, decode() ONLY:

```python
# OLD
                x = x.reshape(-1, self.t_dim // 4, *x.shape[1:])
# NEW
                x = x.reshape(
                    -1, self.t_dim // (2 ** self.n_3d_stages), *x.shape[1:]
                )
```

CAESAR_decompressor.py already accepts grouped [B,K,C,L,L]. Leave its commented
reshape commented. Its self.t_dim=8 is not used by this active decode path.

Replace the example_inputs assignment near the bottom of each file:

CAESAR_compressor.py:

```python
    example_inputs = (
        torch.randn(8, 1, input_frames, 256, 256, device=device).float(),
    )
```

CAESAR_hyper_decompressor.py:

```python
    example_inputs = (
        torch.randn(8, 64, hyper_size, hyper_size, device=device).float(),
    )
```

CAESAR_decompressor.py:

```python
    example_inputs = (
        torch.randn(10, latent_frames, 64, latent_size, latent_size, device=device).float(),
    )
```

Keep the existing export/AOTI packaging calls, dynamic batch specification, and
probability-table writers. Do not hard-code checkpoint paths or bypass
export_context() when using this complete registered installation route.

## 4. CAESAR/models/model_metadata.h: allow the same architectures in C++

Inside read_model_metadata(), replace this final condition in the large identity
validation `if`:

```cpp
      metadata.architecture != "caesar-bcrn-v1")
```

with:

```cpp
      (metadata.architecture != "caesar-bcrn-v1" &&
       metadata.architecture != "caesar-depth32-t16-v1" &&
       metadata.architecture != "caesar-depth16-t16-v1" &&
       metadata.architecture != "caesar-depth8-t16-v1"))
```

Keep all registration-number, SHA-256, and model-ID checks.

## 5. CAESAR/models/model_utils.h and model_utils.cpp: expose dimensions

Append to model_utils.h, after the existing get_model_metadata declaration:

```cpp
struct ModelDimensions {
  int64_t input_frames;
  int64_t latent_frames;
  int64_t latent_size;
  int64_t hyper_size;
};

ModelDimensions get_model_dimensions();
```

Append to model_utils.cpp:

```cpp
ModelDimensions get_model_dimensions() {
  const auto &architecture = get_model_metadata().architecture;
  if (architecture == "caesar-bcrn-v1")
    return {8, 2, 16, 4};
  if (architecture == "caesar-depth32-t16-v1")
    return {16, 8, 32, 8};
  if (architecture == "caesar-depth16-t16-v1")
    return {16, 4, 16, 4};
  if (architecture == "caesar-depth8-t16-v1")
    return {16, 2, 8, 2};
  throw std::runtime_error("Unsupported model architecture: " + architecture);
}
```

## 6. CAESAR/models/caesar_compress.cpp

Replace:

```cpp
  if (config.n_frame != 8)
    throw std::invalid_argument(
        "n_frame is required and must be 8 for the installed architecture");
```

with:

```cpp
  const auto dimensions = get_model_dimensions();
  if (config.n_frame != dimensions.input_frames)
    throw std::invalid_argument(
        "n_frame must be " + std::to_string(dimensions.input_frames) +
        " for the installed architecture");
```

In the SAME function, replace the reconstruction grouping:

```cpp
      std::vector<int64_t> new_shape = {-1, 2};
```

with:

```cpp
      std::vector<int64_t> new_shape = {-1, dimensions.latent_frames};
```

This reconstruction is also used to compute correction residuals, so changing
only the decompressor would be insufficient. Keep 256x256 block sizes unchanged.

In caesar_compress.h update the comment that says only eight-frame windows are
supported to say the required window length comes from the installed architecture.

## 7. CAESAR/models/caesar_decompress.cpp

Replace:

```cpp
  if (comp_result.n_frame != 8)
    throw std::invalid_argument(
        "Compressed n_frame must be 8 for this architecture");
```

with:

```cpp
  const auto dimensions = get_model_dimensions();
  const int64_t K = dimensions.latent_frames;
  const int64_t L = dimensions.latent_size;
  const int64_t Q = dimensions.hyper_size;
  if (comp_result.n_frame != dimensions.input_frames)
    throw std::invalid_argument(
        "Compressed n_frame does not match the installed architecture");
```

In this same function make ALL the following targeted substitutions (both CPU
and GPU paths matter):

| Existing code | Replacement |
|---|---|
| `build_indexes_tensor({1, 64, 4, 4})` (both occurrences) | `build_indexes_tensor({1, 64, Q, Q})` |
| `lat_start += (size_t)batch_size * 2` | `lat_start += (size_t)batch_size * K` |
| `lat_start + (size_t)batch_size * 2` | `lat_start + (size_t)batch_size * K` |
| `TORCH_CHECK(cur_latents % 2 == 0, "cur_latents must be even.");` | `TORCH_CHECK(cur_latents % K == 0, "Incomplete latent frame group");` |
| `size_t cur_samples = cur_latents / 2;` | `size_t cur_samples = cur_latents / K;` |
| `size_t sample_start = lat_start / 2;` | `size_t sample_start = lat_start / K;` |
| `hyper_indexes_gpu.expand({(int64_t)cur_latents, 64, 4, 4})` | `hyper_indexes_gpu.expand({(int64_t)cur_latents, 64, Q, Q})` |
| `torch::empty({(long)cur_latents, 64, 4, 4}, cpu_float_opts)` | `torch::empty({(long)cur_latents, 64, Q, Q}, cpu_float_opts)` |
| `hyper_data + i * 64 * 4 * 4` | `hyper_data + i * 64 * Q * Q` |
| `constexpr int64_t latent_elements = 64 * 16 * 16;` | `const int64_t latent_elements = 64 * L * L;` |
| `{(int64_t)cur_latents, 64, 16, 16}` | `{(int64_t)cur_latents, 64, L, L}` |
| `torch::empty({(long)cur_latents, 64, 16, 16}, cpu_float_opts)` | `torch::empty({(long)cur_latents, 64, L, L}, cpu_float_opts)` |
| `std::vector<int64_t> new_shape = {-1, 2};` | `std::vector<int64_t> new_shape = {-1, K};` |

Immediately BEFORE the latent batching loop add:

```cpp
  TORCH_CHECK(batch_size > 0, "batch_size must be positive");
  TORCH_CHECK(comp_result.encoded_latents.size() % K == 0,
              "Incomplete latent frame group in compressed input");
```

Do not globally replace 2, 4, 8, or 16. Unrelated numerical constants must stay.
Leave GAE and normalization algorithms unchanged. Keep require_model() intact.

## 8. CAESAR/CAESAR.cpp: CLI validation

Ensure this file includes model_utils.h, using:

```cpp
#include "models/model_utils.h"
```

Replace:

```cpp
      if (n_frame != 8)
        throw std::invalid_argument("Compression requires --n-frame 8");
```

with:

```cpp
      const auto dimensions = get_model_dimensions();
      if (n_frame != dimensions.input_frames)
        throw std::invalid_argument(
            "Compression requires --n-frame " +
            std::to_string(dimensions.input_frames));
```

Decompression continues reading n_frame and model_id from the compressed file.
No new bitstream dimension fields are needed here because the installed model
identity selects the architecture and the existing file stores n_frame.

## 9. Select, compile, rebuild, run

After replacing the catalog placeholders with your actual registration:

```bash
# Select/download your registered checkpoint. Use the actual registered name.
python model_registry.py hurricane_depth32 --output pretrained/hurricane_depth32

# Compile all three packages and collect all six tables plus metadata.
python compile_model.py cuda \
  --selection pretrained/hurricane_depth32/selected_model.json \
  --output exported_experiments/hurricane_depth32

# Validate installation completeness and registered metadata.
python model_registry.py --validate-installation exported_experiments/hurricane_depth32

# Rebuild an already-configured C++ build directory.
cmake --build build -j6

# Use an absolute installation path in every inference process.
export CAESAR_MODEL_DIR="$PWD/exported_experiments/hurricane_depth32"
```

If the checkpoint already exists locally, `model_registry.py` also accepts
`--source-dir DIRECTORY`. Place the checkpoint there under its catalog filename
and add that argument to the selection command; it still verifies the hash.
For a fresh C++ configuration use the GPU build instructions in README.md with
your cluster's LibTorch/CUDA/dependency paths; do not change dependency versions
just to change latent dimensions.

Use your usual C++ compression command with `--n-frame 16` and the same input
shape/error bound/correction settings as before. Decompression uses the same
CAESAR_MODEL_DIR and obtains frame count from the compressed file. The old
installations remain selectable by changing CAESAR_MODEL_DIR between processes.
Do not swap installation directories inside a running process: metadata/device
selection are cached.

## 10. Required validation before long data runs

1. Verify the checkpoint's Python compressor emits [B*K,64,L,L], hyperlatent
   [B*K,64,Q,Q], and the grouped decoder reconstructs B*16 frames at 256x256.
2. Compare eager component outputs to the exported AOTI components. Strict weight
   loading alone cannot prove correct stride/padding or temporal grouping.
3. Run C++ compression/decompression on a small field with correction enabled.
   Check output shape and max absolute error against the requested bound.
4. Include multiple batches and a final partial batch to exercise grouping.
5. Test CPU entropy decoding and GPU entropy decoding if both are used.
6. Record total bytes across all output files, including GAE correction data.

Keep TORCH_COMPILE_DISABLE=1 for training if needed by your cluster. If inherited
by an export process and it interferes with AOTI compilation, unset it for that
export process. Training's compiled-backward issue is separate from inference
export. Actual AOTI compilation and C++ runtime validation still need to run on
the target software/GPU environment.
