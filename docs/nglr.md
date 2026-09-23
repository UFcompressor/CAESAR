# NGLR correction

NGLR trains a fresh native C++ network for each compression call, using the
original input and the CAESAR foundation-model reconstruction. It does not load
a pretrained NGLR checkpoint. The network follows `tests/NGLR_train.py`: seven
reconstruction feature channels, residual Conv3D/GroupNorm blocks, eight causal
Lorenzo context channels, and a pointwise fusion network.

Select correction with the existing compression call's optional arguments:

```cpp
nglr::NGLRTrainOptions training; // 50 epochs maximum, batch size 8
training.epochs = 50;
auto result = compressor.compress(config, batch_size, rel_eb,
                                 caesar::CorrectionMethod::NGLR, training);
```

`CorrectionMethod` has fixed byte values `GAE=0`, `LBRC=1`, and `NGLR=2`.
GAE remains the default. Decompression uses the method stored in the result.
There is no independent decoder choice that can disagree with the saved data.

Quantization normalizes by the original input's range, checks reconstructed
NRMSE, and selects a feasible step with a bounded search. Reflected tail frames
are removed before NGLR statistics, training, and coding. Inputs already within
the bound need no model; constant fields are reconstructed from their saved
constant value. NGLR currently accepts float32 tensors. Precision or integer-range
failures raise errors instead of looping indefinitely or overflowing.

Training uses AdamW and the Charbonnier objective. Subtracting its constant
`1e-6` floor makes exact zero loss observable without changing gradients.
Training stops after **three consecutive zero-loss epochs**, or the configured
epoch limit; a nonzero epoch resets the counter. The best epoch's weights are
retained. Zero neural loss concerns prediction of quantized residuals; it does
not replace the separate reconstructed-NRMSE check.

Training uses the compressor's device. Strict diagonal correction encoding and
decoding default to CPU float32 network inference. This makes the NGLR
predictor execution path consistent; it does not resolve the foundation model's
existing GPU-to-CPU latent-index portability issue. Training precomputes block
layout and scale estimates, and builds contexts per minibatch to limit retained
context memory. Large block/batch configurations still require substantial memory.

The direct NGLR API accepts an optional `codec_device` for compression and
decompression. Use the same device for both predictor paths; cross-device
prediction equivalence is not established. `test_nglr_compare` requires CUDA
and explicitly uses `cuda:0` for training and both codec inference paths.
Normalization, quantization, and bitplane/Zstd processing still use CPU.
Run this comparison on the GPU host with
`build/tests/test_nglr_compare /path/to/compare_out` after rebuilding.

## Saved metadata

`CompressionResult` contains `nglrMetaData` and `nglr_comp_data`. Metadata schema 1
identifies this native network layout and includes correction/constant flags,
input shape, normalization and quantization scalars, causal context scales,
block dimensions, architecture dimensions, and named float32 weights with their
shapes. Decode validates schema, dimensions, parameter names/shapes/counts, and
finite weights. No model-directory path is required at decode time.

The existing CLI and C/D test metadata writers now carry these fields for NGLR.
Their one-byte correction tags retain the old GAE/LBRC values. New NGLR files
require updated readers. Compression ratios in C/CD include NGLR model and
correction bytes. This extends the current writers; it does not introduce the
planned CAESAR serialization API.

ADIOS operator changes are deferred. The NGLR fields above are available for
that later integration.

## Commands and verification

```sh
build/CAESAR/caesar compress input.bin --shape 1,1,20,256,256 \
  --error-bound 0.00001 --correction nglr --n-frame 8
build/tests/test_caesarC 0.00001
build/tests/test_caesarD
build/tests/test_caesarCD
```

The C/D/CD programs retain their existing dataset paths and error-bound settings.
To test a correction method, edit the clearly marked `correction_method` constant
in `test_caesarC.cpp` or `test_caesarCD.cpp`; it defaults to `GAE`. The D test reads
the method saved by C. There is no correction command-line argument for these tests.
These commands are for later user-run pipeline validation, not unit tests.

CPU unit tests require no trained foundation model execution:

```sh
build/tests/test_correction_method
build/tests/test_nglr_metadata
build/tests/test_NGLR
build/tests/test_nglr_file_metadata
```

The tests cover correction selection, metadata validation, optimizer updates under
an enclosing inference-mode guard, early stopping, weight reload, correction
codec reconstruction on synthetic tensors, constant/no-correction cases, malformed
payloads, and file metadata. They do not
establish full-pipeline error bounds, GPU behavior, or cross-platform portability.
The separate temporary three-error-bound test is intentionally deferred.
