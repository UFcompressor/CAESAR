# CAESAR examples

## Train the VAE compressor

The training entry point uses
`pyCAESAR.models.compress_4_train_modules3d_mid_SR`. This is the trainable
counterpart of `compress_modules3d_mid_SR`; the latter is the deployment model
used for the three AOTI-compiled components and is not modified by training.

1. Copy [config_npz.example.yaml](config_npz.example.yaml) and replace its
   `data_path` with your NPZ file. The file must contain a `data` array arranged as
   `[variable, section, time, height, width]`.
2. Select the dataset names using `--train_set` and `--test_set`.
3. Run the module from the repository root:

```bash
python3 -m pyCAESAR.train_vae3d \
  --config examples/config_npz.example.yaml \
  --save_path snapshots/example-vae \
  --train_set example_npz \
  --test_set example_npz \
  --batch_size 8 \
  --iterations 100 \
  --model_dim 16 \
  --sr_dim 16
```

`--iterations` is expressed in thousands of optimizer steps, so the example
above requests 100,000 steps. To load an existing checkpoint, add
`--pretrain path/to/checkpoint.pt`.

The repository-level `train.sh` contains the same launcher pattern for a real
multi-dataset training run.

### ADIOS input

Use [config_adios.example.yaml](config_adios.example.yaml) for `.bp`, `.bp4`,
or `.bp5` input. Activate your training virtual environment and make sure its
Python can import `adios2` (`python -c 'from adios2 import FileReader'`). Replace
the example path and ranges with values for your file, then run from the
repository root:

```bash
python -m pyCAESAR.train_vae3d \
  --config examples/config_adios.example.yaml \
  --train_set example_adios --test_set example_adios \
  --save_path snapshots/example-adios \
  --batch_size 8 --iterations 100 --model_dim 16 --sr_dim 16
```

For multi-step ADIOS files containing 3D float/double fields, each selected
step becomes a section. The first spatial axis becomes the frame axis; a
16-frame window therefore contains spatial slices from one ADIOS step.
Variables are sorted by name before selection and converted to float32.

| Option | ADIOS multi-step 3D fields | NPZ `data` array |
| --- | --- | --- |
| `variable_idx` | Indices in sorted float/double variable names; `null` loads all | List of indices on the variable axis |
| `section_range` | ADIOS steps | Sections on the second axis |
| `frame_range` | Slices on the first spatial axis | Frames on the third axis |

Ranges use an exclusive end: `[8, 9]` selects only the last step of a
nine-step file. `[0, 1025]` selects all 1,025 slices of a 3D field.
For a single-step ADIOS file, section and frame ranges select axes of the
loaded array instead; use `[0, 1]` for the section of a single 3D field.

Training `n_overlap: 15` with `n_frame: 16` advances one slice/frame per sample.
Testing `n_overlap: 0` advances 16. Incomplete final windows are padded by
reversing tail slices. Training and evaluation keep all windows, without
constant-block filtering. Instance normalization requires nonconstant crops.

Both formats are loaded into CPU RAM, and training and evaluation retain
separate datasets. Evaluation also allocates full reconstruction buffers;
select fewer variables if these exceed available memory. The example evaluates
the same ADIOS selection used for training; choose a separate step or subset
when you need held-out evaluation.

## In-memory C++ round trip

Build with `-DBUILD_EXAMPLES=ON`, then run `build/examples/hello_caesar` with
`CAESAR_MODEL_DIR` pointing to the exported model installation. The example
passes an original 3D tensor with `CompressionConfig::n_frame = 8` and restores
its shape using `decompress(compressed)`. The field has shape `8x256x256`;
the example checks its reconstructed shape and reports range-normalized RMSE
(NRMSE) against a target of `0.001`. Mean/range normalization is internal.

```bash
CAESAR_MODEL_DIR="$PWD/exported_model" build/examples/hello_caesar gae
CAESAR_MODEL_DIR="$PWD/exported_model" build/examples/hello_caesar lbrc
```

See [the API guide](../docs/public_api.md)
for correction methods, 5D variable selection, and CPU validation.
