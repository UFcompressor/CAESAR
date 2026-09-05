# CAESAR examples

## Train the VAE compressor

The training entry point uses
`pyCAESAR.models.compress_4_train_modules3d_mid_SR`. This is the trainable
counterpart of `compress_modules3d_mid_SR`; the latter is the deployment model
used for the three AOTI-compiled components and is not modified by training.

1. Copy `config_vae3d.example.yaml` and replace the two example `data_path`
   values with your NPZ files. Each file must contain a `data` array arranged as
   `[variable, section, time, height, width]`.
2. Select the dataset names using `--train_set` and `--test_set`.
3. Run the module from the repository root:

```bash
python3 -m pyCAESAR.train_vae3d \
  --config examples/config_vae3d.example.yaml \
  --save_path snapshots/example-vae \
  --train_set example_train \
  --test_set example_test \
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
