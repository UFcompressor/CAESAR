"""Serial ADIOS2/NPZ input shared by compression and training."""

import numpy as np


GX_SHAPE = (2, 16, 4, 96, 83, 42, 2)


def read_array(path, variable="G"):
    """Try ADIOS2 first, then an NPZ archive with `data` (or `G`)."""
    try:
        import adios2

        with adios2.FileReader(str(path)) as reader:
            data = reader.read(variable)
            if data is None:
                raise ValueError(f"Variable {variable!r} has no data")
            return np.asarray(data)
    except Exception as exc:
        adios_error = str(exc)
    try:
        with np.load(path, allow_pickle=False) as archive:
            key = "data" if "data" in archive else variable
            return np.asarray(archive[key])
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read {path}: ADIOS2 ({adios_error}); NPZ ({exc})"
        ) from exc


def scientific_layout(data):
    """Return (variable, section, depth, height, width) layout."""
    if data.ndim == 7:
        if data.shape != GX_SHAPE:
            raise ValueError(f"Expected GX shape {GX_SHAPE}, got {data.shape}")
        # Variable order is (species, moment, component, real/imaginary).
        data = data.transpose(0, 1, 2, 6, 3, 4, 5).reshape(256, 1, 96, 83, 42)
    if data.ndim != 5:
        raise ValueError(f"Expected a 5D scientific array or 7D GX G, got {data.shape}")
    return data
