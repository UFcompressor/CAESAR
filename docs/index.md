# CAESAR developer documentation

CAESAR combines a C++17 compression library with Python model training and
export code. This documentation describes the repository and the constraints
that matter when changing it. The core provides versioned buffer serialization
and single-file save/load; external ADIOS2 integration is documented separately.

See the [CAESAR architecture figure](architecture.md) for model installation,
compression, local reconstruction, correction, metadata, and decompression.

```{toctree}
:maxdepth: 2

codebase
architecture
public_api
adios_serialization
nglr
models
model_registration
```
