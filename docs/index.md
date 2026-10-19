# CAESAR developer documentation

CAESAR combines a C++17 compression library with Python model training and
export code. This documentation describes the repository and the constraints
that matter when changing it. The public API and CPU validation workflow are described below; ADIOS buffer
serialization remains separate.

See the [CAESAR workflow figure](workflow.md) for an overview of the
compression and reconstruction pipeline.

```{toctree}
:maxdepth: 2

codebase
workflow
public_api
models
model_registration
```
