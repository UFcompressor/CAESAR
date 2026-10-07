# Migrating CompressCAESAR to the CAESAR serializer

Use `models/serialization.h` from the updated CAESAR installation in the
external ADIOS2 operator. Keep the existing `CompressCAESAR.h` unchanged;
any new decoding helper can be defined inside the `.cpp` file.

The new writer delegates all result fields to one CAESAR call:

```cpp
const auto bytes = caesar::serialize(compressor.compress(config, accuracy));
```

The operator length-prefixes this byte buffer. Its reader validates the length
against the remaining input and delegates to:

```cpp
auto comp = caesar::deserialize(buffer + pos, size_t(length));
```

Pass `comp` to `Decompressor::decompress`. Preserve ADIOS's outer shape, data
type, threshold bypass, and 5D variable slicing. Use a new operator buffer
version for the new payload layout and retain the existing `DecompressV1`
reader for legacy buffers.

Rebuild and install CAESAR before rebuilding ADIOS2 against it. Ensure the
complete serialized payload fits the output allocation advertised by ADIOS2;
otherwise use a raw-data fallback. The external operator must be tested on
HyperGator. ADIOS2 source files are not included in this commit.

For ordinary disk use, there is no ADIOS wrapper:

```cpp
caesar::save(compressor.compress(config, accuracy), "field.cae");
auto restored = decompressor.decompress(caesar::load("field.cae"));
```

The matching model installation remains necessary for decompression.
