# doppler-radar-llm-codex

Starter project for converting downloaded Doppler radar images into embeddings that can be
fed into a generative model pipeline. This version uses Java 21 and the DJL (Deep Java
Library) model zoo.

## Prerequisites

- Java 21
- Maven 3.9+

## Build

```bash
mvn -q -DskipTests package
```

## Embed radar imagery

```bash
mvn -q exec:java -Dexec.args="path/to/radar/images --output embeddings.bin --device cpu"
```

The command saves:
- `embeddings.bin`: float32 little-endian embedding vectors (one row per image)
- `embeddings.bin.json`: file list aligned with the embeddings

## Notes

- Supported extensions: PNG, JPEG, TIFF.
- To use a GPU, pass `--device cuda` and ensure CUDA-enabled DJL PyTorch artifacts are
  available on your system.
