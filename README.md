# doppler-radar-llm-codex

Starter project for converting downloaded Doppler radar images into embeddings that can be
fed into a generative model pipeline.

## Setup

This project now targets **Java 21** and uses the [Deep Java Library (DJL)](https://djl.ai/) to
run a ResNet-50 backbone for image embeddings.

```bash
mvn -q -DskipTests package
```

## Embed radar imagery

```bash
java -jar target/radar-embeddings-0.1.0-shaded.jar path/to/radar/images \
  --output embeddings.npy \
  --batch-size 32 \
  --device cpu
```

The command saves:
- `embeddings.npy`: `(N, 2048)` float array of ResNet-50 image embeddings
- `embeddings.json`: file list aligned with the embeddings

## Notes

- Supported extensions: PNG, JPEG, TIFF.
- DJL will download model artifacts on first run.
- To use a GPU, pass `--device cuda` and ensure the appropriate CUDA-enabled PyTorch engine is available.
