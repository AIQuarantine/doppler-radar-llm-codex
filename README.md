# doppler-radar-llm-codex

Starter project for converting downloaded Doppler radar images into embeddings that can be
fed into a generative model pipeline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Embed radar imagery

```bash
radar-embed path/to/radar/images --output embeddings.npy --batch-size 32 --device cpu
```

The command saves:
- `embeddings.npy`: `(N, 2048)` float array of ResNet-50 image embeddings
- `embeddings.json`: file list aligned with the embeddings

## Notes

- Supported extensions: PNG, JPEG, TIFF.
- To use a GPU, pass `--device cuda` and ensure CUDA-enabled PyTorch is installed.
