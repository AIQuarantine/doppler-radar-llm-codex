"""Embedding pipeline for radar imagery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class EmbeddingResult:
    embeddings: np.ndarray
    paths: List[str]


def load_image_paths(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input path not found: {root}")
    if root.is_file():
        return [root]
    paths = [path for path in root.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(paths)


def build_model(device: torch.device) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    return model.to(device)


def build_preprocess() -> transforms.Compose:
    weights = models.ResNet50_Weights.DEFAULT
    return weights.transforms()


def _batched(iterable: Sequence[Path], batch_size: int) -> Iterable[List[Path]]:
    for index in range(0, len(iterable), batch_size):
        yield list(iterable[index : index + batch_size])


def embed_image_paths(
    image_paths: Sequence[Path],
    batch_size: int = 16,
    device: str | torch.device = "cpu",
) -> EmbeddingResult:
    if not image_paths:
        raise ValueError("No images found to embed.")

    device = torch.device(device)
    model = build_model(device)
    preprocess = build_preprocess()

    embeddings: List[np.ndarray] = []
    recorded_paths: List[str] = []

    with torch.no_grad():
        for batch in tqdm(_batched(list(image_paths), batch_size), desc="Embedding"):
            images = []
            for path in batch:
                image = Image.open(path).convert("RGB")
                images.append(preprocess(image))
                recorded_paths.append(str(path))

            tensor = torch.stack(images).to(device)
            outputs = model(tensor)
            embeddings.append(outputs.cpu().numpy())

    return EmbeddingResult(embeddings=np.concatenate(embeddings, axis=0), paths=recorded_paths)


def save_embeddings(result: EmbeddingResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result.embeddings)
    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump({"paths": result.paths}, handle, indent=2)
