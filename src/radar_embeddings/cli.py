"""Command-line entry points for embedding radar images."""

from __future__ import annotations

import argparse
from pathlib import Path

from radar_embeddings.embedding import embed_image_paths, load_image_paths, save_embeddings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Doppler radar images into embeddings for generative models."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a radar image or directory of images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("embeddings.npy"),
        help="Output .npy file path for embeddings (metadata saved as .json).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of images per batch.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run inference on (cpu or cuda).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    image_paths = load_image_paths(args.input)
    result = embed_image_paths(image_paths, batch_size=args.batch_size, device=args.device)
    save_embeddings(result, args.output)

    print(f"Saved {len(result.paths)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
