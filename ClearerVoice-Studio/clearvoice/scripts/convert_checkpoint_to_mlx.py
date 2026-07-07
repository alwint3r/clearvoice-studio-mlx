#!/usr/bin/env python
"""Convert a ClearerVoice PyTorch checkpoint to MLX safetensors."""

from __future__ import annotations

import argparse

from clearvoice.mlx.checkpoint import convert_torch_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to a PyTorch .pt/.ckpt checkpoint")
    parser.add_argument("output", help="Path to write MLX safetensors")
    parser.add_argument("--model-key", default="model", help="Checkpoint key that contains the model state dict")
    args = parser.parse_args()
    convert_torch_checkpoint(args.checkpoint, args.output, model_key=args.model_key)


if __name__ == "__main__":
    main()
