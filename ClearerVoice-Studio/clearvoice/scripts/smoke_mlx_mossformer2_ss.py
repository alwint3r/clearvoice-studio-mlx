#!/usr/bin/env python
"""Run a shape-only smoke test for the MLX MossFormer2 speech-separation model."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import mlx.core as mx

from clearvoice.mlx.models.mossformer2_ss import MossFormer2SS16K


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--speakers", type=int, default=2)
    args = parser.parse_args()

    model_args = SimpleNamespace(
        encoder_embedding_dim=args.channels,
        mossformer_sequence_dim=args.channels,
        num_mossformer_layer=args.layers,
        encoder_kernel_size=16,
        num_spks=args.speakers,
    )
    model = MossFormer2SS16K(model_args)
    outputs = model(mx.zeros((1, args.samples)))
    mx.eval(*outputs)
    print([tuple(output.shape) for output in outputs])


if __name__ == "__main__":
    main()
