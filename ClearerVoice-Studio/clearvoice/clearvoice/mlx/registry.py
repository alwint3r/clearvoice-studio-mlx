"""Factory for native MLX ClearerVoice models."""

from __future__ import annotations


def create_mlx_model(args):
    if args.network == "MossFormer2_SS_16K":
        from .models.mossformer2_ss import MossFormer2SS16K

        return MossFormer2SS16K(args)
    if args.network == "MossFormer2_SE_48K":
        from .models.mossformer2_se import MossFormer2SE48K

        return MossFormer2SE48K(args)
    raise NotImplementedError(
        f"MLX backend is not implemented for {args.network}. "
        "Currently supported: MossFormer2_SS_16K, MossFormer2_SE_48K."
    )
