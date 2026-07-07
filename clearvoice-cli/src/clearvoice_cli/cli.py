from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence


DEFAULT_OUTPUT_DIR = Path("out")

TASK_MODELS = {
    "enhance": {
        "task": "speech_enhancement",
        "default_model": "MossFormer2_SE_48K",
        "models": ("MossFormer2_SE_48K", "FRCRN_SE_16K", "MossFormerGAN_SE_16K"),
        "suffix": "enhanced",
    },
    "separate": {
        "task": "speech_separation",
        "default_model": "MossFormer2_SS_16K",
        "models": ("MossFormer2_SS_16K",),
        "suffix": "separated",
    },
    "extract": {
        "task": "target_speaker_extraction",
        "default_model": "AV_MossFormer2_TSE_16K",
        "models": ("AV_MossFormer2_TSE_16K",),
        "suffix": "extracted",
    },
}

BACKENDS = ("torch", "mlx")
MLX_MODELS = {"MossFormer2_SS_16K", "MossFormer2_SE_48K"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "models":
        return print_models()

    return run_task(
        command=args.command,
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        backend=args.backend,
        overwrite=args.overwrite,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cvwrap",
        description="Run local ClearVoice enhancement, separation, and target-speaker extraction models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, spec in TASK_MODELS.items():
        command_parser = subparsers.add_parser(command, help=f"Run {spec['task'].replace('_', ' ')}.")
        command_parser.add_argument("input", type=Path, help="Input audio/video file, directory, or .scp list.")
        command_parser.add_argument(
            "-o",
            "--output",
            type=Path,
            help="Output file for single-file audio tasks, or output directory for batch/extraction tasks.",
        )
        command_parser.add_argument(
            "-m",
            "--model",
            default=spec["default_model"],
            choices=spec["models"],
            help=f"Model to run. Default: {spec['default_model']}.",
        )
        command_parser.add_argument(
            "--backend",
            default="torch",
            choices=BACKENDS,
            help="Inference backend. Default: torch.",
        )
        command_parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Allow writing into an existing output path.",
        )

    subparsers.add_parser("models", help="List supported wrapper commands and model names.")
    return parser


def print_models() -> int:
    for command, spec in TASK_MODELS.items():
        print(f"{command}:")
        print(f"  task: {spec['task']}")
        print(f"  default: {spec['default_model']}")
        print(f"  models: {', '.join(spec['models'])}")
        print(f"  backends: {', '.join(backends_for_models(spec['models']))}")
    return 0


def backends_for_models(model_names: Sequence[str]) -> tuple[str, ...]:
    if any(model_name in MLX_MODELS for model_name in model_names):
        return BACKENDS
    return ("torch",)


def run_task(
    *,
    command: str,
    input_path: Path,
    output_path: Path | None,
    model_name: str,
    backend: str,
    overwrite: bool,
) -> int:
    spec = TASK_MODELS[command]
    input_path = input_path.expanduser().resolve()
    validate_backend(command=command, model_name=model_name, backend=backend)

    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")

    output_path = resolve_output_path(
        command=command,
        input_path=input_path,
        output_path=output_path,
        model_name=model_name,
    )
    guard_output_path(output_path, overwrite=overwrite)

    if command == "extract":
        run_online(
            task=spec["task"],
            model_name=model_name,
            backend=backend,
            input_path=input_path,
            output_path=output_path,
        )
        print(f"Wrote extraction outputs under: {output_path / model_name}")
        return 0

    if should_use_online_write(input_path):
        run_online(
            task=spec["task"],
            model_name=model_name,
            backend=backend,
            input_path=input_path,
            output_path=output_path,
        )
        print(f"Wrote batch outputs under: {output_path / model_name}")
        return 0

    if command == "separate":
        guard_separation_outputs(output_path, overwrite=overwrite)

    run_single_file(
        task=spec["task"],
        model_name=model_name,
        backend=backend,
        input_path=input_path,
        output_path=output_path,
    )
    print(f"Wrote output to: {output_path}")
    if command == "separate":
        print("Separation writes one file per speaker using the output stem, for example *_s1.wav and *_s2.wav.")
    return 0


def resolve_output_path(
    *,
    command: str,
    input_path: Path,
    output_path: Path | None,
    model_name: str,
) -> Path:
    if output_path is not None:
        return output_path.expanduser().resolve()

    spec = TASK_MODELS[command]

    if command == "extract" or should_use_online_write(input_path):
        return (DEFAULT_OUTPUT_DIR / f"{spec['suffix']}_{model_name}").resolve()

    return (DEFAULT_OUTPUT_DIR / f"{input_path.stem}_{spec['suffix']}_{model_name}{input_path.suffix}").resolve()


def should_use_online_write(input_path: Path) -> bool:
    if input_path.is_dir():
        return True
    return input_path.suffix.lower() in {".scp", ".txt", ".lst"}


def guard_output_path(output_path: Path, *, overwrite: bool) -> None:
    if overwrite:
        return

    if output_path.exists():
        raise SystemExit(f"Output already exists, pass --overwrite to reuse it: {output_path}")

    parent = output_path.parent
    if parent.exists() and not os.access(parent, os.W_OK):
        raise SystemExit(f"Output parent is not writable: {parent}")


def guard_separation_outputs(output_path: Path, *, overwrite: bool) -> None:
    if overwrite:
        return

    for speaker_index in (1, 2):
        speaker_output = output_path.with_name(f"{output_path.stem}_s{speaker_index}{output_path.suffix}")
        if speaker_output.exists():
            raise SystemExit(f"Speaker output already exists, pass --overwrite to reuse it: {speaker_output}")


def validate_backend(*, command: str, model_name: str, backend: str) -> None:
    if backend == "torch":
        return

    if model_name in MLX_MODELS:
        return

    raise SystemExit(
        f"MLX backend is not implemented for `{command}` with `{model_name}`. "
        "Currently supported: `MossFormer2_SS_16K`, `MossFormer2_SE_48K`."
    )


def run_single_file(*, task: str, model_name: str, backend: str, input_path: Path, output_path: Path) -> None:
    model = load_model_or_exit(task=task, model_name=model_name, backend=backend)
    model.process(input_path=str(input_path), online_write=False)
    model.write(output_path=str(output_path))


def run_online(*, task: str, model_name: str, backend: str, input_path: Path, output_path: Path) -> None:
    model = load_model_or_exit(task=task, model_name=model_name, backend=backend)
    model.process(input_path=str(input_path), online_write=True, output_path=str(output_path))


def load_model_or_exit(*, task: str, model_name: str, backend: str):
    try:
        return load_model(task=task, model_name=model_name, backend=backend)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


def load_model(*, task: str, model_name: str, backend: str):
    try:
        from clearvoice.network_wrapper import network_wrapper
    except ImportError as error:
        raise SystemExit(
            "Could not import clearvoice. Run `uv sync` from the clearvoice-cli directory first."
        ) from error

    wrapper = network_wrapper()
    wrapper.model_name = model_name

    if task == "speech_enhancement":
        wrapper.load_args_se()
    elif task == "speech_separation":
        wrapper.load_args_ss()
    elif task == "target_speaker_extraction":
        wrapper.load_args_tse()
    else:
        raise SystemExit(f"Unsupported task: {task}")

    wrapper.args.task = task
    wrapper.args.network = model_name
    wrapper.args.backend = backend

    if backend == "mlx":
        if model_name == "MossFormer2_SS_16K":
            from clearvoice.networks import CLS_MLX_MossFormer2_SS_16K

            return CLS_MLX_MossFormer2_SS_16K(wrapper.args)
        if model_name == "MossFormer2_SE_48K":
            from clearvoice.networks import CLS_MLX_MossFormer2_SE_48K

            return CLS_MLX_MossFormer2_SE_48K(wrapper.args)
        raise SystemExit(f"Unsupported MLX model: {model_name}")

    if model_name == "FRCRN_SE_16K":
        from clearvoice.networks import CLS_FRCRN_SE_16K

        return CLS_FRCRN_SE_16K(wrapper.args)
    if model_name == "MossFormer2_SE_48K":
        from clearvoice.networks import CLS_MossFormer2_SE_48K

        return CLS_MossFormer2_SE_48K(wrapper.args)
    if model_name == "MossFormerGAN_SE_16K":
        from clearvoice.networks import CLS_MossFormerGAN_SE_16K

        return CLS_MossFormerGAN_SE_16K(wrapper.args)
    if model_name == "MossFormer2_SS_16K":
        from clearvoice.networks import CLS_MossFormer2_SS_16K

        return CLS_MossFormer2_SS_16K(wrapper.args)
    if model_name == "AV_MossFormer2_TSE_16K":
        from clearvoice.networks import CLS_AV_MossFormer2_TSE_16K

        return CLS_AV_MossFormer2_TSE_16K(wrapper.args)

    raise SystemExit(f"Unsupported model: {model_name}")


if __name__ == "__main__":
    raise SystemExit(main())
