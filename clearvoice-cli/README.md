# ClearVoice CLI

Small CLI wrapper around the local `ClearerVoice-Studio/clearvoice` package.

The project lives outside `ClearerVoice-Studio` and uses a local editable dependency, so the model code stays owned by the upstream checkout.

## Setup

```bash
cd /Users/alwin/explores/audio-cleaning/clearvoice-cli
uv sync
```

The first model run may download checkpoints through ClearVoice.

## Commands

```bash
uv run cvwrap enhance ../audio_only.m4a
uv run cvwrap enhance ../audio_only.m4a --backend mlx
uv run cvwrap separate ../audio_only.m4a
uv run cvwrap separate ../audio_only.m4a --backend mlx
uv run cvwrap extract ./path-to-video-or-video-list.scp -o ./out/extracted
uv run cvwrap models
```

Draw a stacked spectrogram comparison for any number of audio files:

```bash
uv run cvspectrogram \
  ../audio_only.m4a \
  out/audio_only_separated_MossFormer2_SS_16K_ported_s1.m4a \
  enhance.wav \
  -o out/analysis/audio_only_ported_s1_enhance_spectrograms.png
```

Defaults:

- `enhance`: `speech_enhancement` with `MossFormer2_SE_48K`
- `separate`: `speech_separation` with `MossFormer2_SS_16K`
- `extract`: `target_speaker_extraction` with `AV_MossFormer2_TSE_16K`

Backends:

- `--backend torch`: default backend for all commands
- `--backend mlx`: supported for `separate` with `MossFormer2_SS_16K` and `enhance` with `MossFormer2_SE_48K`

The local ClearVoice checkout currently implements MLX for MossFormer2 speech separation and the 48 kHz MossFormer2 speech-enhancement mask model. The 16 kHz enhancement models are different architectures (`FRCRN_SE_16K`, `MossFormerGAN_SE_16K`) and still run through the Torch backend. The CLI rejects unsupported MLX combinations before loading a model.

For single-file enhancement, `-o/--output` is an output audio file. For single-file separation, the path is a base output file and ClearVoice writes speaker files like `_s1.wav` and `_s2.wav`.

For directories, `.scp`, `.txt`, and `.lst` inputs, `-o/--output` is an output directory. ClearVoice writes into a model-named subdirectory, for example `out/separated_MossFormer2_SS_16K/MossFormer2_SS_16K/`.

Use `--overwrite` to allow an existing output path.
