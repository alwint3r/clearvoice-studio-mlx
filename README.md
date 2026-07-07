# Audio Cleaning with ClearVoice

This repository contains a small local workflow for speech enhancement, speech separation, and spectrogram comparison using [ClearerVoice-Studio][clearervoice-github]. It keeps the upstream ClearVoice code in `ClearerVoice-Studio/` and provides a lightweight wrapper CLI in `clearvoice-cli/`.

## What is included

- `ClearerVoice-Studio/` — local checkout of the upstream ClearerVoice toolkit.
- `clearvoice-cli/` — project-specific CLI wrapper around the local ClearVoice package.

## Upstream project and model citations

This work depends on ClearerVoice-Studio, an open-source speech processing toolkit that supports speech enhancement, speech separation, speech super-resolution, target speaker extraction, and speech-quality scoring [1]. The upstream toolkit exposes pre-trained ClearVoice models including:

- `MossFormer2_SS_16K` for 16 kHz speech separation.
- `MossFormer2_SE_48K` for 48 kHz speech enhancement.
- `FRCRN_SE_16K` for 16 kHz speech enhancement.
- `MossFormer2_SR_48K` for 48 kHz speech super-resolution.
- `AV_MossFormer2_TSE_16K` for audio-visual target-speaker extraction.

When reusing the generated audio or this workflow in a report, paper, or product, cite the upstream ClearerVoice-Studio repository and paper, and cite the underlying model papers where applicable [1–4].

## Setup

The wrapper uses [`uv`](https://docs.astral.sh/uv/) and an editable dependency on the local ClearVoice package.

```bash
cd clearvoice-cli
uv sync
```

The first model run may download required model files through ClearVoice. ClearVoice also requires `ffmpeg` for broad audio codec support, including `m4a`, `mp3`, `opus`, and related formats [1].

## Usage

Run commands from `clearvoice-cli/`.

List supported commands and models:

```bash
uv run cvwrap models
```

Enhance one local input file with the default 48 kHz enhancement model:

```bash
uv run cvwrap enhance ../path/to/input.m4a
```

Separate two speakers with the default 16 kHz separation model:

```bash
uv run cvwrap separate ../path/to/input.m4a
```

Use the native MLX backend for supported MossFormer2 models on Apple Silicon:

```bash
uv run cvwrap enhance ../path/to/input.m4a --backend mlx
uv run cvwrap separate ../path/to/input.m4a --backend mlx
```

Draw a stacked spectrogram comparison:

```bash
uv run cvspectrogram \
  ../path/to/original.m4a \
  ../path/to/processed.wav \
  -o ../path/to/spectrograms.png
```

## Notes on sample rates

`MossFormer2_SS_16K` performs speaker separation at 16 kHz. Any 48 kHz separated files produced from that model are resampled compatibility outputs; they do not recover true high-frequency detail above the 16 kHz separation model's effective bandwidth. For actual bandwidth extension/speech super-resolution, use a super-resolution model such as `MossFormer2_SR_48K` from ClearerVoice-Studio [1].

## License and attribution

This repository's wrapper code is separate from the upstream ClearerVoice-Studio project. ClearerVoice-Studio is included as a local checkout and remains governed by its own license and attribution requirements. Review [`ClearerVoice-Studio/LICENSE`](ClearerVoice-Studio/LICENSE) before redistribution.

## References

[1]: ModelScope. "ClearerVoice-Studio." GitHub, https://github.com/modelscope/ClearerVoice-Studio

[2]: ClearerVoice-Studio authors. "ClearerVoice-Studio" paper, arXiv:2506.19398, https://arxiv.org/abs/2506.19398

[3]: Zhao et al. "FRCRN: Boosting Feature Representation Using Frequency Recurrence for Monaural Speech Enhancement." arXiv:2206.07293, https://arxiv.org/abs/2206.07293

[4]: Zhao and Ma. "MossFormer: Pushing the Performance Limit of Monaural Speech Separation using Gated Single-Head Transformer with Convolution-Augmented Joint Self-Attentions." arXiv:2302.11824, https://arxiv.org/abs/2302.11824

[clearervoice-github]: https://github.com/modelscope/ClearerVoice-Studio
