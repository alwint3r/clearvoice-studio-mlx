from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment
from scipy import signal


DEFAULT_OUTPUT = Path("out/analysis/spectrograms.png")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = [path.expanduser().resolve() for path in args.inputs]
    output = args.output.expanduser().resolve()
    draw_spectrograms(
        inputs=inputs,
        output=output,
        title=args.title,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
        db_min=args.db_min,
        db_max=args.db_max,
        peak_normalize=not args.no_peak_normalize,
    )
    print(f"Wrote spectrogram comparison to: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cvspectrogram",
        description="Draw stacked spectrograms for one or more audio files.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Audio files to plot, in display order.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path. Default: {DEFAULT_OUTPUT}.",
    )
    parser.add_argument("--title", default="Spectrogram comparison", help="Figure title.")
    parser.add_argument("--panel-width", type=int, default=1280, help="Width of each spectrogram panel.")
    parser.add_argument("--panel-height", type=int, default=260, help="Height of each spectrogram panel.")
    parser.add_argument("--db-min", type=float, default=-80.0, help="Minimum dB value for the color scale.")
    parser.add_argument("--db-max", type=float, default=0.0, help="Maximum dB value for the color scale.")
    parser.add_argument(
        "--no-peak-normalize",
        action="store_true",
        help="Do not peak-normalize each audio file before computing the spectrogram.",
    )
    return parser


def draw_spectrograms(
    *,
    inputs: Sequence[Path],
    output: Path,
    title: str = "Spectrogram comparison",
    panel_width: int = 1280,
    panel_height: int = 260,
    db_min: float = -80.0,
    db_max: float = 0.0,
    peak_normalize: bool = True,
) -> None:
    if not inputs:
        raise ValueError("At least one input file is required.")
    for input_path in inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"Input does not exist: {input_path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for input_path in inputs:
        audio, sample_rate = load_audio(input_path)
        db = compute_spectrogram_db(audio, sample_rate, peak_normalize=peak_normalize, db_min=db_min, db_max=db_max)
        rows.append(
            {
                "path": input_path,
                "label": input_path.name,
                "audio": audio,
                "sample_rate": sample_rate,
                "db": db,
            }
        )

    image = render_stacked_spectrograms(
        rows,
        title=title,
        panel_width=panel_width,
        panel_height=panel_height,
        db_min=db_min,
        db_max=db_max,
        peak_normalize=peak_normalize,
    )
    image.save(output)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    segment = AudioSegment.from_file(path)
    sample_rate = int(segment.frame_rate)
    channels = int(segment.channels)
    samples = np.array(segment.get_array_of_samples()).astype(np.float32)
    if channels > 1:
        samples = samples.reshape((-1, channels)).mean(axis=1)
    scale = float(1 << (8 * segment.sample_width - 1))
    return samples / scale, sample_rate


def compute_spectrogram_db(
    audio: np.ndarray,
    sample_rate: int,
    *,
    peak_normalize: bool,
    db_min: float,
    db_max: float,
) -> np.ndarray:
    if audio.size == 0:
        raise ValueError("Cannot draw a spectrogram for empty audio.")

    audio = np.asarray(audio, dtype=np.float32)
    if peak_normalize:
        peak = max(float(np.max(np.abs(audio))), 1e-9)
        audio = audio / peak

    nperseg = max(256, int(round(sample_rate * 0.040)))
    noverlap = int(round(nperseg * 0.75))
    _, _, zxx = signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    db = 20 * np.log10(np.maximum(np.abs(zxx), 1e-8))
    if peak_normalize:
        db -= db.max()
    return np.clip(db, db_min, db_max)


def render_stacked_spectrograms(
    rows: Sequence[dict],
    *,
    title: str,
    panel_width: int,
    panel_height: int,
    db_min: float,
    db_max: float,
    peak_normalize: bool,
) -> Image.Image:
    left_margin = 78
    right_pad = 34
    top_margin = 124
    row_gap = 74
    bottom_margin = 82

    canvas_width = left_margin + panel_width + right_pad
    canvas_height = top_margin + len(rows) * panel_height + (len(rows) - 1) * row_gap + bottom_margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    font_title, font, font_small = load_fonts()

    draw.text((left_margin, 20), title, fill=(24, 24, 24), font=font_title)
    normalization = "peak-normalized per file" if peak_normalize else "absolute input scale"
    subtitle = f"STFT magnitude, mono mixdown, {normalization}, shared {db_min:g} dB to {db_max:g} dB color scale"
    draw.text((left_margin, 52), subtitle, fill=(70, 70, 70), font=font_small)

    for index, row in enumerate(rows):
        x0 = left_margin
        y0 = top_margin + index * (panel_height + row_gap)
        rgb = colorize(row["db"], db_min=db_min, db_max=db_max)
        canvas.paste(resize_spectrogram(rgb, panel_width, panel_height), (x0, y0))
        draw.rectangle((x0, y0, x0 + panel_width, y0 + panel_height), outline=(35, 35, 35), width=1)

        duration = len(row["audio"]) / row["sample_rate"]
        header = f"{index + 1}. {row['label']}  |  {row['sample_rate'] / 1000:.1f} kHz  |  {duration:.2f}s"
        draw.text((x0, y0 - 26), header, fill=(20, 20, 20), font=font)
        draw_time_axis(draw, x0, y0, panel_width, panel_height, duration, font_small)
        draw_frequency_axis(draw, x0, y0, panel_height, row["sample_rate"], font_small)

    draw.text((18, top_margin + panel_height // 2 - 35), "frequency", fill=(60, 60, 60), font=font_small)
    draw.text((left_margin + panel_width // 2 - 28, canvas_height - 52), "time", fill=(60, 60, 60), font=font_small)
    draw_color_bar(draw, canvas, left_margin, canvas_height - 28, db_min, db_max, font_small)
    return canvas


def colorize(db: np.ndarray, *, db_min: float, db_max: float) -> np.ndarray:
    x = np.clip((db - db_min) / (db_max - db_min), 0, 1)
    stops = np.array(
        [
            [8, 12, 32],
            [28, 74, 132],
            [38, 166, 154],
            [246, 214, 92],
            [250, 250, 235],
        ],
        dtype=np.float32,
    )
    position = x * (len(stops) - 1)
    low = np.floor(position).astype(int)
    high = np.clip(low + 1, 0, len(stops) - 1)
    fraction = position - low
    return (stops[low] * (1 - fraction[..., None]) + stops[high] * fraction[..., None]).astype(np.uint8)


def resize_spectrogram(rgb: np.ndarray, width: int, height: int) -> Image.Image:
    image = Image.fromarray(np.flipud(rgb), mode="RGB")
    return image.resize((width, height), Image.Resampling.BICUBIC)


def draw_time_axis(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    panel_width: int,
    panel_height: int,
    duration: float,
    font,
) -> None:
    for fraction, label in [
        (0, "0"),
        (0.25, f"{duration * 0.25:.1f}"),
        (0.5, f"{duration * 0.5:.1f}"),
        (0.75, f"{duration * 0.75:.1f}"),
        (1, f"{duration:.1f}s"),
    ]:
        tick_x = x0 + int(panel_width * fraction)
        draw.line((tick_x, y0 + panel_height, tick_x, y0 + panel_height + 5), fill=(60, 60, 60))
        draw.text((tick_x - 14, y0 + panel_height + 9), label, fill=(60, 60, 60), font=font)


def draw_frequency_axis(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    panel_height: int,
    sample_rate: int,
    font,
) -> None:
    nyquist = sample_rate / 2
    for hz in [0, 4000, 8000, 12000, 16000, 20000, 24000]:
        if hz <= nyquist + 1:
            tick_y = y0 + panel_height - int(panel_height * (hz / nyquist if nyquist else 0))
            draw.line((x0 - 5, tick_y, x0, tick_y), fill=(60, 60, 60))
            label = "0" if hz == 0 else f"{hz // 1000}k"
            draw.text((x0 - 48, tick_y - 7), label, fill=(60, 60, 60), font=font)


def draw_color_bar(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    x: int,
    y: int,
    db_min: float,
    db_max: float,
    font,
) -> None:
    width = 360
    height = 14
    gradient = np.linspace(db_min, db_max, width, dtype=np.float32)[None, :]
    bar = Image.fromarray(colorize(gradient, db_min=db_min, db_max=db_max).repeat(height, axis=0), mode="RGB")
    canvas.paste(bar, (x, y))
    draw.rectangle((x, y, x + width, y + height), outline=(80, 80, 80), width=1)
    draw.text((x, y - 20), f"{db_min:g} dB", fill=(70, 70, 70), font=font)
    draw.text((x + width - 38, y - 20), f"{db_max:g} dB", fill=(70, 70, 70), font=font)


def load_fonts():
    try:
        return (
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24),
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16),
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13),
        )
    except OSError:
        fallback = ImageFont.load_default()
        return fallback, fallback, fallback


if __name__ == "__main__":
    raise SystemExit(main())
