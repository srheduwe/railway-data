import argparse
import io
import zlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "railway-data-augmentation"
SOURCE = DATA_ROOT / "IMAGES" / "dataset_split_original" / "test"
KINDS = (
    "rain", "sunlight", "fog", "low_light", "perspective",
    "motion_blur", "contrast", "gaussian_noise", "jpeg",
)


def corrupt(im, kind, s, rng):
    w, h = im.size

    if kind == "low_light":
        return ImageEnhance.Brightness(im).enhance((0.75, 0.55, 0.35)[s - 1])

    if kind == "contrast":
        return ImageEnhance.Contrast(im).enhance((0.8, 0.6, 0.4)[s - 1])

    if kind == "perspective":
        d = w * (0.02, 0.04, 0.06)[s - 1]
        return im.transform(
            im.size, Image.Transform.QUAD,
            (d, 0, 0, h, w, h, w - d, 0),
            resample=Image.Resampling.BICUBIC,
        )

    if kind == "jpeg":
        buffer = io.BytesIO()
        im.save(buffer, format="JPEG", quality=(75, 45, 20)[s - 1])
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")

    if kind == "rain":
        overlay = Image.new("RGBA", im.size)
        draw = ImageDraw.Draw(overlay)
        length = max(3, int(min(w, h) * 0.025))
        for _ in range(max(1, w * h * s // 3500)):
            x, y = int(rng.integers(w)), int(rng.integers(h))
            draw.line(
                (x, y, x - length // 3, y + length),
                fill=(190, 205, 220, 40 + 20 * s),
            )
        base = ImageEnhance.Brightness(im).enhance(1 - 0.06 * s)
        return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")

    a = np.asarray(im, dtype=np.float32)

    if kind == "motion_blur":
        length = max(2, round(min(w, h) * (0.01, 0.02, 0.04)[s - 1]))
        result = np.zeros_like(a)
        for offset in np.linspace(-length // 2, length // 2, 7).astype(int):
            result += np.take(
                a, np.clip(np.arange(w) + offset, 0, w - 1), axis=1
            ) / 7
        a = result

    elif kind == "gaussian_noise":
        a += rng.normal(0, (12, 25, 45)[s - 1], a.shape)

    elif kind == "sunlight":
        y, x = np.ogrid[:h, :w]
        glow = np.exp(-((x / w - 0.8)**2 + (y / h - 0.2)**2) / 0.15)
        a = a * (1 + 0.08 * s)
        a += 255 * 0.08 * s * glow[..., None] * [1.0, 0.95, 0.8]

    elif kind == "fog":
        noise = Image.fromarray(rng.integers(0, 256, (8, 8), dtype=np.uint8))
        noise = np.asarray(
            noise.resize(im.size, Image.Resampling.BICUBIC), dtype=np.float32
        ) / 255
        alpha = 0.12 * s * (0.6 + 0.4 * np.clip(noise[..., None], 0, 1))
        a = a * (1 - alpha) + 255 * alpha

    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = args.out or args.src.with_name(args.src.name + "_corrupted")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = sorted(
        p for p in args.src.glob("*/*")
        if p.is_file() and p.suffix.lower() in extensions
    )
    if not files:
        parser.error(f"No images found in {args.src}/<class>/")

    for p in files:
        with Image.open(p) as image:
            im = ImageOps.exif_transpose(image).convert("RGB")

        for severity in (1, 2, 3):
            dest = out / f"severity_{severity}" / p.parent.name
            dest.mkdir(parents=True, exist_ok=True)

            for kind in KINDS:
                key = f"{args.seed}/{p.relative_to(args.src)}/{kind}/{severity}"
                rng = np.random.default_rng(zlib.crc32(key.encode()))
                corrupt(im, kind, severity, rng).save(
                    dest / f"{p.name}__{kind}.png"
                )

    print(f"Saved {len(files) * 3 * len(KINDS)} images to {out}")


if __name__ == "__main__":
    main()