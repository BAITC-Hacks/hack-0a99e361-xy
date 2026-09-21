"""Generate repeatable, unlabeled inspection samples for the rehearsal track."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def generate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)

    # Both samples start with the same solid square on a white background.
    # Large features survive the detector's 64-pixel representation.
    normal = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(normal).rectangle((96, 96, 415, 415), fill=(32, 32, 32))
    normal.save(output / "ok.png")

    damaged = normal.copy()
    draw = ImageDraw.Draw(damaged)
    # Remove material at the edge and draw a branching crack through the face.
    draw.polygon([(320, 96), (376, 96), (352, 160)], fill="white")
    draw.line([(96, 256), (192, 224), (264, 304), (344, 272)],
              fill="white", width=16)
    draw.line([(192, 224), (208, 160)], fill="white", width=12)
    damaged.save(output / "defect.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)
    print(f"Created {args.output_dir / 'ok.png'} and {args.output_dir / 'defect.png'}")
