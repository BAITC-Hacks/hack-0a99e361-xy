"""Ask Jev to classify a text representation of one rehearsal image."""

import argparse
import io
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal
import warnings

from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from typesafe_sdk import (
    Choice,
    RetryPolicy,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeClient,
    TypeSafeError,
)

GRID_SIZE = 64
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000
Label = Literal["OK", "DEFECT"]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Decision(BaseModel):
    """Validate the decision even though the request already constrains its type."""

    model_config = ConfigDict(strict=True, extra="forbid")
    type: Literal["choice"]
    choice: Label
    confidence: Probability
    probabilities: dict[Label, Probability]

    @model_validator(mode="after")
    def check_distribution(self) -> "Decision":
        if set(self.probabilities) != {"OK", "DEFECT"}:
            raise ValueError("Both class probabilities are required.")
        if not math.isclose(sum(self.probabilities.values()), 1, abs_tol=1e-5):
            raise ValueError("Class probabilities must sum to one.")
        if self.probabilities[self.choice] < max(self.probabilities.values()):
            raise ValueError("The choice must have the highest probability.")
        return self


class Answers(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    inspection: Decision


class InspectionResponse(BaseModel):
    # The API also returns usage and may add metadata; only these fields are used.
    model_config = ConfigDict(strict=True, extra="ignore")
    model: str = Field(min_length=1)
    answers: Answers


def image_state(path: Path) -> dict:
    """Decode pixels locally; send no filename, label, EXIF, or hidden metadata."""
    # A bounded read prevents a large input from being fully loaded into memory.
    with path.open("rb") as source:
        data = source.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("Image exceeds the 10 MiB file limit.")

    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Use a PNG, JPEG, or WebP image.")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Image exceeds the 16 megapixel limit.")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Use a single-frame image, not an animation.")

            # Normalize orientation and composite transparency against white.
            rgba = ImageOps.exif_transpose(image).convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            grayscale = Image.alpha_composite(background, rgba).convert("L")
            # Area averaging avoids ringing that can look like cracks or halos.
            grid = ImageOps.pad(grayscale, (GRID_SIZE, GRID_SIZE),
                                method=Image.Resampling.BOX, color=255)

    # Jev is text-only. These digits represent luminance, not a local verdict.
    # ponytail: 64x64 grayscale loses small/color defects; use a validated vision
    # extractor before Jev when moving beyond large synthetic shape damage.
    pixels = list(grid.get_flattened_data())
    rows = [
        "".join(str(value * 9 // 255) for value in pixels[start:start + GRID_SIZE])
        for start in range(0, len(pixels), GRID_SIZE)
    ]
    return {
        "representation": "64x64 grayscale raster; one digit per pixel",
        "legend": "0=black, 9=white; rows top to bottom, columns left to right",
        "rows": rows,
    }


def classify(client: TypeSafeClient, state: dict) -> tuple[InspectionResponse, float]:
    """Use Jev's native Choice primitive, rather than a free-text chat prompt."""
    started = perf_counter()
    response = client.system_one(
        state=state,
        questions={
            "inspection": Choice(
                instructions=(
                    "Inspect the grayscale raster in state.rows as a spatial grid. "
                    "This rehearsal task depicts one dark filled square on a white "
                    "background. Decide whether that square is intact or damaged. "
                    "Look for white cracks inside the square and missing material "
                    "along its edges. Ignore smooth antialiasing at the boundary. "
                    "Choose only from the supplied criteria."
                ),
                criteria={
                    "OK": "One intact, uniformly filled square with continuous edges.",
                    "DEFECT": "The square has a crack, hole, chip, or missing material.",
                },
            ),
        },
        response_model=InspectionResponse,
    )
    # Wall-clock request time includes SDK validation and any retries.
    return response, (perf_counter() - started) * 1000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="PNG, JPEG, or WebP image path")
    parser.add_argument("--plain", action="store_true", help="Hide success diagnostics")
    args = parser.parse_args(argv)
    console = Console(stderr=True)

    # Resolve .env relative to this script; exported variables take precedence.
    load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
    key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not key or key == "your_typesafe_api_key_here" or not key.isascii() or any(
        character.isspace() or not character.isprintable() for character in key
    ):
        console.print("Set TYPESAFE_API_KEY in .env or the environment.", style="red")
        return 2

    try:
        state = image_state(args.image)
        if not args.plain:
            console.print(Panel(Text(str(args.image)), title="Jev · Shape inspection",
                                subtitle="Experimental text-grid input", border_style="cyan"))

        # Explicit host keeps credentials on TypeSafe's documented endpoint.
        # Reuse the SDK's retry policy instead of implementing another retry loop.
        with TypeSafeClient(
            api_key=key,
            base_url="https://api.typesafe.ai",
            model=os.getenv("TYPESAFE_MODEL", "jev-latest"),
            timeout=10.0,
            retry=RetryPolicy(max_retries=2, timeout=30.0),
        ) as client:
            if args.plain:
                response, latency_ms = classify(client, state)
            else:
                with console.status("Jev is evaluating the image state…", spinner="dots"):
                    response, latency_ms = classify(client, state)

        label = response.answers.inspection.choice
        if not args.plain:
            color = "green" if label == "OK" else "red"
            console.print(Panel(Text(label, style=f"bold {color}", justify="center"),
                                border_style=color, expand=False))
            console.print(f"API round trip: {latency_ms:.1f} ms", style="dim")
            console.print(Text(f"Model: {response.model}", style="dim"))

        # This is the only success output to stdout: safe for pipes and graders.
        print(label)
        return 0
    except TypeSafeAPIResponseValidationError:
        message = "TypeSafe returned an invalid decision; no label was emitted."
    except TypeSafeAPITimeoutError:
        message = "TypeSafe timed out. Check connectivity and try again."
    except TypeSafeAPIError as error:
        # Server bodies may echo submitted data; show a safe, actionable summary.
        hint = {
            401: "Check TYPESAFE_API_KEY.",
            403: "Check your account's API access.",
            404: "Check TYPESAFE_MODEL and your account's model access.",
            429: "Rate limit reached; try again later.",
        }.get(error.status, "Check the TypeSafe console and try again.")
        message = f"TypeSafe HTTP {error.status}. {hint}"
    except TypeSafeError:
        message = "TypeSafe request failed. Check connectivity and your configuration."
    except (OSError, ValueError, UnidentifiedImageError,
            Image.DecompressionBombError, Image.DecompressionBombWarning):
        message = "Cannot read input: use a valid single-frame PNG/JPEG/WebP, <=10 MiB and <=16 MP."
    except KeyboardInterrupt:
        console.print("Cancelled.", style="yellow")
        return 130

    # An outage, invalid response, or unreadable image is never an OK/DEFECT guess.
    console.print(message, style="red", markup=False)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
