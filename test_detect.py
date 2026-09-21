"""Offline contract checks: real SDK and CLI, mocked HTTP; no model accuracy claim."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx2
from PIL import Image

import detect
from generator import generate


def response(label: str) -> dict:
    return {
        "model": "jev-test",
        "answers": {"inspection": {
            "type": "choice", "choice": label, "confidence": 0.9,
            "probabilities": {"OK": 0.95 if label == "OK" else 0.05,
                              "DEFECT": 0.95 if label == "DEFECT" else 0.05},
        }},
        "usage": {"input_tokens": 100, "output_tokens": 1},
    }


def run_cli(path: Path, handler, *, plain=False, key="offline-test-key"):
    stdout, stderr = io.StringIO(), io.StringIO()
    real_client = detect.TypeSafeClient

    def client(**kwargs):
        # Keep the application's retry count, but remove waiting from this check.
        kwargs["retry"] = replace(kwargs["retry"], backoff_initial=0,
                                  backoff_max=0, backoff_jitter=0)
        return real_client(**kwargs, transport=httpx2.MockTransport(handler))

    with patch.dict("os.environ", {"TYPESAFE_API_KEY": key}, clear=True), \
         patch.object(detect, "load_dotenv"), \
         patch.object(detect, "TypeSafeClient", side_effect=client), \
         redirect_stdout(stdout), redirect_stderr(stderr):
        code = detect.main([str(path)] + (["--plain"] if plain else []))
    return code, stdout.getvalue(), stderr.getvalue()


def main():
    checks = 0
    with TemporaryDirectory() as directory:
        root = Path(directory)
        generate(root)
        ok, defect = root / "ok.png", root / "defect.png"
        for path in (ok, defect):
            with Image.open(path) as image:
                assert image.size == (512, 512) and image.format == "PNG"
        assert detect.image_state(ok) != detect.image_state(defect)
        renamed = root / "misleading-defect-name.png"
        renamed.write_bytes(ok.read_bytes())
        assert detect.image_state(ok) == detect.image_state(renamed)
        checks += 1

        # A grid-aligned, uniform square must not acquire halos or internal bands.
        assert set("".join(detect.image_state(ok)["rows"])) == {"1", "9"}
        checks += 1

        # Deliberately swap expected image labels: only the API answer controls stdout.
        for path, label, plain in ((ok, "DEFECT", False), (defect, "OK", True)):
            requests = []

            def success(request):
                requests.append(request)
                payload = json.loads(request.content)
                assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
                assert request.method == "POST"
                assert request.headers["authorization"] == "Bearer offline-test-key"
                assert payload["model"] == "jev-latest"
                question = payload["questions"]["inspection"]
                assert question["type"] == "choice"
                assert set(question["criteria"]) == {"OK", "DEFECT"}
                assert payload["state"] == detect.image_state(path)
                assert path.name not in request.content.decode()
                return httpx2.Response(200, json=response(label))

            code, out, err = run_cli(path, success, plain=plain)
            assert code == 0 and out == label + "\n", (code, out, err)
            assert len(requests) == 1
            assert err == "" if plain else "API round trip:" in err and str(path) in err
            checks += 1

        changes = [
            {"choice": "MAYBE"}, {"choice": "OK\nextra text"}, {"type": "noul"},
            {"confidence": 2.0}, {"confidence": "0.9"},
            {"probabilities": {"OK": 1.0}},
            {"probabilities": {"OK": 0.9, "DEFECT": 0.9}},
            {"probabilities": {"OK": 0.1, "DEFECT": 0.9}},
            {"commentary": "unexpected extra output"},
        ]
        for change in changes:
            body = deepcopy(response("OK"))
            body["answers"]["inspection"].update(change)
            code, out, err = run_cli(ok, lambda _: httpx2.Response(200, json=body))
            assert code == 1 and out == "" and "invalid decision" in err
            checks += 1

        for body in ({}, {"model": "jev-test", "answers": {}}, "not JSON"):
            def invalid(_):
                return (httpx2.Response(200, text=body) if isinstance(body, str)
                        else httpx2.Response(200, json=body))
            code, out, err = run_cli(ok, invalid)
            assert code == 1 and out == "" and "invalid decision" in err
            checks += 1

        for status, attempts in ((401, 1), (403, 1), (429, 3), (503, 3)):
            requests = []

            def failure(request):
                requests.append(request)
                return httpx2.Response(status, json={"error": "do-not-echo-this-secret"})

            code, out, err = run_cli(ok, failure)
            assert code == 1 and out == "" and f"HTTP {status}" in err
            assert len(requests) == attempts and "do-not-echo" not in err
            checks += 1

        def timeout(request):
            raise httpx2.ReadTimeout("timeout", request=request)

        code, out, err = run_cli(ok, timeout)
        assert code == 1 and out == "" and "timed out" in err
        checks += 1

        def unexpected_request(_):
            raise AssertionError("Invalid input must not reach the API")

        bad = root / "bad.png"
        bad.write_text("not an image")
        large = root / "large.png"
        with large.open("wb") as file:
            file.truncate(detect.MAX_BYTES + 1)
        oversized = root / "oversized.png"
        Image.new("1", (4001, 4000)).save(oversized)
        animated = root / "animated.png"
        Image.new("RGB", (8, 8), "white").save(
            animated, save_all=True, append_images=[Image.new("RGB", (8, 8), "black")])
        for path in (bad, large, oversized, animated, root / "missing.png", root):
            code, out, err = run_cli(path, unexpected_request)
            assert code == 1 and out == "" and "Cannot read input" in err
            checks += 1
        for key in ("", "your_typesafe_api_key_here", "bad\nkey"):
            code, out, err = run_cli(ok, unexpected_request, key=key)
            assert code == 2 and out == "" and "TYPESAFE_API_KEY" in err
            checks += 1

    print(f"PASS: {checks} offline checks (real SDK, mocked HTTP; no live accuracy test).")


if __name__ == "__main__":
    main()
