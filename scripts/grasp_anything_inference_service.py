#!/usr/bin/env python3
"""Localhost-only HTTP wrapper for the official Grasp-Anything RGB model.

The service returns 2-D parallel-jaw proposals only.  It has no ROS, TF,
MoveIt, Gazebo, or robot-control dependency.  The ROS adapter is responsible
for registered-depth reconstruction and all downstream safety gates.
"""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
import torch


PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 8 * 1024 * 1024
EXPECTED_MODEL_SHA256 = (
    "65984ef3364790c1ece107f22bcbeb67dc8fba21784087bb3d8ff183a3582e0a"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GraspAnythingModel:
    def __init__(
        self,
        vendor_dir: Path,
        model_path: Path,
        max_candidates: int,
        peak_min_distance: int,
        minimum_score: float,
    ) -> None:
        if not (vendor_dir / "inference").is_dir():
            raise RuntimeError(f"official Grasp-Anything source missing: {vendor_dir}")
        if not model_path.is_file():
            raise RuntimeError(f"Grasp-Anything model missing: {model_path}")
        self.model_path = model_path
        self.model_sha256 = _sha256(model_path)
        if self.model_sha256 != EXPECTED_MODEL_SHA256:
            raise RuntimeError(
                "official model checksum mismatch: "
                f"expected {EXPECTED_MODEL_SHA256}, got {self.model_sha256}"
            )
        # The upstream checkpoint uses Python pickle serialization. Verify the
        # pinned file before deserializing any of its contents.
        sys.path.insert(0, str(vendor_dir))
        self.model = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model.eval()
        if int(self.model.conv1.in_channels) != 3:
            raise RuntimeError("expected the official RGB-only 3-channel model")
        self.max_candidates = max_candidates
        self.peak_min_distance = peak_min_distance
        self.minimum_score = minimum_score
        self.lock = threading.Lock()

    def infer(self, encoded_image: bytes) -> dict:
        start = time.perf_counter()
        with Image.open(io.BytesIO(encoded_image)) as source:
            rgb = source.convert("RGB")
            source_width, source_height = rgb.size
            resized = rgb.resize((224, 224), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array -= float(array.mean())
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
        with self.lock, torch.no_grad():
            prediction = self.model.predict(tensor)

        quality = prediction["pos"].cpu().numpy().squeeze()
        angle = (
            torch.atan2(prediction["sin"], prediction["cos"]) / 2.0
        ).cpu().numpy().squeeze()
        opening = prediction["width"].cpu().numpy().squeeze() * 150.0
        quality = gaussian_filter(quality, sigma=2.0)
        angle = gaussian_filter(angle, sigma=2.0)
        opening = gaussian_filter(opening, sigma=1.0)
        peaks = peak_local_max(
            quality,
            min_distance=self.peak_min_distance,
            threshold_abs=self.minimum_score,
            num_peaks=self.max_candidates,
        )

        scale_x = 224.0 / source_width
        scale_y = 224.0 / source_height
        candidates = []
        for row, column in peaks:
            model_angle = float(angle[row, column])
            du_source = math.cos(model_angle) / scale_x
            dv_source = -math.sin(model_angle) / scale_y
            source_vector_norm = math.hypot(du_source, dv_source)
            source_angle = math.atan2(-dv_source, du_source)
            candidates.append(
                {
                    "u": float(column / scale_x),
                    "v": float(row / scale_y),
                    "angle_rad": source_angle,
                    "opening_px": float(opening[row, column] * source_vector_norm),
                    "score": float(quality[row, column]),
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "model": "Grasp-Anything ICRA 2024 model_grasp_anything",
            "model_sha256": self.model_sha256,
            "input_width": source_width,
            "input_height": source_height,
            "preprocess": "full-frame anisotropic resize to 224x224; angle/opening mapped back",
            "peak_min_distance": self.peak_min_distance,
            "minimum_score": self.minimum_score,
            "inference_seconds": time.perf_counter() - start,
            "candidates": candidates,
        }


def make_handler(model: GraspAnythingModel):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GraspAnythingLocal/1"

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/health":
                self._json(404, {"success": False, "reason": "not_found"})
                return
            self._json(
                200,
                {
                    "success": True,
                    "protocol_version": PROTOCOL_VERSION,
                    "backend": "official_grasp_anything_cpu",
                    "model_sha256": model.model_sha256,
                    "motion_capable": False,
                },
            )

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/v1/grasp":
                self._json(404, {"success": False, "reason": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= MAX_REQUEST_BYTES:
                self._json(413, {"success": False, "reason": "invalid_request_size"})
                return
            body = self.rfile.read(length)
            try:
                result = model.infer(body)
            except Exception as error:
                self._json(
                    422,
                    {
                        "success": False,
                        "reason": f"inference_failed:{error.__class__.__name__}",
                    },
                )
                return
            self._json(200, {"success": True, **result})

        def log_message(self, format_string: str, *args) -> None:
            print(
                f"grasp-anything-http {self.address_string()} "
                f"{format_string % args}",
                file=sys.stderr,
                flush=True,
            )

    return Handler


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=project_root / "assets/vendor/grasp_anything",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root / "assets/vendor/grasp_anything/weights/model_grasp_anything",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--peak-min-distance", type=int, default=15)
    parser.add_argument("--minimum-score", type=float, default=0.20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("Refusing a non-loopback Grasp-Anything service bind", file=sys.stderr)
        return 2
    if not 1 <= args.max_candidates <= 20:
        print("--max-candidates must be in [1, 20]", file=sys.stderr)
        return 2
    if not 1 <= args.peak_min_distance <= 100:
        print("--peak-min-distance must be in [1, 100]", file=sys.stderr)
        return 2
    if not 0.0 <= args.minimum_score <= 1.0:
        print("--minimum-score must be in [0, 1]", file=sys.stderr)
        return 2
    model = GraspAnythingModel(
        args.vendor_dir,
        args.model,
        args.max_candidates,
        args.peak_min_distance,
        args.minimum_score,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model))
    print(
        json.dumps(
            {
                "event": "grasp_anything_service_ready",
                "url": f"http://{args.host}:{args.port}",
                "model_sha256": model.model_sha256,
                "motion_capable": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
