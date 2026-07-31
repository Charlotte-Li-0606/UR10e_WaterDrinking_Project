from __future__ import annotations

import json
import math
import threading
import unittest
import urllib.error
import urllib.request

from live_pendant.server import PendantHTTPServer, PendantState, quaternion_to_rotation_vector


class RotationVectorTest(unittest.TestCase):
    def test_identity(self) -> None:
        self.assertEqual(quaternion_to_rotation_vector(0.0, 0.0, 0.0, 1.0), [0.0, 0.0, 0.0])

    def test_quarter_turn_about_z(self) -> None:
        result = quaternion_to_rotation_vector(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], math.pi / 2)


class PendantStateTest(unittest.TestCase):
    def test_connection_requires_all_core_streams(self) -> None:
        state = PendantState()
        state.update_robot({"mode": "RUNNING"}, "robot_mode")
        self.assertEqual(state.snapshot()["connection_state"], "offline")
        state.update_robot({"safety_mode": "NORMAL"}, "safety_mode")
        state.update("joints", [], "joints")
        self.assertEqual(state.snapshot()["connection_state"], "partial")
        state.update("tcp", {}, "tcp")
        snapshot = state.snapshot()
        self.assertTrue(snapshot["connected"])
        self.assertNotIn("monotonic", snapshot["streams"]["tcp"])


class HTTPServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = PendantState(source="test")
        self.server = PendantHTTPServer(("127.0.0.1", 0), self.state)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)

    def test_health_and_static_page(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/api/health") as response:
            payload = json.load(response)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["read_only"])
            self.assertEqual(response.headers["Cache-Control"], "no-store")
        with urllib.request.urlopen(f"{self.base_url}/") as response:
            page = response.read().decode("utf-8")
            self.assertIn("Live Pendant", page)
            self.assertIn("READ ONLY", page)
            self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])

    def test_camera_returns_404_until_a_frame_arrives(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as captured:
            urllib.request.urlopen(f"{self.base_url}/api/camera.jpg")
        self.assertEqual(captured.exception.code, 404)
        self.state.update_camera(b"jpeg", 2, 1, "rgb8")
        with urllib.request.urlopen(f"{self.base_url}/api/camera.jpg") as response:
            self.assertEqual(response.read(), b"jpeg")
            self.assertEqual(response.headers["Content-Type"], "image/jpeg")


if __name__ == "__main__":
    unittest.main()
