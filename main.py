#!/usr/bin/env python3
import argparse
import collections
import io
import queue
import threading
import time
from functools import partial
from pathlib import Path
from typing import Optional

import cv2
from flask import Flask, Response, jsonify, render_template_string

from hailo_apps.python.core.common.defines import MAX_INPUT_QUEUE_SIZE, MAX_ASYNC_INFER_JOBS
from hailo_apps.python.core.common.hailo_inference import HailoInfer
from hailo_apps.python.core.common.toolbox import (
    InputContext,
    get_labels,
    init_input_source,
    load_json_file,
    preprocess,
)
from hailo_apps.python.standalone_apps.object_detection.object_detection_post_process import (
    extract_detections,
    draw_detections,
)

WEB_PAGE = """
<!doctype html>
<html>
  <head>
    <title>Hailo Object Detection</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; }
      img { max-width: 100%; height: auto; border: 1px solid #ccc; }
      pre { background: #f8f8f8; padding: 12px; border-radius: 6px; }
    </style>
  </head>
  <body>
    <h1>Hailo Object Detection</h1>
    <p>Source: <strong>{{ source_name }}</strong></p>
    <img id="frame" src="/video_feed" alt="Detection stream" />
    <h2>Detections</h2>
    <pre id="detections">Loading...</pre>
    <script>
      async function refresh() {
        try {
          const res = await fetch('/detections');
          const data = await res.json();
          document.getElementById('detections').textContent = JSON.stringify(data, null, 2);
        } catch (err) {
          document.getElementById('detections').textContent = 'Error: ' + err;
        }
      }
      setInterval(refresh, 1000);
      refresh();
    </script>
  </body>
</html>
"""


class SimpleObjectDetector:
    """A small wrapper over hailo-apps helpers to run object detection from a source."""

    def __init__(
        self,
        hef_path: str,
        source: str = "usb",
        labels_path: Optional[str] = None,
        batch_size: int = 1,
    ) -> None:
        self.labels = get_labels(labels_path)
        self.input_context = InputContext(
            input_src=source,
            batch_size=batch_size,
        )
        self.input_context = init_input_source(self.input_context)

        self.hailo_inference = HailoInfer(str(Path(hef_path)), batch_size)
        self.input_height, self.input_width, _ = self.hailo_inference.get_input_shape()

        self.config_data = load_json_file(
            str(
                Path(__file__).resolve().parent
                / "hailo-apps"
                / "hailo_apps"
                / "python"
                / "standalone_apps"
                / "object_detection"
                / "config.json"
            )
        )

        self.input_queue: queue.Queue = queue.Queue(MAX_INPUT_QUEUE_SIZE)
        self.stop_event = threading.Event()
        self._output_lock = threading.Lock()
        self._last_detections = {
            "detection_boxes": [],
            "detection_classes": [],
            "detection_scores": [],
            "num_detections": 0,
        }
        self._last_frame_bgr = None

        self._preprocess_thread: Optional[threading.Thread] = None
        self._infer_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the capture/preprocess and inference threads."""
        self._preprocess_thread = threading.Thread(
            target=preprocess,
            args=(
                self.input_context,
                self.input_queue,
                self.input_width,
                self.input_height,
                None,
                self.stop_event,
            ),
            name="preprocess-thread",
            daemon=True,
        )
        self._infer_thread = threading.Thread(
            target=self._infer_loop,
            name="infer-thread",
            daemon=True,
        )

        self._preprocess_thread.start()
        self._infer_thread.start()
    

    def stop(self) -> None:
        """Stop pipeline threads and release Hailo resources."""
        self.stop_event.set()
        if self.input_queue is not None:
            self.input_queue.put(None)

        if self._preprocess_thread is not None:
            self._preprocess_thread.join(timeout=5.0)
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=5.0)

    def _infer_loop(self) -> None:
        pending_jobs = collections.deque()

        while True:
            item = self.input_queue.get()
            if item is None:
                break
            if self.stop_event.is_set():
                continue

            input_batch, preprocessed_batch = item
            inference_callback_fn = partial(
                self._inference_callback,
                input_batch=input_batch,
            )

            while len(pending_jobs) >= MAX_ASYNC_INFER_JOBS:
                pending_jobs.popleft().wait(10000)

            job = self.hailo_inference.run(preprocessed_batch, inference_callback_fn)
            pending_jobs.append(job)

        self.hailo_inference.close()

    def _inference_callback(self, completion_info, bindings_list, input_batch):
        if completion_info.exception:
            return

        for frame_rgb, bindings in zip(input_batch, bindings_list):
            if len(bindings._output_names) == 1:
                infer_results = bindings.output().get_buffer()
            else:
                infer_results = {
                    name: bindings.output(name).get_buffer()
                    for name in bindings._output_names
                }

            detections = extract_detections(frame_rgb, infer_results, self.config_data)
            annotated_rgb = draw_detections(detections, frame_rgb.copy(), self.labels)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            with self._output_lock:
                self._last_detections = detections
                self._last_frame_bgr = annotated_bgr

    def get_latest_detections(self) -> dict:
        with self._output_lock:
            boxes = self._last_detections["detection_boxes"]
            classes = self._last_detections["detection_classes"]
            scores = self._last_detections["detection_scores"]
            num_detections = int(self._last_detections["num_detections"])

        return {
            "detection_boxes": [
                [int(float(coord)) for coord in box]
                for box in boxes
            ],
            "detection_classes": [int(cls) for cls in classes],
            "detection_scores": [float(score) for score in scores],
            "num_detections": num_detections,
        }

    def get_latest_frame_jpeg(self) -> bytes:
        with self._output_lock:
            frame = self._last_frame_bgr
        if frame is None:
            return b""

        success, jpeg = cv2.imencode(".jpg", frame)
        if not success:
            return b""
        return jpeg.tobytes()


def generate_mjpeg(detector: SimpleObjectDetector):
    """Generate a multipart MJPEG stream from the latest camera frames."""
    while True:
        frame = detector.get_latest_frame_jpeg()
        if frame:
            yield b"--frame\r\n"
            yield b"Content-Type: image/jpeg\r\n\r\n"
            yield frame
            yield b"\r\n"
        else:
            time.sleep(0.05)


def create_app(detector: SimpleObjectDetector, source: str) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(WEB_PAGE, source_name=source)

    @app.route("/video_feed")
    def video_feed():
        return Response(
            generate_mjpeg(detector),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/detections")
    def detections():
        return jsonify(detector.get_latest_detections())

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple Hailo object detection web demo"
    )
    parser.add_argument(
        "--hef-path",
        required=True,
        help="Path to the Hailo HEF model file.",
    )
    parser.add_argument(
        "--source",
        default="usb",
        help=(
            "Input source: 'usb', camera index (0, 1), '/dev/video0', "
            "video file path, image directory, or stream URL."
        ),
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional path to a labels file. If omitted, COCO labels are used.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Web server host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Web server port.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = SimpleObjectDetector(
        hef_path=args.hef_path,
        source=args.source,
        labels_path=args.labels,
        batch_size=args.batch_size,
    )

    detector.start()
    app = create_app(detector, args.source)

    try:
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    finally:
        detector.stop()


if __name__ == "__main__":
    main()
