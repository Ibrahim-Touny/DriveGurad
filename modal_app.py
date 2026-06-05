"""DriveGuard web app for uploading an image or video and returning an
annotated result.

This exposes the existing detection pipeline through a Modal-hosted FastAPI
application with a simple browser UI.
"""

import tempfile
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Tuple
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import modal
import sys

sys.path.append("/root/project")


app = modal.App("driveguard-api")

JOB_STORE = modal.Dict.from_name("driveguard-progress", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .apt_install(
        "libgl1",
        "libglib2.0-0"
    )
    .pip_install(
        "torch",
        "torchvision",
        "ultralytics",
        "opencv-python-headless",
        "numpy",
        "fastapi",
        "python-multipart",
        "easyocr",
        "pillow",
        "arabic-reshaper",
        "python-bidi",
    )
    .add_local_file(
        "detect_objects.py",
        remote_path="/root/project/detect_objects.py"
    )
    .add_local_dir(
        "Models",
        remote_path="/root/project/Models"
    )
)

BASE_DIR = Path(__file__).resolve().parent
PROCESS_LOCK = threading.Lock()


def resolve_model_path(name: str) -> Path:
    return Path("/root/project/Models") / name


def load_ocr_reader(device: str):
        import easyocr

        use_gpu_ocr = device == "cuda"
        try:
                return easyocr.Reader(["ar"], gpu=use_gpu_ocr, verbose=False)
        except Exception:
                return easyocr.Reader(["ar"], gpu=False, verbose=False)


@lru_cache(maxsize=1)
def get_runtime():
    from detect_objects import ArabicTextRenderer, DetectionPipeline, resolve_device
    from ultralytics import YOLO

    device = resolve_device("auto")

    car_model = YOLO(str(resolve_model_path("yolo11n.pt"))).to(device)
    seatbelt_model = YOLO(str(resolve_model_path("SeatBeltModel2.pt"))).to(device)
    lp_model = YOLO(str(resolve_model_path("LicensePlateModel.pt"))).to(device)
    ocr_reader = load_ocr_reader(device)

    pipeline = DetectionPipeline(
        car_model=car_model,
        seatbelt_model=seatbelt_model,
        lp_model=lp_model,
        ocr_reader=ocr_reader,
        vehicle_classes={"car", "van", "truck", "bus"},
        conf=0.25,
        iou=0.45,
        device=device,
        imgsz=640,
        sub_imgsz=320,
        ocr_every=5,
        max_vehicles=8,
        profile=False,
    )
    pipeline.warmup()
    return pipeline, ArabicTextRenderer()


def _encode_image(frame, suffix: str) -> Tuple[bytes, str]:
    import cv2

    suffix = suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"
    if suffix in {".jpg", ".jpeg"}:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        content_type = "image/jpeg"
    elif suffix == ".png":
        ok, encoded = cv2.imencode(".png", frame)
        content_type = "image/png"
    elif suffix == ".webp":
        ok, encoded = cv2.imencode(".webp", frame)
        content_type = "image/webp"
    else:
        ok, encoded = cv2.imencode(".bmp", frame)
        content_type = "image/bmp"

    if not ok:
        raise RuntimeError("Failed to encode processed image")
    return encoded.tobytes(), content_type


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def _load_job(job_id: str) -> dict:
    job = JOB_STORE.get(_job_key(job_id))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _save_job(job_id: str, **updates) -> dict:
    job = JOB_STORE.get(_job_key(job_id), {}) or {}
    job.update(updates)
    JOB_STORE.put(_job_key(job_id), job)
    return job


def _count_video_frames(video_path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:
        capture.release()
        return total

    total = 0
    try:
        while True:
            success, _ = capture.read()
            if not success:
                break
            total += 1
    finally:
        capture.release()
    return total


def process_image_bytes(file_bytes: bytes, filename: str, force_ocr: bool = False):
    import cv2
    import numpy as np
    from detect_objects import draw_detections

    pipeline, text_renderer = get_runtime()
    frame = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image")

    with PROCESS_LOCK:
        detections = pipeline.infer(frame, force_ocr=force_ocr)
        annotated = draw_detections(frame.copy(), detections, text_renderer)

    return _encode_image(annotated, Path(filename).suffix or ".jpg")


def process_video_file(input_path: Path, output_path: Path, progress_callback=None):
    import cv2
    from detect_objects import draw_detections

    pipeline, text_renderer = get_runtime()

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="Could not open the uploaded video")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise HTTPException(status_code=400, detail="Invalid video dimensions")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        capture.release()
        total_frames = _count_video_frames(input_path)
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail="Could not reopen the uploaded video")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    try:
        processed_frames = 0
        while True:
            success, frame = capture.read()
            if not success:
                break

            with PROCESS_LOCK:
                detections = pipeline.infer(frame)
                annotated = draw_detections(frame.copy(), detections, text_renderer)

            writer.write(annotated)
            processed_frames += 1
            if progress_callback is not None:
                progress_callback(processed_frames, total_frames)
    finally:
        capture.release()
        writer.release()

    return output_path.read_bytes(), "video/mp4", total_frames


def _run_image_job(job_id: str, file_bytes: bytes, filename: str) -> None:
    try:
        _save_job(job_id, status="processing", processed_frames=0, total_frames=1, message="Processing image")
        output_bytes, media_type = process_image_bytes(file_bytes, filename, force_ocr=True)
        _save_job(
            job_id,
            status="done",
            processed_frames=1,
            total_frames=1,
            progress=100,
            result_bytes=output_bytes,
            result_media_type=media_type,
            result_filename=f"{Path(filename).stem}_processed{Path(filename).suffix or '.png'}",
            message="Ready",
        )
    except Exception as exc:
        _save_job(job_id, status="failed", error=str(exc), message="Processing failed")


def _run_video_job(job_id: str, file_bytes: bytes, filename: str) -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"driveguard-{job_id}-"))
    input_path = temp_dir / filename
    output_path = temp_dir / f"{Path(filename).stem}_processed.mp4"
    input_path.write_bytes(file_bytes)

    def _progress(done: int, total: int) -> None:
        percent = int((done / max(total, 1)) * 100)
        _save_job(
            job_id,
            status="processing",
            processed_frames=done,
            total_frames=total,
            progress=percent,
            message=f"Processing frame {done} of {total}",
        )

    try:
        _save_job(job_id, status="processing", processed_frames=0, total_frames=0, progress=0, message="Starting video")
        output_bytes, media_type, total_frames = process_video_file(input_path, output_path, progress_callback=_progress)
        _save_job(
            job_id,
            status="done",
            processed_frames=total_frames,
            total_frames=total_frames,
            progress=100,
            result_bytes=output_bytes,
            result_media_type=media_type,
            result_filename=f"{Path(filename).stem}_processed.mp4",
            message="Ready",
        )
    except Exception as exc:
        _save_job(job_id, status="failed", error=str(exc), message="Processing failed")
    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def build_page() -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>DriveGuard Upload</title>
    <style>
        :root {
            color-scheme: dark;
            --bg: #08111f;
            --bg2: #111c31;
            --card: rgba(12, 18, 33, 0.82);
            --card-border: rgba(255, 255, 255, 0.08);
            --text: #f4f7fb;
            --muted: #97a6c4;
            --accent: #60a5fa;
            --accent2: #22c55e;
            --warn: #f59e0b;
            --shadow: 0 30px 80px rgba(0, 0, 0, 0.45);
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(96, 165, 250, 0.18), transparent 30%),
                radial-gradient(circle at 85% 10%, rgba(34, 197, 94, 0.15), transparent 24%),
                linear-gradient(145deg, var(--bg), var(--bg2));
        }

        .shell {
            max-width: 1180px;
            margin: 0 auto;
            padding: 32px 20px 40px;
        }

        .hero {
            display: grid;
            gap: 18px;
            grid-template-columns: 1.3fr 0.7fr;
            align-items: end;
            margin-bottom: 24px;
        }

        .headline {
            padding: 28px;
            border: 1px solid var(--card-border);
            border-radius: 28px;
            background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
            box-shadow: var(--shadow);
            backdrop-filter: blur(16px);
        }

        .eyebrow {
            display: inline-flex;
            gap: 8px;
            align-items: center;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(96, 165, 250, 0.12);
            color: #cfe4ff;
            font-size: 12px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        h1 {
            margin: 14px 0 10px;
            font-size: clamp(2.4rem, 6vw, 4.8rem);
            line-height: 0.96;
            letter-spacing: -0.05em;
        }

        .subcopy {
            max-width: 58ch;
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.6;
            margin: 0;
        }

        .stats {
            display: grid;
            gap: 14px;
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .stat {
            padding: 18px;
            border-radius: 22px;
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--card-border);
            box-shadow: var(--shadow);
        }

        .stat strong {
            display: block;
            font-size: 1.5rem;
            margin-bottom: 6px;
        }

        .stat span {
            color: var(--muted);
            font-size: 0.92rem;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1.15fr;
            gap: 20px;
            align-items: start;
        }

        .panel {
            padding: 24px;
            border-radius: 28px;
            background: var(--card);
            border: 1px solid var(--card-border);
            box-shadow: var(--shadow);
            backdrop-filter: blur(14px);
        }

        .panel h2 {
            margin: 0 0 8px;
            font-size: 1.2rem;
        }

        .panel p {
            margin: 0 0 18px;
            color: var(--muted);
            line-height: 1.55;
        }

        .dropzone {
            position: relative;
            display: grid;
            place-items: center;
            min-height: 260px;
            padding: 24px;
            border-radius: 24px;
            border: 1.5px dashed rgba(96, 165, 250, 0.45);
            background: linear-gradient(180deg, rgba(96, 165, 250, 0.08), rgba(255,255,255,0.03));
            cursor: pointer;
            transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
        }

        .dropzone:hover {
            transform: translateY(-2px);
            border-color: rgba(96, 165, 250, 0.8);
            background: linear-gradient(180deg, rgba(96, 165, 250, 0.12), rgba(255,255,255,0.04));
        }

        .dropzone.dragover {
            border-color: rgba(34, 197, 94, 0.85);
            background: linear-gradient(180deg, rgba(34, 197, 94, 0.12), rgba(255,255,255,0.04));
        }

        .upload-copy {
            text-align: center;
            max-width: 34ch;
        }

        .upload-copy strong {
            display: block;
            margin-bottom: 10px;
            font-size: 1.1rem;
        }

        .upload-copy span {
            display: block;
            color: var(--muted);
            line-height: 1.5;
        }

        input[type="file"] {
            position: absolute;
            inset: 0;
            opacity: 0;
            cursor: pointer;
        }

        .controls {
            display: flex;
            gap: 12px;
            align-items: center;
            margin-top: 18px;
            flex-wrap: wrap;
        }

        .button {
            appearance: none;
            border: 0;
            border-radius: 999px;
            padding: 14px 18px;
            font-weight: 700;
            color: #04101d;
            background: linear-gradient(135deg, #93c5fd, #22c55e);
            box-shadow: 0 16px 30px rgba(34, 197, 94, 0.22);
            cursor: pointer;
        }

        .button:disabled {
            opacity: 0.55;
            cursor: progress;
        }

        .hint {
            color: var(--muted);
            font-size: 0.92rem;
        }

        .result-wrap {
            display: grid;
            gap: 14px;
        }

        .status {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: center;
            padding: 14px 16px;
            border-radius: 18px;
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--card-border);
            color: var(--muted);
            min-height: 56px;
        }

        .status strong { color: var(--text); }

        .progress-wrap {
            display: none;
            gap: 8px;
            padding: 14px 16px;
            border-radius: 18px;
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--card-border);
        }

        .progress-row {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: center;
            font-size: 0.92rem;
            color: var(--muted);
        }

        .progress-track {
            position: relative;
            height: 12px;
            border-radius: 999px;
            overflow: hidden;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.08);
        }

        .progress-fill {
            position: absolute;
            inset: 0 auto 0 0;
            width: 0%;
            border-radius: inherit;
            background: linear-gradient(90deg, #60a5fa, #22c55e, #93c5fd);
            transition: width 180ms ease;
            box-shadow: 0 0 22px rgba(96, 165, 250, 0.35);
        }

        .progress-fill.indeterminate {
            width: 35%;
            animation: progress-sweep 1.15s ease-in-out infinite;
            transform-origin: left center;
        }

        @keyframes progress-sweep {
            0% { transform: translateX(-120%); }
            100% { transform: translateX(300%); }
        }

        .processing .status {
            border-color: rgba(96, 165, 250, 0.35);
        }

        .processing .progress-wrap {
            display: grid;
        }

        .preview {
            width: 100%;
            max-height: 640px;
            border-radius: 24px;
            border: 1px solid var(--card-border);
            background: rgba(0,0,0,0.28);
            overflow: hidden;
            display: grid;
            place-items: center;
        }

        .preview img, .preview video {
            display: block;
            width: 100%;
            max-height: 640px;
            object-fit: contain;
            background: #050b14;
        }

        .download {
            display: inline-flex;
            width: fit-content;
            padding: 12px 16px;
            border-radius: 999px;
            text-decoration: none;
            color: #d8f2ff;
            border: 1px solid rgba(96, 165, 250, 0.35);
            background: rgba(96, 165, 250, 0.12);
        }

        @media (max-width: 980px) {
            .hero, .grid { grid-template-columns: 1fr; }
            .stats { grid-template-columns: 1fr 1fr; }
        }

        @media (max-width: 640px) {
            .shell { padding: 18px 14px 28px; }
            .headline, .panel { padding: 18px; border-radius: 22px; }
            h1 { font-size: 2.4rem; }
            .stats { grid-template-columns: 1fr; }
            .dropzone { min-height: 220px; }
        }
    </style>
</head>
<body>
    <div class="shell">
        <section class="hero">
            <div class="headline">
                <div class="eyebrow">DriveGuard inference studio</div>
                <h1>Upload an image or video and get the annotated result back.</h1>
                <p class="subcopy">
                    The uploaded media is processed with the deployed vehicle, seatbelt, and license-plate models.
                    Processed output is returned directly in the browser so you can preview or download it immediately.
                </p>
            </div>
            <div class="stats">
                <div class="stat"><strong>Image</strong><span>Annotates a single frame and preserves the result as an image.</span></div>
                <div class="stat"><strong>Video</strong><span>Runs the full pipeline frame-by-frame and returns an annotated MP4.</span></div>
            </div>
        </section>

        <section class="grid">
            <div class="panel">
                <h2>Upload</h2>
                <p>Drop a file here or click to browse. Supported formats are common image types and MP4 video.</p>
                <form id="upload-form">
                    <label class="dropzone" id="dropzone">
                        <input id="file-input" name="file" type="file" accept="image/*,video/*" required />
                        <div class="upload-copy">
                            <strong>Choose a file to process</strong>
                            <span>Image or video. The processed version will replace this preview panel once the request completes.</span>
                        </div>
                    </label>
                    <div class="controls">
                        <button class="button" id="submit-btn" type="submit">Process upload</button>
                        <span class="hint" id="file-name">No file selected</span>
                    </div>
                </form>
            </div>

            <div class="panel result-wrap">
                <h2>Result</h2>
                <div class="status" id="status-box">
                    <span>Waiting for a file.</span>
                </div>
                <div class="progress-wrap" id="progress-wrap" aria-hidden="true">
                    <div class="progress-row">
                        <span id="progress-label">Processing upload...</span>
                        <span id="progress-percent">0%</span>
                    </div>
                    <div class="progress-track" aria-label="Processing progress">
                        <div class="progress-fill" id="progress-fill"></div>
                    </div>
                </div>
                <div class="preview" id="preview">
                    <div class="hint" style="padding: 24px; text-align: center;">Your processed media will appear here.</div>
                </div>
                <a class="download" id="download-link" href="#" download style="display:none;">Download processed file</a>
            </div>
        </section>
    </div>

    <script>
        const form = document.getElementById('upload-form');
        const input = document.getElementById('file-input');
        const dropzone = document.getElementById('dropzone');
        const statusBox = document.getElementById('status-box');
        const progressWrap = document.getElementById('progress-wrap');
        const progressFill = document.getElementById('progress-fill');
        const progressLabel = document.getElementById('progress-label');
        const progressPercent = document.getElementById('progress-percent');
        const preview = document.getElementById('preview');
        const submitBtn = document.getElementById('submit-btn');
        const fileName = document.getElementById('file-name');
        const downloadLink = document.getElementById('download-link');
        let currentObjectUrl = null;
        let progressTimer = null;
        let progressValue = 0;

        function clearPreview() {
            if (currentObjectUrl) {
                URL.revokeObjectURL(currentObjectUrl);
                currentObjectUrl = null;
            }
            preview.innerHTML = '<div class="hint" style="padding: 24px; text-align: center;">Your processed media will appear here.</div>';
            downloadLink.style.display = 'none';
            downloadLink.href = '#';
        }

        function setProgress(processed, total, message) {
            document.body.classList.add('processing');
            const safeTotal = Math.max(total || 0, 1);
            const safeProcessed = Math.min(Math.max(processed || 0, 0), safeTotal);
            const percent = Math.min(Math.round((safeProcessed / safeTotal) * 100), 100);
            progressLabel.textContent = message || `Processing ${safeProcessed} of ${safeTotal} frames`;
            progressPercent.textContent = `${safeProcessed}/${safeTotal} frames (${percent}%)`;
            progressFill.classList.remove('indeterminate');
            progressFill.style.width = `${percent}%`;
            progressWrap.setAttribute('aria-hidden', 'false');
            progressWrap.style.display = 'grid';
        }

        function resetProgress() {
            document.body.classList.remove('processing');
            progressWrap.style.display = 'none';
            progressWrap.setAttribute('aria-hidden', 'true');
            progressFill.style.width = '0%';
            progressPercent.textContent = '0/0 frames (0%)';
            progressLabel.textContent = 'Processing upload...';
        }

        function setStatus(message, kind = 'idle') {
            const accent = kind === 'error' ? '#f59e0b' : kind === 'busy' ? '#60a5fa' : '#97a6c4';
            statusBox.innerHTML = `<span style="color:${accent}; font-weight:700;">${message}</span>`;
        }

        input.addEventListener('change', () => {
            const file = input.files && input.files[0];
            fileName.textContent = file ? file.name : 'No file selected';
            clearPreview();
            resetProgress();
            if (file) {
                setStatus(`Ready to process ${file.name}.`);
            }
        });

        ['dragenter', 'dragover'].forEach((eventName) => {
            dropzone.addEventListener(eventName, (event) => {
                event.preventDefault();
                event.stopPropagation();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach((eventName) => {
            dropzone.addEventListener(eventName, (event) => {
                event.preventDefault();
                event.stopPropagation();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (event) => {
            const file = event.dataTransfer.files && event.dataTransfer.files[0];
            if (!file) return;
            const dt = new DataTransfer();
            dt.items.add(file);
            input.files = dt.files;
            input.dispatchEvent(new Event('change'));
        });

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            const file = input.files && input.files[0];
            if (!file) {
                setStatus('Please choose a file first.', 'error');
                return;
            }

            submitBtn.disabled = true;
            setStatus('Starting processing job...', 'busy');
            resetProgress();

            try {
                const formData = new FormData();
                formData.append('file', file);
                const response = await fetch('/process', { method: 'POST', body: formData });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(errorText || 'Processing failed');
                }

                const startInfo = await response.json();
                const jobId = startInfo.job_id;
                const statusUrl = `/status/${jobId}`;

                await new Promise((resolve, reject) => {
                    const pollStatus = async () => {
                        try {
                            const statusResponse = await fetch(statusUrl);
                            if (!statusResponse.ok) {
                                throw new Error('Could not read job status.');
                            }

                            const status = await statusResponse.json();
                            if (status.status === 'failed') {
                                throw new Error(status.error || 'Processing failed');
                            }

                            setProgress(status.processed_frames, status.total_frames, status.message || 'Processing...');

                            if (status.status === 'done') {
                                const resultResponse = await fetch(`/result/${jobId}`);
                                if (!resultResponse.ok) {
                                    throw new Error('Processed file is not ready yet.');
                                }

                                const blob = await resultResponse.blob();
                                const contentType = resultResponse.headers.get('content-type') || blob.type || '';
                                const objectUrl = URL.createObjectURL(blob);
                                if (currentObjectUrl) {
                                    URL.revokeObjectURL(currentObjectUrl);
                                }
                                currentObjectUrl = objectUrl;
                                preview.innerHTML = '';

                                if (contentType.startsWith('image/')) {
                                    const image = document.createElement('img');
                                    image.src = objectUrl;
                                    image.alt = 'Processed result';
                                    preview.appendChild(image);
                                } else if (contentType.startsWith('video/')) {
                                    const video = document.createElement('video');
                                    video.src = objectUrl;
                                    video.controls = true;
                                    video.playsInline = true;
                                    preview.appendChild(video);
                                } else {
                                    const link = document.createElement('a');
                                    link.href = objectUrl;
                                    link.textContent = 'Open processed file';
                                    link.className = 'download';
                                    preview.appendChild(link);
                                }

                                const extension = contentType.startsWith('video/') ? '.mp4' : '.png';
                                downloadLink.href = objectUrl;
                                downloadLink.download = file.name.replace(/\\.[^.]+$/, '') + '_processed' + extension;
                                downloadLink.style.display = 'inline-flex';
                                setStatus('Processing complete.', 'idle');
                                resolve();
                                return;
                            }

                            window.setTimeout(pollStatus, 750);
                        } catch (error) {
                            reject(error);
                        }
                    };

                    pollStatus();
                });
            } catch (error) {
                resetProgress();
                clearPreview();
                setStatus(error.message || 'Processing failed.', 'error');
            } finally {
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>"""


@app.function(image=image, gpu="T4", timeout=60 * 30)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def fastapi_app():
    web_app = FastAPI(title="DriveGuard")

    @web_app.get("/", response_class=HTMLResponse)
    def home():
        return HTMLResponse(build_page())

    @web_app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    @web_app.post("/process")
    async def process(file: UploadFile = File(...)):
        filename = file.filename or "upload"
        suffix = Path(filename).suffix.lower()
        content_type = (file.content_type or "").lower()
        payload = await file.read()

        is_image = content_type.startswith("image/") or suffix in {
            ".jpg", ".jpeg", ".png", ".webp", ".bmp"
        }

        is_video = content_type.startswith("video/") or suffix in {
            ".mp4", ".mov", ".avi", ".mkv", ".webm"
        }

        job_id = uuid.uuid4().hex
        if is_image:
            JOB_STORE.put(_job_key(job_id), {
                "status": "queued",
                "type": "image",
                "filename": filename,
                "processed_frames": 0,
                "total_frames": 1,
                "progress": 0,
                "message": "Queued",
                "result_bytes": None,
                "result_media_type": None,
                "result_filename": None,
                "error": None,
            })
            threading.Thread(target=_run_image_job, args=(job_id, payload, filename), daemon=True).start()
            return JSONResponse({"job_id": job_id, "type": "image", "total_frames": 1})

        if is_video:
            total_frames = 0
            JOB_STORE.put(_job_key(job_id), {
                "status": "queued",
                "type": "video",
                "filename": filename,
                "processed_frames": 0,
                "total_frames": total_frames,
                "progress": 0,
                "message": "Queued",
                "result_bytes": None,
                "result_media_type": None,
                "result_filename": None,
                "error": None,
            })
            threading.Thread(target=_run_video_job, args=(job_id, payload, filename), daemon=True).start()
            return JSONResponse({"job_id": job_id, "type": "video", "total_frames": total_frames})

        raise HTTPException(
            status_code=400,
            detail="Upload an image or video file."
        )

    @web_app.get("/status/{job_id}")
    def status(job_id: str):
        job = _load_job(job_id)
        return {
            "job_id": job_id,
            "status": job.get("status", "unknown"),
            "type": job.get("type"),
            "filename": job.get("filename"),
            "processed_frames": job.get("processed_frames", 0),
            "total_frames": job.get("total_frames", 0),
            "progress": job.get("progress", 0),
            "message": job.get("message", ""),
            "error": job.get("error"),
        }

    @web_app.get("/result/{job_id}")
    def result(job_id: str):
        job = _load_job(job_id)
        if job.get("status") != "done":
            raise HTTPException(status_code=202, detail="Result is not ready yet")

        result_bytes = job.get("result_bytes")
        media_type = job.get("result_media_type") or "application/octet-stream"
        result_filename = job.get("result_filename") or "driveguard_processed.bin"
        return Response(
            content=result_bytes,
            media_type=media_type,
            headers={"Content-Disposition": f'inline; filename="{result_filename}"'},
        )

    return web_app