"""DriveGuard web app for uploading an image or video and returning an
annotated result.

This exposes the existing detection pipeline through a Modal-hosted FastAPI
application with a simple browser UI.
"""

import tempfile          # for creating temporary directories during video processing
import threading         # to run jobs in background threads without blocking requests
import uuid              # to generate unique job IDs
from functools import lru_cache   # cache the loaded models so they are only loaded once
from pathlib import Path          # convenient file path handling
from typing import Optional, Tuple
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import modal   # Modal cloud platform: runs this app on a GPU container
import sys

sys.path.append("/root/project")  # make detect_objects.py importable inside the container


# Modal app and cloud resource setup
app = modal.App("driveguard-api")  # the Modal app that hosts all functions and the web endpoint

JOB_STORE = modal.Dict.from_name("driveguard-progress", create_if_missing=True)    # stores job status/progress across requests
RESULT_VOLUME = modal.Volume.from_name("driveguard-results", create_if_missing=True)  # persistent disk for saving processed files
RESULT_DIR = Path("/root/project/results")  # path inside the container where results are written

# Container image definition
# Builds the Docker-like image Modal will use for every function call.
image = (
    modal.Image.debian_slim()  # start from a minimal Debian base
    .apt_install(
        "libgl1",        # required by OpenCV for image decoding
        "libglib2.0-0"   # required by OpenCV on headless Linux
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

BASE_DIR = Path(__file__).resolve().parent   # directory of this script (used for local file references)
PROCESS_LOCK = threading.Lock()             # ensures only one request runs inference at a time (models are not thread-safe)


# Model loading helpers
def resolve_model_path(name: str) -> Path:
    """Return the full path to a model file inside the container's Models folder."""
    return Path("/root/project/Models") / name


def load_ocr_reader(device: str):
    """Load EasyOCR for Arabic and English, then run a blank warmup to avoid first-request lag."""
    import easyocr
    import numpy as np

    use_gpu_ocr = device == "cuda"  # use GPU if the detection device is also GPU
    try:
        reader = easyocr.Reader(["ar", "en"], gpu=use_gpu_ocr, verbose=False)
    except Exception:
        reader = easyocr.Reader(["ar", "en"], gpu=False, verbose=False)  # fall back to CPU if GPU init fails

    # Prime OCR once so the first detected plate does not pay model warmup cost.
    try:
        warmup_crop = np.zeros((48, 192, 3), dtype=np.uint8)  # blank image to trigger internal model loading
        reader.readtext(warmup_crop, detail=0, paragraph=False)
    except Exception:
        pass
    return reader


@lru_cache(maxsize=1)  # load models only once; reuse the same objects for all requests
def get_runtime():
    """Load and warm up all models. Cached so this only runs once per container."""
    from detect_objects import ArabicTextRenderer, DetectionPipeline, resolve_device
    from ultralytics import YOLO

    device = resolve_device("auto")  # use GPU if available, otherwise CPU

    # Load the three YOLO models.
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
    pipeline.warmup()  # run dummy inference to prepare CUDA/cuDNN before real requests arrive
    return pipeline, ArabicTextRenderer()


# Utility functions (encoding, job store, flag parsing)
def _encode_image(frame, suffix: str) -> Tuple[bytes, str]:
    """Encode an OpenCV frame to bytes using the format matching the file suffix."""
    import cv2

    suffix = suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"  # default to JPEG for unknown formats
    if suffix in {".jpg", ".jpeg"}:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # 95% quality JPEG
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
    return encoded.tobytes(), content_type  # return raw bytes and the MIME type


def _job_key(job_id: str) -> str:
    """Return the key used to store a job's data in JOB_STORE."""
    return f"job:{job_id}"


def _load_job(job_id: str) -> dict:
    """Load a job's data from JOB_STORE, raising 404 if it doesn't exist."""
    job = JOB_STORE.get(_job_key(job_id))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _save_job(job_id: str, **updates) -> dict:
    """Merge updates into a job's stored data and write it back."""
    job = JOB_STORE.get(_job_key(job_id), {}) or {}  # fetch existing data, default to empty dict
    job.update(updates)  # apply the new fields
    JOB_STORE.put(_job_key(job_id), job)
    return job


def _parse_flag(value: Optional[str], default: bool = True) -> bool:
    """Convert a form string like '1' or 'true' to a boolean."""
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _count_video_frames(video_path: Path) -> int:
    """Count total frames in a video (fallback when CAP_PROP_FRAME_COUNT is unreliable)."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:  # fast path: metadata is available and reliable
        capture.release()
        return total

    # Slow path: count frames manually by reading through the whole video.
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


# Core processing functions (image and video)
def process_image_bytes(
    file_bytes: bytes,
    filename: str,
    force_ocr: bool = False,
    enable_seatbelt: bool = True,
    enable_license_plate: bool = True,
    enable_ocr: bool = True,
):
    """Run the detection pipeline on a single image and return (encoded_bytes, content_type)."""
    import cv2
    import numpy as np
    from detect_objects import draw_detections

    pipeline, text_renderer = get_runtime()  # get the cached model objects
    frame = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)  # decode uploaded bytes to a BGR image
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image")

    with PROCESS_LOCK:  # only one request may run inference at a time
        if hasattr(pipeline, "reset_temporal_state"):
            pipeline.reset_temporal_state()  # clear any leftover plate-tracking state from a previous request

        # Temporarily disable models that the user turned off via the UI toggles.
        original_seatbelt_model = pipeline.seatbelt_model
        original_lp_model = pipeline.lp_model
        original_ocr_reader = pipeline.ocr_reader
        try:
            if not enable_seatbelt:
                pipeline.seatbelt_model = None
            if not enable_license_plate:
                pipeline.lp_model = None
                pipeline.ocr_reader = None
            elif not enable_ocr:
                pipeline.ocr_reader = None

            detections = pipeline.infer(frame, force_ocr=force_ocr and enable_ocr and enable_license_plate)
        finally:
            # Always restore the original models even if inference raised an exception.
            pipeline.seatbelt_model = original_seatbelt_model
            pipeline.lp_model = original_lp_model
            pipeline.ocr_reader = original_ocr_reader

        annotated = draw_detections(frame.copy(), detections, text_renderer)  # draw boxes on a copy of the frame

    return _encode_image(annotated, Path(filename).suffix or ".jpg")  # return the annotated image as bytes


def process_video_file(
    input_path: Path,
    output_path: Path,
    progress_callback=None,
    enable_seatbelt: bool = True,
    enable_license_plate: bool = True,
    enable_ocr: bool = True,
):
    """Run the detection pipeline on every frame and write an annotated video file."""
    import cv2
    from detect_objects import draw_detections

    pipeline, text_renderer = get_runtime()

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="Could not open the uploaded video")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0   # fall back to 30 fps if metadata is missing
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:   # sanity check before creating a writer
        capture.release()
        raise HTTPException(status_code=400, detail="Invalid video dimensions")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:  # some containers don't store frame count; count manually
        capture.release()
        total_frames = _count_video_frames(input_path)
        capture = cv2.VideoCapture(str(input_path))  # reopen after manual count
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail="Could not reopen the uploaded video")

    # Try codecs in order; use the first one that opens successfully on this system.
    writer = None
    output_media_type = "video/mp4"
    output_path = output_path.with_suffix(".mp4")
    for codec, ext, media_type in (
        ("avc1", ".mp4", "video/mp4"),
        ("H264", ".mp4", "video/mp4"),
        ("VP80", ".webm", "video/webm"),
        ("vp80", ".webm", "video/webm"),
        ("mp4v", ".mp4", "video/mp4"),
    ):
        candidate_path = output_path.with_suffix(ext)
        candidate = cv2.VideoWriter(
            str(candidate_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (frame_width, frame_height),
        )
        if candidate.isOpened():   # this codec works on the current system
            writer = candidate
            output_path = candidate_path
            output_media_type = media_type
            break
        candidate.release()  # codec not supported; try the next one

    if writer is None:
        capture.release()
        raise HTTPException(status_code=500, detail="Could not create output video writer")

    try:
        processed_frames = 0
        with PROCESS_LOCK:  # only one request may run inference at a time
            if hasattr(pipeline, "reset_temporal_state"):
                pipeline.reset_temporal_state()  # clear plate-tracking state before this video

            # Temporarily disable models the user turned off.
            original_seatbelt_model = pipeline.seatbelt_model
            original_lp_model = pipeline.lp_model
            original_ocr_reader = pipeline.ocr_reader
            try:
                if not enable_seatbelt:
                    pipeline.seatbelt_model = None
                if not enable_license_plate:
                    pipeline.lp_model = None
                    pipeline.ocr_reader = None
                elif not enable_ocr:
                    pipeline.ocr_reader = None

                # Prime model+OCR path on a real frame, then restart from frame 0.
                warm_ok, warm_frame = capture.read()
                if warm_ok:
                    pipeline.infer(
                        warm_frame,
                        force_ocr=(enable_ocr and enable_license_plate),
                    )
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # seek back to the beginning
                    if hasattr(pipeline, "reset_temporal_state"):
                        pipeline.reset_temporal_state()  # clear warmup state so real tracking starts fresh

                while True:
                    success, frame = capture.read()
                    if not success:   # end of video
                        break

                    detections = pipeline.infer(frame)
                    annotated = draw_detections(frame.copy(), detections, text_renderer)

                    writer.write(annotated)   # write the annotated frame to the output file
                    processed_frames += 1
                    if progress_callback is not None:
                        progress_callback(processed_frames, total_frames)  # update job progress
            finally:
                # Restore models even if an exception occurred mid-video.
                pipeline.seatbelt_model = original_seatbelt_model
                pipeline.lp_model = original_lp_model
                pipeline.ocr_reader = original_ocr_reader
    finally:
        capture.release()   # always release the video reader
        writer.release()    # flush and close the output file

    return output_path.read_bytes(), output_media_type, total_frames  # return file bytes to the job runner


# Background job runners (called from daemon threads)
def _run_image_job(
    job_id: str,
    file_bytes: bytes,
    filename: str,
    enable_seatbelt: bool,
    enable_license_plate: bool,
    enable_ocr: bool,
) -> None:
    """Process one image job end-to-end and update JOB_STORE with the result."""
    try:
        _save_job(job_id, status="processing", processed_frames=0, total_frames=1, message="Processing image")
        output_bytes, media_type = process_image_bytes(
            file_bytes,
            filename,
            force_ocr=True,
            enable_seatbelt=enable_seatbelt,
            enable_license_plate=enable_license_plate,
            enable_ocr=enable_ocr,
        )
        result_path = RESULT_DIR / f"{job_id}{Path(filename).suffix or '.jpg'}"  # unique filename per job
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_bytes(output_bytes)
        RESULT_VOLUME.commit()  # flush the volume so the file is visible to other containers
        _save_job(
            job_id,
            status="done",
            processed_frames=1,
            total_frames=1,
            progress=100,
            result_media_type=media_type,
            result_filename=f"{Path(filename).stem}_processed{Path(filename).suffix or '.png'}",
            result_path=str(result_path),
            message="Ready",
        )
    except Exception as exc:
        _save_job(job_id, status="failed", error=str(exc), message="Processing failed")


def _run_video_job(
    job_id: str,
    file_bytes: bytes,
    filename: str,
    enable_seatbelt: bool,
    enable_license_plate: bool,
    enable_ocr: bool,
) -> None:
    """Process one video job end-to-end and update JOB_STORE with the result."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"driveguard-{job_id}-"))  # isolated temp folder per job
    input_path = temp_dir / filename
    output_path = temp_dir / f"{Path(filename).stem}_processed.mp4"
    input_path.write_bytes(file_bytes)  # save the uploaded bytes to a real file for OpenCV

    def _progress(done: int, total: int) -> None:
        """Compute percent done and write it to JOB_STORE so the UI can poll it."""
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
        output_bytes, media_type, total_frames = process_video_file(
            input_path,
            output_path,
            progress_callback=_progress,
            enable_seatbelt=enable_seatbelt,
            enable_license_plate=enable_license_plate,
            enable_ocr=enable_ocr,
        )
        output_ext = ".webm" if media_type == "video/webm" else ".mp4"
        result_path = RESULT_DIR / f"{job_id}{output_ext}"  # unique filename per job
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_bytes(output_bytes)
        RESULT_VOLUME.commit()  # flush the volume so the file is visible to other containers
        _save_job(
            job_id,
            status="done",
            processed_frames=total_frames,
            total_frames=total_frames,
            progress=100,
            result_media_type=media_type,
            result_filename=f"{Path(filename).stem}_processed{output_ext}",
            result_path=str(result_path),
            message="Ready",
        )
    except Exception as exc:
        _save_job(job_id, status="failed", error=str(exc), message="Processing failed")
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)  # clean up the temp folder regardless of success or failure


# Browser UI (single-page HTML, CSS, and JavaScript)
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

        .toggle-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin-top: 14px;
            margin-bottom: 14px;
        }

        .toggle {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 14px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.04);
            color: var(--text);
            user-select: none;
        }

        .toggle input {
            width: 18px;
            height: 18px;
            accent-color: var(--accent);
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
            .toggle-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="shell">
        <section class="hero">
            <div class="headline">
                <div class="eyebrow">DriveGuard inference studio</div>
                <h1>Upload an image or video and get the processed result back.</h1>
                <p class="subcopy">
                    The uploaded media is processed with the deployed vehicle, seatbelt, and license-plate models.
                    Processed output is returned directly, you can preview or download it immediately.
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
                    <div class="toggle-grid">
                        <label class="toggle"><input id="ocr-toggle" type="checkbox" checked /> <span>OCR</span></label>
                        <label class="toggle"><input id="lp-toggle" type="checkbox" checked /> <span>License plate</span></label>
                        <label class="toggle"><input id="seatbelt-toggle" type="checkbox" checked /> <span>Seatbelt</span></label>
                    </div>
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
        const ocrToggle = document.getElementById('ocr-toggle');
        const lpToggle = document.getElementById('lp-toggle');
        const seatbeltToggle = document.getElementById('seatbelt-toggle');
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
                formData.append('enable_ocr', ocrToggle.checked ? '1' : '0');
                formData.append('enable_license_plate', lpToggle.checked ? '1' : '0');
                formData.append('enable_seatbelt', seatbeltToggle.checked ? '1' : '0');
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

                                const extension = contentType.includes('webm') ? '.webm' : (contentType.startsWith('video/') ? '.mp4' : '.png');
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


# FastAPI application and routes
# Deploys on a T4 GPU, allows up to 20 concurrent requests, and serves as an ASGI app.
@app.function(image=image, gpu="T4", timeout=60 * 30, volumes={"/root/project/results": RESULT_VOLUME})
@modal.concurrent(max_inputs=20)  # handle up to 20 simultaneous requests per container
@modal.asgi_app()                  # expose this function as an ASGI-compatible web server
def fastapi_app():
    # Pre-warm up models at startup to avoid cold start freeze
    print("[INFO] Pre-warming up models on startup...")
    try:
        pipeline, _ = get_runtime()
        print("[INFO] Model warmup complete. Ready for requests.")
    except Exception as e:
        print(f"[WARN] Model warmup failed: {e}")
    
    web_app = FastAPI(title="DriveGuard")

    @web_app.get("/", response_class=HTMLResponse)
    def home():
        """Serve the single-page upload UI."""
        return HTMLResponse(build_page())

    @web_app.get("/favicon.ico")
    def favicon():
        """Return an empty 204 so browsers don't log a 404 for the favicon."""
        return Response(status_code=204)

    @web_app.post("/process")
    async def process(
        file: UploadFile = File(...),
        enable_ocr: str = Form("1"),
        enable_license_plate: str = Form("1"),
        enable_seatbelt: str = Form("1"),
    ):
        """Accept an upload, create a job, start a background thread, and return the job ID."""
        filename = file.filename or "upload"
        suffix = Path(filename).suffix.lower()
        content_type = (file.content_type or "").lower()
        payload = await file.read()  # read the uploaded file into memory
        ocr_enabled = _parse_flag(enable_ocr)
        lp_enabled = _parse_flag(enable_license_plate)
        seatbelt_enabled = _parse_flag(enable_seatbelt)

        is_image = content_type.startswith("image/") or suffix in {
            ".jpg", ".jpeg", ".png", ".webp", ".bmp"
        }

        is_video = content_type.startswith("video/") or suffix in {
            ".mp4", ".mov", ".avi", ".mkv", ".webm"
        }

        job_id = uuid.uuid4().hex  # unique ID for tracking this specific job
        if is_image:
            JOB_STORE.put(_job_key(job_id), {  # create the initial job record before starting the thread
                "status": "queued",
                "type": "image",
                "filename": filename,
                "processed_frames": 0,
                "total_frames": 1,
                "progress": 0,
                "message": "Queued",
                "result_media_type": None,
                "result_filename": None,
                "result_path": None,
                "error": None,
            })
            threading.Thread(
                target=_run_image_job,
                args=(job_id, payload, filename, seatbelt_enabled, lp_enabled, ocr_enabled),
                daemon=True,   # thread exits automatically when the process exits
            ).start()
            return JSONResponse({"job_id": job_id, "type": "image", "total_frames": 1})

        if is_video:
            total_frames = 0   # actual count is determined inside the job runner
            JOB_STORE.put(_job_key(job_id), {
                "status": "queued",
                "type": "video",
                "filename": filename,
                "processed_frames": 0,
                "total_frames": total_frames,
                "progress": 0,
                "message": "Queued",
                "result_media_type": None,
                "result_filename": None,
                "result_path": None,
                "error": None,
            })
            threading.Thread(
                target=_run_video_job,
                args=(job_id, payload, filename, seatbelt_enabled, lp_enabled, ocr_enabled),
                daemon=True,   # thread exits automatically when the process exits
            ).start()
            return JSONResponse({"job_id": job_id, "type": "video", "total_frames": total_frames})

        raise HTTPException(
            status_code=400,
            detail="Upload an image or video file."
        )

    @web_app.get("/status/{job_id}")
    def status(job_id: str):
        """Return the current progress and status for a job."""
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
        """Stream the finished file back to the browser for preview or download."""
        job = _load_job(job_id)
        if job.get("status") != "done":
            raise HTTPException(status_code=202, detail="Result is not ready yet")  # 202 = accepted but not complete

        media_type = job.get("result_media_type") or "application/octet-stream"
        result_filename = job.get("result_filename") or "driveguard_processed.bin"
        result_path = job.get("result_path")
        if not result_path:
            raise HTTPException(status_code=404, detail="Processed file is missing")

        return FileResponse(
            path=result_path,
            media_type=media_type,
            filename=result_filename,
        )

    return web_app
