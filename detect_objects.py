#!/usr/bin/env python3
"""
Vehicle → Seatbelt + Egyptian License-Plate detection pipeline (YOLO11).

Three-stage, per-vehicle detection with decoupled display/inference threads:

    Stage 1  Car model       → vehicle boxes on the full frame
    Stage 2  Seatbelt model  → seatbelt / no-seatbelt on each vehicle crop
    Stage 3  LP model        → plate boxes on each vehicle crop
             EasyOCR (Arabic) → plate text, tracked across frames

Architecture
------------
* Display thread (main) : reads every frame and shows it immediately at source
                          FPS, overlaying the latest available detections —
                          playback stays smooth even when inference is slow.
* Inference thread      : runs the full pipeline on the freshest frame only
                          (a 1-slot queue drops stale frames automatically).

Key design choices that make this fast and correct
---------------------------------------------------
* Batched sub-model inference: all vehicle crops are sent to the seatbelt and
  LP models in ONE call each, instead of one call per vehicle. This is the main
  reason CUDA now beats CPU — tiny per-crop calls were dominated by launch/
  transfer overhead and left the GPU idle.
* PlateTracker: OCR is expensive, so it runs only every N frames. Detected
  plates are matched to previous ones by proximity, and the last good OCR text
  follows the plate as the car moves — so the text stays on screen instead of
  flickering for a single frame.
* Single PIL pass per frame for Arabic text rendering, instead of a full-frame
  color conversion per plate.

Example
-------
python detect_objects.py --input input_2.mp4 --show-live --show-original \
       --fullscreen --imgsz 320 --realtime
  --car-weights CarDetectionModel.pt  switch to the KITTI fine-tuned detector
  --no-lp                             disable license-plate detection
  --no-ocr                            detect plates but skip OCR (faster)
"""

import argparse
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ── Optional dependencies ──────────────────────────────────────────────────────
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[WARN] easyocr not installed. OCR disabled. Run: pip install easyocr")

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_RENDER_AVAILABLE = True
except ImportError:
    ARABIC_RENDER_AVAILABLE = False
    print("[WARN] arabic_reshaper / python-bidi not installed; Arabic may render reversed.")
    print("       Run: pip install arabic-reshaper python-bidi")


# ════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════
SEATBELT_COLORS = {
    'seatbelt':    (0, 200, 0),    # green  — belt detected
    'no-seatbelt': (0, 0, 220),    # red    — no belt
}
NO_DETECTION_COLOR = (0, 140, 255)  # orange — vehicle found but no seatbelt result
LP_BOX_COLOR       = (0, 220, 255)  # yellow — license-plate box
LP_TEXT_BG_COLOR   = (20, 20, 20)   # dark background behind plate text

# Fonts tried (in order) for rendering Arabic plate text.
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/times.ttf",
]


# ════════════════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════════════════
Box = Tuple[float, float, float, float]


@dataclass
class Plate:
    """A detected license plate in full-frame coordinates."""
    box: Box
    text: str = ""


@dataclass
class Detection:
    """One detected vehicle plus its seatbelt and plate sub-results."""
    box: Box
    conf: float
    cls: int
    name: str
    seatbelt: Optional[Tuple[str, float]] = None   # (label, conf)
    plates: List[Plate] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════════
# Arabic text rendering (PIL)
# ════════════════════════════════════════════════════════════════════════════
class ArabicTextRenderer:
    """Renders UTF-8 / Arabic strings onto BGR frames using PIL.

    OpenCV's putText cannot draw Arabic, so we composite text via PIL. To keep
    it cheap, all text labels for a frame are drawn in a SINGLE conversion
    pass (BGR→PIL→BGR once per frame, not once per label).
    """

    def __init__(self) -> None:
        self._font_cache: dict = {}

    def _font(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self._font_cache:
            font = None
            for path in _FONT_CANDIDATES:
                if os.path.exists(path):
                    try:
                        font = ImageFont.truetype(path, size)
                        break
                    except Exception:
                        continue
            self._font_cache[size] = font or ImageFont.load_default()
        return self._font_cache[size]

    @staticmethod
    def _shape(text: str) -> str:
        """Reshape + apply BiDi so Arabic glyphs connect and order correctly."""
        if text and ARABIC_RENDER_AVAILABLE:
            try:
                return get_display(arabic_reshaper.reshape(text))
            except Exception:
                pass
        return text

    def draw_labels(self, frame_bgr: np.ndarray,
                    labels: List[Tuple[str, int, int]],
                    font_size: int = 18,
                    text_color=(255, 255, 255),
                    bg_color=LP_TEXT_BG_COLOR,
                    padding: int = 4) -> np.ndarray:
        """Draw a list of (text, x, y) labels in one pass. Returns new frame."""
        if not labels:
            return frame_bgr

        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = self._font(font_size)
        h, w = frame_bgr.shape[:2]

        for text, x, y in labels:
            text = self._shape(text)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            # Clamp so the label stays fully inside the frame.
            x = max(0, min(x, w - tw - padding * 2))
            y = max(0, min(y, h - th - padding * 2))
            draw.rectangle([x, y, x + tw + padding * 2, y + th + padding * 2], fill=bg_color)
            draw.text((x + padding, y + padding), text, font=font, fill=text_color)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ════════════════════════════════════════════════════════════════════════════
# Plate tracker — keeps OCR text on screen as the plate moves
# ════════════════════════════════════════════════════════════════════════════
class PlateTracker:
    """Associates plate detections across frames by proximity.

    OCR runs only every few frames (it is slow). Between runs, we match each
    new plate box to the nearest known plate and reuse its last *non-empty*
    text. This is what keeps the plate text visible instead of flashing for a
    single frame — the previous keying-by-exact-coordinates approach never hit
    because a moving plate shifts every frame.
    """

    def __init__(self, ttl: int = 45) -> None:
        self._tracks: List[dict] = []   # {box, text, age}
        self._ttl = ttl                 # frames a track survives without a match

    @staticmethod
    def _center(box: Box) -> Tuple[float, float]:
        return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

    def resolve(self, box: Box, new_text: str) -> str:
        """Match `box` to a track; update text if `new_text` is non-empty.

        Returns the text that should be displayed for this plate.
        """
        cx, cy = self._center(box)
        width = box[2] - box[0]
        # Match radius scales with plate size; min 40 px for tiny far plates.
        max_dist = max(40.0, 0.9 * width)

        best, best_dist = None, float('inf')
        for track in self._tracks:
            tcx, tcy = self._center(track['box'])
            dist = math.hypot(cx - tcx, cy - tcy)
            if dist < best_dist:
                best, best_dist = track, dist

        if best is not None and best_dist <= max_dist:
            best['box'] = box
            best['age'] = 0
            if new_text:                       # only overwrite with a real read
                best['text'] = new_text
            return best['text']

        # No nearby track → start a new one.
        self._tracks.append({'box': box, 'text': new_text, 'age': 0})
        return new_text

    def end_frame(self) -> None:
        """Age all tracks and drop ones not seen for `ttl` frames."""
        for track in self._tracks:
            track['age'] += 1
        self._tracks = [t for t in self._tracks if t['age'] <= self._ttl]


# ════════════════════════════════════════════════════════════════════════════
# Detection pipeline
# ════════════════════════════════════════════════════════════════════════════
class DetectionPipeline:
    """Runs car → seatbelt → license-plate → OCR on a single frame.

    All sub-model inference is BATCHED: every vehicle crop is sent to the
    seatbelt and LP models in one call each, regardless of how many vehicles
    are present. This keeps the GPU busy and is the key to CUDA outperforming
    CPU on this workload.
    """

    def __init__(self, car_model, seatbelt_model, lp_model, ocr_reader,
                 vehicle_classes, conf, iou, device, imgsz, sub_imgsz,
                 ocr_every, max_vehicles=8, profile=False):
        self.car_model = car_model
        self.seatbelt_model = seatbelt_model
        self.lp_model = lp_model
        self.ocr_reader = ocr_reader
        self.vehicle_classes = vehicle_classes
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz              # image size for full-frame car detection
        self.sub_imgsz = sub_imgsz      # smaller size for seatbelt/LP crops
        self.half = (device == 'cuda')
        self.ocr_every = max(1, ocr_every)
        self.max_vehicles = max(1, max_vehicles)  # cap crops/OCR per frame for speed
        self.profile = profile

        self.car_names = car_model.names
        self._tracker = PlateTracker()
        self._frame_idx = 0
        # Blank tile used to pad sub-model batches to a constant size (see infer).
        self._pad_tile = np.zeros((sub_imgsz, sub_imgsz, 3), dtype=np.uint8)

    # ── Common predict wrapper ────────────────────────────────────────────────
    def _predict(self, model, source, imgsz=None):
        return model.predict(
            source, conf=self.conf, iou=self.iou, imgsz=imgsz or self.imgsz,
            device=self.device, half=self.half, verbose=False,
        )

    def warmup(self) -> None:
        """Run dummy inferences so CUDA/cuDNN tune for the exact shapes used at
        run time. Critically, the sub-models are warmed at the FULL fixed batch
        size (max_vehicles) — the same shape infer() always feeds them — so the
        one-time per-shape tuning cost is paid here, not as a spike mid-video."""
        big = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        batch = [self._pad_tile] * self.max_vehicles
        self._predict(self.car_model, big, self.imgsz)
        for model in (self.seatbelt_model, self.lp_model):
            if model is not None:
                self._predict(model, batch, self.sub_imgsz)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _clip_crop(frame, x1, y1, x2, y2):
        """Return (crop, (x_offset, y_offset)) clipped to frame bounds, or
        (None, None) if the region is empty."""
        h, w = frame.shape[:2]
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(w, int(x2)), min(h, int(y2))
        if cx2 <= cx1 or cy2 <= cy1:
            return None, None
        return frame[cy1:cy2, cx1:cx2], (cx1, cy1)

    @staticmethod
    def _best_label(result, model) -> Optional[Tuple[str, float]]:
        """Highest-confidence (label, conf) from a Results object, or None."""
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None
        i = int(boxes.conf.argmax().item())
        label = model.names.get(int(boxes.cls[i].item()), "unknown")
        return label, float(boxes.conf[i].item())

    def _ocr(self, frame, box: Box) -> str:
        """Run EasyOCR on a plate crop (CPU). Egyptian plates: Arabic letters
        + numerals (RTL). Returns joined text or ''."""
        if self.ocr_reader is None:
            return ""
        crop, _ = self._clip_crop(frame, *box)
        if crop is None:
            return ""
        # Upscale small plates — OCR accuracy drops sharply below ~32 px tall.
        ch = crop.shape[0]
        if ch < 32:
            scale = 32 / ch
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        try:
            tokens = self.ocr_reader.readtext(crop, detail=0, paragraph=False)
            return ' '.join(tokens).strip()
        except Exception:
            return ""

    # ── Main entry ────────────────────────────────────────────────────────────
    def infer(self, frame) -> List[Detection]:
        """Full pipeline on one frame → list of Detection."""
        self._frame_idx += 1
        run_ocr = (self._frame_idx % self.ocr_every == 0)
        t = time.perf_counter
        t_car = t_sb = t_lp = t_ocr = 0.0

        # Stage 1 — vehicle detection on the full frame.
        t0 = t()
        car_result = self._predict(self.car_model, frame, self.imgsz)[0]
        t_car = t() - t0

        detections: List[Detection] = []
        crops: List[np.ndarray] = []
        crop_meta: List[Tuple[int, int, int]] = []   # (detection_idx, x_off, y_off)

        if car_result.boxes is not None:
            # Process the largest/closest vehicles first, capped at max_vehicles —
            # bounds the per-frame sub-model + OCR cost in busy scenes.
            boxes_sorted = sorted(
                car_result.boxes,
                key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]),
                reverse=True,
            )
            for box in boxes_sorted:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls.item())
                name = self.car_names.get(cls, "") if isinstance(self.car_names, dict) \
                    else (self.car_names[cls] if cls < len(self.car_names) else "")

                det = Detection(box=(x1, y1, x2, y2), conf=float(box.conf.item()),
                                cls=cls, name=name)
                detections.append(det)

                # Only crop vehicles we care about (saves sub-model work).
                if name.lower() in self.vehicle_classes and len(crops) < self.max_vehicles:
                    crop, offset = self._clip_crop(frame, x1, y1, x2, y2)
                    if crop is not None:
                        crops.append(crop)
                        crop_meta.append((len(detections) - 1, offset[0], offset[1]))

        # Pad the crop list to a CONSTANT batch size with blank tiles. Sub-models
        # then always receive the same input shape, so cuDNN tunes once (at
        # warmup) instead of re-tuning on every batch-size change — which caused
        # the 1.5–3 s spikes. Padded results are sliced off via [:n_real].
        n_real = len(crops)
        batch = (crops + [self._pad_tile] * (self.max_vehicles - n_real)) if n_real else None

        # Stage 2 — seatbelt on the fixed-size batch (at sub_imgsz).
        if batch is not None and self.seatbelt_model is not None:
            t0 = t()
            results = self._predict(self.seatbelt_model, batch, self.sub_imgsz)[:n_real]
            for (idx, _, _), result in zip(crop_meta, results):
                detections[idx].seatbelt = self._best_label(result, self.seatbelt_model)
            t_sb = t() - t0

        # Stage 3 — license plates on the fixed-size batch (at sub_imgsz).
        n_plates = 0
        if batch is not None and self.lp_model is not None:
            t0 = t()
            lp_results = self._predict(self.lp_model, batch, self.sub_imgsz)[:n_real]
            t_lp = t() - t0
            for (idx, ox, oy), result in zip(crop_meta, lp_results):
                if result.boxes is None:
                    continue
                for pbox in result.boxes:
                    px1, py1, px2, py2 = pbox.xyxy[0].tolist()
                    # Translate crop-relative coords back to full-frame coords.
                    full_box: Box = (px1 + ox, py1 + oy, px2 + ox, py2 + oy)
                    n_plates += 1
                    t0 = t()
                    new_text = self._ocr(frame, full_box) if run_ocr else ""
                    t_ocr += t() - t0
                    text = self._tracker.resolve(full_box, new_text)
                    detections[idx].plates.append(Plate(box=full_box, text=text))

        self._tracker.end_frame()

        if self.profile:
            total = (t_car + t_sb + t_lp + t_ocr) * 1000
            print(f"[infer #{self._frame_idx}] vehicles={len(crops)} plates={n_plates} "
                  f"ocr={'Y' if run_ocr else 'n'} | car={t_car*1000:.0f} sb={t_sb*1000:.0f} "
                  f"lp={t_lp*1000:.0f} ocr={t_ocr*1000:.0f} total={total:.0f} ms", flush=True)

        return detections


# ════════════════════════════════════════════════════════════════════════════
# Rendering
# ════════════════════════════════════════════════════════════════════════════
def draw_detections(frame, detections: List[Detection],
                    text_renderer: ArabicTextRenderer) -> np.ndarray:
    """Draw vehicle boxes, car/seatbelt labels, and plate boxes + OCR text.

    All OpenCV primitives are drawn first; Arabic plate text is composited in a
    single PIL pass at the end for efficiency.
    """
    plate_labels: List[Tuple[str, int, int]] = []

    for det in detections:
        x1, y1, x2, y2 = map(int, det.box)
        name = det.name or "Vehicle"

        # Vehicle box color reflects seatbelt status.
        if det.seatbelt is not None:
            sb_label, sb_conf = det.seatbelt
            color = SEATBELT_COLORS.get(sb_label, NO_DETECTION_COLOR)
        else:
            sb_label, sb_conf = None, None
            color = NO_DETECTION_COLOR

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Car label — above the box.
        car_text = f"{name} {round(det.conf, 1):.1f}"
        (tw, tht), base = cv2.getTextSize(car_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        top = max(0, y1 - tht - 8)
        cv2.rectangle(frame, (x1, top), (x1 + tw, top + tht + base), color, cv2.FILLED)
        cv2.putText(frame, car_text, (x1, top + tht),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Seatbelt label — bottom-left of the box.
        if sb_label is not None:
            sb_text = f"{sb_label} {round(sb_conf, 1):.1f}"
            (sw, sh), _ = cv2.getTextSize(sb_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            sb_top = max(0, y2 - sh - 8)
            cv2.rectangle(frame, (x1, sb_top), (x1 + sw, sb_top + sh + 4), color, cv2.FILLED)
            cv2.putText(frame, sb_text, (x1, sb_top + sh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Plate boxes (drawn now) + text (queued for the single PIL pass).
        for plate in det.plates:
            px1, py1, px2, py2 = map(int, plate.box)
            cv2.rectangle(frame, (px1, py1), (px2, py2), LP_BOX_COLOR, 2)
            if plate.text:
                plate_labels.append((plate.text, px1, py2 + 2))

    return text_renderer.draw_labels(frame, plate_labels)


# ════════════════════════════════════════════════════════════════════════════
# Display helpers
# ════════════════════════════════════════════════════════════════════════════
def get_screen_resolution(default=(1280, 720)) -> Tuple[int, int]:
    """Best-effort screen size for fullscreen layout (falls back to default)."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return default


# ════════════════════════════════════════════════════════════════════════════
# Threaded video application
# ════════════════════════════════════════════════════════════════════════════
class VideoApp:
    """Decoupled display (main thread) + inference (worker thread) player."""

    # ASCII-only title: a non-ASCII char (e.g. '·') makes OpenCV on Windows
    # encode the name differently in namedWindow vs imshow, opening two windows.
    WINDOW_NAME = 'Vehicle - Seatbelt - Plate Detection'

    def __init__(self, pipeline: DetectionPipeline, input_path, output_path,
                 show_live, show_original, save_output, display_size,
                 fullscreen, realtime, frame_skip, text_renderer):
        self.pipeline = pipeline
        self.input_path = input_path
        self.output_path = output_path
        self.show_live = show_live
        self.show_original = show_original
        self.save_output = save_output
        self.display_w, self.display_h = display_size
        self.fullscreen = fullscreen
        self.realtime = realtime
        self.frame_skip = max(1, frame_skip)   # enqueue 1 of every N frames for inference
        self.text_renderer = text_renderer

        # Shared state between threads.
        self._detections: List[Detection] = []
        self._lock = threading.Lock()
        self._infer_count = 0
        self._infer_queue: queue.Queue = queue.Queue(maxsize=1)  # 1 slot → drop stale
        self._stop = threading.Event()

        # Playback state (main thread only).
        self._frame_count = 0
        self._start_time = 0.0
        self._last_frame_time = 0.0
        self._paused = False

    # ── Inference worker ──────────────────────────────────────────────────────
    def _inference_worker(self, ready_event: threading.Event):
        # 1. Initialise CUDA context in THIS thread — PyTorch ties the context
        #    to the thread that first calls into CUDA.
        if self.pipeline.device == 'cuda':
            torch.cuda.init()
            # Override benchmark=True set by Ultralytics. With our fixed batch
            # size it is unnecessary; any shape change would re-trigger the
            # multi-second cuDNN search we worked hard to eliminate.
            torch.backends.cudnn.benchmark = False

        # 2. Warm up every model here, in the same thread and CUDA context that
        #    will run real inference. This pays the cold-start cost (CUDA kernel
        #    compilation, cuDNN tuning) before the video window opens, so frame
        #    #1 is already fast when the user sees it.
        print("Warming up models… ", end="", flush=True)
        self.pipeline.warmup()
        print("done.")

        # 3. Signal the main thread that warmup is finished and playback can start.
        ready_event.set()

        while not self._stop.is_set():
            try:
                frame = self._infer_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                detections = self.pipeline.infer(frame)
            except Exception as exc:
                import traceback
                print(f"\n[ERROR] Inference thread: {exc}")
                traceback.print_exc()
                continue
            with self._lock:
                self._detections = detections
                self._infer_count += 1

    def _flush_inference(self):
        """Drop any queued frame and stale results (after seek/restart)."""
        try:
            self._infer_queue.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            self._detections = []

    # ── Display composition ───────────────────────────────────────────────────
    def _compose(self, original, annotated, screen):
        if self.fullscreen:
            sw, sh = screen
            pw = max(1, sw // 2) if self.show_original else max(1, sw)
            ph = max(1, sh)
        else:
            pw, ph = self.display_w, self.display_h
        ann = cv2.resize(annotated, (pw, ph))
        if not self.show_original:
            return ann
        return cv2.hconcat([cv2.resize(original, (pw, ph)), ann])

    def _fps(self):
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0, 0.0
        with self._lock:
            ic = self._infer_count
        return self._frame_count / elapsed, ic / elapsed

    def _draw_hud(self, disp, status=""):
        stream_fps, infer_fps = self._fps()
        cv2.rectangle(disp, (10, 10), (700, 110 if status else 100), (0, 0, 0), cv2.FILLED)
        cv2.putText(disp, f"Stream FPS: {stream_fps:.2f}", (18, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp, f"Infer FPS:  {infer_fps:.2f}", (18, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(disp, "space pause  a/d seek  r restart  q quit", (18, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if status:
            cv2.putText(disp, status, (520, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # ── Keyboard ──────────────────────────────────────────────────────────────
    def _handle_key(self, cap, key, seek_step) -> bool:
        """Return False to quit."""
        if key == ord('q'):
            return False
        if key == ord(' '):
            self._paused = not self._paused
            if not self._paused:
                self._last_frame_time = time.time()
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._frame_count = 0
            self._start_time = self._last_frame_time = time.time()
            self._flush_inference()
            self._paused = False
        elif key == ord('a'):
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - seek_step))
            self._flush_inference()
        elif key == ord('d'):
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos + seek_step)
            self._flush_inference()
        return True

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video source: {self.input_path}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        seek_step = max(1, int(fps * 2))
        target_interval = 1.0 / fps

        out = None
        if self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (fw, fh))

        # Start the inference thread. It initialises CUDA, warms up all models,
        # then sets ready_event. The main thread blocks here until warmup is
        # complete — so the video window only opens when inference is fully ready
        # and frame #1 will already have fast detection instead of 0 FPS.
        ready_event = threading.Event()
        worker = threading.Thread(target=self._inference_worker,
                                  args=(ready_event,), daemon=True)
        worker.start()
        ready_event.wait()   # blocks until warmup done; then we open the window

        if self.show_live:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            if self.fullscreen:
                cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
        screen = get_screen_resolution()

        self._start_time = self._last_frame_time = time.time()
        annotated = None       # last drawn overlay
        last_frame = None      # last decoded raw frame
        last_infer_seen = -1   # _infer_count value when we last redrew

        # Always pace display to video FPS regardless of --realtime flag.
        # Letting the display run uncapped (e.g. 196 FPS) makes draw_detections
        # (with its PIL pass) dominate the main thread and starves the GPU,
        # cutting inference FPS roughly in half.
        display_interval = 1.0 / fps if fps > 0 else 1.0 / 30.0

        try:
            while True:
                # ── Paused: keep refining the current frame ────────────────────
                # Feed the same frame repeatedly so inference keeps running and
                # OCR fires (every N frames). Redraw only when a new inference
                # result arrives so the PIL pass isn't wasted.
                if self._paused:
                    if last_frame is not None:
                        try:
                            self._infer_queue.put_nowait(last_frame.copy())
                        except queue.Full:
                            pass
                        with self._lock:
                            ic = self._infer_count
                            detections = self._detections
                        if ic != last_infer_seen:   # new result → redraw
                            annotated = draw_detections(last_frame.copy(), detections,
                                                        self.text_renderer)
                            last_infer_seen = ic
                        if self.show_live and annotated is not None:
                            disp = self._compose(last_frame, annotated, screen)
                            self._draw_hud(disp, status="PAUSED")
                            cv2.imshow(self.WINDOW_NAME, disp)
                    if not self._handle_key(cap, cv2.waitKey(50) & 0xFF, seek_step):
                        break
                    continue

                ret, frame = cap.read()
                if not ret:
                    break
                self._frame_count += 1
                last_frame = frame

                # Hand the freshest frame to inference (non-blocking; 1-slot
                # queue drops it if the worker is still busy).
                if self._frame_count % self.frame_skip == 0:
                    try:
                        self._infer_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass

                # Only redraw the overlay when a new inference result is ready.
                # This means draw_detections (PIL pass) runs at inference FPS,
                # not display FPS — removing the main source of GPU contention.
                with self._lock:
                    ic = self._infer_count
                    detections = self._detections
                if ic != last_infer_seen:
                    annotated = draw_detections(frame.copy(), detections, self.text_renderer)
                    last_infer_seen = ic
                elif annotated is None:
                    annotated = frame.copy()   # blank until first inference

                if out is not None:
                    out.write(annotated)

                if self.show_live:
                    disp = self._compose(frame, annotated, screen)
                    self._draw_hud(disp)
                    cv2.imshow(self.WINDOW_NAME, disp)
                    if not self._handle_key(cap, cv2.waitKey(1) & 0xFF, seek_step):
                        break

                # Pace display to video FPS (always on, replaces --realtime flag).
                target = self._last_frame_time + display_interval
                now = time.time()
                if now < target:
                    time.sleep(target - now)
                    self._last_frame_time = target
                else:
                    self._last_frame_time = now
        finally:
            self._stop.set()
            worker.join(timeout=3.0)
            cap.release()
            if out is not None:
                out.release()
            if self.show_live:
                cv2.destroyAllWindows()

        stream_fps, infer_fps = self._fps()
        print(f"Stream FPS:    {stream_fps:.2f}")
        print(f"Inference FPS: {infer_fps:.2f}")
        if self.save_output:
            print(f"Done. Output saved to: {self.output_path}")


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description='Vehicle detection → seatbelt + Egyptian license plate (OCR)')
    p.add_argument('--input', type=str, required=True)
    p.add_argument('--output', type=str, required=False)
    p.add_argument('--car-weights', type=str, default='yolo11n.pt',
                   help='Car detector. yolo11n.pt (COCO) or CarDetectionModel.pt (KITTI).')
    p.add_argument('--seatbelt-weights', type=str, default='SeatBeltModel.pt')
    p.add_argument('--lp-weights', type=str, default='LicensePlateModel.pt',
                   help='Egyptian license-plate detector weights.')
    p.add_argument('--no-sb', action='store_true', help='Disable seatbelt detection entirely.')
    p.add_argument('--no-lp', action='store_true', help='Disable plate detection entirely.')
    p.add_argument('--no-ocr', action='store_true', help='Detect plates but skip OCR (faster).')
    p.add_argument('--ocr-device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto',
                   help="EasyOCR device. 'auto' uses GPU when CUDA is present. Use 'cpu' to "
                        "free VRAM if the detection models stall under GPU memory pressure.")
    p.add_argument('--ocr-every', type=int, default=5,
                   help='Run OCR once every N inference frames; text is tracked between '
                        'runs so it stays on screen. Higher = faster. Default: 5.')
    p.add_argument('--vehicle-classes', type=str, default='car,van,truck,bus',
                   help='Case-insensitive classes to run sub-models on. '
                        'COCO: car,truck,bus | KITTI: Car,Van,Truck')
    p.add_argument('--conf-threshold', type=float, default=0.25)
    p.add_argument('--iou-threshold', type=float, default=0.45)
    p.add_argument('--show-live', action='store_true')
    p.add_argument('--show-original', action='store_true')
    p.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--imgsz', type=int, default=640,
                   help='Image size for full-frame car detection. Default: 640.')
    p.add_argument('--sub-imgsz', type=int, default=320,
                   help='Image size for seatbelt/LP models (run on small crops). '
                        'Lower = faster. Default: 320.')
    p.add_argument('--max-vehicles', type=int, default=8,
                   help='Max vehicles (largest first) to run sub-models/OCR on per frame.')
    p.add_argument('--profile', action='store_true',
                   help='Print per-stage timing (car/seatbelt/LP/OCR ms) each inference.')
    p.add_argument('--frame-skip', type=int, default=1,
                   help='Enqueue 1 of every N frames for inference (display still shows all).')
    p.add_argument('--no-save', action='store_true')
    p.add_argument('--display-width', type=int, default=640)
    p.add_argument('--display-height', type=int, default=360)
    p.add_argument('--fullscreen', action='store_true')
    p.add_argument('--realtime', action='store_true',
                   help='Deprecated — display is now always capped to video FPS. '
                        'Kept for backward compatibility.')
    return p.parse_args()


def resolve_device(requested: str) -> str:
    if requested == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif requested == 'cuda' and not torch.cuda.is_available():
        print('CUDA requested but not available. Falling back to CPU.')
        device = 'cpu'
    else:
        device = requested
    print(f"Using device: {device}")
    return device


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # NOTE on cuDNN: Ultralytics sets torch.backends.cudnn.benchmark = True, which
    # makes cuDNN re-run an expensive (~1.5–3 s) algorithm search on every new
    # input shape. We pin it to False (see VideoApp._inference_worker) AND feed the
    # sub-models a constant batch size, so shapes never change — eliminating the
    # mid-video spikes entirely.

    # Path helpers: bare names resolve under the project's standard folders.
    def resolve_input(name):
        return name if os.path.dirname(name) else os.path.join('Detection Video Validation', name)

    def resolve_model(name):
        return name if os.path.dirname(name) else os.path.join('Models', name)

    # ── Load models ────────────────────────────────────────────────────────────
    car_path = resolve_model(args.car_weights)
    print(f"Loading car model:      {car_path}")
    car_model = YOLO(car_path).to(device)

    seatbelt_model = None
    if not args.no_sb:
        sb_path = resolve_model(args.seatbelt_weights)
        print(f"Loading seatbelt model: {sb_path}")
        seatbelt_model = YOLO(sb_path).to(device)
    else:
        print("Seatbelt detection disabled (--no-sb).")

    lp_model, ocr_reader = None, None
    if not args.no_lp:
        lp_path = resolve_model(args.lp_weights)
        print(f"Loading LP model:       {lp_path}")
        lp_model = YOLO(lp_path).to(device)

        if not args.no_ocr:
            if EASYOCR_AVAILABLE:
                # Use the GPU for OCR when available — recognition is much faster
                # there. Fall back to CPU if the GPU reader fails to initialise
                # (e.g. limited VRAM), so the pipeline always keeps working.
                if args.ocr_device == 'auto':
                    use_gpu_ocr = (device == 'cuda')
                else:
                    use_gpu_ocr = (args.ocr_device == 'cuda')
                print(f"Initialising EasyOCR (Arabic, {'GPU' if use_gpu_ocr else 'CPU'})… "
                      "(first run downloads ~500 MB)")
                try:
                    ocr_reader = easyocr.Reader(['ar'], gpu=use_gpu_ocr, verbose=False)
                except Exception as exc:
                    print(f"[WARN] EasyOCR GPU init failed ({exc}); falling back to CPU.")
                    ocr_reader = easyocr.Reader(['ar'], gpu=False, verbose=False)
                print("EasyOCR ready.")
            else:
                print("[WARN] EasyOCR not available — OCR skipped.")
    else:
        print("License-plate detection disabled (--no-lp).")

    vehicle_classes = {c.strip().lower() for c in args.vehicle_classes.split(',')}
    print(f"Seatbelt/LP on classes (case-insensitive): {vehicle_classes}")

    # ── Resolve paths ──────────────────────────────────────────────────────────
    input_path = resolve_input(args.input)
    if args.output:
        output_path = resolve_input(args.output)
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_annotated{ext}"

    # ── Build pipeline + app and run ───────────────────────────────────────────
    pipeline = DetectionPipeline(
        car_model=car_model, seatbelt_model=seatbelt_model, lp_model=lp_model,
        ocr_reader=ocr_reader, vehicle_classes=vehicle_classes,
        conf=args.conf_threshold, iou=args.iou_threshold,
        device=device, imgsz=args.imgsz, sub_imgsz=args.sub_imgsz,
        ocr_every=args.ocr_every, max_vehicles=args.max_vehicles,
        profile=args.profile,
    )

    app = VideoApp(
        pipeline=pipeline, input_path=input_path, output_path=output_path,
        show_live=args.show_live, show_original=args.show_original,
        save_output=(not args.no_save),
        display_size=(max(1, args.display_width), max(1, args.display_height)),
        fullscreen=args.fullscreen, realtime=args.realtime,
        frame_skip=args.frame_skip, text_renderer=ArabicTextRenderer(),
    )
    app.run()


if __name__ == "__main__":
    main()
