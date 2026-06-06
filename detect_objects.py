#!/usr/bin/env python3
"""
Vehicle, seatbelt, and Egyptian license-plate detection pipeline.

The script runs vehicle detection first, then runs seatbelt and plate models on
vehicle crops. OCR text is tracked across frames so plate text stays visible
without running OCR on every frame.
"""

import argparse          # command-line argument parsing
import math              # for hypot (Euclidean distance)
import os               # file path checks
import queue            # thread-safe queue for passing frames between threads
import threading        # background inference thread
import time             # FPS timing and sleep
from dataclasses import dataclass, field   # clean data container definitions
from typing import List, Optional, Tuple   # type hints

import cv2              # OpenCV: video I/O, drawing, resizing
import numpy as np      # array operations on image data
import torch            # PyTorch: needed to check CUDA availability and init
from ultralytics import YOLO               # YOLO model loader and inference
from PIL import Image, ImageDraw, ImageFont  # PIL: used to render Arabic text correctly

# Optional dependencies
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

# Constants
SEATBELT_COLORS = {
    'seatbelt':    (0, 200, 0),    # green: belt detected
    'no-seatbelt': (0, 0, 220),    # red: no belt
}
NO_DETECTION_COLOR = (0, 140, 255)  # orange: vehicle found but no seatbelt result
LP_BOX_COLOR       = (0, 220, 255)  # yellow: license-plate box
LP_TEXT_BG_COLOR   = (20, 20, 20)   # dark background behind plate text

# Fonts tried (in order) for rendering Arabic plate text.
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/times.ttf",
]

# Supported still-image extensions — used to decide whether to run image or video mode.
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# Seatbelt label normalization across different model taxonomies.
# Map to canonical labels for coloring; None means ignore the class.
SEATBELT_LABEL_MAP = {
    'seatbelt': 'seatbelt',
    'no-seatbelt': 'no-seatbelt',
    'no seatbelt': 'no-seatbelt',
    'no_seatbelt': 'no-seatbelt',
    'mobile': 'no-seatbelt',
    'windshield': None,
}

# Data structures
Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2) bounding box in pixels

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
    plates: List[Plate] = field(default_factory=list)  # list of plates found inside this vehicle crop

# Arabic text rendering (PIL)
class ArabicTextRenderer:
    """Draws Arabic and UTF-8 labels on OpenCV frames using PIL.

    OpenCV text cannot shape Arabic correctly, so labels are rendered with PIL.
    """

    def __init__(self) -> None:
        """Create a font cache so repeated label drawing stays fast."""
        self._font_cache: dict = {}

    def _font(self, size: int) -> ImageFont.FreeTypeFont:
        """Return a cached font for the requested size."""
        if size not in self._font_cache:          # only load the font once per size
            font = None
            for path in _FONT_CANDIDATES:          # try each candidate path in order
                if os.path.exists(path):            # skip paths that don't exist on this OS
                    try:
                        font = ImageFont.truetype(path, size)  # load TrueType font at requested size
                        break                       # stop at the first one that loads successfully
                    except Exception:
                        continue                    # try next candidate if this one fails
            # Fall back to PIL's built-in default font if none of the paths worked.
            self._font_cache[size] = font or ImageFont.load_default()
        return self._font_cache[size]

    @staticmethod
    def _shape(text: str) -> str:
        """Shape Arabic text so glyphs connect and display in the right order."""
        if text and ARABIC_RENDER_AVAILABLE:
            try:
                # reshape joins Arabic letters correctly, get_display fixes right-to-left order
                return get_display(arabic_reshaper.reshape(text))
            except Exception:
                pass                # if shaping fails, fall through and return the raw text
        return text                 # return as-is if libraries are not available

    def draw_labels(self, frame_bgr: np.ndarray,
                    labels: List[Tuple[str, int, int]],
                    font_size: int = 18,
                    text_color=(255, 255, 255),
                    bg_color=LP_TEXT_BG_COLOR,
                    padding: int = 4) -> np.ndarray:
        """Draw all queued text labels with one PIL conversion pass."""
        if not labels:
            return frame_bgr       # nothing to draw; return the frame unchanged

        # OpenCV uses BGR but PIL uses RGB, so convert before drawing.
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)   # PIL drawing context
        font = self._font(font_size)     # get (or load) the font at the right size
        h, w = frame_bgr.shape[:2]       # frame dimensions used for clamping

        for text, x, y in labels:
            text = self._shape(text)     # reshape Arabic glyphs so they render correctly
            bbox = draw.textbbox((0, 0), text, font=font)  # measure rendered text size
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]  # text width and height in pixels
            # Clamp label position so the text box stays inside the frame.
            x = max(0, min(x, w - tw - padding * 2))
            y = max(0, min(y, h - th - padding * 2))
            # Draw dark background rectangle first, then the text on top.
            draw.rectangle([x, y, x + tw + padding * 2, y + th + padding * 2], fill=bg_color)
            draw.text((x + padding, y + padding), text, font=font, fill=text_color)

        # Convert PIL image back to BGR for OpenCV to use.
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Plate tracking
class PlateTracker:
    """Tracks plate boxes so OCR text can follow moving plates between reads."""

    def __init__(self, ttl: int = 45) -> None:
        """Create an empty track list with a frame-based time-to-live."""
        self._tracks: List[dict] = []   # {box, text, age, last_ocr_frame}
        self._ttl = ttl                 # frames a track survives without a match

    @staticmethod
    def _center(box: Box) -> Tuple[float, float]:
        """Return the center point of a bounding box."""
        return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

    def _find_match(self, box: Box) -> Optional[dict]:
        """Find the nearest recent plate track for the current box."""
        cx, cy = self._center(box)
        width = box[2] - box[0]
        # Wider plates can move farther between frames; tiny plates still get 40 px.
        max_dist = max(40.0, 0.9 * width)

        best, best_dist = None, float('inf')
        for track in self._tracks:
            if track.get('age', 0) > 1:  # skip tracks missing for more than 1 frame
                continue
            tcx, tcy = self._center(track['box'])
            dist = math.hypot(cx - tcx, cy - tcy)  # Euclidean distance between centers
            if dist < best_dist:
                best, best_dist = track, dist

        if best is not None and best_dist <= max_dist:
            return best
        return None  # no match found within the allowed range

    @staticmethod
    def _should_ocr(track: Optional[dict], frame_idx: int, every: int) -> bool:
        """Decide whether OCR should run for this plate on this frame."""
        if track is None:
            return True
        last = track.get('last_ocr_frame')
        if last is None:
            return True
        return (frame_idx - last) >= every

    def update(self, box: Box, track: Optional[dict], new_text: str,
               frame_idx: int, ocr_ran: bool) -> str:
        """Update a plate track and return the text that should be displayed."""
        if track is None:
            # No existing track: create a new one with this box and the OCR result.
            self._tracks.append({
                'box': box,
                'text': new_text,
                'age': 0,
                # Only record a frame number if OCR actually produced text.
                'last_ocr_frame': frame_idx if (ocr_ran and bool(new_text)) else None,
            })
            return new_text

        track['box'] = box   # update box position to the latest detection
        track['age'] = 0     # reset age so this track isn't removed
        if ocr_ran:
            if new_text:
                # First successful read: switch to regular cadence.
                track['last_ocr_frame'] = frame_idx
                track['text'] = new_text
            elif not track.get('text'):
                # No text yet: retry OCR on the very next frame.
                track['last_ocr_frame'] = None
            else:
                # Keep cadence once the track already has known text.
                track['last_ocr_frame'] = frame_idx
        return track['text']  # return the best text seen so far for this plate

    def end_frame(self) -> None:
        """Age every track and remove tracks that have been missing too long."""
        for track in self._tracks:
            track['age'] += 1   # increment every frame this track wasn't matched
        # Keep only tracks that haven't exceeded the time-to-live limit.
        self._tracks = [t for t in self._tracks if t['age'] <= self._ttl]

# Detection pipeline
class DetectionPipeline:
    """Runs vehicle, seatbelt, plate, and OCR inference on one frame."""

    def __init__(self, car_model, seatbelt_model, lp_model, ocr_reader,
                 vehicle_classes, conf, iou, device, imgsz, sub_imgsz,
                 ocr_every, max_vehicles=8, profile=False):
        """Store model objects, inference settings, and per-stream state."""
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
        self.half = (device == 'cuda') # use half precision on GPU for speed
        self.ocr_every = max(1, ocr_every)          # enforce at least 1 frame between OCR runs
        self.max_vehicles = max(1, max_vehicles)  # cap crops/OCR per frame for speed
        self.profile = profile                       # if True, print timing breakdown each frame

        self.car_names = car_model.names             # class index -> name mapping from the car model
        self._tracker = PlateTracker()               # tracks plate boxes across frames
        self._frame_idx = 0                          # counts how many frames have been inferred
        # Used to keep sub-model batch size constant even with fewer vehicles.
        self._pad_tile = np.zeros((sub_imgsz, sub_imgsz, 3), dtype=np.uint8)  # blank image used as a batch padding slot

    def _predict(self, model, source, imgsz=None):
        """Run one Ultralytics prediction call with shared thresholds/settings."""
        return model.predict(
            source, conf=self.conf, iou=self.iou, imgsz=imgsz or self.imgsz,
            device=self.device, half=self.half, verbose=False,
        ) # half: use half precision on GPU for speed

    def warmup(self) -> None: # First inference is slow due to CUDA context init and cuDNN autotuning, so run dummy inference to warm up.
        """Run dummy inference so CUDA/cuDNN prepares the shapes used later."""
        big = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)   # blank frame at full detection size
        batch = [self._pad_tile] * self.max_vehicles  # full padded batch shape
        self._predict(self.car_model, big, self.imgsz)                 # warm car model
        for model in (self.seatbelt_model, self.lp_model):
            if model is not None:
                self._predict(model, batch, self.sub_imgsz)            # warm seatbelt and plate models

        if self.ocr_reader is not None:
            # Warm OCR on blank and plate-like crops before the first real plate.
            try:
                blank = np.zeros((48, 192, 3), dtype=np.uint8)         # plain black image
                plate_like = np.full((64, 240, 3), 235, dtype=np.uint8)  # light grey plate-shaped image
                cv2.rectangle(plate_like, (4, 4), (235, 59), (0, 0, 0), 2)   # border around the fake plate
                cv2.putText(
                    plate_like,
                    "ABC 123",     # dummy text to give OCR something realistic to process
                    (18, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                for ocr_warm in (blank, plate_like):
                    self.ocr_reader.readtext(ocr_warm, detail=0, paragraph=False)  # run OCR to trigger model load
            except Exception:
                pass  # warmup failures are non-critical; real inference will still work

    def reset_temporal_state(self) -> None: # reset after warmup 
        """Reset per-stream tracking state before processing a new file."""
        self._tracker = PlateTracker()
        self._frame_idx = 0

    @staticmethod
    def _clip_crop(frame, x1, y1, x2, y2):
        """Clip a crop to frame bounds and return the crop plus its offset."""
        h, w = frame.shape[:2]                          # frame height and width
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))    # clamp top-left to frame edge
        cx2, cy2 = min(w, int(x2)), min(h, int(y2))    # clamp bottom-right to frame edge
        if cx2 <= cx1 or cy2 <= cy1:                    # box is zero-size or outside frame
            return None, None
        return frame[cy1:cy2, cx1:cx2], (cx1, cy1)     # return crop array and its top-left position

    @staticmethod
    def _normalize_seatbelt_label(label: str) -> Optional[str]:
        """Map raw model labels into the two labels used by the renderer."""
        key = label.strip().lower()    # normalize to lowercase for consistent lookup
        if key in SEATBELT_LABEL_MAP:
            return SEATBELT_LABEL_MAP[key]  # return the canonical label ('seatbelt' or 'no-seatbelt')
        return None   # unrecognized label; caller will skip it

    def _best_label(self, result, model, label_normalizer=None) -> Optional[Tuple[str, float]]:
        """Return the highest-confidence usable label from one model result."""
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None   # model found nothing in this crop

        best_label = None
        best_conf = -1.0
        for i in range(len(boxes)):
            raw = model.names.get(int(boxes.cls[i].item()), "unknown")  # get class name by index
            label = label_normalizer(raw) if label_normalizer else raw   # optionally remap to canonical name
            if label is None:
                continue   # skip labels that the normalizer says to ignore
            conf = float(boxes.conf[i].item())   # confidence score for this detection
            if conf > best_conf:                  # keep only the highest-confidence label
                best_conf = conf
                best_label = label

        if best_label is None:
            return None   # all detected labels were filtered out
        return best_label, best_conf

    def _ocr(self, frame, box: Box) -> str:
        """Run EasyOCR on a plate crop and return joined text, or empty text."""
        if self.ocr_reader is None:
            return ""   # OCR is disabled
        crop, _ = self._clip_crop(frame, *box)   # extract the plate region from the full frame
        if crop is None:
            return ""   # box was outside frame bounds
        # Upscale tiny plate crops because OCR is weak below about 32 px tall.
        ch = crop.shape[0]   # crop height in pixels
        if ch < 32:
            scale = 32 / ch  # scale factor to bring the height up to at least 32 px
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        try:
            tokens = self.ocr_reader.readtext(crop, detail=0, paragraph=False)  # returns list of text strings
            return ' '.join(tokens).strip()   # join multiple tokens into one plate string
        except Exception:
            return ""   # OCR failed; return empty so the tracker keeps the last known text

    def infer(self, frame, force_ocr: bool = False) -> List[Detection]:
        """Run the full frame pipeline and return vehicle detections."""
        self._frame_idx += 1          # advance the frame counter used by the OCR cadence
        t = time.perf_counter         # high-resolution timer function (stored to avoid repeated lookups)
        t_car = t_sb = t_lp = t_ocr = 0.0   # per-stage timing accumulators (milliseconds)
        ocr_runs = 0                  # how many times OCR actually ran this frame

        # Stage 1: detect vehicles on the full frame.
        t0 = t()
        car_result = self._predict(self.car_model, frame, self.imgsz)[0]
        t_car = t() - t0

        detections: List[Detection] = []            # results returned to the caller
        crops: List[np.ndarray] = []                 # vehicle image crops for sub-models
        crop_meta: List[Tuple[int, int, int]] = []   # (detection_idx, x_off, y_off)

        if car_result.boxes is not None:
            # Sort largest vehicles first, then cap sub-model work for busy frames.
            boxes_sorted = sorted(
                car_result.boxes,
                key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]),  # box area
                reverse=True,
            )
            for box in boxes_sorted:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # absolute pixel coords
                cls = int(box.cls.item())
                # Support both dict and list model.names formats.
                name = self.car_names.get(cls, "") if isinstance(self.car_names, dict) \
                    else (self.car_names[cls] if cls < len(self.car_names) else "")

                det = Detection(box=(x1, y1, x2, y2), conf=float(box.conf.item()),
                                cls=cls, name=name)
                detections.append(det)

                # Only crop target vehicle classes, then send crops to sub-models.
                if name.lower() in self.vehicle_classes and len(crops) < self.max_vehicles:
                    crop, offset = self._clip_crop(frame, x1, y1, x2, y2)
                    if crop is not None:
                        crops.append(crop)              # raw pixel crop
                        crop_meta.append((len(detections) - 1, offset[0], offset[1]))  # store detection index and top-left offset

        # Pad crops so sub-models always receive the same batch shape.
        n_real = len(crops)   # number of actual vehicle crops this frame
        # Fill remaining batch slots with black tiles; [:n_real] later discards padding results.
        batch = (crops + [self._pad_tile] * (self.max_vehicles - n_real)) if n_real else None

        # Stage 2: run seatbelt detection on the fixed-size crop batch.
        if batch is not None and self.seatbelt_model is not None:
            t0 = t()
            results = self._predict(self.seatbelt_model, batch, self.sub_imgsz)[:n_real]  # ignore padding results
            for (idx, _, _), result in zip(crop_meta, results):
                # Attach the best seatbelt label to the corresponding Detection object.
                detections[idx].seatbelt = self._best_label(
                    result,
                    self.seatbelt_model,
                    label_normalizer=self._normalize_seatbelt_label,
                )
            t_sb = t() - t0   # record seatbelt stage time

        # Stage 3: detect license plates on the same crop batch.
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
                    # Convert crop coordinates back into full-frame coordinates.
                    full_box: Box = (px1 + ox, py1 + oy, px2 + ox, py2 + oy)
                    n_plates += 1
                    track = self._tracker._find_match(full_box)  # look for an existing track
                    run_ocr = force_ocr or self._tracker._should_ocr(
                        track, self._frame_idx, self.ocr_every
                    )
                    t0 = t()
                    new_text = self._ocr(frame, full_box) if run_ocr else ""  # skip OCR when cadence hasn't elapsed
                    t_ocr += t() - t0
                    if run_ocr:
                        ocr_runs += 1
                    # Persist text in the tracker so it survives frames where OCR doesn't run.
                    text = self._tracker.update(
                        full_box, track, new_text, self._frame_idx, run_ocr
                    )
                    detections[idx].plates.append(Plate(box=full_box, text=text))

        self._tracker.end_frame()   # age all tracks and remove stale ones

        if self.profile:
            total = (t_car + t_sb + t_lp + t_ocr) * 1000   # total ms for this frame
            print(f"[infer #{self._frame_idx}] vehicles={len(crops)} plates={n_plates} "
                f"ocr_runs={ocr_runs} | car={t_car*1000:.0f} sb={t_sb*1000:.0f} "
                f"lp={t_lp*1000:.0f} ocr={t_ocr*1000:.0f} total={total:.0f} ms", flush=True)

        return detections

# Rendering
def draw_detections(frame, detections: List[Detection],
                    text_renderer: ArabicTextRenderer) -> np.ndarray:
    """Draw detection boxes and labels on a frame."""
    plate_labels: List[Tuple[str, int, int]] = []   # collected plate texts to draw via PIL at the end

    for det in detections:
        x1, y1, x2, y2 = map(int, det.box)   # convert float coords to integer pixels for drawing
        name = det.name or "Vehicle"           # fall back to 'Vehicle' if the class name is empty

        # Pick vehicle color from seatbelt status.
        if det.seatbelt is not None:
            sb_label, sb_conf = det.seatbelt
            color = SEATBELT_COLORS.get(sb_label, NO_DETECTION_COLOR)  # green/red based on belt result
        else:
            sb_label, sb_conf = None, None
            color = NO_DETECTION_COLOR   # orange when seatbelt model returned nothing

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)   # draw vehicle bounding box

        # Draw car label above the vehicle box.
        car_text = f"{name} {round(det.conf, 1):.1f}"   # e.g. "car 0.9"
        (tw, tht), base = cv2.getTextSize(car_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)  # measure text size
        top = max(0, y1 - tht - 8)   # position label above the box, clamped to frame top
        cv2.rectangle(frame, (x1, top), (x1 + tw, top + tht + base), color, cv2.FILLED)  # label background
        cv2.putText(frame, car_text, (x1, top + tht),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)  # white text on colored background

        # Draw seatbelt label at the bottom-left of the vehicle box.
        if sb_label is not None:
            sb_text = f"{sb_label} {round(sb_conf, 1):.1f}"   # e.g. "no-seatbelt 0.8"
            (sw, sh), _ = cv2.getTextSize(sb_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            sb_top = max(0, y2 - sh - 8)   # position near the bottom of the vehicle box
            cv2.rectangle(frame, (x1, sb_top), (x1 + sw, sb_top + sh + 4), color, cv2.FILLED)
            cv2.putText(frame, sb_text, (x1, sb_top + sh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw plate boxes now and queue OCR text for one PIL pass later.
        for plate in det.plates:
            px1, py1, px2, py2 = map(int, plate.box)
            cv2.rectangle(frame, (px1, py1), (px2, py2), LP_BOX_COLOR, 2)  # yellow plate box
            if plate.text:
                plate_labels.append((plate.text, px1, py2 + 2))   # queue text just below the plate box

    # Render all plate text labels in one PIL pass (handles Arabic correctly).
    return text_renderer.draw_labels(frame, plate_labels)

# Display helpers
def get_screen_resolution(default=(1280, 720)) -> Tuple[int, int]:
    """Return screen size for fullscreen layout, or a default fallback."""
    try:
        import tkinter as tk
        root = tk.Tk()       # create a hidden Tk window just to read the display dimensions
        root.withdraw()      # hide the window immediately so it doesn't appear on screen
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()       # clean up the Tk instance
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass   # tkinter may not be available in headless environments
    return default   # fall back to 1280×720 if screen size can't be determined

# Threaded video application
class VideoApp:
    """Runs video display in the main thread and inference in a worker thread."""

    # Keep the window title ASCII so OpenCV uses the same name on Windows.
    WINDOW_NAME = 'Vehicle - Seatbelt - Plate Detection'

    def __init__(self, pipeline: DetectionPipeline, input_path, output_path,
                 show_live, show_original, save_output, display_size,
                 fullscreen, realtime, frame_skip, text_renderer):
        """Store playback options and initialize shared thread state."""
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

        # Shared inference state — read/written by both threads, protected by _lock.
        self._detections: List[Detection] = []          # latest detections from the worker
        self._lock = threading.Lock()                    # guards _detections and _infer_count
        self._infer_count = 0                            # incremented each time worker finishes a frame
        self._infer_queue: queue.Queue = queue.Queue(maxsize=1)  # one slot drops stale frames
        self._stop = threading.Event()                   # set to True to tell the worker thread to exit

        # Playback state used only by the main thread.
        self._frame_count = 0        # total frames read from the video so far
        self._start_time = 0.0       # wall-clock time when playback started (for FPS calculation)
        self._last_frame_time = 0.0  # time the last frame was displayed (used for FPS pacing)
        self._paused = False         # whether the user has paused playback

    def _prepare_inference(self) -> None:
        """Initialize CUDA when needed and warm all models before playback."""
        if self.pipeline.device == 'cuda':
            torch.cuda.init()                         # explicitly initialize the CUDA context in this thread
            torch.backends.cudnn.benchmark = False    # disable autotuning so shapes don't cause re-searches
        print("Warming up models… ", end="", flush=True)
        self.pipeline.warmup()    # run dummy inference through all models
        print("done.")

    def _prime_video_start(self) -> None:
        """Run one real frame before opening the playback window."""
        cap = cv2.VideoCapture(self.input_path)   # open the video just to grab the first frame
        if not cap.isOpened():
            return   # can't open video; skip priming
        try:
            ok, frame = cap.read()    # read only the very first frame
            if ok:
                self.pipeline.infer(frame, force_ocr=True)   # run a real inference so all internals are warm
        finally:
            cap.release()                              # close the temporary capture
            self.pipeline.reset_temporal_state()       # clear the plate tracker so real playback starts fresh

    def _inference_worker(self, ready_event: threading.Event):
        """Consume fresh frames from the queue and publish detections."""
        # Initialize CUDA in this worker thread and warm models before playback.
        self._prepare_inference()
        self._prime_video_start()   # run one real frame so the models are fully ready

        # Signal the main thread that warmup is finished and playback can start.
        ready_event.set()

        while not self._stop.is_set():   # keep running until the main thread signals us to stop
            try:
                frame = self._infer_queue.get(timeout=0.05)  # wait up to 50 ms for a new frame
            except queue.Empty:
                continue   # no frame yet; check the stop flag and try again
            try:
                detections = self.pipeline.infer(frame)   # run the full detection pipeline
            except Exception as exc:
                import traceback
                print(f"\n[ERROR] Inference thread: {exc}")
                traceback.print_exc()
                continue   # log the error and keep running
            with self._lock:
                self._detections = detections   # publish new results for the main thread
                self._infer_count += 1          # let the main thread know a new result is available

    def _flush_inference(self):
        """Drop any queued frame and stale results (after seek/restart)."""
        try:
            self._infer_queue.get_nowait()   # discard any frame that is waiting in the queue
        except queue.Empty:
            pass   # nothing was queued; that's fine
        with self._lock:
            self._detections = []   # clear old detections so stale boxes don't linger on screen

    def _compose(self, original, annotated, screen):
        """Resize and combine original/annotated views for display."""
        if self.fullscreen:
            sw, sh = screen   # actual monitor resolution
            # Split the screen in half when showing both views side by side.
            pw = max(1, sw // 2) if self.show_original else max(1, sw)
            ph = max(1, sh)
        else:
            pw, ph = self.display_w, self.display_h   # use the user-specified window size
        ann = cv2.resize(annotated, (pw, ph))   # resize annotated frame to the display panel size
        if not self.show_original:
            return ann   # single-panel mode: just return the annotated view
        # Side-by-side mode: put the original on the left, annotated on the right.
        return cv2.hconcat([cv2.resize(original, (pw, ph)), ann])

    def _run_image(self):
        """Run the pipeline once for a still image input."""
        frame = cv2.imread(self.input_path)   # load the image from disk
        if frame is None:
            raise ValueError(f"Error opening image: {self.input_path}")

        self._prepare_inference()   # warm up models (same as video mode)
        detections = self.pipeline.infer(frame, force_ocr=True)  # run detection with OCR forced on
        annotated = draw_detections(frame.copy(), detections, self.text_renderer)  # draw boxes on a copy

        if self.save_output:
            cv2.imwrite(self.output_path, annotated)   # save the annotated image to disk
            print(f"Done. Output saved to: {self.output_path}")

        if self.show_live:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)   # resizable window
            if self.fullscreen:
                cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            screen = get_screen_resolution()
            disp = self._compose(frame, annotated, screen)   # build the display frame
            while True:
                cv2.imshow(self.WINDOW_NAME, disp)
                key = cv2.waitKey(50) & 0xFF   # wait 50 ms for a key press
                if key in (ord('q'), 27, ord(' ')):   # q, Escape, or Space to close
                    break
            cv2.destroyAllWindows()

    def _fps(self):
        """Return display FPS and inference FPS based on elapsed time."""
        elapsed = time.time() - self._start_time   # seconds since playback started
        if elapsed <= 0:
            return 0.0, 0.0   # avoid division by zero at startup
        with self._lock:
            ic = self._infer_count   # snapshot the counter under the lock
        return self._frame_count / elapsed, ic / elapsed   # (display FPS, inference FPS)

    def _draw_hud(self, disp, status=""):
        """Draw playback and inference stats on the display frame."""
        stream_fps, infer_fps = self._fps()
        # Draw a semi-transparent black background panel for the HUD text.
        cv2.rectangle(disp, (10, 10), (700, 110 if status else 100), (0, 0, 0), cv2.FILLED)
        cv2.putText(disp, f"Stream FPS: {stream_fps:.2f}", (18, 35),   # display thread frame rate
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp, f"Infer FPS:  {infer_fps:.2f}", (18, 65),    # inference thread frame rate
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(disp, "space pause  a/d seek  r restart  q quit", (18, 92),   # keyboard shortcuts hint
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if status:   # show optional status text (e.g. "PAUSED") in the top-right
            cv2.putText(disp, status, (520, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    def _handle_key(self, cap, key, seek_step) -> bool:
        """Handle playback hotkeys and return False when playback should stop."""
        if key == ord('q'):            # Q: quit
            return False
        if key == ord(' '):            # Space: toggle pause
            self._paused = not self._paused
            if not self._paused:       # resuming: reset the frame timer to avoid a speed burst
                self._last_frame_time = time.time()
        elif key == ord('r'):          # R: restart from the beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # seek video back to frame 0
            self._frame_count = 0
            self._start_time = self._last_frame_time = time.time()  # reset FPS counters
            self._flush_inference()    # discard stale queue and detections
            self._paused = False
        elif key == ord('a'):          # A: seek backward by ~2 seconds
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))   # current frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - seek_step))   # don't seek before frame 0
            self._flush_inference()
        elif key == ord('d'):          # D: seek forward by ~2 seconds
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos + seek_step)
            self._flush_inference()
        return True   # keep playing

    def run(self):
        """Run image mode or video playback mode from the selected input."""
        # Check the file extension to decide between image and video mode.
        if os.path.splitext(self.input_path)[1].lower() in _IMAGE_EXTS:
            return self._run_image()

        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video source: {self.input_path}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimal buffer keeps frames fresh and reduces latency

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # original video width in pixels
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # original video height in pixels
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fall back to 30 if FPS metadata is missing
        seek_step = max(1, int(fps * 2))         # 'a'/'d' keys jump ±2 seconds
        target_interval = 1.0 / fps              # seconds between display frames to match source FPS

        out = None
        if self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264-compatible container
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (fw, fh))  # output file matches source resolution

        # Start worker first so playback opens after warmup is complete.
        ready_event = threading.Event()
        worker = threading.Thread(target=self._inference_worker,
                                  args=(ready_event,), daemon=True)  # daemon=True: thread dies with the main process
        worker.start()
        ready_event.wait()   # blocks until warmup done; then we open the window

        if self.show_live:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)   # resizable display window
            if self.fullscreen:
                cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
        screen = get_screen_resolution()   # needed by _compose to calculate panel widths

        self._start_time = self._last_frame_time = time.time()   # start FPS clock
        annotated = None       # last drawn overlay
        last_frame = None      # last decoded raw frame
        last_infer_seen = -1   # _infer_count value when we last redrew

        # Pace display to source FPS so drawing does not starve inference.
        display_interval = 1.0 / fps if fps > 0 else 1.0 / 30.0

        try:
            while True:
                # While paused, keep refining the same frame and redraw on new results.
                if self._paused:
                    if last_frame is not None:
                        try:
                            self._infer_queue.put_nowait(last_frame.copy())   # keep sending the same frame for refinement
                        except queue.Full:
                            pass
                        with self._lock:
                            ic = self._infer_count
                            detections = self._detections
                        if ic != last_infer_seen:   # redraw only for a new inference result
                            annotated = draw_detections(last_frame.copy(), detections,
                                                        self.text_renderer)
                            last_infer_seen = ic
                        if self.show_live and annotated is not None:
                            disp = self._compose(last_frame, annotated, screen)
                            self._draw_hud(disp, status="PAUSED")   # show PAUSED badge in HUD
                            cv2.imshow(self.WINDOW_NAME, disp)
                    # Wait 50 ms for a key; handle it and loop back to the top.
                    if not self._handle_key(cap, cv2.waitKey(50) & 0xFF, seek_step):
                        break
                    continue

                ret, frame = cap.read()
                if not ret:  # end of video
                    break
                self._frame_count += 1
                last_frame = frame   # keep a copy so the paused branch can keep sending it

                # Queue the newest frame; a full queue means the worker is busy.
                if self._frame_count % self.frame_skip == 0:   # obey the frame-skip setting
                    try:
                        self._infer_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass  # drop frame; worker will pick up the next one

                # Redraw overlays only when a new inference result is ready.
                with self._lock:
                    ic = self._infer_count
                    detections = self._detections
                if ic != last_infer_seen:   # new detections available
                    annotated = draw_detections(frame.copy(), detections, self.text_renderer)
                    last_infer_seen = ic
                elif annotated is None:
                    annotated = frame.copy()   # blank until first inference

                if out is not None:
                    out.write(annotated)  # write annotated frame to output video

                if self.show_live:
                    disp = self._compose(frame, annotated, screen)
                    self._draw_hud(disp)   # overlay FPS and controls
                    cv2.imshow(self.WINDOW_NAME, disp)
                    if not self._handle_key(cap, cv2.waitKey(1) & 0xFF, seek_step):  # 1 ms wait keeps display responsive
                        break

                # Sleep until the next source-FPS display slot.
                target = self._last_frame_time + display_interval   # when we should show the next frame
                now = time.time()
                if now < target:
                    time.sleep(target - now)     # sleep the remaining time in this display slot
                    self._last_frame_time = target
                else:
                    self._last_frame_time = now  # already late; no sleep needed
        finally:
            # Always clean up resources, even if an exception occurred.
            self._stop.set()            # signal the inference worker to exit
            worker.join(timeout=3.0)    # wait up to 3 s for the thread to finish
            cap.release()               # release the video capture
            if out is not None:
                out.release()           # flush and close the output video file
            if self.show_live:
                cv2.destroyAllWindows()

        stream_fps, infer_fps = self._fps()
        print(f"Stream FPS:    {stream_fps:.2f}")
        print(f"Inference FPS: {infer_fps:.2f}")
        if self.save_output:
            print(f"Done. Output saved to: {self.output_path}")

# Entry point
def parse_args():
    """Parse command-line options for models, input/output, and display."""
    p = argparse.ArgumentParser(
        description='Vehicle detection → seatbelt + Egyptian license plate (OCR)')
    p.add_argument('--input', type=str, required=True)
    p.add_argument('--output', type=str)
    p.add_argument('--car-weights', type=str, default='yolo11n.pt',
                   help='Car detector. yolo11n.pt (COCO) or CarDetectionModel.pt (KITTI).')
    p.add_argument('--seatbelt-weights', type=str, default='SeatBeltModel2.pt')
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
    p.add_argument('--conf-threshold', type=float, default=0.25, help='Detection confidence threshold.') 
    p.add_argument('--iou-threshold', type=float, default=0.45, help='Detection Intersection over Union threshold.')
    p.add_argument('--show-live', action='store_true', help='Show live video display with detections overlaid.')
    p.add_argument('--show-original', action='store_true', help='Show original video display.')
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
    p.add_argument('--no-save', action='store_true', help='Do not save annotated video output.')
    p.add_argument('--display-width', type=int, default=640)
    p.add_argument('--display-height', type=int, default=360)
    p.add_argument('--fullscreen', action='store_true', help='Start in fullscreen mode (press F to toggle during playback).')
    p.add_argument('--realtime', action='store_true',
                   help='Deprecated — display is now always capped to video FPS. '
                        'Kept for backward compatibility.')
    return p.parse_args()


def resolve_device(requested: str) -> str:
    """Resolve auto/cpu/cuda into the device actually used by the models."""
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
    """Load models, build the pipeline, and run the video/image app."""
    args = parse_args()
    device = resolve_device(args.device)

    # The worker disables cuDNN benchmark and keeps sub-model batches fixed.
    # That avoids repeated CUDA algorithm searches when input shapes change.

    # Path helpers: bare names resolve under the project's standard folders.
    def resolve_input(name):
        """Resolve a bare input name under the validation video folder."""
        return name if os.path.dirname(name) else os.path.join('Detection Video Validation', name)

    def resolve_model(name):
        """Resolve a bare model filename under the Models folder."""
        return name if os.path.dirname(name) else os.path.join('Models', name)

    # Load models.
    car_path = resolve_model(args.car_weights)          # full path to the car detector weights
    print(f"Loading car model:      {car_path}")
    car_model = YOLO(car_path).to(device)               # load YOLO and move to GPU or CPU

    seatbelt_model = None
    if not args.no_sb:
        sb_path = resolve_model(args.seatbelt_weights)
        print(f"Loading seatbelt model: {sb_path}")
        seatbelt_model = YOLO(sb_path).to(device)
    else:
        print("Seatbelt detection disabled (--no-sb).")

    lp_model, ocr_reader = None, None   # both start as None and are set only if enabled
    if not args.no_lp:
        lp_path = resolve_model(args.lp_weights)
        print(f"Loading LP model:       {lp_path}")
        lp_model = YOLO(lp_path).to(device)

        if not args.no_ocr:
            if EASYOCR_AVAILABLE:
                # Prefer GPU OCR when available, then fall back to CPU if needed.
                if args.ocr_device == 'auto':
                    use_gpu_ocr = (device == 'cuda')     # match OCR device to detection device
                else:
                    use_gpu_ocr = (args.ocr_device == 'cuda')  # honour explicit user choice
                print(f"Initialising EasyOCR (Arabic+English, {'GPU' if use_gpu_ocr else 'CPU'}) ")
                try:
                    ocr_reader = easyocr.Reader(['ar', 'en'], gpu=use_gpu_ocr, verbose=False) # verbose False to suppress EasyOCR's own printouts
                except Exception as exc:
                    print(f"[WARN] EasyOCR GPU init failed ({exc}); falling back to CPU.")
                    ocr_reader = easyocr.Reader(['ar', 'en'], gpu=False, verbose=False)  # retry on CPU
                print("EasyOCR ready.")
            else:
                print("[WARN] EasyOCR not available — OCR skipped.")
    else:
        print("License-plate detection disabled (--no-lp).")

    # Parse comma-separated class names into a lowercase set for fast lookup.
    vehicle_classes = {c.strip().lower() for c in args.vehicle_classes.split(',')}
    print(f"Seatbelt/LP on classes (case-insensitive): {vehicle_classes}")

    # Resolve input/output paths.
    input_path = resolve_input(args.input)
    if args.output:
        output_path = resolve_input(args.output)
    else:
        # Auto-generate output path by appending '_annotated' to the input name.
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_annotated{ext}"

    # Build pipeline and run the app.
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
        save_output=(not args.no_save),                                      # save unless --no-save is set
        display_size=(max(1, args.display_width), max(1, args.display_height)),  # ensure size is at least 1x1
        fullscreen=args.fullscreen, realtime=args.realtime,
        frame_skip=args.frame_skip, text_renderer=ArabicTextRenderer(),      # shared renderer for Arabic text
    )
    app.run()   # start the video/image processing loop


if __name__ == "__main__":
    main()
