# DriveGuard

DriveGuard is a computer vision project that detects vehicles, checks seatbelt status, detects license plates, and reads Egyptian license plate text with OCR. It can run locally on images/videos or through a small web app for uploading media and downloading the annotated result.

## What the Project Does

For each image or video frame, DriveGuard runs this pipeline:

1. Detect vehicles in the full frame.
2. Crop each detected vehicle.
3. Run seatbelt detection on the vehicle crop.
4. Run license plate detection on the vehicle crop.
5. Run OCR on detected plates and draw the recognized text on the output.

OCR is optimized so a newly seen plate is read immediately, then OCR is repeated every 5 inference frames while the last good text is tracked between reads.

## Main Files

- `detect_objects.py`  
  Local detection script. It loads the YOLO models, processes images or videos, shows live OpenCV output if requested, and can save annotated output files.

- `modal_app.py`  
  Modal/FastAPI web app. It provides a browser upload page for images and videos, processes the uploaded media, tracks progress, and returns the annotated result.

- `requirements.txt`  
  Python dependencies needed to run the project.

- `Models/`  
  Model weights used by the pipeline:
  - `yolo11n.pt`: general vehicle detector
  - `CarDetectionModel.pt`: alternate/custom car detector
  - `SeatBeltModel1.pt` and `SeatBeltModel2.pt`: seatbelt detectors
  - `LicensePlateModel.pt`: Egyptian license plate detector

- `Training/`  
  Training notebooks for the vehicle, seatbelt, and license plate models.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

EasyOCR may download its recognition models the first time it runs.

## Run Locally

Process a video from `Detection Video Validation/`:

```powershell
python detect_objects.py --input input_video.mp4
```

Save to a specific output path:

```powershell
python detect_objects.py --input input_video.mp4 --output output_video.mp4
```

Show the live detection window:

```powershell
python detect_objects.py --input input_video.mp4 --show-live
```

Show original and annotated video side by side:

```powershell
python detect_objects.py --input input_video.mp4 --show-live --show-original
```

Process an image:

```powershell
python detect_objects.py --input image.jpg
```

## Local Script Arguments

These are the arguments supported by `detect_objects.py`.

- `--input PATH`  
  Required. Image or video to process. If only a filename is provided, the script looks inside `Detection Video Validation/`.

- `--output PATH`  
  Optional output path. If not provided, the script creates a file next to the input with `_annotated` added to the name.

- `--car-weights PATH`  
  Vehicle detector weights. Default: `yolo11n.pt`. If only a filename is provided, the script looks inside `Models/`.

- `--seatbelt-weights PATH`  
  Seatbelt detector weights. Default: `SeatBeltModel2.pt`.

- `--lp-weights PATH`  
  License plate detector weights. Default: `LicensePlateModel.pt`.

- `--no-sb`  
  Disable seatbelt detection.

- `--no-lp`  
  Disable license plate detection completely.

- `--no-ocr`  
  Detect license plates but skip reading the plate text. This is faster.

- `--ocr-device auto|cpu|cuda`  
  Selects the EasyOCR device. Default: `auto`. Use `cpu` if GPU memory is limited.

- `--ocr-every NUMBER`  
  Controls how often OCR is repeated after the first read. Default: `5`. A newly seen plate is read immediately, then OCR repeats every N inference frames.

- `--vehicle-classes LIST`  
  Comma-separated vehicle classes that should run seatbelt and license plate detection. Default: `car,van,truck,bus`.

- `--conf-threshold NUMBER`  
  YOLO confidence threshold. Default: `0.25`. Higher values show fewer detections but may miss weak ones.

- `--iou-threshold NUMBER`  
  YOLO IoU threshold for filtering overlapping boxes. Default: `0.45`.

- `--show-live`  
  Open a live OpenCV window while processing.

- `--show-original`  
  When live view is enabled, show the original frame beside the annotated frame.

- `--device auto|cpu|cuda`  
  Selects the YOLO detection device. Default: `auto`. `auto` uses CUDA if available.

- `--imgsz NUMBER`  
  Image size used for full-frame vehicle detection. Default: `640`. Higher can improve detection but is slower.

- `--sub-imgsz NUMBER`  
  Image size used for seatbelt and license plate detection on vehicle crops. Default: `320`.

- `--max-vehicles NUMBER`  
  Maximum number of vehicles per frame that will run through seatbelt/license plate/OCR stages. Default: `8`.

- `--profile`  
  Print timing for each stage: vehicle detection, seatbelt, license plate, OCR, and total inference time.

- `--frame-skip NUMBER`  
  Enqueue one frame for inference every N frames. Default: `1`. Higher values can improve playback smoothness but detections update less often.

- `--no-save`  
  Do not save an output file. Useful when using only the live preview.

- `--display-width NUMBER`  
  Width of the live display window. Default: `640`.

- `--display-height NUMBER`  
  Height of the live display window. Default: `360`.

- `--fullscreen`  
  Show the live window in fullscreen mode.

- `--realtime`  
  Deprecated compatibility option. Display is already capped to the video FPS.

## Run the Web App

The web app is defined in `modal_app.py` and is intended to run on Modal.

Typical command:

```powershell
modal serve modal_app.py
```

Then open the URL printed by Modal, upload an image or video, choose which detections to enable, and download the annotated result.

The web app packages:

- `detect_objects.py`
- the `Models/` folder
- all Python dependencies needed by the API

## Performance Notes

- The local video app warms the models before playback starts.
- It also performs a hidden first-frame inference before opening playback, then restarts from the first frame. This avoids the visible cold-start freeze at the beginning of a video.
- Sub-model batches are padded to a fixed size to avoid repeated GPU shape tuning delays.
- Plate OCR runs immediately when a plate is first seen, then follows the configured OCR cadence.

## Output

The output is the original media with annotations:

- vehicle box
- seatbelt status, when enabled
- license plate box, when enabled
- OCR text under the plate, when enabled and recognized

For local runs, if `--output` is not provided, the script writes an annotated file next to the input with `_annotated` added to the filename.

## Troubleshooting

If OCR is slow on the first run, EasyOCR may still be downloading or initializing its models.

If CUDA is requested but not available, the script falls back to CPU.

If GPU memory is tight, try:

```powershell
python detect_objects.py --input input_video.mp4 --ocr-device cpu
```

If detection is slow, try lowering sub-model work:

```powershell
python detect_objects.py --input input_video.mp4 --max-vehicles 4
```
