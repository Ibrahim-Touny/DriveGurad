import cv2
import torch
import argparse
import os
from ultralytics import YOLO
import time
import tkinter as tk

# run by running in terminal:
# python detect_objects.py --input input_2.mp4 --show-live --show-original --fullscreen --imgsz 320 --frame-skip 2 --no-save --device cpu


def draw_boxes_on_frame(frame, detections, class_names):
    """
    Draws bounding boxes and labels for detected objects on a video frame.

    Args:
        frame (numpy.ndarray): The video frame to draw annotations on.
        detections (list): The detection results from YOLO model for the frame.
        class_names (list): List of class names corresponding to class IDs.
    Returns:
        numpy.ndarray: Annotated frame with bounding boxes and labels drawn.
    """
    for det in detections:
        # YOLO detection format:
        # [x1, y1, x2, y2, confidence, class_id]
        # Where (x1, y1) is top-left corner and (x2, y2) is bottom-right.
        x1, y1, x2, y2, conf, cls = map(float, det)
        if 0 <= cls < len(class_names):
            class_label = class_names[int(cls)]
        else:
            class_label = "Unknown"
        label = f"{class_label} {conf:.2f}"
        # Drawing the bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Drawing the label background
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        top = int(y1) - label_size[1] - 8
        left = int(x1)
        if top < 0:
            top = int(y1) + label_size[1] + 8
        cv2.rectangle(frame, (left, top),
                      (left + label_size[0], top + label_size[1] + baseline), (0, 255, 0), cv2.FILLED)
        # Drawing the label text
        cv2.putText(frame, label, (left, top + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def get_screen_resolution(default_width=1280, default_height=720):
    """Return primary screen resolution, with a safe fallback when unavailable."""
    try:
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return default_width, default_height


def detect_objects_in_video(input_video_path, output_video_path, model, class_names, conf_threshold, iou_threshold,
                            show_live, show_original, device, imgsz, frame_skip, save_output,
                            display_width, display_height, fullscreen, realtime):
    """
    Performs object detection on a video using a YOLO model, draws bounding boxes and labels, 
    and saves/displays the output.

    Args:
        input_video_path (str): The path to the input video file.
        output_video_path (str): The path to save the annotated output video.
        model (YOLO): The loaded YOLO model.
        class_names (list): List of class names for labels.
        conf_threshold (float): Confidence threshold for detections.
        iou_threshold (float): IoU threshold for Non-Maximum Suppression (NMS).
        show_live (bool): Whether to display the video output in a window.
        device (str): The device used for inference (`cpu` or `cuda`).

    Returns:
        None
    """
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video source {input_video_path}")
    # Keep only the newest frame in the queue to reduce end-to-end latency.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Retrieve video properties for output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # If FPS is zero (e.g., some video files don't specify FPS), set a default
    if fps == 0:
        fps = 30

    out = None
    if save_output:
        # Define the video codec and create VideoWriter object.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    inference_count = 0
    start_time = time.time()  # Start timing for FPS calculation
    last_detections = []
    frame_skip = max(1, frame_skip)
    use_half = (device == 'cuda')
    window_name = 'YOLO Object Detection'
    paused = False
    last_frame_time = time.time()

    # Use source FPS as a base for simple seek controls.
    seek_step_frames = max(1, int(fps * 2))

    if show_live:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    screen_width, screen_height = get_screen_resolution()

    while True:
        if paused:
            if show_live:
                preview_frame = annotated_frame if 'annotated_frame' in locals() else None
                if preview_frame is not None:
                    if fullscreen:
                        if show_original:
                            panel_width = max(1, screen_width // 2)
                            panel_height = max(1, screen_height)
                        else:
                            panel_width = max(1, screen_width)
                            panel_height = max(1, screen_height)
                    else:
                        panel_width = display_width
                        panel_height = display_height

                    resized_original = cv2.resize(frame, (panel_width, panel_height))
                    resized_annotated = cv2.resize(preview_frame, (panel_width, panel_height))
                    if show_original:
                        display_frame = cv2.hconcat([resized_original, resized_annotated])
                    else:
                        display_frame = resized_annotated

                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        live_stream_fps = frame_count / elapsed
                        live_infer_fps = inference_count / elapsed
                    else:
                        live_stream_fps = 0.0
                        live_infer_fps = 0.0

                    line1 = f"Stream FPS: {live_stream_fps:.2f}"
                    line2 = f"Inference FPS: {live_infer_fps:.2f}"
                    line3 = "Controls: space play/pause, a/d seek, r restart, q quit"
                    line4 = "Status: PAUSED"
                    cv2.rectangle(display_frame, (10, 10), (760, 110), (0, 0, 0), cv2.FILLED)
                    cv2.putText(display_frame, line1, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, line2, (18, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, line3, (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(display_frame, line4, (560, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(50) & 0xFF
                if key == ord(' '):
                    paused = False
                    last_frame_time = time.time()
                elif key == ord('q'):
                    break
                elif key == ord('r'):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    inference_count = 0
                    start_time = time.time()
                    last_detections = []
                    paused = False
                    last_frame_time = time.time()
                elif key == ord('a'):
                    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    new_pos = max(0, current_pos - seek_step_frames)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    last_detections = []
                    last_frame_time = time.time()
                elif key == ord('d'):
                    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    new_pos = current_pos + seek_step_frames
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    last_detections = []
                    last_frame_time = time.time()
                continue

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        should_infer = (frame_count - 1) % frame_skip == 0

        # Run detection every N frames and reuse last detections on skipped frames.
        if should_infer:
            inference_count += 1
            results = model.predict(
                frame,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
                device=device,
                imgsz=imgsz,
                half=use_half,
            )

            # Extract detection data from the current inference.
            last_detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf.item() if box.conf is not None else 0
                        cls = box.cls.item() if box.cls is not None else -1
                        last_detections.append([x1, y1, x2, y2, conf, cls])

        annotated_frame = frame.copy()
        if last_detections:
            annotated_frame = draw_boxes_on_frame(annotated_frame, last_detections, class_names)

        if out is not None:
            out.write(annotated_frame)

        if show_live:
            if fullscreen:
                if show_original:
                    panel_width = max(1, screen_width // 2)
                    panel_height = max(1, screen_height)
                else:
                    panel_width = max(1, screen_width)
                    panel_height = max(1, screen_height)
            else:
                panel_width = display_width
                panel_height = display_height

            resized_original = cv2.resize(frame, (panel_width, panel_height))
            resized_annotated = cv2.resize(annotated_frame, (panel_width, panel_height))

            if show_original:
                display_frame = cv2.hconcat([resized_original, resized_annotated])
            else:
                display_frame = resized_annotated

            elapsed = time.time() - start_time
            if elapsed > 0:
                live_stream_fps = frame_count / elapsed
                live_infer_fps = inference_count / elapsed
            else:
                live_stream_fps = 0.0
                live_infer_fps = 0.0

            line1 = f"Stream FPS: {live_stream_fps:.2f}"
            line2 = f"Inference FPS: {live_infer_fps:.2f}"
            line3 = "Controls: space play/pause, a/d seek, r restart, q quit"
            cv2.rectangle(display_frame, (10, 10), (760, 100), (0, 0, 0), cv2.FILLED)
            cv2.putText(display_frame, line1, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, line2, (18, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, line3, (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = True
            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                inference_count = 0
                start_time = time.time()
                last_detections = []
                last_frame_time = time.time()
            elif key == ord('a'):
                current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                new_pos = max(0, current_pos - seek_step_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                last_detections = []
                last_frame_time = time.time()
            elif key == ord('d'):
                current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                new_pos = current_pos + seek_step_frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                last_detections = []
                last_frame_time = time.time()

        if realtime and fps > 0:
            target_interval = 1.0 / fps
            target_time = last_frame_time + target_interval
            now = time.time()
            if now < target_time:
                time.sleep(target_time - now)
                last_frame_time = target_time
            else:
                last_frame_time = now

    end_time = time.time()  # End timing after processing all frames
    total_time = end_time - start_time
    # Print throughput and effective inference rate.
    if total_time > 0:
        stream_fps = frame_count / total_time
        infer_fps = inference_count / total_time
        print(f"Stream FPS: {stream_fps:.2f} frames per second")
        print(f"Inference FPS: {infer_fps:.2f} inferences per second")

    cap.release()
    if out is not None:
        out.release()
    if show_live:
        cv2.destroyAllWindows()

    print(f"Object detection on video completed. Output saved to: {output_video_path}")


def main():
    parser = argparse.ArgumentParser(description='Object detection in a video using YOLO.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output', type=str, required=False,
                        help='Path to the output video file. If not specified, `_annotated` will be added to input filename.')
    parser.add_argument('--weights', type=str, default='yolov11n.pt', help='Path to the YOLO model weights. Default is `yolov11n.pt`.')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='Confidence threshold for object detection.')
    parser.add_argument('--iou-threshold', type=float, default=0.45, help='IoU threshold for Non-Maximum Suppression.')
    parser.add_argument('--show-live', action='store_true', help='If set, display the video output in a window during processing.')
    parser.add_argument('--show-original', action='store_true',
                        help='When used with --show-live, display original and annotated frames side by side.')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='Inference device. Use `auto`, `cpu`, or `cuda`.')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Inference image size. Lower values are faster on edge devices.')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Run inference every N frames (e.g. 2, 3) and reuse boxes between frames.')
    parser.add_argument('--no-save', action='store_true',
                        help='Disable output video writing to reduce I/O overhead.')
    parser.add_argument('--display-width', type=int, default=640,
                        help='Fixed width in pixels for each live preview panel.')
    parser.add_argument('--display-height', type=int, default=360,
                        help='Fixed height in pixels for each live preview panel.')
    parser.add_argument('--fullscreen', action='store_true',
                        help='Show live preview in fullscreen. With --show-original, each view takes half the screen width.')
    parser.add_argument('--realtime', action='store_true',
                        help='Pace playback to the source FPS for real-time viewing.')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA requested but not available. Falling back to CPU.')
        device = 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    model = YOLO(args.weights).to(device)
    class_names = model.names if hasattr(model, 'names') else [
        'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 
        'Cyclist', 'Tram', 'Misc', 'DontCare'
    ]

    if args.output:
        output_video_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_video_path = f"{base}_annotated{ext}"

    detect_objects_in_video(
        input_video_path=args.input,
        output_video_path=output_video_path,
        model=model,
        class_names=class_names,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        show_live=args.show_live,
        show_original=args.show_original,
        device=device,
        imgsz=args.imgsz,
        frame_skip=args.frame_skip,
        save_output=(not args.no_save),
        display_width=max(1, args.display_width),
        display_height=max(1, args.display_height),
        fullscreen=args.fullscreen,
        realtime=args.realtime
    )

if __name__ == "__main__":
    main()
