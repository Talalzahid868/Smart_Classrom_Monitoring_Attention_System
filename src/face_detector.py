# =============================================================================
# face_detector.py
# Detects faces in a video frame using MediaPipe Face Detection.
# Supports frame enhancement for compressed/low-quality videos.
# =============================================================================

import cv2
import mediapipe as mp

# MediaPipe face detection module
mp_face_detection = mp.solutions.face_detection


def enhance_frame(frame):
    """
    Clean up a blurry or compressed video frame before detection.

    Step 1 - Denoise: removes blocky compression artifacts (JPEG/H.264)
    Step 2 - Sharpen: recovers edge detail lost during compression
    """
    # Remove noise using Non-Local Means Denoising
    denoised = cv2.fastNlMeansDenoisingColored(
        frame, None,
        h=6, hColor=6,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # Sharpen using unsharp masking: sharp = original*1.8 - blurred*0.8
    blurred   = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.8, blurred, -0.8, 0)

    return sharpened


class FaceDetector:
    """
    Detects all faces in a frame and returns their bounding boxes.

    Uses MediaPipe Face Detection with model_selection=1 (full-range model)
    which works up to 5 meters — suitable for classroom-distance shots.
    model_selection=0 only works up to 2 meters (selfie/close-up).
    """

    def __init__(self, confidence=0.4):
        # model_selection=1 = full-range model (up to ~5m, good for classrooms)
        # confidence=0.4    = lower threshold to catch distant/angled faces
        self.detector = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=confidence
        )

    def detect_faces(self, frame, enhance=False):
        """
        Find all faces in a frame and return their bounding boxes.

        Args:
            frame:   A BGR image (OpenCV format)
            enhance: Set True for compressed or low-quality video

        Returns:
            List of (x, y, width, height) bounding boxes
        """
        if enhance:
            frame = enhance_frame(frame)

        # MediaPipe requires RGB input
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        boxes = []

        if results.detections:
            frame_h, frame_w = frame.shape[:2]

            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                # Convert relative (0.0–1.0) coordinates to pixel values
                x  = max(0, int(bbox.xmin  * frame_w))
                y  = max(0, int(bbox.ymin  * frame_h))
                bw = int(bbox.width  * frame_w)
                bh = int(bbox.height * frame_h)

                # Skip detections that are too small (likely noise)
                if bw < 20 or bh < 20:
                    continue

                boxes.append((x, y, bw, bh))

        return boxes


# =============================================================================
# Run this file directly to test face detection on a video
# Usage:  python face_detector.py
#         python face_detector.py --video myvideo.mp4 --enhance
# =============================================================================

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Multi-student face detection tester")
    parser.add_argument("--video",      type=str,   default="215475_tiny.mp4")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--enhance",    action="store_true")
    parser.add_argument("--skip",       type=int,   default=2,
                        help="Process every Nth frame (default: every 2nd)")
    parser.add_argument("--save",       type=str,   default=None,
                        help="Save annotated video to this path (optional)")
    args = parser.parse_args()

    # A set of distinct colours — one per detected student
    COLOURS = [
        (0,255,0),   (0,128,255), (255,0,0),   (0,255,255), (255,0,255),
        (255,165,0), (0,200,100), (180,0,255),  (255,80,80), (80,80,255),
    ]

    detector = FaceDetector(confidence=args.confidence)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n  Video      : {args.video}")
    print(f"  Resolution : {vid_w}x{vid_h}  |  FPS: {fps:.1f}")
    print(f"  Frames     : {total_frames}   |  Skip every: {args.skip}")
    print(f"  Confidence : {args.confidence}  |  Enhance: {args.enhance}")
    print(f"  Press Q or ESC to quit\n")

    # Optional: save annotated video to disk
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps / args.skip, (vid_w, vid_h))

    frame_idx    = 0
    processed    = 0
    max_detected = 0
    face_counts  = []
    t_start      = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames for speed (process every Nth frame)
        if frame_idx % args.skip != 0:
            continue

        processed += 1

        # Detect faces and measure how long it took
        t0    = time.time()
        boxes = detector.detect_faces(frame, enhance=args.enhance)
        ms    = (time.time() - t0) * 1000

        # Track statistics
        n = len(boxes)
        face_counts.append(n)
        max_detected = max(max_detected, n)

        # Draw a coloured box + label for each detected face
        for i, (x, y, bw, bh) in enumerate(boxes):
            colour = COLOURS[i % len(COLOURS)]
            label  = f"Student {i + 1}"

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), colour, 2)

            # Filled label background above the box
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), colour, -1)
            cv2.putText(frame, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # HUD — frame info in the top-left corner
        hud_lines = [
            f"Frame  : {frame_idx} / {total_frames}",
            f"Faces  : {n}   (max: {max_detected})",
            f"Time   : {ms:.1f} ms",
        ]
        for i, text in enumerate(hud_lines):
            y_pos = 28 + i * 26
            cv2.rectangle(frame, (8, y_pos - 20), (260, y_pos + 6), (0, 0, 0), -1)
            cv2.putText(frame, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 1, cv2.LINE_AA)

        print(f"  [Frame {frame_idx:>5}]  faces={n}  time={ms:.1f}ms")

        if writer:
            writer.write(frame)

        cv2.imshow("Face Detection  |  Q or ESC to quit", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            print("\n  Quit by user.")
            break

    # Print final summary
    elapsed   = time.time() - t_start
    avg_faces = sum(face_counts) / len(face_counts) if face_counts else 0

    print(f"\n  --- Summary ---")
    print(f"  Frames processed  : {processed}")
    print(f"  Time elapsed      : {elapsed:.1f}s")
    print(f"  Avg faces / frame : {avg_faces:.2f}")
    print(f"  Max faces in frame: {max_detected}")

    cap.release()
    if writer:
        writer.release()
        print(f"  Saved to: {args.save}")
    cv2.destroyAllWindows()














# import cv2
# import mediapipe as mp
# from .video_processor import load_video, read_frames

# mp_face_detection = mp.solutions.face_detection


# def enhance_frame(frame):
#     denoised = cv2.fastNlMeansDenoisingColored(
#         frame, None, 6, 6, 7, 21
#     )

#     blurred = cv2.GaussianBlur(denoised, (0, 0), 2)

#     sharpened = cv2.addWeighted(
#         denoised, 1.8, blurred, -0.8, 0
#     )

#     return sharpened


# class FaceDetector:

#     def __init__(self, confidence=0.4):
#         self.detector = mp_face_detection.FaceDetection(
#             model_selection=1,
#             min_detection_confidence=confidence
#         )

#     def detect_faces(self, frame, enhance=False):

#         if enhance:
#             frame = enhance_frame(frame)

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.detector.process(rgb)

#         boxes = []

#         if not results.detections:
#             return boxes

#         h, w, _ = frame.shape

#         for detection in results.detections:
#             bbox = detection.location_data.relative_bounding_box

#             x = max(0, int(bbox.xmin * w))
#             y = max(0, int(bbox.ymin * h))
#             bw = int(bbox.width * w)
#             bh = int(bbox.height * h)

#             if bw < 20 or bh < 20:
#                 continue

#             boxes.append((x, y, bw, bh))

#         return boxes


# if __name__ == "__main__":

#     video_path = "215475_tiny.mp4"

#     cap = load_video(video_path)

#     detector = FaceDetector()

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             break

#         boxes = detector.detect_faces(
#             frame,
#             enhance=True
#         )

#         for i, (x, y, bw, bh) in enumerate(boxes):

#             cv2.rectangle(
#                 frame,
#                 (x, y),
#                 (x + bw, y + bh),
#                 (0, 255, 0),
#                 2
#             )

#             cv2.putText(
#                 frame,
#                 f"Student {i}",
#                 (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (0, 255, 0),
#                 2
#             )

#         cv2.imshow("Face Detection", frame)

#         if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp
# from datasetloader import load_dataset
# from video_processor import load_video, read_frames

# mp_face_detection = mp.solutions.face_detection
# class FaceDetector:
#     def __init__(self, confidence=0.5):
#         self.face_detection = mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=confidence)

#     def detect_faces(self, frame):
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.face_detection.process(rgb)
#         boxes = []
#         if results.detections:
#             h, w, _ = frame.shape
#             for detection in results.detections:
#                 bbox = detection.location_data.relative_bounding_box
#                 x = int(bbox.xmin * w)
#                 y = int(bbox.ymin * h)
#                 bw = int(bbox.width * w)
#                 bh = int(bbox.height * h)
#                 boxes.append((x, y, bw, bh))
#         return boxes
    
# dataset = load_dataset("DataSet/Attention_labels.csv","DataSet")

# sample = dataset[0]
# cap = load_video(sample["video_path"])
# detector = FaceDetector()

# for frame in read_frames(cap, skip_frames=30):
#     boxes = detector.detect_faces(frame)
#     print("Faces:", len(boxes))
#     for (x, y, w, h) in boxes:
#         cv2.rectangle(frame,(x, y),(x+w, y+h),(0, 255, 0),2)

#     cv2.imshow("Face Detection", frame)

#     if cv2.waitKey(0) == 27:
#         break

# cv2.destroyAllWindows()

