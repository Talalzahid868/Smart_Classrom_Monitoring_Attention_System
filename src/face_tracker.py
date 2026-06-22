# =============================================================================
# face_tracker.py
# Assigns a stable ID to each student face and tracks them across frames.
# Uses centroid matching — each face's "center point" is compared frame
# to frame. If a center point is close enough to a previous one, it is
# considered the same student.
# =============================================================================

import cv2
import math
from .landmarks import FaceMeshDetector
from .face_detector import FaceDetector, enhance_frame


class FaceTracker:
    """
    Tracks multiple faces across video frames using centroid matching.

    How it works:
    - Every face has a "centroid" — the average (x, y) of all its landmarks.
    - Each frame, new centroids are compared to the previous frame's centroids.
    - If the distance is within the threshold, it's the same student → keep ID.
    - If no match is found → new student → assign a new ID.
    - If a student disappears for more than max_disappeared frames → remove them.
    """

    def __init__(self, distance_threshold=80,max_disappeared=10):
        """
        Args:
            distance_threshold: Max pixel distance to consider two centroids
                                the same person (default: 80px)
            max_disappeared:    How many frames a face can be missing before
                                its ID is deleted (default: 10 frames)
        """
        self.next_id           = 0
        self.tracked_faces     = {}   # {face_id: {"centroid": ..., "landmarks": ...}}
        self.disappeared       = {}   # {face_id: number of frames missing}
        self.distance_threshold = distance_threshold
        self.max_disappeared   = max_disappeared

    def get_centroid(self, landmarks):
        """Calculate the center point of a face from its landmarks."""
        x = sum(p[0] for p in landmarks) / len(landmarks)
        y = sum(p[1] for p in landmarks) / len(landmarks)
        return (x, y)

    def distance(self, p1, p2):
        """Euclidean distance between two (x, y) points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def update(self, faces):
        """
        Update tracked faces with new detections from the current frame.

        Args:
            faces: List of landmark arrays — one per detected face.
                   Pass [] or None if no faces were detected.

        Returns:
            Dict of {face_id: {"centroid": (x, y), "landmarks": [...]}}
        """

        # --- No faces detected this frame ---
        if not faces:
            # Increment the "missing" counter for every tracked face
            for face_id in list(self.tracked_faces.keys()):
                self.disappeared[face_id] = self.disappeared.get(face_id, 0) + 1
                # Remove faces that have been missing too long
                if self.disappeared[face_id] > self.max_disappeared:
                    del self.tracked_faces[face_id]
                    del self.disappeared[face_id]
            return self.tracked_faces

        # --- Match new detections to existing tracks ---
        current_frame = {}
        already_matched = set()   # prevents two faces claiming the same ID
        for landmarks in faces:
            centroid = self.get_centroid(landmarks)
            # Find the closest existing track to this new detection
            best_id       = None
            best_distance = float("inf")
            for face_id, data in self.tracked_faces.items():
                if face_id in already_matched:
                    continue  # this track is already taken
                dist = self.distance(centroid, data["centroid"])
                if dist < self.distance_threshold and dist < best_distance:
                    best_id       = face_id
                    best_distance = dist
            # If no existing track matched → this is a new student
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            already_matched.add(best_id)
            self.disappeared[best_id] = 0
            current_frame[best_id] = {"centroid":centroid,"landmarks":landmarks}

        # Keep tracks that weren't matched this frame but haven't timed out
        for face_id, data in self.tracked_faces.items():
            if face_id not in current_frame:
                self.disappeared[face_id] = self.disappeared.get(face_id, 0) + 1
                if self.disappeared[face_id] <= self.max_disappeared:
                    # Carry forward last known position
                    current_frame[face_id] = data
        # Clean up the disappeared registry for removed faces
        for face_id in list(self.disappeared.keys()):
            if face_id not in current_frame:
                del self.disappeared[face_id]
        self.tracked_faces = current_frame
        return current_frame


# =============================================================================
# Helper: crop a face from the frame and get Face Mesh landmarks
# (shared logic used in both face_tracker and multistd_classifier)
# =============================================================================

CROP_PADDING = 0.25   # Add 25% padding around each face crop


def get_padded_crop(frame, box):
    """
    Crop a face region from the frame with padding on all sides.
    Returns (crop, x1, y1, scale) so coordinates can be converted back.
    """
    fh, fw      = frame.shape[:2]
    x, y, bw, bh = box

    # Add padding and clamp to frame boundaries
    px = int(bw * CROP_PADDING)
    py = int(bh * CROP_PADDING)
    x1 = max(0,  x  - px)
    y1 = max(0,  y  - py)
    x2 = min(fw, x + bw + px)
    y2 = min(fh, y + bh + py)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None, x1, y1, 1.0

    # Upscale tiny crops — Face Mesh needs at least ~120px to work reliably
    ch, cw = crop.shape[:2]
    scale  = max(120 / cw, 120 / ch) if (cw < 120 or ch < 120) else 1.0

    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_LINEAR)

    return crop, x1, y1, scale


def landmarks_to_frame_coords(landmarks_crop, x1, y1, scale):
    """
    Convert Face Mesh landmarks from crop coordinates back to full-frame
    coordinates, accounting for the padding offset and upscale factor.
    """
    result = []
    for pt in landmarks_crop:
        lx = pt[0] / scale + x1
        ly = pt[1] / scale + y1
        result.append((lx, ly, pt[2]) if len(pt) == 3 else (lx, ly))
    return result


# =============================================================================
# Run this file directly to test the tracker on a video
# Usage:  python face_tracker.py
# =============================================================================

if __name__ == "__main__":

    COLOURS = [
        (0,255,0),   (0,128,255), (255,0,0),   (0,255,255), (255,0,255),
        (255,165,0), (0,200,100), (180,0,255),  (255,80,80), (80,80,255),
    ]

    video_path   = "215475_tiny.mp4"
    SKIP_FRAMES  = 5

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Stage 1: find face boxes
    detector = FaceDetector(confidence=0.4)

    # Stage 2: get landmarks from each cropped face
    mesh = FaceMeshDetector(
        max_faces=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    tracker   = FaceTracker(distance_threshold=80, max_disappeared=10)
    frame_idx = 0

    print(f"\n  Video: {video_path}  |  Total frames: {total_frames}")
    print(f"  Press Q or ESC to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % SKIP_FRAMES != 0:
            continue

        # --- Stage 1: detect face bounding boxes ---
        boxes = detector.detect_faces(frame, enhance=True)

        # --- Stage 2: get Face Mesh landmarks for each face crop ---
        all_landmarks = []
        for box in boxes:
            crop, x1, y1, scale = get_padded_crop(frame, box)
            if crop is None:
                continue

            faces = mesh.get_landmarks(crop)
            if not faces:
                continue

            # Convert crop coordinates back to full-frame coordinates
            lm = landmarks_to_frame_coords(faces[0], x1, y1, scale)
            all_landmarks.append(lm)

        # --- Update tracker with this frame's landmarks ---
        tracked = tracker.update(all_landmarks)
        ids     = list(tracked.keys())

        print(f"  [Frame {frame_idx:>5} / {total_frames}]  "
              f"boxes={len(boxes)}  landmarks={len(all_landmarks)}  "
              f"tracked IDs={ids}")

        # --- Draw boxes and labels on the frame ---
        for i, (x, y, bw, bh) in enumerate(boxes):
            face_id = ids[i] if i < len(ids) else i
            colour  = COLOURS[face_id % len(COLOURS)]

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), colour, 2)

            # Student label tag
            label = f"Student {face_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw + 6, y), colour, -1)
            cv2.putText(frame, label, (x + 3, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Small dot at the centroid
            if face_id in tracked:
                cx, cy = tracked[face_id]["centroid"]
                cv2.circle(frame, (int(cx), int(cy)), 5, colour, -1)

        # HUD overlay
        hud = [
            f"Frame    : {frame_idx} / {total_frames}",
            f"Detected : {len(boxes)} faces",
            f"Tracked  : {len(ids)} IDs  {ids}",
        ]
        for i, text in enumerate(hud):
            yp = 28 + i * 26
            cv2.rectangle(frame, (8, yp - 20), (320, yp + 6), (0, 0, 0), -1)
            cv2.putText(frame, text, (10, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 1, cv2.LINE_AA)

        cv2.imshow("Face Tracker  |  Q or ESC to quit", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            print("\n  Quit by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
























# import math
# import cv2
# from .video_processor import load_video, read_frames
# from .landmarks import FaceMeshDetector


# class FaceTracker:

#     def __init__(self, distance_threshold=80, max_disappeared=10):
#         self.next_id = 0
#         self.tracked_faces = {}
#         self.disappeared = {}

#         self.distance_threshold = distance_threshold
#         self.max_disappeared = max_disappeared

#     def get_centroid(self, landmarks):

#         x = [point[0] for point in landmarks]
#         y = [point[1] for point in landmarks]

#         return (
#             sum(x) / len(x),
#             sum(y) / len(y)
#         )

#     def distance(self, p1, p2):

#         return math.sqrt(
#             (p1[0] - p2[0]) ** 2 +
#             (p1[1] - p2[1]) ** 2
#         )

#     def update(self, faces):

#         if not faces:

#             for face_id in list(self.tracked_faces.keys()):
#                 self.disappeared[face_id] += 1

#                 if self.disappeared[face_id] > self.max_disappeared:
#                     del self.tracked_faces[face_id]
#                     del self.disappeared[face_id]

#             return self.tracked_faces

#         updated_faces = {}
#         used_ids = set()

#         for landmarks in faces:

#             centroid = self.get_centroid(landmarks)

#             matched_id = None
#             min_distance = float("inf")

#             for face_id, old_data in self.tracked_faces.items():

#                 if face_id in used_ids:
#                     continue

#                 old_centroid = old_data["centroid"]

#                 dist = self.distance(
#                     centroid,
#                     old_centroid
#                 )

#                 if dist < self.distance_threshold and dist < min_distance:
#                     matched_id = face_id
#                     min_distance = dist

#             if matched_id is None:
#                 matched_id = self.next_id
#                 self.next_id += 1

#             updated_faces[matched_id] = {
#                 "centroid": centroid,
#                 "landmarks": landmarks
#             }

#             self.disappeared[matched_id] = 0
#             used_ids.add(matched_id)

#         for face_id, old_data in self.tracked_faces.items():

#             if face_id not in updated_faces:
#                 self.disappeared[face_id] += 1

#                 if self.disappeared[face_id] <= self.max_disappeared:
#                     updated_faces[face_id] = old_data

#         self.tracked_faces = updated_faces

#         return updated_faces


# if __name__ == "__main__":

#     import cv2
#     from face_detector import FaceDetector

#     video_path = "215475_tiny.mp4"

#     cap = load_video(video_path)

#     detector = FaceDetector()
#     mesh = FaceMeshDetector(max_faces=1)
#     tracker = FaceTracker()

#     frame_no = 0

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             break

#         frame_no += 1

#         # Step 1: Detect faces
#         boxes = detector.detect_faces(frame, enhance=True)

#         # Step 2: Get landmarks from each crop
#         all_faces = []

#         for (x, y, bw, bh) in boxes:
#             crop = frame[y:y+bh, x:x+bw]

#             if crop.size == 0:
#                 continue

#             faces = mesh.get_landmarks(crop)

#             if not faces:
#                 continue

#             landmarks = []

#             for point in faces[0]:
#                 landmarks.append(
#                     (point[0] + x, point[1] + y)
#                 )

#             all_faces.append(landmarks)

#         # Step 3: Track faces
#         tracked = tracker.update(all_faces)

#         ids = list(tracked.keys())

#         # Draw boxes
#         for i, (x, y, bw, bh) in enumerate(boxes):

#             face_id = ids[i] if i < len(ids) else i

#             cv2.rectangle(
#                 frame,
#                 (x, y),
#                 (x + bw, y + bh),
#                 (0, 255, 0),
#                 2
#             )

#             cv2.putText(
#                 frame,
#                 f"Student {face_id}",
#                 (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (0, 255, 0),
#                 2
#             )

#         # Draw centroid
#         for face_id, data in tracked.items():

#             cx, cy = data["centroid"]

#             cv2.circle(
#                 frame,
#                 (int(cx), int(cy)),
#                 5,
#                 (0, 0, 255),
#                 -1
#             )

#         # HUD
#         cv2.putText(
#             frame,
#             f"Frame: {frame_no}",
#             (20, 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (255, 255, 255),
#             2
#         )

#         cv2.putText(
#             frame,
#             f"Detected: {len(boxes)}",
#             (20, 60),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (255, 255, 255),
#             2
#         )

#         cv2.putText(
#             frame,
#             f"Tracked IDs: {ids}",
#             (20, 90),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (255, 255, 255),
#             2
#         )

#         cv2.imshow("Face Tracker", frame)

#         if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
#             break

#     cap.release()
#     cv2.destroyAllWindows()










# import math
# from video_processor import load_video, read_frames
# from landmarks import FaceMeshDetector


# class FaceTracker:

#     def __init__(self, distance_threshold=50):

#         self.next_id = 0
#         self.tracked_faces = {}
#         self.distance_threshold = distance_threshold

#     def get_centroid(self, landmarks):

#         x_coords = [point[0] for point in landmarks]
#         y_coords = [point[1] for point in landmarks]
#         cx = sum(x_coords) / len(x_coords)
#         cy = sum(y_coords) / len(y_coords)

#         return (cx, cy)

#     def euclidean_distance(self, p1, p2):

#         return math.sqrt(
#             (p1[0] - p2[0]) ** 2 +
#             (p1[1] - p2[1]) ** 2
#         )

#     def update(self, faces):

#         if faces is None or len(faces) == 0:
#             return self.tracked_faces

#         current_ids = {}

#         for landmarks in faces:

#             centroid = self.get_centroid(landmarks)

#             matched_id = None
#             min_distance = float("inf")

#             for face_id, data in self.tracked_faces.items():

#                 old_centroid = data["centroid"]

#                 distance = self.euclidean_distance(
#                     centroid,
#                     old_centroid
#                 )

#                 if distance < self.distance_threshold and distance < min_distance:
#                     matched_id = face_id
#                     min_distance = distance

#             if matched_id is None:
#                 matched_id = self.next_id
#                 self.next_id += 1

#             current_ids[matched_id] = {
#                 "centroid": centroid,
#                 "landmarks": landmarks
#             }

#         self.tracked_faces = current_ids

#         return current_ids
    

# if __name__=="__main__":

#     video_path = "215475_tiny.mp4"

#     cap = load_video(video_path)

#     mesh = FaceMeshDetector()
#     tracker = FaceTracker()

#     for frame in read_frames(cap, skip_frames=5):

#         faces = mesh.get_landmarks(frame)

#         tracked = tracker.update(faces)

#         print(tracked)

#     cap.release()