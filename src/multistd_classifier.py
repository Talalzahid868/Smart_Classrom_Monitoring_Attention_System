# =============================================================================
# multistd_classifier.py
# Classifies each student's attention level across a classroom video.
#
# Pipeline (per frame):
#   1. FaceDetector     → find bounding boxes of all faces in the wide frame
#   2. FaceMeshDetector → get 468 landmarks from each face crop
#   3. FaceTracker      → assign a stable ID to each student across frames
#   4. EAR + HeadPose   → measure eye openness and head direction
#   5. AttentionScorer  → classify as "Attentive" or "Distracted"
# =============================================================================

import cv2
from .video_processor import load_video, read_frames
from .landmarks import FaceMeshDetector
from .face_detector import FaceDetector, enhance_frame
from .face_tracker import FaceTracker, get_padded_crop, landmarks_to_frame_coords
from .EAR import calculate_ear
from .head_pose import HeadPoseEstimator
from .attention_score import AttentionScorer


# MediaPipe Face Mesh landmark indices for the left and right eye
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# One distinct colour per student ID (up to 20 students)
COLOURS = [
    (0,255,0),   (0,128,255), (255,0,0),   (0,255,255), (255,0,255),
    (255,165,0), (0,200,100), (180,0,255),  (255,80,80), (80,80,255),
    (0,180,180), (180,180,0), (100,255,100),(255,100,200),(200,100,255),
    (50,200,50), (255,200,0), (0,100,255),  (200,50,50), (50,50,200),
]


class MultiStudentClassifier:
    """
    Processes a classroom video and reports the attention level of each student.
    """

    def __init__(self, enhance=False, detection_confidence=0.4):
        """
        Args:
            enhance:              Set True for compressed/low-quality video
            detection_confidence: How confident MediaPipe must be to count
                                  a face as detected (lower = catches more
                                  distant/angled faces, 0.4 is a good default)
        """
        # Stage 1: finds all faces in the wide classroom frame
        self.detector = FaceDetector(confidence=detection_confidence)

        # Stage 2: gets detailed landmarks from each individual face crop
        self.mesh = FaceMeshDetector(
            max_faces=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

        self.pose_estimator = HeadPoseEstimator()
        self.scorer         = AttentionScorer()
        self.tracker        = FaceTracker(distance_threshold=80, max_disappeared=10)
        self.enhance        = enhance

    def _get_all_landmarks(self, frame, boxes):
        """
        For each detected face box, crop that region and extract Face Mesh
        landmarks. Converts landmarks back to full-frame coordinates.

        Returns a list of landmark arrays (one per successfully processed face).
        """
        all_landmarks = []

        for box in boxes:
            # Crop the face with padding and upscale if needed
            crop, x1, y1, scale = get_padded_crop(frame, box)
            if crop is None:
                continue

            # Run Face Mesh on the tight face crop
            faces = self.mesh.get_landmarks(crop)
            if not faces:
                continue

            # Convert crop coordinates back to full-frame coordinates
            lm = landmarks_to_frame_coords(faces[0], x1, y1, scale)
            all_landmarks.append(lm)

        return all_landmarks

    def _classify_student(self, landmarks, frame_shape):
        """
        Calculate EAR and head pose for one student, then classify attention.

        Returns "attentive" or "distracted", or None if something went wrong.
        """
        try:
            # Eye Aspect Ratio (EAR): low EAR = eyes closing = distracted
            left_eye  = [landmarks[i] for i in LEFT_EYE]
            right_eye = [landmarks[i] for i in RIGHT_EYE]
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2

            # Head pose: large yaw/pitch = looking away = distracted
            pose = self.pose_estimator.estimate_pose(landmarks, frame_shape)
            if pose is None:
                return None
            pitch, yaw, roll = pose

            label, _ = self.scorer.classify(ear, yaw, pitch)
            return "attentive" if label == "Attentive" else "distracted"

        except Exception:
            return None

    def _draw_on_frame(self, frame, boxes, tracked, student_records, frame_count):
        """
        Draw bounding boxes, student labels, running attention %, and HUD
        onto the frame in-place for live visualization.
        """
        tracked_ids = list(tracked.keys())

        for i, (x, y, bw, bh) in enumerate(boxes):
            face_id = tracked_ids[i] if i < len(tracked_ids) else i
            colour  = COLOURS[face_id % len(COLOURS)]

            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), colour, 2)

            # Calculate running attention rate for this student
            rec   = student_records.get(face_id, {})
            total = rec.get("attentive", 0) + rec.get("distracted", 0)

            if total > 0:
                rate     = rec["attentive"] / total * 100
                label    = f"{'Attentive' if rate >= 50 else 'Distracted'} {rate:.0f}%"
                text_col = (0, 220, 0) if rate >= 50 else (0, 60, 220)
            else:
                label, text_col = "Detecting...", (180, 180, 180)

            # Student ID tag above the box (filled background for readability)
            id_text = f"Student {face_id}"
            (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw + 6, y), colour, -1)
            cv2.putText(frame, id_text, (x + 3, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Attention label below the box
            cv2.putText(frame, label, (x + 3, y + bh + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_col, 1, cv2.LINE_AA)

            # Small centroid dot
            if face_id in tracked:
                cx, cy = tracked[face_id]["centroid"]
                cv2.circle(frame, (int(cx), int(cy)), 5, colour, -1)

        # HUD in the top-left corner
        hud = [
            f"Frame    : {frame_count}",
            f"Detected : {len(boxes)} faces",
            f"Tracked  : {len(tracked_ids)} IDs  {tracked_ids}",
        ]
        for i, text in enumerate(hud):
            yp = 28 + i * 26
            cv2.rectangle(frame, (8, yp - 20), (340, yp + 6), (0, 0, 0), -1)
            cv2.putText(frame, text, (10, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 1, cv2.LINE_AA)

    def classify_video(self, video_path, show_video=False, save_video=None):
        """
        Process the full video and return attention results for each student.

        Args:
            video_path : Path to the classroom .mp4 video
            show_video : If True, opens a live window with annotations
            save_video : If set, saves the annotated video to this path

        Returns:
            {
                "frames_processed":  int,
                "students_detected": int,
                "per_student": {
                    "student_0": {"attentive_frames": int,
                                  "distracted_frames": int,
                                  "attention_rate": float},
                    ...
                }
            }
        """
        cap = load_video(video_path)

        # Set up video writer if saving is requested
        writer = None
        if save_video:
            raw  = cv2.VideoCapture(video_path)
            fps  = raw.get(cv2.CAP_PROP_FPS) or 25
            vidw = int(raw.get(cv2.CAP_PROP_FRAME_WIDTH))
            vidh = int(raw.get(cv2.CAP_PROP_FRAME_HEIGHT))
            raw.release()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_video, fourcc, fps, (vidw, vidh))

        # {face_id: {"attentive": int, "distracted": int}}
        student_records = {}
        frame_count     = 0

        for frame in read_frames(cap, skip_frames=5):
            frame_count += 1

            # Optionally clean up the frame before processing
            if self.enhance:
                frame = enhance_frame(frame)

            # --- Stage 1: find all face bounding boxes ---
            boxes = self.detector.detect_faces(frame, enhance=False)

            # --- Stage 2: get Face Mesh landmarks for each face ---
            all_landmarks = self._get_all_landmarks(frame, boxes)

            # --- Update tracker: assign/maintain student IDs ---
            tracked = self.tracker.update(all_landmarks)

            # --- Classify each tracked student ---
            for face_id, data in tracked.items():
                if face_id not in student_records:
                    student_records[face_id] = {"attentive": 0, "distracted": 0}

                result = self._classify_student(data["landmarks"], frame.shape)

                if result:
                    student_records[face_id][result] += 1

            # --- Draw and show the annotated frame ---
            if show_video or writer:
                vis = frame.copy()
                self._draw_on_frame(vis, boxes, tracked, student_records, frame_count)

                if writer:
                    writer.write(vis)

                if show_video:
                    cv2.imshow("Multi-Student Classifier  |  Q or ESC to quit", vis)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        print("\n  Quit by user.")
                        break

        cap.release()

        if writer:
            writer.release()
            print(f"  Annotated video saved to: {save_video}")

        if show_video:
            cv2.destroyAllWindows()

        if not student_records:
            return None

        # Build the final per-student report
        report = {}
        for face_id, counts in student_records.items():
            total = counts["attentive"] + counts["distracted"]
            report[f"student_{face_id}"] = {
                "attentive_frames":  counts["attentive"],
                "distracted_frames": counts["distracted"],
                "attention_rate":    round(counts["attentive"] / total * 100, 2)
                                     if total else 0
            }

        return {
            "frames_processed":  frame_count,
            "students_detected": len(student_records),
            "per_student":       report
        }


# =============================================================================
# Run this file directly to process a video
# =============================================================================

if __name__ == "__main__":

    classifier = MultiStudentClassifier(
        enhance=True,               # True for compressed/DAISEE-style video
        detection_confidence=0.4    # Lower = catches more distant faces
    )

    result = classifier.classify_video(
        "215475_tiny.mp4",
        show_video=True,                      # Open live window
        save_video="annotated_output.mp4"     # Remove this line if not needed
    )

    print(result)





















# import cv2
# from .video_processor import load_video, read_frames
# from .face_detector import FaceDetector, enhance_frame
# from .landmarks import FaceMeshDetector
# from .face_tracker import FaceTracker
# from .EAR import calculate_ear
# from .head_pose import HeadPoseEstimator
# from .attention_score import AttentionScorer


# LEFT_EYE = [33, 160, 158, 133, 153, 144]
# RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# CROP_PADDING = 0.25


# def expand_box(x, y, w, h, frame_h, frame_w):

#     pad_x = int(w * CROP_PADDING)
#     pad_y = int(h * CROP_PADDING)

#     x1 = max(0, x - pad_x)
#     y1 = max(0, y - pad_y)
#     x2 = min(frame_w, x + w + pad_x)
#     y2 = min(frame_h, y + h + pad_y)

#     return x1, y1, x2, y2


# class MultiStudentClassifier:

#     def __init__(self, enhance=False):

#         self.detector = FaceDetector()
#         self.mesh = FaceMeshDetector(max_faces=1)
#         self.tracker = FaceTracker()
#         self.pose = HeadPoseEstimator()
#         self.scorer = AttentionScorer()
#         self.enhance = enhance

#     def get_landmarks(self, frame, box):

#         h, w = frame.shape[:2]
#         x, y, bw, bh = box

#         x1, y1, x2, y2 = expand_box(
#             x, y, bw, bh, h, w
#         )

#         crop = frame[y1:y2, x1:x2]

#         if crop.size == 0:
#             return None

#         faces = self.mesh.get_landmarks(crop)

#         if not faces:
#             return None

#         landmarks = []

#         for point in faces[0]:

#             lx = point[0] + x1
#             ly = point[1] + y1

#             if len(point) == 3:
#                 landmarks.append((lx, ly, point[2]))
#             else:
#                 landmarks.append((lx, ly))

#         return landmarks

#     def classify_video(self, video_path):

#         cap = load_video(video_path)

#         student_records = {}

#         for frame in read_frames(cap, skip_frames=5):

#             if self.enhance:
#                 frame = enhance_frame(frame)

#             boxes = self.detector.detect_faces(frame)

#             all_faces = []

#             for box in boxes:
#                 landmarks = self.get_landmarks(frame, box)

#                 if landmarks:
#                     all_faces.append(landmarks)

#             tracked = self.tracker.update(all_faces)

#             for face_id, data in tracked.items():

#                 landmarks = data["landmarks"]

#                 if face_id not in student_records:
#                     student_records[face_id] = {
#                         "attentive": 0,
#                         "distracted": 0
#                     }

#                 try:
#                     left_eye = [landmarks[i] for i in LEFT_EYE]
#                     right_eye = [landmarks[i] for i in RIGHT_EYE]

#                     ear = (
#                         calculate_ear(left_eye) +
#                         calculate_ear(right_eye)
#                     ) / 2

#                     pose = self.pose.estimate_pose(
#                         landmarks,
#                         frame.shape
#                     )

#                     if pose is None:
#                         continue

#                     pitch, yaw, roll = pose

#                     label, score = self.scorer.classify(
#                         ear,
#                         yaw,
#                         pitch
#                     )

#                     if label == "Attentive":
#                         student_records[face_id]["attentive"] += 1
#                     else:
#                         student_records[face_id]["distracted"] += 1

#                 except:
#                     continue

#         cap.release()

#         return student_records


# if __name__ == "__main__":

#     classifier = MultiStudentClassifier(enhance=True)

#     cap = load_video("215475_tiny.mp4")

#     frame_no = 0

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             break

#         frame_no += 1

#         if classifier.enhance:
#             frame = enhance_frame(frame)

#         # Detection
#         boxes = classifier.detector.detect_faces(frame)

#         # Landmarks
#         all_faces = []

#         for box in boxes:
#             landmarks = classifier.get_landmarks(frame, box)

#             if landmarks:
#                 all_faces.append(landmarks)

#         # Tracking
#         tracked = classifier.tracker.update(all_faces)

#         # Classification
#         for i, (face_id, data) in enumerate(tracked.items()):

#             landmarks = data["landmarks"]

#             try:
#                 left_eye = [landmarks[i] for i in LEFT_EYE]
#                 right_eye = [landmarks[i] for i in RIGHT_EYE]

#                 ear = (
#                     calculate_ear(left_eye) +
#                     calculate_ear(right_eye)
#                 ) / 2

#                 pose = classifier.pose.estimate_pose(
#                     landmarks,
#                     frame.shape
#                 )

#                 if pose is None:
#                     continue

#                 pitch, yaw, roll = pose

#                 label, score = classifier.scorer.classify(
#                     ear,
#                     yaw,
#                     pitch
#                 )

#                 x, y, bw, bh = boxes[i]

#                 # Bounding box
#                 cv2.rectangle(
#                     frame,
#                     (x, y),
#                     (x + bw, y + bh),
#                     (0, 255, 0),
#                     2
#                 )

#                 # Student ID
#                 cv2.putText(
#                     frame,
#                     f"Student {face_id}",
#                     (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     (0, 255, 0),
#                     2
#                 )

#                 # Attention label
#                 cv2.putText(
#                     frame,
#                     label,
#                     (x, y + bh + 20),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     (0, 255, 255),
#                     2
#                 )

#             except:
#                 continue

#         # Stats
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
#             f"Students Detected: {len(boxes)}",
#             (20, 60),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (255, 255, 255),
#             2
#         )

#         cv2.putText(
#             frame,
#             f"Tracked Students: {len(tracked)}",
#             (20, 90),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (255, 255, 255),
#             2
#         )

#         cv2.imshow("Multi Student Classifier", frame)

#         if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
#             break

#     cap.release()
#     cv2.destroyAllWindows()







# from video_processor import load_video, read_frames
# from landmarks import FaceMeshDetector
# from EAR import calculate_ear
# from head_pose import HeadPoseEstimator
# from attention_score import AttentionScorer
# from datasetloader import load_dataset
# from face_tracker import FaceTracker
# import cv2


# LEFT_EYE = [33, 160, 158, 133, 153, 144]
# RIGHT_EYE = [362, 385, 387, 263, 373, 380]



# class MultiStudentClassifier:
#     def __init__(self):
#         self.mesh = FaceMeshDetector(max_faces=15,min_detection_confidence=0.3,min_tracking_confidence=0.3 )  # Fix #1
#         self.pose_estimator = HeadPoseEstimator()
#         self.scorer = AttentionScorer()
#         self.tracker = FaceTracker()               # Fix #2

#     def classify_video(self, video_path):
#         cap = load_video(video_path)
        
#         # Per-student tracking: {face_id: {"attentive": 0, "distracted": 0}}
#         student_records = {}
#         frame_count = 0

#         for frame in read_frames(cap, skip_frames=5):  # don't release inside generator
#             frame_count += 1
#             frame = cv2.resize(frame, None, fx=1.5, fy=1.5)
#             faces = self.mesh.get_landmarks(frame)
#             if not faces:
#                 continue

#             tracked = self.tracker.update(faces)  # {face_id: centroid}
            
#             # Map face_id → landmarks by matching order
#             for face_id, data in tracked.items():

#                 landmarks = data["landmarks"]
                
#                 if face_id not in student_records:
#                     student_records[face_id] = {"attentive": 0, "distracted": 0}

#                 left_eye = [landmarks[i] for i in LEFT_EYE]
#                 right_eye = [landmarks[i] for i in RIGHT_EYE]
#                 ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2

#                 pose = self.pose_estimator.estimate_pose(landmarks, frame.shape)
#                 if pose is None:
#                     continue
#                 pitch, yaw, roll = pose

#                 label, score = self.scorer.classify(ear, yaw, pitch)
#                 student_records[face_id][
#                     "attentive" if label == "Attentive" else "distracted"
#                 ] += 1

#         cap.release()  # Fix #3 — release only here

#         if not student_records:
#             return None

#         # Build per-student report
#         report = {}
#         for face_id, counts in student_records.items():
#             total = counts["attentive"] + counts["distracted"]
#             report[f"student_{face_id}"] = {
#                 "attentive_frames": counts["attentive"],
#                 "distracted_frames": counts["distracted"],
#                 "attention_rate": round((counts["attentive"] / total) * 100, 2) if total else 0
#             }

#         return {
#             "frames_processed": frame_count,
#             "students_detected": len(student_records),
#             "per_student": report
#         }
    

# if __name__=="__main__":    
#     # dataset = load_dataset("DataSet/Attention_labels.csv","DataSet")

#     # sample = dataset[0]
#     # classifier = MultiStudentClassifier()
#     # result = classifier.classify_video(sample["video_path"])

#     classifier = MultiStudentClassifier()
#     result = classifier.classify_video("215475_tiny.mp4")

#     print(result)




