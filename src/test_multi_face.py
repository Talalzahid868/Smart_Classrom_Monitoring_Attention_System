# test_multi_face.py
from video_processor import load_video, read_frames
from landmarks import FaceMeshDetector
from face_tracker import FaceTracker
import cv2

cap = load_video("215475_tiny.mp4")
mesh = FaceMeshDetector(max_faces=10)
tracker = FaceTracker()

frame_num = 0
for frame in read_frames(cap, skip_frames=5):
    frame_num += 1
    faces = mesh.get_landmarks(frame)
    tracked = tracker.update(faces)
    print(f"Faces detected: {len(faces)} | Tracked IDs: {list(tracked.keys())}")

    # Save first 5 failed frames as images to inspect
    if len(faces) == 0 and frame_num <= 5:
        cv2.imwrite(f"debug_frame_{frame_num}.jpg", frame)

cap.release()