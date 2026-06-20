from video_processor import load_video, read_frames
from landmarks import FaceMeshDetector
from EAR import calculate_ear
from head_pose import HeadPoseEstimator
from attention_score import AttentionScorer
from datasetloader import load_dataset
from face_tracker import FaceTracker
import cv2


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]



class MultiStudentClassifier:
    def __init__(self):
        self.mesh = FaceMeshDetector(max_faces=15,min_detection_confidence=0.3,min_tracking_confidence=0.3 )  # Fix #1
        self.pose_estimator = HeadPoseEstimator()
        self.scorer = AttentionScorer()
        self.tracker = FaceTracker()               # Fix #2

    def classify_video(self, video_path):
        cap = load_video(video_path)
        
        # Per-student tracking: {face_id: {"attentive": 0, "distracted": 0}}
        student_records = {}
        frame_count = 0

        for frame in read_frames(cap, skip_frames=5):  # don't release inside generator
            frame_count += 1
            frame = cv2.resize(frame, None, fx=1.5, fy=1.5)
            faces = self.mesh.get_landmarks(frame)
            if not faces:
                continue

            tracked = self.tracker.update(faces)  # {face_id: centroid}
            
            # Map face_id → landmarks by matching order
            for face_id, data in tracked.items():

                landmarks = data["landmarks"]
                
                if face_id not in student_records:
                    student_records[face_id] = {"attentive": 0, "distracted": 0}

                left_eye = [landmarks[i] for i in LEFT_EYE]
                right_eye = [landmarks[i] for i in RIGHT_EYE]
                ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2

                pose = self.pose_estimator.estimate_pose(landmarks, frame.shape)
                if pose is None:
                    continue
                pitch, yaw, roll = pose

                label, score = self.scorer.classify(ear, yaw, pitch)
                student_records[face_id][
                    "attentive" if label == "Attentive" else "distracted"
                ] += 1

        cap.release()  # Fix #3 — release only here

        if not student_records:
            return None

        # Build per-student report
        report = {}
        for face_id, counts in student_records.items():
            total = counts["attentive"] + counts["distracted"]
            report[f"student_{face_id}"] = {
                "attentive_frames": counts["attentive"],
                "distracted_frames": counts["distracted"],
                "attention_rate": round((counts["attentive"] / total) * 100, 2) if total else 0
            }

        return {
            "frames_processed": frame_count,
            "students_detected": len(student_records),
            "per_student": report
        }
    

if __name__=="__main__":    
    dataset = load_dataset("DataSet/Attention_labels.csv","DataSet")

    sample = dataset[0]
    classifier = MultiStudentClassifier()
    result = classifier.classify_video(sample["video_path"])

    # classifier = MultiStudentClassifier()
    # result = classifier.classify_video("215475_tiny.mp4")

    print(result)




