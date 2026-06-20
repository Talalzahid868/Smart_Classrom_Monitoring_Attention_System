import cv2
from datasetloader import load_dataset
from video_processor import load_video, read_frames
from landmarks import FaceMeshDetector
from EAR import calculate_ear
from head_pose import HeadPoseEstimator


class AttentionScorer:

    def __init__(
        self,
        ear_threshold=0.23,
        yaw_threshold=10,
        pitch_threshold=10
    ):

        self.ear_threshold = ear_threshold
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold

    def classify(self, ear, yaw, pitch):

        score = 0

        # Eye Status
        if ear >= self.ear_threshold:
            score += 1

        # Looking Left/Right?
        if abs(yaw) <= self.yaw_threshold:
            score += 1

        # Looking Up/Down?
        if abs(pitch) <= self.pitch_threshold:
            score += 1

        if score >=2:
            return "Attentive", score

        return "Distracted", score
    


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

dataset = load_dataset("DataSet/Attention_labels.csv","DataSet")

sample = dataset[0]
cap = load_video(sample["video_path"])
mesh = FaceMeshDetector()
pose_estimator = HeadPoseEstimator()
scorer = AttentionScorer()
for frame in read_frames(cap, skip_frames=30):
    faces = mesh.get_landmarks(frame)
    if len(faces) > 0:
        landmarks = faces[0]
        # EAR
        left_eye = [landmarks[i] for i in LEFT_EYE]
        right_eye = [landmarks[i] for i in RIGHT_EYE]
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        ear = (left_ear + right_ear) / 2

        # Head Pose
        pose = pose_estimator.estimate_pose(landmarks,frame.shape)

        if pose:
            pitch, yaw, roll = pose
            label, score = scorer.classify(ear,yaw,pitch)

            print(f"EAR:{ear:.3f} "f"Yaw:{yaw:.2f} "f"Pitch:{pitch:.2f} "f"-> {label}")

            cv2.putText(frame,label,(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Attention Monitor", frame)
    key = cv2.waitKey(0)
    if key == 27:
        break
cv2.destroyAllWindows()    