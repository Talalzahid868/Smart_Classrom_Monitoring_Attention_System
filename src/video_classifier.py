from .video_processor import load_video, read_frames
from .landmarks import FaceMeshDetector
from .EAR import calculate_ear
from .head_pose import HeadPoseEstimator
from .attention_score import AttentionScorer
from .datasetloader import load_dataset



LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


class VideoClassifier:

    def __init__(self):

        self.mesh = FaceMeshDetector()
        self.pose_estimator = HeadPoseEstimator()
        self.scorer = AttentionScorer()

    def classify_video(self, video_path):

        cap = load_video(video_path)

        attentive_count = 0
        distracted_count = 0
        total_processed = 0

        for frame in read_frames(cap, skip_frames=5):

            faces = self.mesh.get_landmarks(frame)

            if len(faces) == 0:
                continue

           # landmarks = faces[0]
            for landmarks in faces:

                # EAR
                left_eye = [landmarks[i] for i in LEFT_EYE]
                right_eye = [landmarks[i] for i in RIGHT_EYE]
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2

                # Head Pose
                pose = self.pose_estimator.estimate_pose(
                    landmarks,
                    frame.shape
                )

                if pose is None:
                    continue

                pitch, yaw, roll = pose

                # Attention Decision
                label, score = self.scorer.classify(
                    ear,
                    yaw,
                    pitch
                )

                if label == "Attentive":
                    attentive_count += 1
                else:
                    distracted_count += 1

                total_processed += 1

            print(f"EAR={ear:.3f}, "f"Yaw={yaw:.2f}, "f"Pitch={pitch:.2f}, "f"Label={label}")


        if total_processed == 0:
            return None

        attention_rate = (
            attentive_count / total_processed
        ) * 100

        final_label = (
            "Attentive"
            if attentive_count >= distracted_count
            else "Distracted"
        )

        return {
            "total_frames": total_processed,
            "attentive_frames": attentive_count,
            "distracted_frames": distracted_count,
            "attention_rate": round(attention_rate, 2),
            "final_label": final_label
        }


      
if __name__=="__main__":


    dataset = load_dataset(
        "DataSet/Attention_labels.csv",
        "DataSet"
    )

    sample = dataset[0]

    print("Video:", sample["Clip_id"])
    print("Ground Truth:", sample["label"])

    classifier = VideoClassifier()

    result = classifier.classify_video(
        sample["video_path"]
    )

    print(result)



