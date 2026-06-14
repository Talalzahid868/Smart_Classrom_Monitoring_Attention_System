import cv2

from video_processor import load_video, get_fps
from landmarks import FaceMeshDetector
from EAR import calculate_ear
from head_pose import HeadPoseEstimator
from attention_score import AttentionScorer
from datasetloader import load_dataset


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


class AnnotatedVideoGenerator:

    def __init__(self):

        self.mesh = FaceMeshDetector()
        self.pose_estimator = HeadPoseEstimator()
        self.scorer = AttentionScorer()

    def generate(self, input_path, output_path):

        cap = load_video(input_path)

        fps = get_fps(cap)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            faces = self.mesh.get_landmarks(frame)

            if len(faces) > 0:

                landmarks = faces[0]

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

                if pose:

                    pitch, yaw, roll = pose

                    # Attention classification
                    label, score = self.scorer.classify(
                        ear,
                        yaw,
                        pitch
                    )

                    # Draw text
                    cv2.putText(
                        frame,
                        f"EAR: {ear:.2f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Yaw: {yaw:.2f}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Pitch: {pitch:.2f}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Label: {label}",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Score: {score}",
                        (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 0),
                        2
                    )

            out.write(frame)

            print(f"Processing frame {frame_count}")

        cap.release()
        out.release()

        print("Annotated video saved successfully.")






dataset = load_dataset(
    "DataSet/Attention_labels.csv",
    "DataSet"
)

sample = dataset[0]

generator = AnnotatedVideoGenerator()

generator.generate(
    sample["video_path"],
    "output_annotated.avi"
)