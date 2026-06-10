import cv2

from datasetloader import load_dataset
from video_processor import load_video, read_frames
from landmarks import FaceMeshDetector
from head_pose import HeadPoseEstimator


dataset = load_dataset(
    "DataSet/Attention_labels.csv",
    "DataSet"
)

sample = dataset[0]

cap = load_video(sample["video_path"])

mesh = FaceMeshDetector()
head_pose = HeadPoseEstimator()

for frame in read_frames(cap, skip_frames=30):

    faces = mesh.get_landmarks(frame)

    if len(faces) > 0:

        landmarks = faces[0]

        pose = head_pose.estimate_pose(
            landmarks,
            frame.shape
        )

        if pose:

            pitch, yaw, roll = pose

            print(
                f"Pitch: {pitch:.2f}, "
                f"Yaw: {yaw:.2f}, "
                f"Roll: {roll:.2f}"
            )

            cv2.putText(
                frame,
                f"Yaw:{yaw:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                f"Pitch:{pitch:.1f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

    cv2.imshow("Head Pose", frame)

    key = cv2.waitKey(0)

    if key == 27:
        break

cv2.destroyAllWindows()