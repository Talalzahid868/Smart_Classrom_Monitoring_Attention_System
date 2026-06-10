import cv2

from datasetloader import load_dataset
from video_processor import load_video, read_frames
from landmarks import FaceMeshDetector
from EAR import calculate_ear


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


dataset = load_dataset(
    "DataSet/Attention_labels.csv",
    "DataSet"
)

sample = dataset[0]

cap = load_video(sample["video_path"])

mesh = FaceMeshDetector()

for frame in read_frames(cap, skip_frames=30):

    faces = mesh.get_landmarks(frame)

    if len(faces) > 0:

        landmarks = faces[0]

        left_eye = [landmarks[i] for i in LEFT_EYE]
        right_eye = [landmarks[i] for i in RIGHT_EYE]

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        ear = (left_ear + right_ear) / 2

        print("EAR:", round(ear, 3))

        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("EAR Test", frame)

    key = cv2.waitKey(0)

    if key == 27:
        break

cv2.destroyAllWindows()



