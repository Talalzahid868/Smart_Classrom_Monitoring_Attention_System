import math
import cv2
from .datasetloader import load_dataset
from .video_processor import load_video, read_frames
from .landmarks import FaceMeshDetector

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 +(p1[1] - p2[1])**2)


def calculate_ear(eye_points):
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

LEFT_EYE = [33, 160, 158, 133, 153, 144]  # these are eyes indices in mediapipe according to mediapipe documentation
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

dataset = load_dataset("DataSet/Attention_labels.csv","DataSet")

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

        cv2.putText(frame,f"EAR: {ear:.3f}",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)

    cv2.imshow("EAR Test", frame)
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()
