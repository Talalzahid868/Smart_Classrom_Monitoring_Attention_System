from datasetloader import load_dataset
from video_processor import load_video, read_frames
from face_detector import FaceDetector
import cv2

dataset = load_dataset(
    "DataSet/Attention_labels.csv",
    "DataSet"
)

sample = dataset[0]

cap = load_video(sample["video_path"])

detector = FaceDetector()

for frame in read_frames(cap, skip_frames=30):

    boxes = detector.detect_faces(frame)

    print("Faces:", len(boxes))

    for (x, y, w, h) in boxes:

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(0) == 27:
        break

cv2.destroyAllWindows()

