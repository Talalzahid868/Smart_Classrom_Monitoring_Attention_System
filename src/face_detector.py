import cv2
import mediapipe as mp
from datasetloader import load_dataset
from video_processor import load_video, read_frames

mp_face_detection = mp.solutions.face_detection
class FaceDetector:
    def __init__(self, confidence=0.5):
        self.face_detection = mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=confidence)

    def detect_faces(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)
        boxes = []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                boxes.append((x, y, bw, bh))
        return boxes
    
dataset = load_dataset("DataSet/Attention_labels.csv","DataSet")

sample = dataset[0]
cap = load_video(sample["video_path"])
detector = FaceDetector()

for frame in read_frames(cap, skip_frames=30):
    boxes = detector.detect_faces(frame)
    print("Faces:", len(boxes))
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame,(x, y),(x+w, y+h),(0, 255, 0),2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(0) == 27:
        break

cv2.destroyAllWindows()

