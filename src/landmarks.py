import cv2
import mediapipe as mp
from datasetloader import load_dataset
from video_processor import load_video, read_frames


mp_face_mesh = mp.solutions.face_mesh
class FaceMeshDetector:
    def __init__(self,
                 static_mode=False,
                 max_faces=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def get_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        all_landmarks = []
        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for lm in face_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    landmarks.append((x, y))
                all_landmarks.append(landmarks)
        return all_landmarks



dataset = load_dataset(
    "DataSet/Attention_labels.csv",
    "DataSet"
)

sample = dataset[0]
print("Testing:", sample["Clip_id"])

cap = load_video(sample["video_path"])
mesh_detector = FaceMeshDetector()
for frame in read_frames(cap, skip_frames=30):
    faces = mesh_detector.get_landmarks(frame)
    print("Faces:", len(faces))
    for face in faces:
        print("Landmarks:", len(face))
        for (x, y) in face:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Face Mesh", frame)
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()