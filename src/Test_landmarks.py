import cv2
from datasetloader import load_dataset
from video_processor import load_video, read_frames
from landmarks import FaceMeshDetector


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




