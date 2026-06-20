import math
from video_processor import load_video, read_frames
from landmarks import FaceMeshDetector


class FaceTracker:

    def __init__(self, distance_threshold=50):

        self.next_id = 0
        self.tracked_faces = {}
        self.distance_threshold = distance_threshold

    def get_centroid(self, landmarks):

        x_coords = [point[0] for point in landmarks]
        y_coords = [point[1] for point in landmarks]
        cx = sum(x_coords) / len(x_coords)
        cy = sum(y_coords) / len(y_coords)

        return (cx, cy)

    def euclidean_distance(self, p1, p2):

        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    def update(self, faces):

        if faces is None or len(faces) == 0:
            return self.tracked_faces

        current_ids = {}

        for landmarks in faces:

            centroid = self.get_centroid(landmarks)

            matched_id = None
            min_distance = float("inf")

            for face_id, data in self.tracked_faces.items():

                old_centroid = data["centroid"]

                distance = self.euclidean_distance(
                    centroid,
                    old_centroid
                )

                if distance < self.distance_threshold and distance < min_distance:
                    matched_id = face_id
                    min_distance = distance

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1

            current_ids[matched_id] = {
                "centroid": centroid,
                "landmarks": landmarks
            }

        self.tracked_faces = current_ids

        return current_ids
    

if __name__=="__main__":

    video_path = "215475_tiny.mp4"

    cap = load_video(video_path)

    mesh = FaceMeshDetector()
    tracker = FaceTracker()

    for frame in read_frames(cap, skip_frames=5):

        faces = mesh.get_landmarks(frame)

        tracked = tracker.update(faces)

        print(tracked)

    cap.release()