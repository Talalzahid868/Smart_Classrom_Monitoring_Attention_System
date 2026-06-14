# import cv2

# video_path = r"DataSet/Test/500044/5000441001/5000441001.avi"

# cap = cv2.VideoCapture(video_path)

# print("FPS:", cap.get(cv2.CAP_PROP_FPS))
# print("Frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

# cap.release()




# import os

# count = 0

# for root, dirs, files in os.walk("DataSet"):
#     for file in files:
#         if file.endswith(".avi"):
#             count += 1

# print("Total videos:", count)


import cv2


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    return cap


def get_fps(cap):
    return cap.get(cv2.CAP_PROP_FPS)


def get_total_frames(cap):
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def get_duration(cap):
    fps = get_fps(cap)
    if fps == 0:
        return 0
    return get_total_frames(cap) / fps


def read_frames(cap, skip_frames=5):
    """
    Generator that yields every nth frame.
    """

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % skip_frames == 0:
            yield frame
        frame_count += 1
    cap.release()


video_path = r"DataSet/Test/500044/5000441001/5000441001.avi"
cap = load_video(video_path)
print("FPS:", get_fps(cap))
print("Frames:", get_total_frames(cap))
print("Duration:", get_duration(cap))
processed = 0
for frame in read_frames(cap, skip_frames=5):
    processed += 1

print("Processed Frames:", processed)
