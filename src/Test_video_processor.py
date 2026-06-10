
from video_processor import *

video_path = r"DataSet/Test/500044/5000441001/5000441001.avi"

cap = load_video(video_path)

print("FPS:", get_fps(cap))
print("Frames:", get_total_frames(cap))
print("Duration:", get_duration(cap))

processed = 0

for frame in read_frames(cap, skip_frames=5):
    processed += 1

print("Processed Frames:", processed)






