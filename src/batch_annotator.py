import os
from datasetloader import load_dataset
from annotated_video import AnnotatedVideoGenerator


dataset = load_dataset(
    "DataSet/Attention_labels.csv",
    "DataSet"
)

generator = AnnotatedVideoGenerator()

output_folder = "Annotated_Videos"

os.makedirs(output_folder, exist_ok=True)

for sample in dataset[:10]:   # first 10 videos

    clip_id = sample["Clip_id"]

    output_path = os.path.join(
        output_folder,
        f"annotated_{clip_id}"
    )

    print(f"Processing {clip_id}...")

    generator.generate(
        sample["video_path"],
        output_path
    )