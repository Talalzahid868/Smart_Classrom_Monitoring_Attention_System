import pandas as pd
from .video_classifier import VideoClassifier
from .datasetloader import load_dataset


class ReportGenerator:
    def __init__(self):
        self.classifier = VideoClassifier()

    def generate_report(self, dataset, output_file="report.csv"):
        report_data = []
        for sample in dataset:
            clip_id = sample["Clip_id"]
            video_path = sample["video_path"]
            ground_truth = sample["label"]
            print(f"Processing {clip_id}...")
            result = self.classifier.classify_video(video_path)
            if result is None:
                continue

            report_data.append({
                "clip_id": clip_id,
                "ground_truth": ground_truth,
                "predicted_label": result["final_label"],
                "total_frames": result["total_frames"],
                "attentive_frames": result["attentive_frames"],
                "distracted_frames": result["distracted_frames"],
                "attention_rate": result["attention_rate"]
            })

        df = pd.DataFrame(report_data)
        df.to_csv(output_file, index=False)
        print(f"Report saved to {output_file}")
        return df

dataset = load_dataset("DataSet/Attention_labels.csv","DataSet")

generator= ReportGenerator()

report=generator.generate_report(dataset[:10],"attention_report.csv")
print(report) 



