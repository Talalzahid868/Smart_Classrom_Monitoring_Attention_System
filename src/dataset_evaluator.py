from datasetloader import load_dataset
from video_classifier import VideoClassifier
from sklearn.metrics import accuracy_score, classification_report
import random


dataset = load_dataset(
    "DataSet/Attention_labels.csv",
    "DataSet"
)

random.shuffle(dataset)

classifier = VideoClassifier()

y_true = []
y_pred = []

# Test first 300 videos
for sample in dataset[:300]:

    print("Processing:", sample["Clip_id"])

    result = classifier.classify_video(
        sample["video_path"]
    )

    if result is None:
        continue

    y_true.append(sample["label"])
    y_pred.append(result["final_label"])

print("\nAccuracy:")
print(accuracy_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))