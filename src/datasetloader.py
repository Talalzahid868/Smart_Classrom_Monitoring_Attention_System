import pandas as pd
import os

df = pd.read_csv("DataSet/Attention_labels.csv")
print(df.head())

# def find_video_path(dataset_root,clip_id):
#     for root, dirs, files in os.walk(dataset_root):
#         print("root :",root,"dirs :",dirs,"files :",files)
#         if clip_id in files:
#             return os.path.join(root,clip_id)
#     return None
def find_video_path(dataset_root):
    video_map={}
    for root, _, files in os.walk(dataset_root):
        for file in files:
            if file.endswith(".avi"):
                video_map[file]=os.path.join(root,file)
    return video_map
     
def load_dataset(csv_path,dataset_root):
    df = pd.read_csv(csv_path)
    dataset = []
    print("Indexing video directory....")
    video_map=find_video_path(dataset_root)
    print(f"Indexed {len(video_map)} files")
    for _, row in df.iterrows():
            clip_id= row["ClipID"]
            label= row["Attention"]
            video_path=video_map.get(clip_id)
            if video_path is not None:
                 dataset.append({"Clip_id": clip_id, "video_path": video_path, "label": label})
    return dataset

if __name__=="__main__":
    d=load_dataset("DataSet/Attention_labels.csv","Dataset")
    print(d[:3])    









