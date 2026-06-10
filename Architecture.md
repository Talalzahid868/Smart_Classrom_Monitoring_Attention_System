SmartClassroomAttention/
│
├── dataset/
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── labels/
│
├── src/
│   ├── video_processor.py
│   ├── face_detector.py
│   ├── face_tracker.py
│   ├── landmarks.py
│   ├── ear.py
│   ├── head_pose.py
│   ├── attention_score.py
│   ├── report_generator.py
│   └── main.py
│
├── outputs/
│   ├── reports/
│   ├── graphs/
│   └── annotated_videos/
│
├── notebooks/
│
├── requirements.txt
│
└── README.md



Video
  ↓
Frame Processing
  ↓
Face Detection
  ↓
MediaPipe Face Mesh
  ↓
EAR Calculation
  ↓
Head Pose Estimation
  ↓
Attention Features
  ↓
Attentive / Distracted
  ↓
Compare with DAiSEE Label


Recorded Classroom Video
            │
            ▼
     Video Upload
            │
            ▼
     Frame Extraction
            │
            ▼
      Face Detection
            │
            ▼
 Student Face Tracking
            │
            ▼
 MediaPipe Face Mesh
            │
            ▼
 ┌─────────────────────┐
 │ EAR Calculation     │
 │ Head Pose Estimation│
 │ Face Direction      │
 └─────────────────────┘
            │
            ▼
    Attention Scoring
            │
            ▼
Attentive / Distracted
            │
            ▼
 Statistics Generation
            │
            ▼
     Final Report