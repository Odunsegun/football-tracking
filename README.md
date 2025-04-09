# Football Player and Ball Tracking System

This project is a Computer Vision pipeline that processes football (soccer) match footage to extract analytics such as player tracking, ball detection, team possession, speed/distance stats, and more.
# Result
Result of the video project is placed in **output_videos**. Shows the training and how I used it

## Folder Structure
project/ 
│ 
├── input_videos/ # Place the raw football video here (e.g., Spain_vs_France_2024.mp4) 
├── output_videos/ # The output annotated video will be saved here (e.g., output_video.avi) 
├── stubs/ # Cached data such as tracked objects and camera motion (saves time) 
├── models/ # Trained YOLOv8 model weights (best.pt) 
├── team_assigner/ # Handles team classification using color clustering 
├── trackers/ # Core detection & tracking logic 
├── utils/ # Utility functions (video IO, bbox ops, math) 
  ├── bbox_utils.py 
  └── video_utils.py 
├── main.py # Main entrypoint to run the pipeline 
├── requirements.txt # Required Python packages 
└── README.md # Here


## ▶️ How to Run

1. **Setup the Environment**  
   Ensure you are in a Python environment with the necessary dependencies.

   You can install them via:

   ```bash
   pip install -r requirements.txt

2. **Prepare Input**  
   Place your football match video in the input_videos/ folder

   input_videos/Spain_vs_France_2024.mp4


3. **Run Script**  
   From the root of the project, run:

   ```bash
   python main.py

4. **Check the Output**  
   After processing, the annotated video with:

   Player tracking

   Ball pointer

   Team colors

   Speed/distance overlays

   Possession stats

   will be saved to:

   ```bash
   output_videos/output_video.avi

# Notes
The model used is a fine-tuned YOLOv8 trained on football-specific data.

Stub files are used to save processing time and avoid retraining or re-tracking.

Possession is smoothed over frames to reduce flickering.

Referees are automatically excluded from analytics based on their ID.

Ball trails and interpolations are used when the ball is not detected in some frames.
