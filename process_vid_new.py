import cv2
import mediapipe as mp
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
VIDEO_FOLDER = "videos" # Update this to your folder of .mp4 files
OUTPUT_FILE = "skeleton_dataset.pt"
GESTURE_CLASSES = ["IDLE", "SHOOT", "PASS", "MOVE"]
NUM_FRAMES = 15

# The 12 Torso and Leg joints we care about
CORE_JOINTS = [11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30]

def extract_skeletons():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    
    video_paths = list(Path(VIDEO_FOLDER).glob("*.mp4"))
    if not video_paths:
        print(f"❌ No .mp4 videos found in {VIDEO_FOLDER}")
        return

    X_data, y_data = [], []

    print(f"--- Extracting skeletons from {len(video_paths)} videos ---")
    for vid_path in tqdm(video_paths):
        # 1. Get Label from filename (e.g., SHOOT_01.mp4)
        label_str = vid_path.name.split("_")[0].upper()
        if label_str not in GESTURE_CLASSES: continue
        label_idx = GESTURE_CLASSES.index(label_str)

        # 2. Read all frames
        cap = cv2.VideoCapture(str(vid_path))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()

        n = len(frames)
        if n == 0: continue

        # 3. Temporal Striding (Select exactly 15 frames)
        step = n / NUM_FRAMES
        indices = [int(i * step) for i in range(NUM_FRAMES)]
        
        video_tensor = np.zeros((3, NUM_FRAMES, len(CORE_JOINTS)), dtype=np.float32)
        
        # 4. Extract MediaPipe landmarks for selected frames
        for t, idx in enumerate(indices):
            # Mirror the frame to align with game physics
            img = cv2.flip(frames[idx], 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            
            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                for v, joint_idx in enumerate(CORE_JOINTS):
                    video_tensor[0, t, v] = lms[joint_idx].x
                    video_tensor[1, t, v] = lms[joint_idx].y
                    video_tensor[2, t, v] = lms[joint_idx].visibility
                    
        X_data.append(video_tensor)
        y_data.append(label_idx)

    pose.close()

    # Save the dataset
    X_tensor = torch.tensor(np.array(X_data))
    y_tensor = torch.tensor(np.array(y_data), dtype=torch.long)
    torch.save({'X': X_tensor, 'y': y_tensor}, OUTPUT_FILE)
    
    print(f"\n✅ Saved {len(X_data)} sequences to {OUTPUT_FILE}")
    print(f"Tensor Shape: {X_tensor.shape} -> (Batch, Channels, Time, Nodes)")

if __name__ == "__main__":
    extract_skeletons()