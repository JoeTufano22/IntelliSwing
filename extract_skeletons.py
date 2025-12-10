"""
Extract MediaPipe skeletons from GolfDB videos.
Processes all videos in data/videos_160/ and saves skeleton sequences to data/skeletons.pkl
"""

import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import pickle
from tqdm import tqdm


def extract_skeleton_from_video(video_path, pose):
    """
    Extract MediaPipe skeleton landmarks from a video file.
    
    Args:
        video_path: Path to video file
        pose: MediaPipe Pose object
    
    Returns:
        skeletons: np.array of shape [num_frames, 33, 4] where 4 = [x, y, z, visibility]
                  Returns None if video cannot be opened
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    skeletons = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract landmarks: 33 points, each with x, y, z, visibility
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            skeletons.append(landmarks)
        else:
            # No detection - use zeros (will be handled during normalization)
            skeletons.append([[0.0, 0.0, 0.0, 0.0] for _ in range(33)])
    
    cap.release()
    
    if len(skeletons) == 0:
        return None
    
    return np.array(skeletons, dtype=np.float32)


def interpolate_missing_detections(skeletons):
    """
    Interpolate missing skeleton detections (where visibility is 0).
    Uses linear interpolation between valid frames.
    
    Args:
        skeletons: np.array of shape [num_frames, 33, 4]
    
    Returns:
        interpolated: np.array with interpolated values
    """
    interpolated = skeletons.copy()
    num_frames, num_landmarks, num_features = skeletons.shape
    
    # For each landmark, interpolate missing detections
    for landmark_idx in range(num_landmarks):
        # Check visibility (feature index 3)
        visibility = skeletons[:, landmark_idx, 3]
        
        # Find frames with valid detections
        valid_mask = visibility > 0.5
        
        if not np.any(valid_mask):
            # No valid detections for this landmark - skip interpolation
            continue
        
        # For each feature (x, y, z)
        for feature_idx in range(3):
            values = skeletons[:, landmark_idx, feature_idx]
            
            # Only interpolate if there are some valid values
            if np.any(valid_mask):
                # Create interpolation function
                valid_indices = np.where(valid_mask)[0]
                valid_values = values[valid_indices]
                
                # Interpolate for all frames
                if len(valid_indices) > 1:
                    interpolated[:, landmark_idx, feature_idx] = np.interp(
                        np.arange(num_frames),
                        valid_indices,
                        valid_values
                    )
                else:
                    # Only one valid frame - fill all with that value
                    interpolated[:, landmark_idx, feature_idx] = valid_values[0]
    
    return interpolated


def main():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Load GolfDB dataframe
    golfdb_path = 'data/golfDB.pkl'
    if not os.path.exists(golfdb_path):
        print(f"Error: {golfdb_path} not found. Run data/generate_splits.py first.")
        return
    
    df = pd.read_pickle(golfdb_path)
    vid_dir = 'data/videos_160/'
    
    if not os.path.exists(vid_dir):
        print(f"Error: Video directory {vid_dir} not found.")
        return
    
    # Dictionary to store skeletons: {video_id: np.array([num_frames, 33, 4])}
    skeletons_dict = {}
    
    print(f"Processing videos from GolfDB...")
    print(f"Video directory: {vid_dir}")
    
    failed_videos = []
    
    # Process each video
    for idx in tqdm(range(len(df)), desc="Extracting skeletons"):
        video_id = df.loc[idx, 'id']
        video_path = os.path.join(vid_dir, f'{video_id}.mp4')
        
        if not os.path.exists(video_path):
            continue
        
        # Extract skeletons
        skeletons = extract_skeleton_from_video(video_path, pose)
        
        if skeletons is None or len(skeletons) == 0:
            failed_videos.append(video_id)
            continue
        
        # Interpolate missing detections
        skeletons = interpolate_missing_detections(skeletons)
        
        # Store in dictionary
        skeletons_dict[video_id] = skeletons
    
    # Close MediaPipe pose
    pose.close()
    
    # Save skeletons to pickle file
    output_path = 'data/skeletons.pkl'
    os.makedirs('data', exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(skeletons_dict, f)
    
    print()
    print("Extraction complete!")
    print(f"   Processed: {len(skeletons_dict)} videos")
    print(f"   Failed: {len(failed_videos)} videos")
    print(f"   Saved to: {output_path}")


if __name__ == '__main__':
    main()

