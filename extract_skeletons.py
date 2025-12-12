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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            skeletons.append(landmarks)
        else:
            skeletons.append([[0.0, 0.0, 0.0, 0.0] for _ in range(33)])
    
    cap.release()
    
    if len(skeletons) == 0:
        return None
    
    return np.array(skeletons, dtype=np.float32)


def interpolate_missing_detections(skeletons):
    """
    Interpolate missing skeleton detections using linear interpolation between valid frames.
    
    Args:
        skeletons: np.array of shape [num_frames, 33, 4]
    
    Returns:
        interpolated: np.array with interpolated values
    """
    interpolated = skeletons.copy()
    num_frames, num_landmarks, num_features = skeletons.shape
    
    for landmark_idx in range(num_landmarks):
        visibility = skeletons[:, landmark_idx, 3]
        valid_mask = visibility > 0.5
        
        if not np.any(valid_mask):
            continue
        for feature_idx in range(3):
            values = skeletons[:, landmark_idx, feature_idx]
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                valid_values = values[valid_indices]
                if len(valid_indices) > 1:
                    interpolated[:, landmark_idx, feature_idx] = np.interp(
                        np.arange(num_frames),
                        valid_indices,
                        valid_values
                    )
                else:
                    interpolated[:, landmark_idx, feature_idx] = valid_values[0]
    
    return interpolated


def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    golfdb_path = 'data/golfDB.pkl'
    if not os.path.exists(golfdb_path):
        print(f"Error: {golfdb_path} not found. Run data/generate_splits.py first.")
        return
    
    df = pd.read_pickle(golfdb_path)
    vid_dir = 'data/videos_160/'
    
    if not os.path.exists(vid_dir):
        print(f"Error: Video directory {vid_dir} not found.")
        return
    
    skeletons_dict = {}
    
    print(f"Processing videos from GolfDB...")
    print(f"Video directory: {vid_dir}")
    
    failed_videos = []
    
    for idx in tqdm(range(len(df)), desc="Extracting skeletons"):
        video_id = df.loc[idx, 'id']
        video_path = os.path.join(vid_dir, f'{video_id}.mp4')
        
        if not os.path.exists(video_path):
            continue
        
        skeletons = extract_skeleton_from_video(video_path, pose)
        
        if skeletons is None or len(skeletons) == 0:
            failed_videos.append(video_id)
            continue
        
        skeletons = interpolate_missing_detections(skeletons)
        skeletons_dict[video_id] = skeletons
    
    pose.close()
    
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

