import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


class SkeletonDataset(Dataset):
    """Dataset for MediaPipe skeleton sequences from GolfDB."""
    
    def __init__(self, data_file, skeletons_file, seq_length=None, transform=None, train=True):
        self.df = pd.read_pickle(data_file)
        self.seq_length = seq_length
        self.transform = transform
        self.train = train
        
        if not os.path.exists(skeletons_file):
            raise FileNotFoundError(f"Skeletons file not found: {skeletons_file}")
        
        with open(skeletons_file, 'rb') as f:
            self.skeletons_dict = pickle.load(f)
        
        available_ids = set(self.skeletons_dict.keys())
        self.df = self.df[self.df['id'].isin(available_ids)].reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} videos with skeleton data from {data_file}")
    
    def __len__(self):
        return len(self.df)
    
    def normalize_skeleton(self, skeleton):
        normalized = skeleton.copy()
        
        x_coords = skeleton[:, 0]
        y_coords = skeleton[:, 1]
        visibility = skeleton[:, 3]
        
        left_hip = skeleton[LEFT_HIP]
        right_hip = skeleton[RIGHT_HIP]
        
        if left_hip[3] > 0.5 and right_hip[3] > 0.5:
            hip_center_x = (left_hip[0] + right_hip[0]) / 2.0
            hip_center_y = (left_hip[1] + right_hip[1]) / 2.0
        elif left_hip[3] > 0.5:
            hip_center_x = left_hip[0]
            hip_center_y = left_hip[1]
        elif right_hip[3] > 0.5:
            hip_center_x = right_hip[0]
            hip_center_y = right_hip[1]
        else:
            visible_mask = visibility > 0.5
            if np.any(visible_mask):
                hip_center_x = np.mean(x_coords[visible_mask])
                hip_center_y = np.mean(y_coords[visible_mask])
            else:
                hip_center_x = 0.5
                hip_center_y = 0.5
        
        normalized[:, 0] = x_coords - hip_center_x
        normalized[:, 1] = y_coords - hip_center_y
        
        left_shoulder = skeleton[LEFT_SHOULDER]
        right_shoulder = skeleton[RIGHT_SHOULDER]
        
        scale = 1.0
        if (left_shoulder[3] > 0.5 and left_hip[3] > 0.5):
            dist_shoulder_hip = np.sqrt(
                (left_shoulder[0] - left_hip[0])**2 + 
                (left_shoulder[1] - left_hip[1])**2
            )
            if dist_shoulder_hip > 0:
                scale = dist_shoulder_hip
        elif (right_shoulder[3] > 0.5 and right_hip[3] > 0.5):
            dist_shoulder_hip = np.sqrt(
                (right_shoulder[0] - right_hip[0])**2 + 
                (right_shoulder[1] - right_hip[1])**2
            )
            if dist_shoulder_hip > 0:
                scale = dist_shoulder_hip
        
        if scale > 0:
            normalized[:, 0] = normalized[:, 0] / scale
            normalized[:, 1] = normalized[:, 1] / scale
            normalized[:, 2] = skeleton[:, 2] / scale
        
        normalized[:, 3] = visibility
        
        return normalized
    
    def __getitem__(self, idx):
        a = self.df.loc[idx, :]
        video_id = a['id']
        
        skeleton_seq = self.skeletons_dict[video_id]
        
        normalized_frames = []
        for frame in skeleton_seq:
            normalized_frame = self.normalize_skeleton(frame)
            normalized_frames.append(normalized_frame.flatten())
        
        normalized_frames = np.array(normalized_frames)
        
        if self.seq_length is None:
            seq_length = len(normalized_frames)
        else:
            seq_length = self.seq_length
            
            if self.train:
                if len(normalized_frames) > seq_length:
                    start_idx = np.random.randint(0, len(normalized_frames) - seq_length + 1)
                    normalized_frames = normalized_frames[start_idx:start_idx + seq_length]
                elif len(normalized_frames) < seq_length:
                    padding = np.zeros((seq_length - len(normalized_frames), 132))
                    normalized_frames = np.vstack([normalized_frames, padding])
            else:
                if len(normalized_frames) > seq_length:
                    normalized_frames = normalized_frames[:seq_length]
                elif len(normalized_frames) < seq_length:
                    padding = np.zeros((seq_length - len(normalized_frames), 132))
                    normalized_frames = np.vstack([normalized_frames, padding])
        
        skeleton_tensor = torch.from_numpy(normalized_frames).float()
        
        if self.transform:
            skeleton_tensor = self.transform(skeleton_tensor)
        
        return skeleton_tensor


