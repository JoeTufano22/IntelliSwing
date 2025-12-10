"""
PyTorch Dataset for loading MediaPipe skeleton sequences from GolfDB.
Handles normalization (centering around hip, scaling by body size) and padding.
"""

import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


# MediaPipe Pose landmark indices
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


class SkeletonDataset(Dataset):
    """
    Dataset for loading MediaPipe skeleton sequences from GolfDB.
    
    Normalizes skeletons by:
    1. Centering around hip joint (average of left and right hip)
    2. Scaling by body size (distance between shoulders and hips)
    
    Returns padded sequences of shape [seq_length, 132] where 132 = 33 landmarks × 4 values
    """
    
    def __init__(self, data_file, skeletons_file, seq_length=None, transform=None, train=True):
        """
        Args:
            data_file: Path to GolfDB split pickle file (e.g., 'data/train_split_1.pkl')
            skeletons_file: Path to skeletons pickle file (e.g., 'data/skeletons.pkl')
            seq_length: Maximum sequence length. If None, uses full sequence length.
            transform: Optional transform to apply to skeleton sequences
            train: If True, use random sampling; if False, use full sequences
        """
        self.df = pd.read_pickle(data_file)
        self.seq_length = seq_length
        self.transform = transform
        self.train = train
        
        # Load skeletons dictionary
        if not os.path.exists(skeletons_file):
            raise FileNotFoundError(f"Skeletons file not found: {skeletons_file}")
        
        with open(skeletons_file, 'rb') as f:
            self.skeletons_dict = pickle.load(f)
        
        # Filter out videos that don't have skeletons
        available_ids = set(self.skeletons_dict.keys())
        self.df = self.df[self.df['id'].isin(available_ids)].reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} videos with skeleton data from {data_file}")
    
    def __len__(self):
        return len(self.df)
    
    def normalize_skeleton(self, skeleton):
        """
        Normalize a single skeleton frame.
        
        Args:
            skeleton: np.array of shape [33, 4] (landmarks × [x, y, z, visibility])
        
        Returns:
            normalized: np.array of shape [33, 4] normalized skeleton
        """
        normalized = skeleton.copy()
        
        # Extract x, y coordinates (ignore z and visibility for normalization)
        x_coords = skeleton[:, 0]
        y_coords = skeleton[:, 1]
        visibility = skeleton[:, 3]
        
        # Calculate hip center (average of left and right hip)
        left_hip = skeleton[LEFT_HIP]
        right_hip = skeleton[RIGHT_HIP]
        
        # Use hip center if both are visible, otherwise use available one
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
            # No hip detected - use center of bounding box of visible points
            visible_mask = visibility > 0.5
            if np.any(visible_mask):
                hip_center_x = np.mean(x_coords[visible_mask])
                hip_center_y = np.mean(y_coords[visible_mask])
            else:
                # Fallback: use image center
                hip_center_x = 0.5
                hip_center_y = 0.5
        
        # Center around hip
        normalized[:, 0] = x_coords - hip_center_x
        normalized[:, 1] = y_coords - hip_center_y
        
        # Calculate body scale (distance between shoulders and hips)
        left_shoulder = skeleton[LEFT_SHOULDER]
        right_shoulder = skeleton[RIGHT_SHOULDER]
        
        # Calculate shoulder-hip distance for scale normalization
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
        
        # Normalize by scale (only if scale is valid)
        if scale > 0:
            normalized[:, 0] = normalized[:, 0] / scale
            normalized[:, 1] = normalized[:, 1] / scale
            # Also normalize z by scale (depth)
            normalized[:, 2] = skeleton[:, 2] / scale
        
        # Keep visibility unchanged
        normalized[:, 3] = visibility
        
        return normalized
    
    def __getitem__(self, idx):
        """
        Get a skeleton sequence.
        
        Returns:
            skeleton_seq: torch.Tensor of shape [seq_length, 132]
                         (132 = 33 landmarks × 4 values flattened)
        """
        a = self.df.loc[idx, :]
        video_id = a['id']
        
        # Get skeleton sequence
        skeleton_seq = self.skeletons_dict[video_id]  # [num_frames, 33, 4]
        
        # Normalize each frame
        normalized_frames = []
        for frame in skeleton_seq:
            normalized_frame = self.normalize_skeleton(frame)
            # Flatten: [33, 4] -> [132]
            normalized_frames.append(normalized_frame.flatten())
        
        normalized_frames = np.array(normalized_frames)  # [num_frames, 132]
        
        # Handle sequence length
        if self.seq_length is None:
            # Use full sequence
            seq_length = len(normalized_frames)
        else:
            seq_length = self.seq_length
            
            if self.train:
                # Random starting position for training
                if len(normalized_frames) > seq_length:
                    start_idx = np.random.randint(0, len(normalized_frames) - seq_length + 1)
                    normalized_frames = normalized_frames[start_idx:start_idx + seq_length]
                elif len(normalized_frames) < seq_length:
                    # Pad if shorter
                    padding = np.zeros((seq_length - len(normalized_frames), 132))
                    normalized_frames = np.vstack([normalized_frames, padding])
            else:
                # Validation: use full sequence or pad/truncate
                if len(normalized_frames) > seq_length:
                    # Take first seq_length frames
                    normalized_frames = normalized_frames[:seq_length]
                elif len(normalized_frames) < seq_length:
                    # Pad with zeros
                    padding = np.zeros((seq_length - len(normalized_frames), 132))
                    normalized_frames = np.vstack([normalized_frames, padding])
        
        # Convert to tensor
        skeleton_tensor = torch.from_numpy(normalized_frames).float()
        
        if self.transform:
            skeleton_tensor = self.transform(skeleton_tensor)
        
        return skeleton_tensor

