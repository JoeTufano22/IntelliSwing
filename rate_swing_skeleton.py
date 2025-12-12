"""
Rate golf swing videos based on spatial quality using skeleton autoencoder.
Extracts MediaPipe skeletons and compares them to professional swing patterns.
"""

import argparse
import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from skeleton_autoencoder import SkeletonAutoencoder
from skeleton_dataloader import SkeletonDataset
from torch.utils.data import DataLoader


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
    """Interpolate missing skeleton detections."""
    interpolated = skeletons.copy()
    num_frames, num_landmarks, num_features = skeletons.shape
    
    for landmark_idx in range(num_landmarks):
        visibility = skeletons[:, landmark_idx, 3]
        valid_mask = visibility > 0.5
        
        if not np.any(valid_mask):
            continue
        
        for feature_idx in range(3):
            values = skeletons[:, landmark_idx, feature_idx]
            valid_indices = np.where(valid_mask)[0]
            valid_values = values[valid_indices]
            
            if len(valid_indices) > 1:
                interpolated[:, landmark_idx, feature_idx] = np.interp(
                    np.arange(num_frames),
                    valid_indices,
                    valid_values
                )
            elif len(valid_indices) == 1:
                interpolated[:, landmark_idx, feature_idx] = valid_values[0]
    
    return interpolated


def normalize_skeleton(skeleton):
    """
    Normalize a single skeleton frame (same as in SkeletonDataset).
    """
    normalized = skeleton.copy()
    
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    
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


def calculate_error_distribution(model, dataloader, device):
    """
    Calculate reconstruction error distribution on training set.
    Used to map errors to 1-100 quality scores.
    """
    model.eval()
    errors = []
    
    with torch.no_grad():
        for skeleton_seq in dataloader:
            skeleton_seq = skeleton_seq.to(device)
            reconstructed = model(skeleton_seq)
            
            mse = torch.mean((reconstructed - skeleton_seq) ** 2, dim=(1, 2))
            errors.extend(mse.cpu().numpy().tolist())
    
    errors = np.array(errors)
    return {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'min': np.min(errors),
        'max': np.max(errors),
        'percentiles': {
            5: np.percentile(errors, 5),
            25: np.percentile(errors, 25),
            50: np.percentile(errors, 50),
            75: np.percentile(errors, 75),
            95: np.percentile(errors, 95)
        }
    }


LANDMARK_NAMES = [
    'Nose', 'Left Eye Inner', 'Left Eye', 'Left Eye Outer', 'Right Eye Inner',
    'Right Eye', 'Right Eye Outer', 'Left Ear', 'Right Ear', 'Mouth Left',
    'Mouth Right', 'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Pinky', 'Right Pinky', 'Left Index',
    'Right Index', 'Left Thumb', 'Right Thumb', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle', 'Left Heel',
    'Right Heel', 'Left Foot Index', 'Right Foot Index'
]


def get_landmark_name(landmark_idx):
    """Get human-readable name for a MediaPipe landmark index."""
    if 0 <= landmark_idx < len(LANDMARK_NAMES):
        return LANDMARK_NAMES[landmark_idx]
    return f'Landmark {landmark_idx}'


def calculate_per_landmark_error(original, reconstructed):
    """Calculate reconstruction error per landmark."""
    orig_reshaped = original.view(-1, 33, 4)
    recon_reshaped = reconstructed.view(-1, 33, 4)
    
    errors = torch.mean((orig_reshaped - recon_reshaped) ** 2, dim=(0, 2))
    
    return errors.cpu().numpy()


def identify_worst_landmark(landmark_errors):
    """
    Identify which landmark has the highest reconstruction error.
    
    Args:
        landmark_errors: np.array of shape [33] with error per landmark
    
    Returns:
        worst_idx: Index of worst-performing landmark
        worst_error: Error value for worst landmark
        worst_name: Name of worst-performing landmark
    """
    worst_idx = np.argmax(landmark_errors)
    worst_error = landmark_errors[worst_idx]
    worst_name = get_landmark_name(worst_idx)
    
    return worst_idx, worst_error, worst_name


def identify_top_problem_areas(landmark_errors, top_n=5):
    """Identify top N body parts with highest reconstruction errors."""
    sorted_indices = np.argsort(landmark_errors)[::-1]
    
    top_problems = []
    for i in range(min(top_n, len(sorted_indices))):
        idx = sorted_indices[i]
        error = landmark_errors[idx]
        name = get_landmark_name(idx)
        top_problems.append((idx, error, name))
    
    return top_problems


def error_to_score(reconstruction_error, error_stats):
    """Convert reconstruction error to quality score (1-100) using percentiles."""
    p5 = error_stats['percentiles'][5]
    p95 = error_stats['percentiles'][95]
    
    if reconstruction_error <= p5:
        score = 100.0
    elif reconstruction_error >= p95:
        score = 1.0
    else:
        score = 100.0 - ((reconstruction_error - p5) / (p95 - p5)) * 99.0
    
    return max(1.0, min(100.0, score))


def rate_swing_spatial(video_path, model_path='models/skeleton_autoencoder_best.pth.tar', 
                       skeletons_file='data/skeletons.pkl', seq_length=64):
    """
    Rate a golf swing video based on spatial quality.
    
    Args:
        video_path: Path to input video
        model_path: Path to trained autoencoder model
        skeletons_file: Path to skeletons.pkl (for error distribution calculation)
        seq_length: Sequence length for processing
    
    Returns:
        score: Quality score (1-100)
        reconstruction_error: Raw reconstruction error
        worst_landmark_name: Name of worst-performing body part
        worst_landmark_error: Error value for worst body part
    """
    print()
    print(f'Analyzing spatial quality of: {video_path}')
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        print("Please train the model first using train_skeleton_autoencoder.py")
        return None, None, None, None
    
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint.get('model_config', {
        'hidden_dim': 128,
        'latent_dim': 64,
        'num_lstm_layers': 2,
        'dropout': 0.2
    })
    
    model = SkeletonAutoencoder(
        input_dim=132,
        hidden_dim=model_config['hidden_dim'],
        latent_dim=model_config['latent_dim'],
        num_lstm_layers=model_config['num_lstm_layers'],
        dropout=model_config['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f'Loaded model from {model_path}')
    
    print('Extracting MediaPipe skeletons from video...')
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    skeletons = extract_skeleton_from_video(video_path, pose)
    pose.close()
    
    if skeletons is None or len(skeletons) == 0:
        print("❌ Error: Could not extract skeletons from video")
        return None, None, None, None
    
    print(f'Extracted {len(skeletons)} frames')
    
    skeletons = interpolate_missing_detections(skeletons)
    
    print('Normalizing skeletons...')
    normalized_frames = []
    for frame in skeletons:
        normalized_frame = normalize_skeleton(frame)
        normalized_frames.append(normalized_frame.flatten())
    
    normalized_frames = np.array(normalized_frames)
    
    print('Processing with autoencoder...')
    all_errors = []
    all_landmark_errors = np.zeros(33)
    
    with torch.no_grad():
        step_size = seq_length // 2
        
        for start_idx in range(0, len(normalized_frames), step_size):
            end_idx = min(start_idx + seq_length, len(normalized_frames))
            seq = normalized_frames[start_idx:end_idx]
            
            if len(seq) < seq_length:
                padding = np.zeros((seq_length - len(seq), 132))
                seq = np.vstack([seq, padding])
            
            seq_tensor = torch.from_numpy(seq).float().unsqueeze(0).to(device)
            reconstructed = model(seq_tensor)
            
            mse = torch.mean((reconstructed - seq_tensor) ** 2).item()
            all_errors.append(mse)
            
            landmark_errors = calculate_per_landmark_error(
                seq_tensor.squeeze(0),
                reconstructed.squeeze(0)
            )
            all_landmark_errors += landmark_errors
    
    reconstruction_error = np.mean(all_errors)
    
    all_landmark_errors = all_landmark_errors / len(all_errors) if len(all_errors) > 0 else all_landmark_errors
    
    worst_idx, worst_error, worst_name = identify_worst_landmark(all_landmark_errors)
    top_problems = identify_top_problem_areas(all_landmark_errors, top_n=5)
    
    print('Calculating quality score...')
    try:
        train_dataset = SkeletonDataset(
            data_file='data/train_split_1.pkl',
            skeletons_file=skeletons_file,
            seq_length=seq_length,
            train=True
        )
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        error_stats = calculate_error_distribution(model, train_loader, device)
        
        score = error_to_score(reconstruction_error, error_stats)
        
    except Exception as e:
        print(f"Warning: Could not calculate error distribution: {e}")
        print("Using simple error-to-score mapping...")
        score = max(1.0, min(100.0, 100.0 - reconstruction_error * 1000))
    
    print()
    print('=' * 80)
    print('SPATIAL QUALITY ANALYSIS RESULTS:')
    print('=' * 80)
    print(f'Reconstruction Error: {reconstruction_error:.6f}')
    print(f'Spatial Quality Score: {score:.1f}/100')
    
    if score >= 80:
        quality = "Excellent - Professional Spatial Patterns"
    elif score >= 70:
        quality = "Good - Near Professional Patterns"
    elif score >= 60:
        quality = "Fair - Acceptable Patterns"
    elif score >= 40:
        quality = "Below Average - Patterns Need Work"
    else:
        quality = "Poor - Significant Spatial Issues"
    
    print(f'Quality Assessment: {quality}')
    print('=' * 80)
    
    print()
    print('BODY PART ANALYSIS:')
    print('=' * 80)
    avg_error = np.mean(all_landmark_errors)
    print(f'Top Problem Areas (furthest from professional patterns):')
    print('-' * 80)
    for rank, (idx, error, name) in enumerate(top_problems, 1):
        deviation = error / avg_error if avg_error > 0 else 0
        print(f'{rank}. {name:25s} | Error: {error:.6f} | {deviation:.1f}x average')
    print('=' * 80)
    
    return score, reconstruction_error, worst_name, worst_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Rate golf swing spatial quality using skeleton autoencoder'
    )
    parser.add_argument('-p', '--path', help='Path to video', default='test_video.mp4')
    parser.add_argument('-m', '--model', help='Path to model checkpoint', 
                       default='models/skeleton_autoencoder_best.pth.tar')
    
    args = parser.parse_args()
    
    result = rate_swing_spatial(args.path, args.model)
    
    if result is not None and result[0] is not None:
        score, error, worst_part, worst_error = result
        print()
        print('Analysis complete!')
    else:
        print()
        print('Analysis failed. Check error messages above.')

