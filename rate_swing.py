"""
Simple script to rate golf swing videos for your school project
Usage: python rate_swing.py -p path/to/video.mp4

This script rates swings based on temporal alignment with professional swing patterns from GolfDB.
"""

import argparse
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataloader import ToTensor, Normalize
from model import EventDetector
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os
import mediapipe as mp

event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing (arm parallel)',
    3: 'Top',
    4: 'Mid-downswing (arm parallel)',
    5: 'Impact',
    6: 'Mid-follow-through (shaft parallel)',
    7: 'Finish'
}


def load_professional_tempo_stats():
    """
    Load GolfDB and calculate professional swing tempo statistics.
    Returns mean and std of frame intervals between consecutive events.
    
    Note: GolfDB has 10 events total, but only events[1:9] are the 8 swing events.
    Event 0 and event 9 are video start/end markers.
    """
    golfdb_path = 'data/golfDB.pkl'
    
    if not os.path.exists(golfdb_path):
        print(f"⚠️  Warning: {golfdb_path} not found. Cannot compare to professional tempo.")
        return None, None
    
    df = pd.read_pickle(golfdb_path)
    
    # Calculate intervals between consecutive events for all swings
    all_intervals = [[] for _ in range(7)]  # 7 intervals between 8 events
    
    for idx in range(len(df)):
        full_events = df.loc[idx, 'events']
        # Use only the 8 swing events (exclude first and last markers)
        events = full_events[1:9]
        
        if len(events) == 8:  # Ensure all 8 events are present
            for i in range(7):
                interval = events[i+1] - events[i]
                all_intervals[i].append(interval)
    
    # Calculate mean and std for each interval
    interval_means = np.array([np.mean(intervals) for intervals in all_intervals])
    interval_stds = np.array([np.std(intervals) for intervals in all_intervals])
    
    return interval_means, interval_stds


def calculate_tempo_score(events, interval_means, interval_stds):
    """
    Calculate how well the swing tempo matches professional patterns.
    
    Args:
        events: Detected event frame numbers (8 events)
        interval_means: Mean intervals from professional swings
        interval_stds: Standard deviations of intervals
    
    Returns:
        tempo_score: Overall tempo alignment score (0-100)
        interval_scores: Individual scores for each interval
        detected_intervals: The actual intervals in the swing
    """
    if interval_means is None or interval_stds is None:
        return None, None, None
    
    # Check for chronological order
    if not all(events[i] < events[i+1] for i in range(7)):
        print("⚠️  Warning: Events are not in chronological order!")
        return 0.0, [0.0]*7, None
    
    # Calculate actual intervals
    detected_intervals = np.array([events[i+1] - events[i] for i in range(7)])
    
    # Calculate z-scores (how many standard deviations away from mean)
    z_scores = np.abs((detected_intervals - interval_means) / interval_stds)
    
    # Convert z-scores to scores (0-100 scale)
    # z=0 (exactly average) → 100, z=1 → ~60, z=2 → ~20, z≥3 → 0
    interval_scores = np.maximum(0, 100 * np.exp(-0.5 * z_scores))
    
    # Overall tempo score is weighted average
    # Give more weight to critical intervals (backswing, downswing, impact)
    weights = np.array([1.0, 1.5, 1.5, 2.0, 2.0, 1.5, 1.0])  # Emphasize downswing and impact
    tempo_score = np.average(interval_scores, weights=weights)
    
    return tempo_score, interval_scores, detected_intervals


def visualize_pose(video_path, scale_factor=3.0):
    """
    Visualize golf swing video with MediaPipe pose detection overlay.
    Displays skeleton connections in real-time in a larger window.
    
    Args:
        video_path: Path to video file
        scale_factor: Factor to scale up the video display (default: 3.0 for 3x larger)
    """
    print(f'\n🎬 Starting pose visualization for: {video_path}')
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate display dimensions (scaled up)
    display_width = int(frame_width * scale_factor)
    display_height = int(frame_height * scale_factor)
    
    print(f'📹 Video info: {frame_width}x{frame_height} @ {fps:.2f} FPS, {total_frames} frames')
    print(f'🖥️  Display size: {display_width}x{display_height} ({scale_factor}x scale)')
    print('💡 Press "q" to quit visualization\n')
    
    frame_count = 0
    
    # Create window with a specific name
    window_name = 'Golf Swing - Pose Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print(f'\n✅ Reached end of video ({frame_count} frames processed)')
            break
        
        frame_count += 1
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = pose.process(rgb_frame)
        
        # Draw pose landmarks and connections on the frame
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            # Use smaller, less cluttered landmarks
            # Landmarks: smaller radius (1px), thinner lines, more subtle color
            # Connections: slightly thicker for visibility when scaled up
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                # Landmark drawing spec: smaller radius, thinner, more subtle green
                mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1, circle_radius=1),
                # Connection drawing spec: thicker red lines for visibility
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        
        # Resize frame for larger display
        display_frame = cv2.resize(annotated_frame, (display_width, display_height), 
                                   interpolation=cv2.INTER_LINEAR)
        
        # Scale text size for larger display
        text_scale = scale_factor * 0.5
        text_thickness = int(max(1, scale_factor))
        
        # Add frame counter and instructions
        cv2.putText(display_frame, f'Frame: {frame_count}/{total_frames}', 
                   (int(10 * scale_factor), int(30 * scale_factor)), 
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness)
        cv2.putText(display_frame, 'Press "q" to quit', 
                   (int(10 * scale_factor), int(display_height - 20 * scale_factor)), 
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness)
        
        # Display the frame
        cv2.imshow(window_name, display_frame)
        
        # Check for 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f'\n⏹️  Visualization stopped by user at frame {frame_count}')
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print('✅ Visualization complete\n')


class SampleVideo(Dataset):
    def __init__(self, path, input_size=160, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)
        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # preprocess and return frames
        images = []
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, img = cap.read()
            if img is None:
                break
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)

            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()
        labels = np.zeros(len(images))  # only for compatibility with transforms
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


def rate_swing(video_path, seq_length=64):
    """
    Rate a golf swing video based on temporal alignment with professional swings.
    Returns: events (frame numbers), tempo score, individual interval scores
    """
    print(f'\n📹 Analyzing video: {video_path}')
    
    # Load professional tempo statistics
    print('📊 Loading professional swing tempo data from GolfDB...')
    interval_means, interval_stds = load_professional_tempo_stats()
    
    if interval_means is not None:
        print(f'✅ Loaded tempo data from {len(interval_means)+1} swing phases')
    
    # Load pre-trained model
    device = torch.device('cpu')  # Use CPU to avoid overwhelming system
    print(f'🖥️  Using device: {device}')
    
    model = EventDetector(pretrain=False,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    try:
        save_dict = torch.load('models/swingnet_1800.pth.tar', weights_only=False, map_location=device)
        model.load_state_dict(save_dict['model_state_dict'])
        print('✅ Loaded pre-trained model weights')
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure 'models/swingnet_1800.pth.tar' exists")
        return None, None, None

    model.to(device)
    model.eval()

    # Prepare video
    ds = SampleVideo(video_path, transform=transforms.Compose([ToTensor(),
                                Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])]))

    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    print('🏌️  Detecting swing events...')
    
    with torch.no_grad():
        for sample in dl:
            images = sample['images']
            # Process in batches to avoid memory issues
            batch = 0
            while batch * seq_length < images.shape[1]:
                if (batch + 1) * seq_length > images.shape[1]:
                    image_batch = images[:, batch * seq_length:, :, :, :]
                else:
                    image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
                logits = model(image_batch.to(device))
                if batch == 0:
                    probs = F.softmax(logits.data, dim=1).cpu().numpy()
                else:
                    probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
                batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    
    # Calculate model confidence scores (for reference)
    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    avg_confidence = np.mean(confidence)
    
    # Calculate tempo-based rating
    tempo_score, interval_scores, detected_intervals = calculate_tempo_score(
        events, interval_means, interval_stds
    )
    
    print('\n📊 SWING ANALYSIS RESULTS:')
    print('=' * 80)
    print(f'{"Event":35s} | {"Frame":>5s} | {"Confidence":>10s}')
    print('-' * 80)
    for i, (event_frame, conf) in enumerate(zip(events, confidence)):
        print(f'{event_names[i]:35s} | {event_frame:5d} | {conf:10.3f}')
    print('=' * 80)
    
    if tempo_score is not None:
        print('\n⏱️  TEMPORAL ALIGNMENT ANALYSIS:')
        print('=' * 80)
        print(f'{"Interval":40s} | {"Frames":>7s} | {"Pro Avg":>8s} | {"Score":>6s}')
        print('-' * 80)
        
        interval_names = [
            f'{event_names[i]} → {event_names[i+1]}' for i in range(7)
        ]
        
        for i, (name, frames, score) in enumerate(zip(interval_names, detected_intervals, interval_scores)):
            pro_avg = interval_means[i]
            print(f'{name:40s} | {frames:7.0f} | {pro_avg:8.1f} | {score:6.1f}')
        
        print('=' * 80)
        print(f'\n⭐ OVERALL TEMPO RATING: {tempo_score:.1f}/100')
        print(f'   (Based on alignment with professional swing patterns)')
        print(f'   Model Detection Confidence: {avg_confidence:.3f}')
        
        # Qualitative rating based on tempo score
        if tempo_score >= 80:
            quality = "Excellent - Professional Tempo 🏆"
        elif tempo_score >= 70:
            quality = "Good - Near Professional Tempo ✅"
        elif tempo_score >= 60:
            quality = "Fair - Acceptable Tempo 👍"
        elif tempo_score >= 40:
            quality = "Below Average - Tempo Needs Work 💪"
        else:
            quality = "Poor - Significant Tempo Issues ⚠️"
        
        print(f'   Quality Assessment: {quality}\n')
        
        return events, tempo_score, interval_scores
    else:
        # Fallback to confidence-based rating if tempo data unavailable
        print(f'\n⚠️  Tempo comparison unavailable - using detection confidence')
        rating = avg_confidence * 100
        print(f'\n⭐ DETECTION CONFIDENCE: {rating:.1f}/100')
        
        if rating >= 80:
            quality = "High Confidence 🏆"
        elif rating >= 70:
            quality = "Good Confidence ✅"
        elif rating >= 60:
            quality = "Fair Confidence 👍"
        else:
            quality = "Low Confidence 💪"
        
        print(f'   Quality Assessment: {quality}\n')
        
        return events, rating, confidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Rate a golf swing video based on temporal alignment with professional patterns'
    )
    parser.add_argument('-p', '--path', help='Path to video', default='test_video.mp4')
    parser.add_argument('--no-visualize', action='store_true', 
                       help='Skip pose visualization (faster processing)')
    args = parser.parse_args()
    
    # Visualize pose detection first (unless disabled)
    if not args.no_visualize:
        visualize_pose(args.path)
    
    # Then calculate swing rating
    events, rating, scores = rate_swing(args.path)
    
    if rating is not None:
        print('✅ Analysis complete!')
        print(f'\nTo analyze another video, run:')
        print(f'python rate_swing.py -p path/to/your/video.mp4')
        print(f'python rate_swing.py -p path/to/your/video.mp4 --no-visualize  # Skip visualization')
    else:
        print('❌ Analysis failed. Check error messages above.')

