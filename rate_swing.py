import argparse
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision import transforms
from dataloader import ToTensor, Normalize
from model import EventDetector
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os
import mediapipe as mp
from skeleton_autoencoder import SkeletonAutoencoder
from rate_swing_skeleton import (
    extract_skeleton_from_video,
    interpolate_missing_detections,
    normalize_skeleton,
    calculate_per_landmark_error,
    identify_top_problem_areas,
    calculate_error_distribution,
    error_to_score,
)
from skeleton_dataloader import SkeletonDataset

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


def generalize_landmark_name(name):
    lower = name.lower()
    side = "Left" if "left" in lower else "Right" if "right" in lower else ""

    if any(k in lower for k in ["pinky", "index", "thumb", "wrist", "hand"]):
        core = "Hand/Arm"
    elif "elbow" in lower or "shoulder" in lower:
        core = "Arm/Shoulder"
    elif "hip" in lower:
        core = "Hips/Pelvis"
    elif any(k in lower for k in ["knee", "ankle", "heel", "foot"]):
        core = "Lower Body/Leg"
    else:
        return name

    return f"{side} {core}".strip()


def load_professional_tempo_stats():
    golfdb_path = 'data/golfDB.pkl'
    
    if not os.path.exists(golfdb_path):
        print(f"Warning: {golfdb_path} not found. Cannot compare to professional tempo.")
        return None, None
    
    df = pd.read_pickle(golfdb_path)
    
    all_intervals = [[] for _ in range(7)]
    
    for idx in range(len(df)):
        full_events = df.loc[idx, 'events']
        events = full_events[1:9]
        
        if len(events) == 8:
            for i in range(7):
                interval = events[i+1] - events[i]
                all_intervals[i].append(interval)
    
    interval_means = np.array([np.mean(intervals) for intervals in all_intervals])
    interval_stds = np.array([np.std(intervals) for intervals in all_intervals])
    
    return interval_means, interval_stds


def get_improvement_tips(tempo_score, spatial_score, top_problems, interval_scores=None, interval_means=None, detected_intervals=None):
    tips = []
    avg_score = (tempo_score + spatial_score) / 2 if tempo_score is not None else spatial_score

    if avg_score < 60:
        tips.append("Focus on fundamentals: work on basic swing mechanics and consistency.")
    elif avg_score < 70:
        tips.append("Good foundation: refine technique for better consistency.")
    elif avg_score < 80:
        tips.append("Strong swing: minor adjustments can elevate your game.")
    else:
        tips.append("Excellent technique: maintain your current form.")

    if interval_scores is not None and len(interval_scores) == 7:
        worst_idx = int(np.argmin(interval_scores))
        stage_name = f"{event_names[worst_idx]} → {event_names[worst_idx + 1]}"
        stage_score = interval_scores[worst_idx]
        if detected_intervals is not None and len(detected_intervals) == 7 and interval_means is not None:
            actual = detected_intervals[worst_idx]
            pro_avg = interval_means[worst_idx]
            delta = actual - pro_avg
            direction = "longer" if delta > 0 else "shorter"
            tips.append(f"Tempo: biggest gap is {stage_name} ({actual:.0f} frames vs {pro_avg:.1f} pro avg, {abs(delta):.0f} frames {direction}). Smooth that transition.")
        else:
            tips.append(f"Tempo: {stage_name} is the least aligned (score {stage_score:.1f}). Focus on that transition.")
    elif tempo_score is not None and tempo_score < 70:
        tips.append("Tempo: keep rhythm consistent; practice with a metronome or counting cadence.")

    if top_problems:
        raw_problem = top_problems[0][2]
        main_problem = generalize_landmark_name(raw_problem)
        tips.append(f"Spatial: main issue appears around the {main_problem.lower()} ({raw_problem}).")

        lower_problem = main_problem.lower()
        if "hand/arm" in lower_problem:
            tips.append("Keep grip pressure neutral and maintain a stable wrist/forearm line through takeaway and impact.")
        elif "shoulder" in lower_problem:
            tips.append("Work on shoulder turn while keeping posture stable; avoid early opening or collapsing lead shoulder.")
        elif "hip" in lower_problem or "pelvis" in lower_problem:
            tips.append("Focus on hip rotation and weight shift; avoid sliding and keep a centered pivot.")
        elif any(k in lower_problem for k in ["knee", "ankle", "foot", "leg"]):
            tips.append("Stabilize lower body: maintain flex, ground contact, and balanced weight transfer.")
        elif "elbow" in lower_problem:
            tips.append("Maintain arm structure; avoid excessive lead elbow bend or trail elbow flying out.")

    return tips


def calculate_tempo_score(events, interval_means, interval_stds):
    if interval_means is None or interval_stds is None:
        return None, None, None
    
    if not all(events[i] < events[i+1] for i in range(7)):
        print("Warning: Events are not in chronological order.")
        return 0.0, [0.0]*7, None
    
    detected_intervals = np.array([events[i+1] - events[i] for i in range(7)])
    
    z_scores = np.abs((detected_intervals - interval_means) / interval_stds)
    
    interval_scores = np.maximum(0, 100 * np.exp(-0.3 * z_scores))
    interval_scores = np.maximum(interval_scores, 30 * np.exp(-0.5 * np.maximum(0, z_scores - 2)))
    
    weights = np.array([1.0, 1.5, 1.5, 2.0, 2.0, 1.5, 1.0])
    tempo_score = np.average(interval_scores, weights=weights)
    
    return tempo_score, interval_scores, detected_intervals


def get_current_stage(frame_num, events):
    if len(events) != 8:
        return "Unknown", -1
    
    if not all(events[i] <= events[i+1] for i in range(7)):
        return "Unknown", -1
    
    if frame_num < events[0]:
        return "Before Address", -1
    
    for i in range(7):
        if frame_num == events[i]:
            return event_names[i], i
        if events[i] < frame_num < events[i + 1]:
            return event_names[i], i
    
    if frame_num >= events[7]:
        return event_names[7], 7
    
    return "Unknown", -1


def visualize_pose(video_path, scale_factor=3.0):
    print()
    print(f'Starting pose visualization for: {video_path}')
    
    print('Detecting swing stages...')
    device = torch.device('cpu')
    
    model = EventDetector(pretrain=False,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    
    try:
        save_dict = torch.load('models/swingnet_1800.pth.tar', weights_only=False, map_location=device)
        model.load_state_dict(save_dict['model_state_dict'])
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Warning: Could not load model for stage detection: {e}")
        print("Visualization will continue without stage detection")
        model = None
        events = None
    else:
        ds = SampleVideo(video_path, transform=transforms.Compose([ToTensor(),
                                    Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])]))
        dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
        
        seq_length = 64
        with torch.no_grad():
            for sample in dl:
                images = sample['images']
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
        print(f'Detected swing stages at frames: {events}')
        for i, (frame, name) in enumerate(zip(events, [event_names[i] for i in range(8)])):
            print(f'   {name}: frame {frame}')
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    display_width = int(frame_width * scale_factor)
    display_height = int(frame_height * scale_factor)
    
    print(f'Video info: {frame_width}x{frame_height} @ {fps:.2f} FPS, {total_frames} frames')
    print(f'Display size: {display_width}x{display_height} ({scale_factor}x scale)')
    print('Press "q" to quit visualization')
    
    frame_count = 0
    
    window_name = 'Golf Swing - Pose Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print()
            print(f'Reached end of video ({frame_count} frames processed)')
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = pose.process(rgb_frame)
        
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        
        display_frame = cv2.resize(annotated_frame, (display_width, display_height), 
                                   interpolation=cv2.INTER_LINEAR)
        
        text_scale = scale_factor * 0.4
        text_thickness = int(max(2, scale_factor * 0.8))
        outline_thickness = int(max(3, scale_factor * 1.2))
        
        current_stage = "Unknown"
        if events is not None:
            current_stage, _ = get_current_stage(frame_count, events)
        
        text_color = (0, 255, 255)
        outline_color = (0, 0, 0) 
        
        frame_text = f'Frame: {frame_count}/{total_frames}'
        frame_pos = (int(10 * scale_factor), int(35 * scale_factor))
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    cv2.putText(display_frame, frame_text,
                               (frame_pos[0] + dx, frame_pos[1] + dy),
                               cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                               outline_color, outline_thickness)
        
        cv2.putText(display_frame, frame_text, frame_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                   text_color, text_thickness)
        
        stage_text = f'Stage: {current_stage}'
        stage_pos = (int(10 * scale_factor), int(70 * scale_factor))
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    cv2.putText(display_frame, stage_text,
                               (stage_pos[0] + dx, stage_pos[1] + dy),
                               cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                               outline_color, outline_thickness)
        
        cv2.putText(display_frame, stage_text, stage_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                   text_color, text_thickness)
        
        cv2.imshow(window_name, display_frame)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print()
            print(f'Visualization stopped by user at frame {frame_count}')
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    if model is not None:
        del model
    print('Visualization complete')


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

        images = []
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, img = cap.read()
            if img is None:
                break
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])

            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()
        labels = np.zeros(len(images))
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


def rate_swing(video_path, seq_length=64):
    print()
    print(f'Analyzing video: {video_path}')
    
    print('Loading professional swing tempo data from GolfDB...')
    interval_means, interval_stds = load_professional_tempo_stats()
    
    if interval_means is not None:
        print(f'Loaded tempo data from {len(interval_means)+1} swing phases')
    
    device = torch.device('cpu') 
    print(f'Using device: {device}')
    
    model = EventDetector(pretrain=False,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    try:
        save_dict = torch.load('models/swingnet_1800.pth.tar', weights_only=False, map_location=device)
        model.load_state_dict(save_dict['model_state_dict'])
        print('Loaded pre-trained model weights')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure 'models/swingnet_1800.pth.tar' exists")
        return None, None, None

    model.to(device)
    model.eval()

    ds = SampleVideo(video_path, transform=transforms.Compose([ToTensor(),
                                Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])]))

    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    print('Detecting swing events...')
    
    with torch.no_grad():
        for sample in dl:
            images = sample['images']
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
    
    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    avg_confidence = np.mean(confidence)
    
    tempo_score, interval_scores, detected_intervals = calculate_tempo_score(
        events, interval_means, interval_stds
    )
    
    print()
    print('SWING ANALYSIS RESULTS:')
    print('=' * 80)
    print(f'{"Event":35s} | {"Frame":>5s} | {"Confidence":>10s}')
    print('-' * 80)
    for i, (event_frame, conf) in enumerate(zip(events, confidence)):
        print(f'{event_names[i]:35s} | {event_frame:5d} | {conf:10.3f}')
    print('=' * 80)
    
    if tempo_score is not None:
        print()
        print('TEMPORAL ALIGNMENT ANALYSIS:')
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
        print()
        print(f'OVERALL TEMPO RATING: {tempo_score:.1f}/100')
        print('   Based on alignment with professional swing patterns')
        print(f'   Model detection confidence: {avg_confidence:.3f}')
        
        if tempo_score >= 80:
            quality = "Excellent - Professional Tempo"
        elif tempo_score >= 70:
            quality = "Good - Near Professional Tempo"
        elif tempo_score >= 60:
            quality = "Fair - Acceptable Tempo"
        elif tempo_score >= 40:
            quality = "Below Average - Tempo Needs Work"
        else:
            quality = "Poor - Significant Tempo Issues"
        
        print(f'   Quality Assessment: {quality}')
        
        return events, tempo_score, interval_scores
    else:
        print()
        print('Tempo comparison unavailable - using detection confidence')
        rating = avg_confidence * 100
        print()
        print(f'DETECTION CONFIDENCE: {rating:.1f}/100')
        
        if rating >= 80:
            quality = "High Confidence"
        elif rating >= 70:
            quality = "Good Confidence"
        elif rating >= 60:
            quality = "Fair Confidence"
        else:
            quality = "Low Confidence"
        
        print(f'   Quality Assessment: {quality}')
        
        return events, rating, confidence


def analyze_swing(video_path, scale_factor=3.0, seq_length=64):
    print()
    print(f'Starting swing analysis for: {video_path}')

    interval_means, interval_stds = load_professional_tempo_stats()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    tempo_model = EventDetector(pretrain=False,
                                width_mult=1.,
                                lstm_layers=1,
                                lstm_hidden=256,
                                bidirectional=True,
                                dropout=False)
    try:
        save_dict = torch.load('models/swingnet_1800.pth.tar', weights_only=False, map_location=device)
        tempo_model.load_state_dict(save_dict['model_state_dict'])
        tempo_model.to(device)
        tempo_model.eval()
        print('Loaded tempo detection model')
    except Exception as e:
        print(f"❌ Error loading tempo model: {e}")
        return

    ds = SampleVideo(video_path, transform=transforms.Compose([ToTensor(),
                                Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])]))
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    probs = None
    with torch.no_grad():
        for sample in dl:
            images = sample['images']
            batch = 0
            while batch * seq_length < images.shape[1]:
                if (batch + 1) * seq_length > images.shape[1]:
                    image_batch = images[:, batch * seq_length:, :, :, :]
                else:
                    image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
                logits = tempo_model(image_batch.to(device))
                if batch == 0:
                    probs = F.softmax(logits.data, dim=1).cpu().numpy()
                else:
                    probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
                batch += 1

    if probs is None:
        print("Could not detect swing events.")
        return

    events = np.argmax(probs, axis=0)[:-1]

    tempo_score, interval_scores, detected_intervals = calculate_tempo_score(
        events, interval_means, interval_stds
    )

    spatial_model_path = 'models/skeleton_autoencoder_best.pth.tar'
    if not os.path.exists(spatial_model_path):
        print(f"Error: Spatial model not found: {spatial_model_path}")
        return

    checkpoint = torch.load(spatial_model_path, map_location=device, weights_only=False)
    model_config = checkpoint.get('model_config', {
        'hidden_dim': 128, 'latent_dim': 64, 'num_lstm_layers': 2, 'dropout': 0.2
    })

    spatial_model = SkeletonAutoencoder(
        input_dim=132, hidden_dim=model_config['hidden_dim'],
        latent_dim=model_config['latent_dim'],
        num_lstm_layers=model_config['num_lstm_layers'],
        dropout=model_config['dropout']
    )
    spatial_model.load_state_dict(checkpoint['model_state_dict'])
    spatial_model.to(device)
    spatial_model.eval()
    print('Loaded spatial analysis model')

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
        print("Error: Could not extract skeletons from video")
        return

    skeletons = interpolate_missing_detections(skeletons)
    normalized_frames = []
    for frame in skeletons:
        normalized_frame = normalize_skeleton(frame)
        normalized_frames.append(normalized_frame.flatten())
    normalized_frames = np.array(normalized_frames)

    all_spatial_errors = []
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
            reconstructed = spatial_model(seq_tensor)
            mse = torch.mean((reconstructed - seq_tensor) ** 2).item()
            all_spatial_errors.append(mse)
            landmark_errors = calculate_per_landmark_error(
                seq_tensor.squeeze(0), reconstructed.squeeze(0)
            )
            all_landmark_errors += landmark_errors

    avg_spatial_error = np.mean(all_spatial_errors)
    all_landmark_errors = all_landmark_errors / len(all_spatial_errors) if len(all_spatial_errors) > 0 else all_landmark_errors

    try:
        train_dataset = SkeletonDataset(
            data_file='data/train_split_1.pkl',
            skeletons_file='data/skeletons.pkl',
            seq_length=seq_length,
            train=True
        )
        train_loader = TorchDataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0)
        error_stats = calculate_error_distribution(spatial_model, train_loader, device)
        spatial_score = error_to_score(avg_spatial_error, error_stats)
    except Exception as e:
        print(f"Warning: Could not calculate error distribution: {e}")
        spatial_score = max(1.0, min(100.0, 100.0 - avg_spatial_error * 1000))

    top_problems = identify_top_problem_areas(all_landmark_errors, top_n=5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    display_width = int(frame_width * scale_factor)
    display_height = int(frame_height * scale_factor)

    delay_play_ms = max(1, int(round(1000.0 / fps))) if fps and fps > 0 else 33
    delay_pause_ms = 30

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose_viz = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    window_name = 'IntelliSwing'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)

    playing = True
    button_rect = (10, 10, 110, 40)
    scrub_state = {'seeking': False, 'updating': False, 'target': 0}

    def on_mouse(event, x, y, flags, param):
        nonlocal playing
        if event == cv2.EVENT_LBUTTONDOWN:
            bx, by, bw, bh = button_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                playing = not playing

    cv2.setMouseCallback(window_name, on_mouse)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    def on_trackbar(val):
        if scrub_state['updating']:
            return 
        scrub_state['seeking'] = True
        scrub_state['target'] = val

    cv2.createTrackbar('Frame', window_name, 0, max(total_frames - 1, 1), on_trackbar)

    frame_count = 0
    last_frame = None
    while True:
        if scrub_state['seeking']:
            cap.set(cv2.CAP_PROP_POS_FRAMES, scrub_state['target'])
            ret, frame = cap.read()
            scrub_state['seeking'] = False
            playing = False 
        elif playing:
            ret, frame = cap.read()
        else:
            ret, frame = True, last_frame

        if not ret or frame is None:
            if last_frame is None:
                break 
            frame = last_frame
            playing = False

        frame_count = max(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1, 0)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_viz.process(rgb_frame)

        annotated_frame = frame.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        display_frame = cv2.resize(annotated_frame, (display_width, display_height),
                                   interpolation=cv2.INTER_LINEAR)

        text_scale = scale_factor * 0.4
        text_thickness = int(max(2, scale_factor * 0.8))
        outline_thickness = int(max(3, scale_factor * 1.2))
        text_color = (0, 255, 255)  
        outline_color = (0, 0, 0)   

        current_stage, _ = get_current_stage(frame_count, events) if events is not None else ("Unknown", -1)

        frame_text = f'Frame: {frame_count}/{total_frames}'
        frame_pos = (int(10 * scale_factor), int(35 * scale_factor))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    cv2.putText(display_frame, frame_text,
                               (frame_pos[0] + dx, frame_pos[1] + dy),
                               cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                               outline_color, outline_thickness)
        cv2.putText(display_frame, frame_text, frame_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                   text_color, text_thickness)

        stage_text = f'Stage: {current_stage}'
        stage_pos = (int(10 * scale_factor), int(70 * scale_factor))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    cv2.putText(display_frame, stage_text,
                               (stage_pos[0] + dx, stage_pos[1] + dy),
                               cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                               outline_color, outline_thickness)
        cv2.putText(display_frame, stage_text, stage_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                   text_color, text_thickness)

        bx, by, bw, bh = button_rect
        cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (30, 30, 30), -1)
        cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)
        btn_text = "Pause" if playing else "Play"
        cv2.putText(display_frame, btn_text, (bx + 10, by + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        scrub_state['updating'] = True
        cv2.setTrackbarPos('Frame', window_name, min(frame_count, total_frames - 1))
        scrub_state['updating'] = False

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(delay_play_ms if playing else delay_pause_ms) & 0xFF
        if key == ord('q'):
            print()
            print('Skipped to results')
            break
        if key == ord(' '): 
            playing = not playing

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print()
            print('Window closed by user')
            break

        last_frame = frame.copy()

    cap.release()
    cv2.destroyAllWindows()
    pose_viz.close()

    print()
    print('=' * 80)
    print('COMPREHENSIVE SWING ANALYSIS RESULTS')
    print('=' * 80)

    print()
    print('TEMPORAL (TEMPO) ANALYSIS:')
    print('-' * 80)
    if tempo_score is not None:
        print(f'Overall Tempo Score: {tempo_score:.1f}/100')
        print()
        print('Detected Swing Stages:')
        for name, frame in zip(event_names.values(), events):
            print(f'  {name:35s} | Frame {frame:4d}')
    else:
        print('Tempo analysis unavailable')
        tempo_score = 0

    print()
    print('SPATIAL (BODY POSITION) ANALYSIS:')
    print('-' * 80)
    print(f'Overall Spatial Score: {spatial_score:.1f}/100')
    print()
    print('Top Problem Areas:')
    avg_landmark_error = np.mean(all_landmark_errors) if len(all_landmark_errors) else 0
    for rank, (idx, error, name) in enumerate(top_problems, 1):
        display_name = generalize_landmark_name(name)
        deviation = error / avg_landmark_error if avg_landmark_error > 0 else 0
        print(f'  {rank}. {display_name:25s} ({name}) | Error: {error:.6f} | {deviation:.1f}x average')

    print()
    print('COMBINED ANALYSIS:')
    print('-' * 80)
    avg_score = (tempo_score + spatial_score) / 2 if tempo_score else spatial_score
    print(f'Average Score: {avg_score:.1f}/100')
    if avg_score >= 80:
        overall_quality = "Excellent - Professional Level"
    elif avg_score >= 70:
        overall_quality = "Good - Near Professional"
    elif avg_score >= 60:
        overall_quality = "Fair - Acceptable"
    elif avg_score >= 40:
        overall_quality = "Below Average - Needs Work"
    else:
        overall_quality = "Poor - Significant Issues"
    print(f'Overall Quality: {overall_quality}')

    print()
    print('IMPROVEMENT TIPS:')
    print('-' * 80)
    tips = get_improvement_tips(
        tempo_score,
        spatial_score,
        top_problems,
        interval_scores=interval_scores,
        interval_means=interval_means,
        detected_intervals=detected_intervals
    )
    for tip in tips:
        print(f'  {tip}')
    print()
    print('=' * 80)
    print('Analysis complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Unified golf swing analysis with video display and combined tempo + spatial scoring'
    )
    parser.add_argument('-p', '--path', help='Path to video', default='test_video.mp4')
    parser.add_argument('--scale', type=float, default=3.0, help='Display scale factor')
    args = parser.parse_args()

    analyze_swing(args.path, args.scale)

