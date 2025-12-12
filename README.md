# IntelliSwing Quickstart

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Rate a Swing
```bash
python rate_swing.py -p test_video.mp4 --scale 3.0
```
- `-p` points to the video file.
- `--scale` controls on-screen display size (3.0 = 3x).

## Notes
- Press `q` to skip to results while the window is open.
- Spacebar toggles play/pause; arrow keys step frames; trackbar scrubs.

