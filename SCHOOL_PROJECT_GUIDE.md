# Golf Swing Rating - School Project Guide

## ✅ Setup Complete!

Everything is ready to use! You have:
- ✅ All dependencies installed (PyTorch, OpenCV, etc.)
- ✅ Pre-trained model downloaded (`models/swingnet_1800.pth.tar`)
- ✅ Ready-to-use rating script (`rate_swing.py`)

## 🎯 Quick Start - Rate a Golf Swing Video

### Basic Usage:
```bash
python rate_swing.py -p path/to/your/video.mp4
```

### Example (with included test video):
```bash
python rate_swing.py -p test_video.mp4
```

## 📊 What You Get

The script analyzes a golf swing video and provides:

1. **Event Detection** - Identifies 8 key swing events:
   - Address
   - Toe-up
   - Mid-backswing (arm parallel)
   - Top
   - Mid-downswing (arm parallel)
   - Impact
   - Mid-follow-through (shaft parallel)
   - Finish

2. **Confidence Scores** - How confident the model is about each event (0-1)

3. **Overall Rating** - A 0-100 score based on average confidence

4. **Quality Assessment** - Qualitative rating:
   - 80-100: Excellent 🏆
   - 70-79: Good ✅
   - 60-69: Fair 👍
   - Below 60: Needs Work 💪

## 📹 Video Requirements

**Important:** Videos must be:
- Cropped to show just the golfer
- Trimmed to contain only ONE complete swing
- Reasonable quality (doesn't need to be HD)

### Free Online Tools:
- **Crop**: https://ezgif.com/crop-video
- **Trim**: https://online-video-cutter.com/

## 🔧 Other Available Scripts

### 1. Test Video Display (shows event frames with GUI)
```bash
python test_video.py -p your_video.mp4
```
*Note: This opens GUI windows showing detected events*

### 2. Evaluate Model Performance
```bash
python eval.py
```
*Note: This evaluates on the validation dataset and gives PCE score*

### 3. Train Your Own Model (Optional)
```bash
# Light training (won't overwhelm your laptop)
python train_light.py

# Full training (very resource intensive!)
python train.py
```
**⚠️ Warning:** Training is VERY resource-intensive. The pre-trained model is recommended for your school project!

## 💡 Tips for Your Project

1. **Use the Pre-trained Model** - It's already trained and works well!

2. **Compare Multiple Swings** - Rate different golfers or different swings to show comparison

3. **Document Your Results** - Save screenshots of the analysis output

4. **Understand the Metrics**:
   - High confidence (>0.7) = Model is very sure about that event
   - Low confidence (<0.5) = Model is uncertain (might be video quality issue)

5. **Video Quality Matters**:
   - Better lighting = better results
   - Stable camera = better results
   - Clear view of golfer = better results

## 📝 Example Output

```
📹 Analyzing video: test_video.mp4
🖥️  Using device: cpu
✅ Loaded pre-trained model weights
🏌️  Detecting swing events...

📊 SWING ANALYSIS RESULTS:
============================================================
Address                             | Frame:   74 | Confidence: 0.091
Toe-up                              | Frame:   86 | Confidence: 0.571
Mid-backswing (arm parallel)        | Frame:   98 | Confidence: 0.848
Top                                 | Frame:  114 | Confidence: 0.707
Mid-downswing (arm parallel)        | Frame:  132 | Confidence: 0.867
Impact                              | Frame:  143 | Confidence: 0.974
Mid-follow-through (shaft parallel) | Frame:  151 | Confidence: 0.743
Finish                              | Frame:  236 | Confidence: 0.174
============================================================

⭐ OVERALL SWING RATING: 62.2/100
   Average Confidence: 0.622
   Quality Assessment: Fair 👍
```

## 🆘 Troubleshooting

### "Model weights not found"
- Make sure `models/swingnet_1800.pth.tar` exists
- The file should be about 60MB

### "Video not found"
- Check your video path is correct
- Use forward slashes (/) even on Windows

### Script runs slowly
- This is normal on CPU
- Processing takes 10-30 seconds per video depending on length
- Be patient!

### Low ratings on all videos
- Check video quality (lighting, clarity, framing)
- Make sure video shows ONE complete swing
- Verify video is properly cropped to just the golfer

## 📚 For Your Report

Key points to mention:
- Uses deep learning (CNN + LSTM architecture)
- MobileNetV2 for feature extraction
- LSTM for temporal sequence modeling
- Trained on GolfDB dataset (1,400+ videos)
- Detects 8 key events in golf swing
- Provides confidence scores and overall rating

## 🎓 Project Success!

You now have a working golf swing analysis system without needing to:
- ❌ Train a model from scratch
- ❌ Overwhelm your laptop
- ❌ Deal with complex setup

Just run `rate_swing.py` on your videos and analyze the results! 🏌️⛳

Good luck with your project! 🎉



