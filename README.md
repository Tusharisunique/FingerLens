# FingerScope

**FingerScope** turns your fingers into magical lenses. Pinch your thumb and index finger to frame and explore parts of the world, invert colors, or isolate regions, creating an imaginary camera controlled entirely by your hands.

## Features

- Real-time hand tracking using **MediaPipe**.
- Two interactive modes:
  1. **Main Mode:** Show only the region framed by your fingers on a black background.
  2. **Inverted Colors Mode:** Invert colors inside the framed region.
- Switch between modes using a simple pinch gesture over on-screen buttons.
- Visual feedback with landmarks and region outlines.

## Requirements

- Python 3.7+
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)
- NumPy (`pip install numpy`)

## Usage

1. Clone this repository or copy the script.
2. Run the script:
   ```bash
   python fingerscope.py
