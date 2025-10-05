# FingerScope

**FingerScope** turns your hands into a virtual camera, letting your fingers act as magical lenses. By pinching your thumb and index finger, you can frame and explore any part of the world in a new way, zoom in on details, invert colors, or isolate regions as if seeing through an invisible lens. Switch between modes with a simple gesture and experience the world from the perspective of your fingertips.

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
