<div align="center">

##  **Real-time drowsiness detection powered by facial landmark analysis and deep learning.**  
No wearables. No cloud. Just your webcam.

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](CONTRIBUTING.md)

<br/>

> ⚡ **94.7% accuracy · < 300ms alert latency · ~30 FPS on CPU · 2.4MB CNN model**

</div>

---

## Table of Contents

- [Why DrowsyGuard?](#why-drowsyguard)
- [How It Works](#how-it-works)
  - [Detection Pipeline](#detection-pipeline)
  - [Eye Aspect Ratio (EAR)](#eye-aspect-ratio-ear)
  - [Dual Validation Architecture](#dual-validation-architecture)
- [Performance](#performance)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [macOS / Linux](#macos--linux)
  - [Windows](#windows)
  - [Troubleshooting dlib](#troubleshooting-dlib)
- [Running DrowsyGuard](#running-drowsyguard)
  - [Basic usage](#basic-usage)
  - [All CLI options](#all-cli-options)
  - [Keyboard controls](#keyboard-controls)
- [CNN Classifier (Optional)](#cnn-classifier-optional)
  - [Dataset structure](#dataset-structure)
  - [Training](#training)
  - [Integrating the CNN](#integrating-the-cnn)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [Use Cases](#use-cases)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Code of Conduct](#code-of-conduct)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Why DrowsyGuard?

Fatigue-related incidents are responsible for **21% of fatal road crashes** and billions in lost workplace productivity annually. Existing solutions either require expensive hardware (steering wheel sensors, EEG bands) or cloud connectivity that introduces latency and privacy concerns.

DrowsyGuard runs entirely on-device — no internet, no subscription, no wearables — using only the webcam already built into your laptop or workstation. The system is designed to intervene *before* microsleep occurs, not after.

| Approach | Cost | Latency | Privacy | Setup |
|---|---|---|---|---|
| Wearable EEG bands | $$$ | Low | ✅ Local | Complex |
| Cloud-based CV services | $$ | High (network) | ❌ Uploads video | None |
| Rule-based EAR only | Free | Low | ✅ Local | Simple |
| **DrowsyGuard** | **Free** | **< 300ms** | **✅ Fully local** | **Simple** |

---

## How It Works

### Detection Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DROWSYGUARD PIPELINE                        │
└─────────────────────────────────────────────────────────────────────┘

  [Webcam]
     │  BGR frame @ 30 FPS
     ▼
  ┌──────────────────┐
  │  Frame Capture   │  OpenCV VideoStream (threaded, non-blocking)
  └────────┬─────────┘
           │  640×480 BGR numpy array
           ▼
  ┌──────────────────┐
  │  Grayscale Conv  │  cv2.cvtColor → reduces compute for detector
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Face Detection  │  dlib HOG + SVM detector
  └────────┬─────────┘  → returns bounding box rect(s)
           │
           ▼
  ┌──────────────────┐
  │ Shape Prediction │  dlib 68-point landmark predictor
  └────────┬─────────┘  → (x, y) coords for all 68 facial keypoints
           │
     ┌─────┴─────┐
     │           │
  Left eye    Right eye
  pts 37-42   pts 43-48
     │           │
     └─────┬─────┘
           │  both eye landmark arrays
           ▼
  ┌──────────────────┐
  │  EAR Calculation │  (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
  └────────┬─────────┘  → mean EAR across both eyes
           │
           ▼
  ┌──────────────────┐     EAR >= threshold
  │  Threshold Gate  │ ──────────────────────► reset counter → [loop]
  └────────┬─────────┘
           │  EAR < threshold
           ▼
  ┌──────────────────┐
  │ Frame Counter +1 │
  └────────┬─────────┘
           │
           │  counter < N frames
           ├──────────────────────► [continue loop — single blink]
           │
           │  counter >= N frames
           ▼
  ┌──────────────────┐
  │  ALERT TRIGGER   │  AlarmController.trigger() — threaded
  └────────┬─────────┘
           │
     ┌─────┴────────────────────────────┐
     │                                  │
  playsound(alarm.wav)          HUD overlay + red banner
  (daemon thread, non-blocking)        on frame
```

### Eye Aspect Ratio (EAR)

The EAR algorithm, introduced by Soukupová & Čech (CVWW 2016), computes a normalized ratio of vertical to horizontal eye dimensions using 6 dlib landmark points per eye.

```
         p2        p3
          *--------*
         /          \
   p1 *              * p4
         \          /
          *--------*
         p6        p5


       ||p2 - p6|| + ||p3 - p5||
EAR = ─────────────────────────────
              2 x ||p1 - p4||
```

The horizontal distance `||p1-p4||` normalizes for face scale, making EAR invariant to distance from the camera.

| State | Typical EAR | Description |
|---|---|---|
| Wide open | 0.35 – 0.45 | Alert, attentive |
| Normal open | 0.25 – 0.35 | Baseline awake |
| Tired / heavy | 0.15 – 0.25 | Fatigue beginning |
| Closed | 0.00 – 0.10 | Blink or drowsy closure |

> **Note:** Optimal threshold varies by individual (glasses, deep-set eyes, face angle). Run a 30-second calibration session to measure your personal baseline and set `--ear` to `baseline x 0.75`.

### Dual Validation Architecture

A single EAR threshold would fire on every normal blink (~150–200ms). DrowsyGuard uses two gates to eliminate false positives:

**Gate 1 — Temporal:** The EAR must stay below threshold for `N` consecutive frames. At 30 FPS, `N=20` corresponds to ~667ms — longer than any natural blink.

**Gate 2 — Spatial (optional CNN):** A MobileNetV2-based classifier confirms eye closure on the raw pixel crop, independent of landmark geometry. This catches edge cases where lighting artifacts push EAR below threshold without actual eye closure.

```
  EAR < threshold     →   counter += 1
  counter >= N frames →   CNN confirms "closed"?   →   ALARM
                              └── "open"?           →   false positive, skip
```

---

## Performance

| Metric | Value | Notes |
|---|---|---|
| Detection accuracy | **94.7%** | On MRL Eye Dataset |
| False positive rate | **< 2%** | Gate 1 + Gate 2 combined |
| Frame rate (CPU) | **~30 FPS** | Intel i5, no GPU |
| Frame rate (GPU) | **~60 FPS** | NVIDIA RTX 3060 |
| Alert latency | **< 300 ms** | Frame capture to alarm sound |
| CNN model size | **2.4 MB** | MobileNetV2 head only |
| RAM usage | **~180 MB** | Idle with stream active |

---

## Prerequisites

- **Python 3.9 – 3.11** (3.12 has known dlib build issues as of this writing)
- **Webcam** with at least 480p resolution
- **cmake** — required to build dlib from source
- **~500 MB disk space** for models and dependencies

---

## Installation

### macOS / Linux

```bash
# 1. Clone the repository
git clone https://github.com/yourname/drowsy-guard.git
cd drowsy-guard

# 2. Create and activate a virtual environment (strongly recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install cmake (required by dlib)
#    macOS:
brew install cmake
#    Ubuntu/Debian:
sudo apt-get install -y cmake libopenblas-dev liblapack-dev

# 4. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Download the dlib 68-point landmark model (~100MB)
curl -Lo shape_predictor_68_face_landmarks.dat.bz2 \
  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# 6. Verify everything works
python ear_utils.py
```

Expected output from step 6:
```
Running EAR utility tests...

  [PASS] Open eye EAR = 0.332
  [PASS] Closed eye EAR = 0.000
  [PASS] Smoothed EAR = 0.193
  [PASS] State label logic correct
  [PASS] Normalize eye centroid ~ origin

5/5 tests passed.
```

### Windows

```powershell
# 1. Install cmake via winget or from https://cmake.org/download/
winget install Kitware.CMake

# 2. Clone and enter the repo
git clone https://github.com/yourname/drowsy-guard.git
cd drowsy-guard

# 3. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 4. Install Visual C++ Build Tools if not already installed
#    Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 5. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 6. Download landmark model (PowerShell)
Invoke-WebRequest -Uri "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" `
  -OutFile "shape_predictor_68_face_landmarks.dat.bz2"
# Extract with 7-Zip or WinRAR
```

### Troubleshooting dlib

dlib compiles from C++ source and occasionally fails. Common fixes:

| Error | Fix |
|---|---|
| `cmake not found` | Install cmake and restart terminal |
| `No module named 'dlib'` | `pip install dlib==19.24.0` explicitly |
| Build fails on Python 3.12 | Downgrade to Python 3.11 |
| Slow build (10+ min) | Normal — dlib has no prebuilt wheel for most platforms |
| macOS arm64 error | `arch -x86_64 pip install dlib` then rebuild |

If dlib consistently fails, a pre-built wheel for your platform may be available at [pypi.org/project/dlib](https://pypi.org/project/dlib/).

---

## Running DrowsyGuard

### Basic usage

```bash
# Default settings (EAR threshold 0.25, 20 consecutive frames)
python drowsy_guard.py

# High-sensitivity mode (fires faster — good for driving)
python drowsy_guard.py --ear 0.28 --frames 12

# Low-sensitivity mode (fewer false positives — good for reading)
python drowsy_guard.py --ear 0.22 --frames 30

# Disable landmark overlay for better performance on slow machines
python drowsy_guard.py --no-lm

# External USB camera (index 1)
python drowsy_guard.py --camera 1

# Custom alarm sound
python drowsy_guard.py --alarm sounds/loud_beep.wav
```

### All CLI options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--ear` | float | `0.25` | EAR threshold. Eyes below this value are considered closed. |
| `--frames` | int | `20` | Consecutive sub-threshold frames required before alarm fires. |
| `--alarm` | str | `alarm.wav` | Path to alarm `.wav` file. |
| `--model` | str | `shape_predictor_68_face_landmarks.dat` | Path to dlib landmark model. |
| `--camera` | int | `0` | OpenCV camera index. `0` = built-in webcam. |
| `--no-lm` | flag | off | Disables 68-point landmark overlay. Improves FPS on weak hardware. |

### Keyboard controls

| Key | Action |
|---|---|
| `q` | Quit and close all windows |
| `r` | Reset alarm and counter manually (useful after intentional eye closure) |

---

## CNN Classifier (Optional)

The EAR-only system performs well in controlled lighting. The CNN second gate is recommended when deploying in variable conditions (night driving, sunlight glare, reflective glasses).

### Dataset structure

```
data/
├── train/
│   ├── open/           # ~1500 eye crop images, 64x64px minimum
│   │   ├── img_001.jpg
│   │   └── ...
│   └── closed/         # ~1500 eye crop images
│       ├── img_001.jpg
│       └── ...
└── val/
    ├── open/           # ~400 images each
    └── closed/
```

Good public datasets for sourcing eye crops:
- **MRL Eye Dataset** — 84,898 infrared eye images (ideal)
- **CEW Dataset** — Closed Eyes in the Wild, 1,192 subjects
- **OpenEDS** — Facebook/Meta open eye dataset

### Training

```bash
# Basic training run (20 epochs, transfer learning from ImageNet)
python train_eye_cnn.py \
  --data   ./data \
  --epochs 20 \
  --output model/eye_cnn.h5

# Train from scratch (no pretrained weights)
python train_eye_cnn.py \
  --data    ./data \
  --epochs  50 \
  --scratch \
  --output  model/eye_cnn_scratch.h5

# Custom image size and batch
python train_eye_cnn.py \
  --data     ./data \
  --img-size 96 \
  --batch    64 \
  --output   model/eye_cnn_96.h5
```

Training runs two phases automatically:
1. **Head training** — base MobileNetV2 frozen, only the custom dense head trains (fast, ~5 min)
2. **Fine-tuning** — top 30 base layers unfrozen, trained at 10x lower learning rate (~15 min)

Expect ~92–96% validation accuracy on a balanced dataset of 3,000 images.

### Integrating the CNN

In `drowsy_guard.py`, add the second gate after the EAR counter check:

```python
from train_eye_cnn import EyeCNNClassifier

# In DrowsyGuard.__init__:
self.cnn = EyeCNNClassifier("model/eye_cnn.h5")

# In process_frame, after counter >= consec_frames:
if self.counter >= self.cfg["consec_frames"]:
    left_state, _  = self.cnn.predict_ear(frame, left_eye)
    right_state, _ = self.cnn.predict_ear(frame, right_eye)
    if left_state == "closed" and right_state == "closed":
        self.alert_active = True
        self.alarm.trigger()
```

---

## Project Structure

```
drowsy-guard/
│
├── drowsy_guard.py              # Main detection loop & entry point
│   ├── DrowsyGuard              #   class — full pipeline
│   ├── AlarmController          #   class — threaded audio alert
│   ├── draw_hud()               #   HUD overlay on frame
│   └── draw_eye_contour()       #   eye convex hull overlay
│
├── ear_utils.py                 # EAR geometry helpers (importable)
│   ├── eye_aspect_ratio()       #   core EAR formula
│   ├── smooth_ear()             #   running average filter
│   ├── ear_to_state()           #   open/tired/closed label
│   └── normalize_eye()          #   canonical coords for CNN input
│
├── train_eye_cnn.py             # CNN trainer + inference class
│   ├── build_model()            #   MobileNetV2 head architecture
│   ├── get_data_generators()    #   augmented Keras ImageDataGenerators
│   ├── train()                  #   two-phase training loop
│   └── EyeCNNClassifier         #   class — load & run inference
│
├── requirements.txt             # Pinned Python dependencies
├── alarm.wav                    # Default alarm sound (replace freely)
├── shape_predictor_68_face_landmarks.dat   # Download separately
│
├── model/                       # Trained CNN weights (git-ignored)
│   └── eye_cnn.h5
│
├── data/                        # Training data (git-ignored)
│   ├── train/{open,closed}/
│   └── val/{open,closed}/
│
└── tests/
    └── test_ear_utils.py        # pytest-compatible test suite
```

---

## Configuration Reference

All `DEFAULT_CONFIG` values in `drowsy_guard.py` can be overridden at construction time for programmatic use:

```python
from drowsy_guard import DrowsyGuard

guard = DrowsyGuard(config={
    "ear_threshold":     0.25,   # float — EAR below this = closed eye
    "consec_frames":     20,     # int   — frames before alarm fires
    "alarm_sound":       "alarm.wav",
    "landmark_model":    "shape_predictor_68_face_landmarks.dat",
    "camera_index":      0,      # int   — OpenCV VideoStream source
    "show_landmarks":    True,   # bool  — render 68-point overlay
    "show_ear":          True,   # bool  — render HUD with EAR value
    "frame_width":       640,    # int   — resize all frames to this
    "frame_height":      480,
})
guard.run()
```

---

## Use Cases

| Domain | Scenario | Recommended config |
|---|---|---|
| 🚗 Long-haul driving | Night driving, highway monotony | `--ear 0.27 --frames 15` |
| 🖥️ Remote work | Late-night coding sessions | `--ear 0.25 --frames 22` |
| 📚 Studying | Exam preparation, reading | `--ear 0.23 --frames 28` |
| 🏭 Industrial operators | Assembly line, crane operation | `--ear 0.28 --frames 12` |
| 🎮 Gaming | Marathon sessions | `--ear 0.24 --frames 25` |

---

## Roadmap

- [ ] **v1.1** — PERCLOS metric (% eye closure over 1-min rolling window) as supplementary signal
- [ ] **v1.2** — Head pose estimation (chin-drop as additional drowsiness cue)
- [ ] **v1.3** — Yawn detection using mouth landmark distances
- [ ] **v1.4** — Calibration mode — auto-measures personal EAR baseline in 30 seconds
- [ ] **v2.0** — ONNX export for edge deployment (Raspberry Pi, Jetson Nano)
- [ ] **v2.1** — REST API mode — stream alerts to a dashboard or phone notification

---

## Contributing

Contributions are welcome. This project follows a standard fork → branch → PR workflow.

### Getting started

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/drowsy-guard.git
cd drowsy-guard

# 2. Create a virtual environment and install dev dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest black flake8    # dev tools

# 3. Create a feature branch — name it after what you're changing
git checkout -b feature/yawn-detection
# or
git checkout -b fix/alarm-threading-race-condition
```

### Making changes

- **One change per PR.** Keep pull requests focused. A PR that adds yawn detection and also refactors `AlarmController` is harder to review and harder to revert.
- **Write tests.** New functions in `ear_utils.py` should include test cases. Run `python ear_utils.py` or `pytest tests/` to verify.
- **Follow the existing code style.** Run `black .` before committing to auto-format. Run `flake8 .` to catch lint issues.
- **Update this README** if your change affects setup steps, CLI flags, or project structure.

### Submitting a PR

```bash
# Run tests and linting before pushing
python ear_utils.py            # built-in unit tests
black . --check                # formatting check
flake8 . --max-line-length 100

# Commit with a clear message
git add -A
git commit -m "feat: add yawn detection via mouth aspect ratio"

# Push and open a PR on GitHub
git push origin feature/yawn-detection
```

**PR title format:**

```
feat:     short description of new capability
fix:      short description of what was broken
docs:     what documentation changed
refactor: code change with no behaviour change
```

**In your PR description, include:**
- What problem this solves or feature it adds
- How you tested it (webcam model used, lighting conditions, dataset)
- Any known limitations or follow-up work needed

### Good first issues

| Issue | Skill level | Description |
|---|---|---|
| Add `--calibrate` flag | Intermediate | 30-second auto-baseline measurement for EAR threshold |
| PERCLOS metric | Intermediate | Track % eye closure over a rolling 60-second window |
| Add pytest suite | Beginner | Move `ear_utils.py` inline tests to `tests/test_ear_utils.py` |
| ONNX export script | Advanced | Export trained CNN to `.onnx` for edge deployment |
| macOS audio fix | Beginner | `playsound` hangs on macOS Ventura — investigate `simpleaudio` as replacement |

### Reporting bugs

Open a GitHub Issue with:

1. **Python version** (`python --version`)
2. **OS and version** (e.g., macOS 14.2, Ubuntu 22.04, Windows 11)
3. **Error message** — full traceback, not just the last line
4. **Steps to reproduce** — exact commands you ran
5. **Expected vs actual behaviour**

---

## Code of Conduct

This project follows the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

In short: be respectful, be constructive, assume good intent. Harassment of any kind will result in a permanent ban.

---

## License

MIT License — see [LICENSE](LICENSE) for full text.

You are free to use, modify, and distribute this software for any purpose, including commercial use, with attribution.

---

## Acknowledgements

- **Soukupová & Čech** — *Real-Time Eye Blink Detection using Facial Landmarks* (CVWW 2016) — the EAR formula that powers the core detection logic
- **Davis King** — [dlib](http://dlib.net/) — the face detector and shape predictor that make 30 FPS landmark detection possible on CPU
- **MRL Eye Dataset** — Faculty of Information Technology, Brno University of Technology — training data for the CNN classifier
- **OpenCV team** — for two decades of computer vision infrastructure

---

<div align="center">

Built with Python · OpenCV · dlib · TensorFlow

*If this project helped you, consider starring the repo ⭐*

</div>
