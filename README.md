# Autonomous RC Car — End-to-End TinyML Steering on a Raspberry Pi Zero 2W

An RC car that steers itself using a **PilotNet CNN** running entirely on a **Raspberry Pi Zero 2W**. The model maps raw camera frames directly to a steering angle — no lane detection, no hand-written control logic. Training happens on a PC; the quantized TensorFlow Lite model runs on the car's CPU in real time. Throttle stays with the human operator through a game controller.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![LiteRT](https://img.shields.io/badge/Runtime-LiteRT%20(TFLite)-4285F4)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-Zero%202W-A22846?logo=raspberrypi&logoColor=white)

| Top view | Front / camera view | Side isometric view |
| --- | --- | --- |
| ![Top view](docs/images/vehicle-top.png) | ![Front view](docs/images/vehicle-front.png) | ![Isometric view](docs/images/vehicle-isometric.png) |

---

## Contents

- [Highlights](#highlights)
- [How it works](#how-it-works)
- [Hardware](#hardware)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model](#model)
- [Troubleshooting](#troubleshooting)
- [Authors](#authors)
- [References](#references)

---

## Highlights

- **End-to-end learning** — one CNN replaces the classic perception → planning → control pipeline.
- **Runs on a $15 board** — post-training-quantized TFLite model on the Pi Zero 2W's Cortex-A53, within the project's 133 ms/frame latency budget.
- **Synchronized data collection** — three parallel processes (camera, controller, disk writer) record 30 FPS image/label pairs and discard any frame whose control reading is more than 100 ms stale.
- **Turn-aware training** — a weighted Huber loss penalizes errors on sharp turns up to 6× more than on straights, countering the heavy bias toward near-zero steering in driving data.
- **Augmentation** — horizontal flips (with negated labels), random brightness and random blur.
- **Adaptive steering smoothing** — an EMA whose α grows with turn strength: steady on straights, responsive in corners.

## How it works

![System architecture](docs/images/system-architecture.jpg)

The system runs in three phases:

1. **Data collection (Pi)** — `data_collection.py` drives the car from a game controller while recording camera frames and steering/throttle values.
2. **Training (PC)** — `train.py` crops, resizes and augments the frames, trains PilotNet, and exports a `.tflite` model. `validation.py` replays a held-out run through the model to check it before deployment.
3. **Autonomous driving (Pi)** — `autonomous_drive.py` feeds live frames to the model and drives the steering servo from its predictions.

![Classic pipeline vs end-to-end learning](docs/images/control-approaches.jpg)

## Hardware

| Component | Role | |
| --- | --- | --- |
| Raspberry Pi Zero 2W | Camera capture, model inference, GPIO control | ![Pi Zero 2W](docs/images/raspberry-pi-zero-2w.jpg) |
| Raspberry Pi Camera Module v1.3 | Front-facing camera, angled down at the track | ![Camera](docs/images/camera-module.jpg) |
| L298N dual H-bridge | Drives the rear motor | ![L298N](docs/images/l298n-driver.jpg) |
| N20 micro gear motor | Propulsion | ![N20 motor](docs/images/n20-motor.jpg) |
| Steering servo | Front-wheel steering | |
| USB/Bluetooth game controller | Manual driving and throttle | |

### Wiring (BCM pin numbers)

| Signal | GPIO |
| --- | --- |
| Steering servo PWM | 18 |
| L298N forward (IN1) | 5 |
| L298N backward (IN2) | 6 |
| L298N enable (ENA, PWM) | 19 |

Servo pulse width is calibrated to **1300–1900 µs**; adjust `min_pulse_width` / `max_pulse_width` in the scripts if your servo's travel differs.

## Repository structure

```
├── data_collection.py    # Pi: manual driving + synchronized dataset recording
├── test_crop.py          # PC: interactive tool to pick crop rows for the camera frame
├── params.py             # Shared config: input size, crop, label column, paths, epochs
├── pilotnet.py           # PilotNet model definition (Keras)
├── train.py              # PC: train, then export .h5 and .tflite
├── validation.py         # PC: evaluate a .tflite model on a held-out run and plot it
├── autonomous_drive.py   # Pi: real-time inference and steering
├── REPORT.pdf            # Full project report
└── docs/images/          # Figures used in this README
```

## Setup

### On the Raspberry Pi

1. Flash **Raspberry Pi OS Lite** with Raspberry Pi Imager, enabling SSH during setup, then SSH in from your PC.
2. Enable and start the pigpio daemon (needed for clean hardware PWM on the servo):
   ```bash
   sudo apt install pigpio python3-pigpio
   sudo systemctl enable --now pigpiod
   ```
3. Create a virtual environment and install dependencies. Picamera2 ships with the OS, so allow system packages:
   ```bash
   sudo apt install python3-picamera2 python3-opencv
   python3 -m venv --system-site-packages ~/rc-env
   source ~/rc-env/bin/activate
   pip install pygame gpiozero pigpio ai-edge-litert numpy
   ```
4. Pair or plug in the game controller.

### On the PC

```bash
pip install tensorflow opencv-python numpy pandas matplotlib ai-edge-litert
```

## Usage

### 1. Collect driving data (Pi)

```bash
python data_collection.py
```

- Each run is saved to a new numbered folder: `dataset/run_001`, `dataset/run_002`, …
- **X button** starts/stops recording. Left stick steers; right trigger drives forward, left trigger reverses.
- Frames where the car is stationary are skipped automatically.
- Press `Ctrl+C` to finish the session.

Resulting layout:

```
dataset/
├── run_001/
│   ├── images/          # frame_000001.jpg, ...
│   └── labels.csv       # frame, timestamp, steering, throttle, smooth_steer, smooth_throttle
├── run_002/
└── ...
run_006_val/             # a held-out run for validation — keep it OUTSIDE dataset/
```

Drive several clean laps in both directions. Record one extra run, move it out of `dataset/`, and set its path as `validation_run_dir` in `params.py`.

### 2. Choose the crop region (PC)

Copy the dataset to your PC, then:

```bash
python test_crop.py
```

Use the trackbars to browse images and adjust the top/bottom crop until only the track is visible (no sky, walls or car body). Press `q`, then copy the printed values into `crop_y1` / `crop_y2` in `params.py`.

### 3. Train (PC)

```bash
python train.py
```

Outputs `<MODEL_BASENAME>.h5` and `<MODEL_BASENAME>.tflite` (also copied to `models/`). Training uses AdamW (lr 1e-4), batch size 256, early stopping and learning-rate reduction on plateau.

### 4. Validate (PC)

```bash
python validation.py
```

Prints raw and smoothed MSE, correlation, and per-direction (left/right/straight) error, then plots predicted steering against the recorded ground truth. Set `MODEL_PATH` at the top of the file to your `.tflite` file.

### 5. Drive autonomously (Pi)

Copy the `.tflite` file next to `autonomous_drive.py` and set `MODEL_PATH` to its filename, then:

```bash
python autonomous_drive.py
```

The model steers; **you control throttle** with the triggers. Press the **X button** (or `Ctrl+C`) to stop.

> ⚠️ Start with a low `MAX_SPEED` and test in an open space. The car has no obstacle detection.

## Configuration

Key settings in `params.py`:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `inputres` | `(68, 68)` | Model input size (width, height) |
| `crop_y1`, `crop_y2` | `60`, `140` | Rows kept from the 360×240 frame before resizing |
| `target_column` | `smooth_steer` | Label to train on (`steering` = raw stick, `smooth_steer` = smoothed) |
| `dataset_dir` | `dataset` | Root folder containing `run_*` subfolders |
| `validation_run_dir` | `run_006_val` | Held-out run used for validation |
| `num_epochs` | `25` | Maximum training epochs |

Settings at the top of the scripts:

| File | Setting | Purpose |
| --- | --- | --- |
| `train.py` | `MODEL_BASENAME` | Output model filename |
| `train.py` | `ZERO_CONTROL_FRAME_DROP_PERCENT` | % of zero-steer/zero-throttle frames to drop |
| `autonomous_drive.py`, `validation.py` | `MODEL_PATH` | `.tflite` file to load |
| both Pi scripts | `MAX_SPEED` | Throttle cap (0–1) |
| `autonomous_drive.py` | `BASE_ALPHA`, `TURN_ALPHA_GAIN` | Steering smoothing: α = 0.12 + 0.38·\|steer\| |
| both Pi scripts | `X_BUTTON_INDEX` | Controller button index — differs between controllers |

## Model

PilotNet (NVIDIA, 2016) scaled down for a 68×68 RGB input:

| Layer | Config | Output |
| --- | --- | --- |
| Rescaling | ÷255 (built into the model) | 68×68×3 |
| Conv2D + Dropout 0.1 | 24 × 5×5, stride 2, ReLU | 32×32×24 |
| Conv2D + Dropout 0.1 | 36 × 5×5, stride 2, ReLU | 14×14×36 |
| Conv2D | 48 × 5×5, stride 2, ReLU | 5×5×48 |
| Conv2D | 64 × 3×3, ReLU | 3×3×64 |
| Conv2D | 64 × 3×3, ReLU | 1×1×64 |
| Dense + Dropout 0.1 | 100, ReLU | 100 |
| Dense | 50, ReLU | 50 |
| Dense | 10, ReLU | 10 |
| Dense | 1 (steering, −1 to 1) | 1 |

About **143k parameters**. Because rescaling is part of the model, the Pi feeds raw `uint8` frames with no extra preprocessing beyond crop and resize.

**Loss:** Huber (δ = 0.1) weighted by `1 + 5·|y|`, so a full-lock turn counts 6× more than driving straight.

## Troubleshooting

- **`Controller not found`** — connect the controller before starting the script; check it with `ls /dev/input/js*`.
- **X button does nothing** — button indices vary by controller. The scripts use `2` (collection) and `0` (driving); print `controller.get_button(i)` values to find yours.
- **Servo jitters** — make sure `pigpiod` is running; the default software PWM is too noisy for servos.
- **Model file not found** — `train.py` saves `MODEL_BASENAME.tflite`, but `autonomous_drive.py` and `validation.py` each load their own `MODEL_PATH`. Make all three match.
- **Model always predicts ~0** — the data is dominated by straight driving. Record more turns, keep the zero-frame drop on, and check the crop shows the track.
- **Works in validation, fails on track** — retrain with runs recorded under the lighting you'll drive in, and re-check the crop.

## Authors

- **Abdullah Aqeel Khan** ([@abdullahaqeelk](https://github.com/abdullahaqeelk))
- **Mohammad Usman Irshad** ([@usman-irshad1](https://github.com/usman-irshad1))

Group project for **EC-310 Microprocessor and Microcontroller Based Design**, College of E&ME, NUST. The [project report](REPORT.pdf) also credits Muhammad Husnain and Fatima Sohail as group members.

## References

1. M. Bojarski et al. (NVIDIA), "End-to-End Learning for Self-Driving Cars," [arXiv:1604.07316](https://arxiv.org/abs/1604.07316), 2016.
2. M. Bechtel, Q. Weng, H. Yun, "DeepPicarMicro: Applying TinyML to Autonomous Cyber Physical Systems," RTCSA 2022. [Code](https://github.com/CSL-KU/DeepPicarMicro)
3. [LiteRT / TensorFlow Lite documentation](https://ai.google.dev/edge/litert)

See [REPORT.pdf](REPORT.pdf) for the full design, results and challenges.
