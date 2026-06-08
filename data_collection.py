import multiprocessing as mp
import csv
import os
import queue
import time
from pathlib import Path

import cv2
import pygame
from gpiozero import AngularServo, Motor
from gpiozero.pins.pigpio import PiGPIOFactory

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


# ================= CONFIG =================
SERVO_PIN = 18
MAX_SPEED = 0.4
THROTTLE_SMOOTHING = 0.05
STEERING_SMOOTHING = 0.05
FRAME_WIDTH = 360
FRAME_HEIGHT = 240
FRAME_RATE = 30
DATASET_ROOT = "dataset"
SESSION_PREFIX = "run_"
IMAGE_DIRNAME = "images"
LABELS_FILENAME = "labels.csv"
SKIP_STATIONARY_FRAMES = True
MIN_SAVE_THROTTLE = 0.05
X_BUTTON_INDEX = 2
os.environ["SDL_AUDIODRIVER"] = "dummy"


def next_session_dir(root_dir):
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith(SESSION_PREFIX):
            suffix = path.name[len(SESSION_PREFIX) :]
            if suffix.isdigit():
                existing.append(int(suffix))
    next_idx = max(existing, default=0) + 1
    session_dir = root / f"{SESSION_PREFIX}{next_idx:03d}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def camera_process(
    frame_queue,
    stop_event,
    start_event,
    recording_event,
    start_time_shared,
    state_lock,
    current_steering,
    current_throttle,
    current_smooth_steer,
    current_smooth_throttle,
    current_state_timestamp,
):
    if Picamera2 is None:
        print(
            "Picamera2 is not installed. Install it on the Pi before collecting data."
        )
        stop_event.set()
        return

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
        controls={"FrameRate": FRAME_RATE},
    )
    picam2.configure(config)
    picam2.start()

    try:
        start_event.wait()
        target_period = 1.0 / FRAME_RATE

        while not stop_event.is_set():
            frame_time = time.perf_counter()
            frame = picam2.capture_array()
            if not recording_event.is_set():
                sleep_time = target_period - (time.perf_counter() - frame_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            elapsed = frame_time - start_time_shared.value

            with state_lock:
                steering = current_steering.value
                throttle = current_throttle.value
                smooth_steer = current_smooth_steer.value
                smooth_throttle = current_smooth_throttle.value
                state_timestamp = current_state_timestamp.value

            # If control data is too old, skip the frame rather than writing
            # a potentially mismatched image/label pair.
            if abs(frame_time - state_timestamp) > 0.1:
                sleep_time = target_period - (time.perf_counter() - frame_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            sample = (
                elapsed,
                frame,
                steering,
                throttle,
                smooth_steer,
                smooth_throttle,
            )

            try:
                frame_queue.put_nowait(sample)
            except queue.Full:
                pass

            sleep_time = target_period - (time.perf_counter() - frame_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        picam2.stop()


def control_process(
    stop_event,
    start_event,
    recording_event,
    start_time_shared,
    state_lock,
    current_steering,
    current_throttle,
    current_smooth_steer,
    current_smooth_throttle,
    current_state_timestamp,
):
    factory = PiGPIOFactory()
    servo = AngularServo(
        SERVO_PIN,
        pin_factory=factory,
        initial_angle=0,
        min_pulse_width=1300 / 1e6,
        max_pulse_width=1900 / 1e6,
    )
    drive_motor = Motor(forward=5, backward=6, enable=19, pin_factory=factory)

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    pygame.joystick.init()

    try:
        controller = pygame.joystick.Joystick(0)
        controller.init()
        print(f"Connected: {controller.get_name()}")

        start_time_shared.value = time.perf_counter()
        start_event.set()
    except pygame.error as exc:
        print(f"Controller not found: {exc}")
        stop_event.set()
        return

    smooth_steering = 0.0
    smooth_throttle = 0.0
    x_button_was_pressed = False

    try:
        while not stop_event.is_set():
            pygame.event.pump()

            x_button_pressed = controller.get_button(X_BUTTON_INDEX)
            if x_button_pressed and not x_button_was_pressed:
                if recording_event.is_set():
                    recording_event.clear()
                    print("Recording stopped.")
                else:
                    start_time_shared.value = time.perf_counter()
                    recording_event.set()
                    print("Recording started.")
            x_button_was_pressed = bool(x_button_pressed)

            steering = controller.get_axis(0)
            if abs(steering) < 0.05:
                steering = 0.0

            fwd = (controller.get_axis(5) + 1) / 2
            rev = (controller.get_axis(2) + 1) / 2
            throttle = (fwd - rev) * MAX_SPEED

            smooth_steering += (steering - smooth_steering) * STEERING_SMOOTHING
            smooth_throttle += (throttle - smooth_throttle) * THROTTLE_SMOOTHING

            with state_lock:
                current_steering.value = steering
                current_throttle.value = throttle
                current_smooth_steer.value = smooth_steering
                current_smooth_throttle.value = smooth_throttle
                current_state_timestamp.value = time.perf_counter()

            servo.value = smooth_steering
            if smooth_throttle > 0.02:
                drive_motor.forward(smooth_throttle)
            elif smooth_throttle < -0.02:
                drive_motor.backward(abs(smooth_throttle))
            else:
                drive_motor.stop()

            time.sleep(0.02)
    finally:
        drive_motor.stop()
        servo.detach()
        pygame.quit()


def writer_process(frame_queue, stop_event, session_dir):
    session_path = Path(session_dir)
    image_dir = session_path / IMAGE_DIRNAME
    image_dir.mkdir(parents=True, exist_ok=True)
    labels_path = session_path / LABELS_FILENAME

    saved_frames = 0
    with labels_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "timestamp",
                "steering",
                "throttle",
                "smooth_steer",
                "smooth_throttle",
            ]
        )

        while not stop_event.is_set() or not frame_queue.empty():
            try:
                elapsed, frame, steering, throttle, smooth_steer, smooth_throttle = (
                    frame_queue.get(timeout=0.5)
                )
            except queue.Empty:
                continue

            if SKIP_STATIONARY_FRAMES and abs(smooth_throttle) < MIN_SAVE_THROTTLE:
                continue

            saved_frames += 1
            frame_name = f"frame_{saved_frames:06d}.jpg"
            frame_path = image_dir / frame_name

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), frame_bgr)
            writer.writerow(
                [
                    frame_name,
                    f"{elapsed:.6f}",
                    f"{steering:.6f}",
                    f"{throttle:.6f}",
                    f"{smooth_steer:.6f}",
                    f"{smooth_throttle:.6f}",
                ]
            )
        f.flush()

    print(f"Saved {saved_frames} frames to {session_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    stop_event = mp.Event()
    start_event = mp.Event()
    recording_event = mp.Event()
    session_dir = next_session_dir(DATASET_ROOT)

    start_time_shared = mp.Value("d", 0.0)
    state_lock = mp.Lock()
    current_steering = mp.Value("d", 0.0)
    current_throttle = mp.Value("d", 0.0)
    current_smooth_steer = mp.Value("d", 0.0)
    current_smooth_throttle = mp.Value("d", 0.0)
    current_state_timestamp = mp.Value("d", 0.0)
    frame_queue = mp.Queue(maxsize=200)

    print(f"Recording session to: {session_dir}")
    print("Press the controller X button to start/stop recording.")

    processes = [
        mp.Process(
            target=camera_process,
            args=(
                frame_queue,
                stop_event,
                start_event,
                recording_event,
                start_time_shared,
                state_lock,
                current_steering,
                current_throttle,
                current_smooth_steer,
                current_smooth_throttle,
                current_state_timestamp,
            ),
        ),
        mp.Process(
            target=control_process,
            args=(
                stop_event,
                start_event,
                recording_event,
                start_time_shared,
                state_lock,
                current_steering,
                current_throttle,
                current_smooth_steer,
                current_smooth_throttle,
                current_state_timestamp,
            ),
        ),
        mp.Process(
            target=writer_process, args=(frame_queue, stop_event, str(session_dir))
        ),
    ]

    for p in processes:
        p.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_event.set()
        for p in processes:
            p.join()
