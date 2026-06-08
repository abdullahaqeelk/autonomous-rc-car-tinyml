from collections import deque
import os

import cv2
import numpy as np
import pygame
from ai_edge_litert.interpreter import Interpreter
from gpiozero import AngularServo, Motor
from gpiozero.pins.pigpio import PiGPIOFactory
from picamera2 import Picamera2

import params


MODEL_PATH = "model_recommended.tflite"
CAMERA_SIZE = (360, 240)
SERVO_PIN = 18
MAX_SPEED = 0.4
THROTTLE_DEADBAND = 0.02
THROTTLE_SMOOTHING = 0.05
X_BUTTON_INDEX = 0

# Steering smoothing only. Throttle remains manual.
BASE_ALPHA = 0.12
TURN_ALPHA_GAIN = 0.38
STACK_SIZE = 1

os.environ["SDL_AUDIODRIVER"] = "dummy"


def preprocess_frame(frame):
    cropped = frame[params.crop_y1 : params.crop_y2, :]
    resized = cv2.resize(cropped, (params.inputres[0], params.inputres[1]))
    return resized


def get_adaptive_alpha(raw_steering):
    turn_strength = min(abs(raw_steering), 1.0)
    return BASE_ALPHA + (TURN_ALPHA_GAIN * turn_strength)


def main():
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    inp_idx = input_details["index"]
    out_idx = output_details["index"]
    inp_dtype = input_details["dtype"]

    frame_buffer = deque(maxlen=STACK_SIZE)
    last_smoothed_steering = 0.0
    smooth_throttle = 0.0

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": CAMERA_SIZE}
    )
    picam2.configure(config)
    picam2.start()

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
    except pygame.error as exc:
        picam2.stop()
        servo.detach()
        drive_motor.close()
        pygame.quit()
        raise RuntimeError(f"Controller not found: {exc}") from exc

    print("Autonomous steering active.")
    print(f"Model: {MODEL_PATH}")
    print(f"Connected: {controller.get_name()}")
    print("Throttle control: manual")
    print("Press the controller X button to stop.")

    try:
        while True:
            pygame.event.pump()

            if controller.get_button(X_BUTTON_INDEX):
                print("Stopping autonomous steering...")
                break

            frame = picam2.capture_array()
            frame = preprocess_frame(frame)
            frame_buffer.append(frame)

            if len(frame_buffer) < STACK_SIZE:
                continue

            model_input = np.expand_dims(frame_buffer[-1].astype(inp_dtype), axis=0)

            interpreter.set_tensor(inp_idx, model_input)
            interpreter.invoke()
            raw_steering = float(interpreter.get_tensor(out_idx)[0][0])

            alpha = get_adaptive_alpha(raw_steering)
            smoothed_steering = (
                alpha * raw_steering + (1.0 - alpha) * last_smoothed_steering
            )
            last_smoothed_steering = smoothed_steering

            fwd = (controller.get_axis(5) + 1) / 2
            rev = (controller.get_axis(2) + 1) / 2
            throttle = (fwd - rev) * MAX_SPEED
            smooth_throttle += (throttle - smooth_throttle) * THROTTLE_SMOOTHING

            servo.value = max(min(smoothed_steering, 1.0), -1.0)
            if smooth_throttle > THROTTLE_DEADBAND:
                drive_motor.forward(smooth_throttle)
            elif smooth_throttle < -THROTTLE_DEADBAND:
                drive_motor.backward(abs(smooth_throttle))
            else:
                drive_motor.stop()

            print(
                f"Steering raw={raw_steering:+.3f} "
                f"smoothed={smoothed_steering:+.3f} alpha={alpha:.2f} "
                f"throttle={smooth_throttle:+.3f}"
            )

    except KeyboardInterrupt:
        print("Stopping autonomous steering...")
    finally:
        drive_motor.stop()
        drive_motor.close()
        servo.detach()
        pygame.quit()
        picam2.stop()


if __name__ == "__main__":
    main()
