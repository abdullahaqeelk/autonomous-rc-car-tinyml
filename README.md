# Autonomous RC Car — End-to-End TinyML Steering on a Raspberry Pi Zero 2W

## Authors

- **Mohammad Usman Irshad** ([usman-irshad1](https://github.com/usman-irshad1))
- **Abdullah Aqeel Khan** ([abdullahaqeelk](https://github.com/abdullahaqeelk))

Developed as a group project for EC-310, Microprocessor and Microcontroller Based Design, at the College of E&ME, NUST. The project report also credits Muhammad Husnain and Fatima Sohail as group members.

The car uses a PilotNet CNN running on a Raspberry Pi Zero 2W to predict steering from camera images. Training runs on a PC, and the resulting TensorFlow Lite model runs on the car. The operator controls throttle with a game controller.

## Assembled vehicle

| Top view | Front and camera view | Side isometric view |
| --- | --- | --- |
| ![Top view showing the vehicle and electronics](docs/images/vehicle-top.png) | ![Front view showing the camera mount](docs/images/vehicle-front.png) | ![Side isometric view of the RC car](docs/images/vehicle-isometric.png) |

Vehicle photographs extracted from Figures 7-9 on page 22 of the [project report](REPORT.pdf).

## System architecture

![Data collection, offline training, and autonomous inference architecture](docs/images/system-architecture.jpg)

Figure 2 from page 8 of the [project report](REPORT.pdf) shows the three phases of the system. The report uses earlier script names. In this repository, data collection uses data_collection.py, training uses train.py and pilotnet.py, and driving uses autonomous_drive.py.

![Standard robotics control pipeline and end-to-end learning approach](docs/images/control-approaches.jpg)

Figure 1 from page 7 of the [project report](REPORT.pdf) compares the standard robotics control pipeline with the end-to-end learning approach used for steering.

## Hardware

| Component | Report figure |
| --- | --- |
| Raspberry Pi Zero 2W for camera capture and inference | ![Raspberry Pi Zero 2W](docs/images/raspberry-pi-zero-2w.jpg) |
| N20 micro gear motor | ![N20 motor with metal gearbox](docs/images/n20-motor.jpg) |
| L298N dual H-bridge motor driver | ![L298N motor driver module](docs/images/l298n-driver.jpg) |
| Raspberry Pi Camera Module v1.3 | ![Raspberry Pi Camera Module v1.3](docs/images/camera-module.jpg) |

Hardware figures extracted from Figures 3-6 on pages 16-19 of the [project report](REPORT.pdf).

## Running the project

AUTONOMOUS CAR USING NVIDIA PILOTNET

STEP 1:
      install rasberry os lite from rasberry imager, enable ssh while installing.
      connect your pc with pi through ssh.
      install opencv, picamera2, pygame, pigpiod-daemon and ai-edge-litert in  virtual enviourment of python.(these are for pi)
STEP 2:
      run data_collection.py on pi and record dataset.
      Dataset format should look like this:
                  dataset-->
                              run_001-->
                                          images
                                          labels.csv
                              run_002-->
                                          images
                                          labels.csv
                              run_003-->
                              ##
                              ##
      Note: keep validation data outside the dataset folder and update the validation data path in params.py
STEP 3:
      import dataset to pc and run teat_crop on pc to test and crop unwanted areas in images.
      make sure only path comes in camera frame.
      After getting the crop values update the params.py according to it.
STEP 4:
      run train.py after that  run validation.py to see that how much your model is close to the real drive.

STEP 5:
      copy the #####.tflite file on the pi where autonomous_drivw.py lives.
STEP 6
      run autonomous_drive.py.
      controll throttle manually.
