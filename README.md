# SUDO_CAR
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
