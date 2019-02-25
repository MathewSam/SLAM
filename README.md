# SLAM
Implement simultaneous localization and mapping (SLAM) using odometry, inertial, 2-D laser
range, and RGBD measurements from a differential-drive robot. Use the IMU, odometry, and laser
measurements to localize the robot and build a 2-D occupancy grid map of the environment. Use the
RGBD information to color the floor of your 2-D map.

## Goals
The goal of the projects is to use an occupancy grid for mapping and a particle filter with a laser-grid correlation model for localization.
- **MAPPING**:Try mapping from the first scan and plot the map to make sure your transforms are
correct before you start estimating the robot pose
- **PREDICTION**:Implement a prediction-only particle filter at first. In other words, use the wheel
odometry and the yaw gyro measurements to estimate the robot trajectory and build a 2-D map
based on this estimate before correcting it with the laser readings.
- **UPDATE**:Once the prediction-only filter works, include an update step that uses scan matching
to correct the robot pose. Remove scan points that are too close, too far, or hit the ground.
- **TEXTURE MAP**:Project colored points from the RGBD sensor onto your occupancy grid in order
to color the floor. Find the ground plane in the transformed data via thresholding on the height
## Libraries used:
- numpy
- matplotlib
- tqdm(for visualizing the progress bar)
- scipy
- math
- argparser

## Files
### SLAM.py
This is the main file in this repo. This file is responsible for synchronizing the timing data from the different sensors. Specifically, the data is loaded into classes designed in the utils folder of the project. The SLAM class which handles the synchronizations and the calls to the particle filter for particle prediction and update is initialized. The default parameters for grid size is xmin=-40, xmax =40,ymin =-40 and ymax =40. To run this code, simply type into command line 

usage: SLAM.py [-h] dataset_id num_particles num_eff Show_texture

positional arguments:
  dataset_id     Expects a number associated with the datastream required. For the code to work the Encoders, IMU and Hokuyo data for dataset id must be present in the current working directory
  num_particles  Indicates number of particles for SLAM
  num_eff        number of effective samples for resampling
  Show_texture   Expects a boolean value to indicate whether to do texture mapping or not

