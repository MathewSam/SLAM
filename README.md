# SLAM
Implement simultaneous localization and mapping (SLAM) using odometry, inertial, 2-D laser
range, and RGBD measurements from a differential-drive robot. Use the IMU, odometry, and laser
measurements to localize the robot and build a 2-D occupancy grid map of the environment. Use the
RGBD information to color the floor of your 2-D map.

**PLEASE extract RGBD.zip for kinect data

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

>>> python SLAM.py 

 arguments to change in SLAM.py for different conditions:</br>
  |dataset_id     |Expects a number associated with the datastream required. For the code to work the Encoders, IMU and Hokuyo data for dataset id must be present in the current working directory</br>
  |num_particles  |Indicates number of particles for SLAM</br>
  |num_eff        |number of effective samples for resampling</br>
  |Show_texture   |Expects a boolean value to indicate whether to do texture mapping or not</br>



Note the files pertaining to encoder, imu , lidar and kinect must be in the same working directory. This file uses all the other files in this project to implement SLAM. Please also create a folder called plots in the working directory to save the plots.

### load_data.py
Sample file provided to show how the data is loaded in the files

### utils/SENSORS.py
This file encapsulates all the classes to load the data providing indexing mechanisms to index sensor data that is processed to provide the specific information that is needed by the SLAM class. Documentation for all classes are attached with the code

### utils/MAPPING.py
This file contains the Observation model class that is a wrapper for the occupancy grid, log odds map and the texture map. Documentation for all the files should be provided with the functions.

### utils/PARTICLE_FILTER.py
This file contains the functions for particle prediction and update after providing necessa 


