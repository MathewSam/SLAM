import os
import math
import numpy as np
from scipy.signal import butter, lfilter

from skimage import color,io

class LIDAR:
    def __init__(self,lidar_angle_min,lidar_angle_max,lidar_angle_increment,lidar_range_min,lidar_range_max,lidar_ranges,lidar_stamps,translation):
        '''
        Initializes class to handle LIDAR measurements
        Args:
        ----
            self:pointer to current instance of the class
            lidar_angle_min:start angle of the scan [rad]
            lidar_angle_max:end angle of the scan [rad]
            lidar_angle_increment:angular distance between measurements [rad]
            lidar_range_min:minimum range value [m]
            lidar_range_max:maximum range value [m]
        '''
        angles = np.arange(lidar_angle_min,lidar_angle_max + lidar_angle_increment,lidar_angle_increment,dtype=np.float64)[:1081]
        range_max = lidar_range_max
        range_min = lidar_range_min
        
        self.ranges = lidar_ranges
        self.time_stamps = lidar_stamps

        self._LIDAR_stats = {"angle_span":angles,"range_max":range_max,"range_min":range_min,"translation":translation}

    @property
    def LIDAR_stats(self):
        '''
        Returns LIDAR stats
        Args:
        ----
            self:pointer to current instance of the class 
        Returns:
        ------
            LIDAR_stats:statistics associated with LIDAR readings
        '''
        return self._LIDAR_stats
    
    def __getitem__(self,index):
        '''
        Design an iterator for efficient generation of laser scan ranges and time_stamps. 
        Args:
        ---
            self:pointer to current instance of the class
        Returns:
        -------
            self:pointer to current instance of the class
        '''
        ranges = self.ranges[:,index]
        indValid = np.logical_and((ranges <= self._LIDAR_stats["range_max"]),(ranges >= self._LIDAR_stats["range_min"]))
        
        ranges = ranges[indValid]
        angles = self._LIDAR_stats["angle_span"][indValid]

        x = ranges*np.cos(angles) + self._LIDAR_stats["translation"][0]
        y = ranges*np.sin(angles) + self._LIDAR_stats["translation"][1]
        Homogenous = np.stack([x, y,np.zeros_like(x),np.ones_like(x)], axis=0)
        return Homogenous

    def __len__(self):
        '''
        Number of time stamps stored from LIDAR
        params:
            self : pointer to current instance of the class
        '''
        return self.time_stamps.shape[0]

class IMU:
    def __init__(self,angular_velocity,time_stamps):
        '''
        Initializes class to handle IMU measurements. Linear acceleration is ignored.
        params:
            angular_velocity : angular velocity at each time step
            time_stamps : time step 
        '''

        self.omega = angular_velocity[2,:]
        self.time_stamps = time_stamps
        fs = 1/(time_stamps[1]-time_stamps[0])
        fc = 10
        B, A = butter(1, fc / (fs / 2), btype='low') 
        self.omega = lfilter(B, A, self.omega, axis=0)
        #self.omega = (self.omega/180)*math.pi


    def __getitem__(self,index):
        '''
        Binding together the time stamp and angular velocity at a time step
        params:
            self : pointer to current instance of class
            index : index of data to extract
        returns:
            yaw,time_stamp: yaw velocity at index, time_stamp at index 
        '''
        return self.omega[index]

    def __len__(self):
        '''
        Number of time stamps stored from IMU
        params:
            self : pointer to current instance of the class
        '''
        return self.time_stamps.shape[0]

class Encoder:
    def __init__(self,encoder_counts,time_stamps,meters_per_tick=0.022):
        '''
        Initializes class to handle IMU measurements. Linear acceleration is ignored.
        params:
            encoder_counts : number of counts
            time_stamps : time step 
        '''
        left_movement = 0.0022*(encoder_counts[0,:] + encoder_counts[2,:])/2
        right_movement = 0.0022*(encoder_counts[1,:] + encoder_counts[3,:])/2
        self.avg_displacement = (left_movement + right_movement)/2
        self.time_stamps = time_stamps

    def __getitem__(self,index):
        '''
        Binding together the time stamp and angular velocity at a time step
        params:
            self : pointer to current instance of classclass
            index : index of data to extract
        returns:
            yaw,time_stamp: yaw velocity at index, time_stamp at index 
        '''
        return self.avg_displacement[index]

    def __len__(self):
        '''
        Number of time stamps stored from Encder
        params:
            self : pointer to current instance of the class
        '''
        return self.time_stamps.shape[0]

class Camera:
    def __init__(self):
        '''
        Describes camera capture model
        '''