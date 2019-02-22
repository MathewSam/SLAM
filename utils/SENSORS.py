import os
import math
import numpy as np
from scipy.signal import butter, lfilter

from PIL import Image

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

class IR:
    def __init__(self,folder,prefix,time_stamps):
        '''
        Describes disparity capture model
        Args:
            self: pointer to current instance of the class
            folder: folder with images
            time_stamps: time stamps corresponding to inputs
        '''
        self._file_names = [prefix + str(file_num)+".png" for file_num in range(1,1+len(os.listdir(folder)))]
        self._file_names.sort() # To ensure that files are sorted in order of time steps
        self.time_stamps = time_stamps
        img = Image.open(self._file_names[0])# to set properties of images that are read in
        sample_img = np.array(img.getdata(),  np.uint16).reshape(img.size[1], img.size[0])
        u_max,v_max = sample_img.shape
        self._shape = sample_img.shape
        x_cord = np.arange(0,u_max)
        y_cord = np.arange(0,v_max)
        self.uv = np.stack(np.meshgrid(y_cord,x_cord) + [np.ones((u_max,v_max))]).reshape(3,-1)#Reshape into 3xnum_grid cells

        fsu = 585.05108211
        fsv = 585.05108211
        fstheta = 0
        cu = 242.94140713
        cv = 315.83800193

        pitch = 0.36
        roll = 0
        yaw = 0.021

        R_pitch =  np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
        R_roll = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
        R_yaw =  np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])

        Rcb = np.dot(np.dot(R_yaw,R_pitch),R_roll)#Body to camera coordinates
        pcb = np.array([0.18,0.005,0.36])
        Tcb = [[Rcb[0,0],Rcb[0,1],Rcb[0,2],pcb[0]],[Rcb[1,0],Rcb[1,1],Rcb[1,2],pcb[1]],[Rcb[2,0],Rcb[2,1],Rcb[2,2],pcb[2]],[0,0,0,1]]
        Roc = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])#camera frame to optical 
        Tob = np.dot(Roc,Tcb)#Translation from body to optical frame
        self.Tbo = np.linalg.inv(Tob)#Translation from optical frame to body frame
        self.K_inv = np.linalg.inv(np.array([[fsu,fstheta,cu],[0,fsv,cv],[0,0,1]]))
        


    def __getitem__(self,index):
        '''
        returns value at index for data at location
        Args:
            index: index corresponding to position relative to time
        '''
        img = Image.open(self._file_names[index])
        disparity_img = np.array(img.getdata(),  np.uint16).reshape(img.size[1], img.size[0])
        disp = (-0.00304*disparity_img)+3.31
        disp = disp.reshape(-1,)
        depth = 1.03/(disp)

        uvd = self.uv[:,depth>=0]*depth[depth>=0]
        uv_indices = self.uv[:,depth>=0]# U:column V:row
        disp = disp[depth>=0]

        #Frame transformation
        optical_coordinates = np.dot(self.K_inv,uvd)#returns optical coordinates
        homogenous_coordinates = np.vstack([optical_coordinates,np.ones(optical_coordinates.shape[1])])
        body_coordinates = np.dot(self.Tbo,homogenous_coordinates)

        #Thresholding and keeping only ground points
        uv_indices = uv_indices[:,body_coordinates[2,:]<1.5]
        disp = disp[body_coordinates[2,:]<1.5]
        body_coordinates = body_coordinates[:,body_coordinates[2,:]<1.5]
        

        rgbi = ((uv_indices[0,:]*526.37 + disp*(-4.5*1750.46) + 19276)/585.051).astype(np.uint16)
        rgbj = ((uv_indices[1,:]*526.37 + 16662.0)/585.051).astype(np.uint16)
        indices = np.logical_and(rgbi<self._shape[1],rgbj<self._shape[0])
        rgbi = rgbi[indices]
        rgbj = rgbj[indices]
        body_coordinates = body_coordinates[:,indices]
        return rgbi,rgbj,body_coordinates
    
class Camera:
    def __init__(self,folder,prefix,time_stamps):
        '''
        Describes disparity capture model
        Args:
            self: pointer to current instance of the class
            folder: folder with images
            time_stamps: time stamps corresponding to inputs
        '''
        self._file_names = [prefix + str(file_num)+".png" for file_num in range(1,1+len(os.listdir(folder)))]
        #self._file_names.sort() # To ensure that files are sorted in order of time steps
        self.time_stamps = time_stamps
        img = Image.open(self._file_names[0])# to set properties of images that are read in
        self._shape =  (img.size[1], img.size[0])

    def __getitem__(self,index):
        '''
        returns value at index for data at location
        Args:
            index: index corresponding to position relative to time
        '''
        img = Image.open(self._file_names[index])
        data = np.array(img)
        return data
    
    @property
    def shape(self):
        img = Image.open(self._file_names[0])
        data = np.array(img)
        return data.shape