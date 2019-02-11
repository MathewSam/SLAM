import numpy as np

class LIDAR:
    def __init__(self,lidar_angle_min,lidar_angle_max,lidar_angle_increment,lidar_range_min,lidar_range_max,lidar_ranges,lidar_stamps):
        '''
        Initializes class to handle LIdar measurements
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

        self._LIDAR_stats = {"angle_span":angles,"range_max":range_max,"range_min":range_min}

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
        Design an iterator for efficient generation of laser scan ranges and time_stamps. Returns object with next function
        Args:
        ---
            self:pointer to current instance of the class
        Returns:
        -------
            self:pointer to current instance of the class
        '''
        return self.ranges[:,index],self.time_stamps[index]


