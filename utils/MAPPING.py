'''
Explains the observation model
'''
import numpy as np
import matplotlib.pyplot as plt 

class ObservationModel:
    def __init__(self,max_right,max_left,max_front,max_behind,res,p11=0.75,p00=0.75):
        '''
        Initializes observation model. This consists of the occupancy grid( a boolean array displaying filled and empty grid cells) and the 
        logodds map(a probability based representation which indicates how likely a cell is filled or not)
        Args:
        ----
            max_right:maximum distance to the right of the 0,0 particle 
            max_left:maximum distance to the left of the 0,0 particle 
            max_front:maximum distance to the front of the 0,0 particle 
            max_behind:maximum distance behind the 0,0 particle 
            res:number of grid cells per unit distance
        Kwargs:
        ------
            p11: probability of grid cell being occupied when it is measured as occupied
            default_value:0.75
            p00: probability of grid cell being free when it is measured as free
            default_value:0.75
        ''' 
        assert max_behind<=0,"Please input as in coordinate frame"
        assert max_front>=0,"Please input as in coordinate frame"
        assert max_right>=0,"Please input as in coordinate frame"
        assert max_left<=0,"Please input as in coordinate frame"
        assert p11>=0 and p11<=1,"Please input probability that makes sense"
        assert p00>=0 and p00<=1,"Please input probability that makes sense"

        self._grid_stats = {"minx":max_left,"miny":max_behind,"maxx":max_right,"maxy":max_front,"res":res}

        self._p11 = p11
        self._p00 = p00 
        self._p01 = 1 - p11
        self._p10 = 1 - p00

        self._shape = (int((self._grid_stats["maxx"] - self._grid_stats["minx"])//self._grid_stats["res"]) + 1,int((self._grid_stats["maxy"] - self._grid_stats["miny"])//self._grid_stats["res"]) + 1)
        self.occupancy_map = np.zeros(self._shape,dtype=np.uint8)
        self.logodds_map = np.zeros(self._shape,dtype=np.float64)

    def generate_map(self,ranges,LIDAR_stats,robot_state,display_map=True):
        '''
        Initializes the map from the first range reading from LIDAR
        params:
            self : pointer to current class instance
            ranges : first LIDAR reading in the form of ranges
            LIDAR_stats : dictionary of LIDAR properties
            robot_state : most likely robot state(x,y,yaw)
        kwargs:
            display_map : option to display map
            default_value : True
        '''
        angle_span = LIDAR_stats["angle_span"]
        range_max = LIDAR_stats["range_max"]
        range_min = LIDAR_stats["range_min"]

        indValid = np.logical_and((ranges < range_max),(ranges> range_min))

        ranges = ranges[indValid]#Removing all unlikely readings/noise
        angles = angle_span[indValid]#Removing all unlikely readings/noise

        #Converting range reading to xy coordinate reading where the bot is assumed to be at the origin
        xs0 = ranges*np.cos(angles)
        ys0 = ranges*np.sin(angles)
        zs0  = np.zeros_like(angles)
        coordinates = np.stack([xs0,ys0,zs0],axis=1)# in LIDAR Frame

        #Converting xy cordinates to grid coordinates
        xis = np.ceil((xs0 - self._grid_stats['minx'])/self._grid_stats['res']).astype(np.int16)-1
        yis = np.ceil((ys0 - self._grid_stats['miny'])/self._grid_stats['res']).astype(np.int16)-1

        #Filling occupancy grid
        indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < self._shape[0])), (yis < self._shape[1]))
        self.occupancy_map[xis[indGood[0]],yis[indGood[0]]]=1
        self.logodds_map = self.occupancy_map*np.log(self._p11/self._p10) + (1-self.occupancy_map)*np.log(self._p00/self._p01)
        if display_map==True:
            plt.figure()
            plt.title("Initialized Map")
            plt.imshow(self.occupancy_map,cmap="hot")
            #plt.xticks([])
            #plt.yticks([])
            plt.show()
        return self.occupancy_map
    
    @property
    def shape(self):
        '''
        Returns shape of maps
        Args:
            self:pointer to current instance of the class
        Returns:
            shape:shape of occupancy grid
        '''
        return self._shape
    
    @property
    def grid_stats(self):
        '''
        Returns properties of map
        Args:
            self:pointer to current instance of the class
        Returns:
            grid_stats: statistics of grid
        '''
        return self._grid_stats

    @property
    def p11(self):
        '''
        Returns probability of grid cell being occupied when it is measured as occupied
        Args:
            self:pointer to current instance of the class
        Returns:
            p11:probability of grid cell being occupied when it is measured as occupied
        '''
        return self._p11

    @property
    def p00(self):
        '''
        Returns probability of grid cell being free when it is measured as free
        Args:
            self:pointer to current instance of the class
        Returns:
            p11:probability of grid cell being free when it is measured as free
        '''
        return self._p00