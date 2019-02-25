'''
Explains the observation model
'''
import numpy as np
import matplotlib.pyplot as plt 
import cv2


def bresenham2D(sx, sy, ex, ey):
    '''
    Bresenham's ray tracing algorithm in 2D.
    Inputs:
        (sx, sy)	start point of ray
        (ex, ey)	end point of ray
    '''
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex-sx)
    dy = abs(ey-sy)
    steep = abs(dy)>abs(dx)
    if steep:
        dx,dy = dy,dx # swap 

    if dy == 0:
        q = np.zeros((dx+1,1))
    else:
        q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
    if steep:
        if sy <= ey:
            y = np.arange(sy,ey+1)
        else:
            y = np.arange(sy,ey-1,-1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx,ex+1)
        else:
            x = np.arange(sx,ex-1,-1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
    return np.vstack((x,y))


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
        self._p01 = 1 - p11
        self._odds_update = np.log(p11/(1-p11))

        self._shape = (int((self._grid_stats["maxx"] - self._grid_stats["minx"])//self._grid_stats["res"]) + 1,int((self._grid_stats["maxy"] - self._grid_stats["miny"])//self._grid_stats["res"]) + 1)
        self._grid_shift_vector = np.array([self._grid_stats['minx'],self._grid_stats['miny']])

        self.occupancy_map = np.zeros(self._shape,dtype=np.uint8)
        self.logodds_map = np.zeros(self._shape,dtype=np.float64)
        self.texture_map = np.zeros((self._shape[0],self._shape[1],3))

    def generate_map(self,particle,LIDAR_reading):
        '''
        Generate map for a specific LIDAR readings
        params:
            self : pointer to current instance of the class
            particle : most likely particle 
            LIDAR_readings : Homogenous map from current LIDAR reading
        '''
        wTb = np.array([[np.cos(particle[-1]),-np.sin(particle[-1]),0,particle[0]],[np.sin(particle[-1]),np.cos(particle[-1]),0,particle[1]],[0,0,1,0],[0,0,0,1]])
        map_coordinates = np.floor((np.dot(wTb,LIDAR_reading)[:2,:] - self._grid_shift_vector.reshape(-1,1))/self._grid_stats["res"]).astype(np.uint16)#Transfer points into world co ordinates
        beam_source = np.floor((np.array([particle[0],particle[1]]) - self._grid_shift_vector)/self._grid_stats["res"]).astype(np.uint16)
        beam_ends = map_coordinates.T.tolist()
        for beam_end in beam_ends:
            scans = bresenham2D(beam_source[0],beam_source[1],beam_end[0],beam_end[1]).astype(np.uint16)
            self.logodds_map[scans[1][-1],scans[0][-1]] = self.logodds_map[scans[1][-1],scans[0][-1]] + self._odds_update
            self.logodds_map[scans[1][1:-1],scans[0][1:-1]] = self.logodds_map[scans[1][1:-1],scans[0][1:-1]] - self._odds_update
        self.logodds_map[scans[1][0],scans[0][0]] = self.logodds_map[scans[1][0],scans[0][0]] - self._odds_update

        P_occupied = 1/(1 + np.exp(-self.logodds_map))
        self.occupancy_map = (P_occupied>=0.95)*(1) + (P_occupied<0.01)*(-1)
        return self.occupancy_map,self.logodds_map

    def generate_texture(self,particle,texture,body_frame):
        '''
        Generate map for a specific LIDAR readings
        params:
            self : pointer to current instance of the class
            particle : most likely particle 
            LIDAR_readings : Homogenous map from current LIDAR reading
        '''
        wTb = np.array([[np.cos(particle[-1]),-np.sin(particle[-1]),0,particle[0]],[np.sin(particle[-1]),np.cos(particle[-1]),0,particle[1]],[0,0,1,0],[0,0,0,1]])
        map_coordinates = np.floor((np.dot(wTb,body_frame)[:2,:] - self._grid_shift_vector.reshape(-1,1))/self._grid_stats["res"]).astype(np.uint16)#Transfer points into world co ordinates
        indices = np.logical_and(map_coordinates[1,:]<self.texture_map.shape[0],map_coordinates[0,:]<self.texture_map.shape[1])
        map_coordinates = map_coordinates[:,indices]
        self.texture_map[map_coordinates[1,:],map_coordinates[0,:],:] = texture/255
        P_occupied = 1/(1 + np.exp(-self.logodds_map))
        self.texture_map[:,:,0] = self.texture_map[:,:,0]*(P_occupied<0.05)
        self.texture_map[:,:,1] = self.texture_map[:,:,1]*(P_occupied<0.05)
        self.texture_map[:,:,2] = self.texture_map[:,:,2]*(P_occupied<0.05)
        return self.texture_map

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
    def grid_shift_vector(self):
        '''
        Returns properties of map
        Args:
            self:pointer to current instance of the class
        Returns:
            grid_stats: statistics of grid
        '''
        return self._grid_shift_vector

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
