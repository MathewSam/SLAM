import numpy as np 
from scipy.stats import multivariate_normal
import math
def mapCorrelation(im, x_im, y_im, vp, xs, ys):
    '''
    INPUT 
    im              the map 
    x_im,y_im       physical x,y positions of the grid map cells
    vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
    xs,ys           physical x,y,positions you want to evaluate "correlation" 

    OUTPUT 
    c               sum of the cell values of all the positions hit by range sensor
    '''
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = x_im[0]
    xmax = x_im[-1]
    xresolution = (xmax-xmin)/(nx-1)
    ymin = y_im[0]
    ymax = y_im[-1]
    yresolution = (ymax-ymin)/(ny-1)
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys))
    for jy in range(0,nys):
        y1 = vp[1,:] + ys[jy] # 1 x 1076
        iy = np.int16(np.round((y1-ymin)/yresolution))
        for jx in range(0,nxs):
            x1 = vp[0,:] + xs[jx] # 1 x 1076
            ix = np.int16(np.round((x1-xmin)/xresolution))
            valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), np.logical_and((ix >=0), (ix < nx)))
            cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
    return cpr

class ParticleFilter:
    def __init__(self,num_particles,threshold):
        '''
        Initializes particles
        params:
            self : pointer to current instance of the class
            num_particles : number of particles
            threshold : value to resample at 
        '''
        self.particles = np.zeros((num_particles,3))
        self.particle_weight = np.ones((num_particles,1))/num_particles
        self.threshold = threshold


    def particle_prediction(self,displacement,angle_shift,var=0.001):
        '''
        Particle prediction from Encoder and IMU angle values
        params:
            displacement : displacement from encoder
            angle_shift : angle shift from IMU
        kwargs:
            var : variance of noise 
        '''
        noise = np.random.randn(self.particles.shape[0],self.particles.shape[1])
        
        if angle_shift!=0:
            self.particles[:,0] = self.particles[:,0] + (displacement*np.sin(angle_shift/2)*(np.cos(self.particles[:,-1] + (angle_shift/2)))/(angle_shift/2)) 
            self.particles[:,1] = self.particles[:,1] + (displacement*np.sin(angle_shift/2)*(np.sin(self.particles[:,-1] + (angle_shift/2)))/(angle_shift/2)) 
            self.particles[:,2] = self.particles[:,2] + angle_shift
        else:
            self.particles[:,0] = self.particles[:,0] + displacement*np.cos(self.particles[:,-1])
            self.particles[:,1] = self.particles[:,1] + displacement*np.sin(self.particles[:,-1])

        self.particles = self.particles + noise*var
        

    def particle_update(self,LIDAR_reading,Obsmodel):
        '''
        Particle filter update for filter weights using readings from LIDAR
        params:
            self : pointer to current instance of the class
            LIDAR_reading : Homographic input from LIDAR module.
            Obsmodel : observation model
        '''
        occ_grid = Obsmodel.occupancy_map#map to be fed into map correlation
        grid_stats = Obsmodel.grid_stats

        ph = np.zeros_like(self.particle_weight)
        x_im = np.arange(grid_stats["minx"],grid_stats["maxx"] + grid_stats["res"],grid_stats["res"])
        y_im = np.arange(grid_stats["miny"],grid_stats["maxy"] + grid_stats["res"],grid_stats["res"])

        for i in range(self.particles.shape[0]):
            x_range = np.arange(-0.2,0.2+0.05,0.05)
            y_range = np.arange(-0.2,0.2+0.05,0.05)
            corr = mapCorrelation(occ_grid,x_im,y_im,LIDAR_reading,x_range,y_range)
            ph[i] = np.max(corr)
            location = np.unravel_index(corr.argmax(), corr.shape)
            self.particles[i,0] = self.particles[i,0] + (location[0] - 4)*grid_stats["res"]
            self.particles[i,1] = self.particles[i,1] + (location[1] - 4)*grid_stats["res"]

        ph = np.exp(ph - np.max(ph))
        ph = ph/np.sum(ph)            
        
        self.particle_weight = ph*self.particle_weight
        self.particle_weight = self.particle_weight/np.sum(self.particle_weight)

    def resample(self):
        '''
        '''
        j = 0
        c = self.particle_weight[0]
        new_particles = self.particles.copy()
        num_particles = self.particles.shape[0]
        for k in range(num_particles):
            u = np.random.uniform(0,1/num_particles)
            beta = u + (k/num_particles)
            while beta > c:
                j = j + 1
                c = c + self.particle_weight[j]
            new_particles[k,:] = self.particles[j,:]
        self.particles = new_particles
        self.particle_weight = np.ones((num_particles,1))/num_particles

    @property
    def most_likely(self):
        return self.particles[np.argmax(self.particle_weight),:]

    @property
    def resample_condn(self):
        '''
        '''
        return (1/np.sum(np.square(self.particle_weight)))<self.threshold


