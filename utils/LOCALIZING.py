import numpy as np 
import matplotlib.pyplot as plt 

class MotionModel:
    def __init__(self,num_particles):
        '''
        params:
            self: pointer to current instance of the class
            num_particles: number of particles
        ''' 
        self.particles = np.zeros((num_particles,3))
        self.particle_weight = np.ones((num_particles,1))/num_particles