import numpy as np 
import matplotlib.pyplot as plt
from tqdm import trange
from matplotlib import animation

from utils.SENSORS import LIDAR,IMU,Encoder,Kinect 
from utils.MAPPING import ObservationModel
from utils.PARTICLE_FILTER import ParticleFilter
class SLAM:
    def __init__(self,particle_filter,Obs_model):
        '''
        Initializes the class that handles slam
        params:
            self : pointer to current instance of the class
            particle_filter : particle filter initialized at time 0
            Obs_model : Observation model corresponding to particle filter
        '''
        self.particle_filter = particle_filter
        self.Obs_model = Obs_model

        
    def __call__(self,LIDAR,IMU,ENCODER):
        '''
        '''
        lidar_tracking = LIDAR.time_stamps
        imu_tracking = IMU.time_stamps
        encoder_tracking = ENCODER.time_stamps
        x = []
        y = []

        for i in trange(lidar_tracking.shape[0]):
            if i==0:
                scan = LIDAR[0]
                state = self.particle_filter.most_likely
                occ_grid,_ =  self.Obs_model.generate_map(state,scan)
            else:
                scan = LIDAR[i]
                imu_indices = np.logical_and(imu_tracking>=lidar_tracking[i-1],imu_tracking<lidar_tracking[i])               
                if np.sum(imu_indices)==0:
                    angle_shift=0
                else:
                    angle_shift = np.mean(IMU[imu_indices])*(lidar_tracking[i] - lidar_tracking[i-1])

                encoder_indices = np.logical_and(encoder_tracking>=lidar_tracking[i-1],encoder_tracking<lidar_tracking[i])
                displacement = np.sum(ENCODER[encoder_indices])

                if displacement!=0 or angle_shift!=0:
                    self.particle_filter.particle_prediction(displacement,angle_shift)
                    self.particle_filter.particle_update(scan,self.Obs_model)
                
                state = self.particle_filter.most_likely
                occ_grid,_ =  self.Obs_model.generate_map(state,scan)
                x = x + [np.floor((state[0] - self.Obs_model.grid_shift_vector[0])/self.Obs_model.grid_stats["res"]).astype(np.uint16)]
                y = y + [np.floor((state[1] - self.Obs_model.grid_shift_vector[1])/self.Obs_model.grid_stats["res"]).astype(np.uint16)]
            if i%10==0:
                plt.scatter(x,y,s=0.25,c='r')
                plt.imshow(occ_grid,cmap='gray')
                plt.savefig("Plots/{}.png".format(i))
                plt.close()
                

       

if __name__ == '__main__':
    dataset = 20

    with np.load("Encoders%d.npz"%dataset) as data:
        encoder_counts = data["counts"] # 4 x n encoder counts
        encoder_stamps = data["time_stamps"] # encoder time stamps
        encoder_reader = Encoder(encoder_counts,encoder_stamps)
    with np.load("Hokuyo%d.npz"%dataset) as data:
        lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
        lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
        lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
        lidar_range_min = data["range_min"] # minimum range value [m]
        lidar_range_max = data["range_max"] # maximum range value [m]
        lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
        lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
        translation = np.array([0,.150,0]) #relative shift of LIDAR wrt center of robot
        Hokuyo_reader = LIDAR(lidar_angle_min,lidar_angle_max,lidar_angle_increment,lidar_range_min,lidar_range_max,lidar_ranges,lidar_stamps,translation)

    with np.load("Imu%d.npz"%dataset) as data:
        imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
        imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
        imu_reader = IMU(imu_angular_velocity,imu_stamps)

    with np.load("Kinect%d.npz"%dataset) as data:
        disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
        rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images
        RGB_folder = "dataRGBD/RGB{}/".format(dataset)
        disp_folder = "dataRGBD/Disparity{}/".format(dataset)
        texture = Kinect(RGB_folder,disp_folder,rgb_stamps,disp_stamps)


    Obs_model = ObservationModel(30,-30,30,-30,0.05,p11=0.8)
    PF = ParticleFilter(100,2)
    robot = SLAM(PF,Obs_model)
    robot(Hokuyo_reader,imu_reader,encoder_reader)
