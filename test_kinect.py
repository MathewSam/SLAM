import numpy as np
from utils.SENSORS import Camera,IR
import matplotlib.pyplot as plt 


if __name__ == '__main__':
    dataset = 20

    with np.load("Kinect%d.npz"%dataset) as data:
        disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
        rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images
        RGB_folder = "dataRGBD/RGB{}/".format(dataset)
        disp_folder = "dataRGBD/Disparity{}/".format(dataset)
        cam = Camera(RGB_folder,rgb_stamps)
        ir = IR(disp_folder,disp_stamps)

        #plt.figure(1)
        #plt.imshow(cam[0])
        #plt.show()
        
        optical_coord = ir[0]
        print(optical_coord.shape)
        #print(ir.shape)
        #print(cam.shape)