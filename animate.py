import imageio
import os
images = []
filenames = os.listdir("Plots/")
for filename in filenames:
    images.append(imageio.imread("Plots/"+filename))
imageio.mimsave('movie.gif', images)