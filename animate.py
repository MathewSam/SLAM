import imageio
import os
images = []
kargs = { 'duration': 5 }
filenames = os.listdir("Plots/")
for filename in filenames:
    images.append(imageio.imread("Plots/"+filename))
imageio.mimsave('movie.gif', images,'GIF',**kargs)