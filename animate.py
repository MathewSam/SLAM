import imageio
import os
images = []
filenames = os.listdir("Plots/")
filenames = list(map(lambda x:int(x[:-4]),filenames))
filenames.sort()
for filename in filenames:
    images.append(imageio.imread("Plots/{}.png".format(filename)))
imageio.mimsave('movie.gif', images)