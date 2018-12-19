import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams.update({'font.size': 14})


x = np.load("pts_3d/x.npy")
y = np.load("pts_3d/y.npy")
z = np.load("pts_3d/z.npy")

x1 = np.load("pts_3d/x1.npy")
y1 = np.load("pts_3d/y1.npy")
z1 = np.load("pts_3d/z1.npy")

fig = plt.figure()
ax = Axes3D(fig)

mean1, mean2, mean3, mean4 = 0, 0, 0, 0
ecart1, ecart2, ecart3, ecart4 = 0, 0, 0, 0
count1, count2, count3, count4 = 0, 0, 0, 0

#mean horizontal
#h1, h2, h3, h4 = 0.99686959, 1.98921916, 2.98230079, 3.97692425

#mean vertical
h1, h2, h3, h4 = 1.00051454, 1.99987299, 3.00042952, 4.00009097

for i in range(len(x1)):
    for j in range(len(x1)):
        if j < i:
            if i-j == 1:
                count1 += 1
                mean1 += np.sqrt((x1[i]-x1[j])**2+(y1[i]-y1[j])**2+(z1[i]-z1[j])**2)
                ecart1 += (np.sqrt((x1[i]-x1[j])**2+(y1[i]-y1[j])**2+(z1[i]-z1[j])**2)-h1)**2
            if i-j == 2:
                count2 += 1
                mean2 += np.sqrt((x1[i]-x1[j])**2+(y1[i]-y1[j])**2+(z1[i]-z1[j])**2)
                ecart2 += (np.sqrt((x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2 + (z1[i] - z1[j]) ** 2) - h2)**2
            if i-j == 3:
                count3 += 1
                mean3 += np.sqrt((x1[i]-x1[j])**2+(y1[i]-y1[j])**2+(z1[i]-z1[j])**2)
                ecart3 += (np.sqrt((x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2 + (z1[i] - z1[j]) ** 2) - h3)**2
            if i-j == 4:
                count4 += 1
                mean4 += np.sqrt((x1[i]-x1[j])**2+(y1[i]-y1[j])**2+(z1[i]-z1[j])**2)
                ecart4 += (np.sqrt((x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2 + (z1[i] - z1[j]) ** 2) - h4)**2

# print(mean1/count1)
# print(mean2/count2)
# print(mean3/count3)
# print(mean4/count4)
#
# print(np.sqrt(ecart1/(count1-1)))
# print(np.sqrt(ecart2/(count2-1)))
# print(np.sqrt(ecart3/(count3-1)))
# print(np.sqrt(ecart4/(count4-1)))

graph = True
if graph == True:
    ax.scatter(x, y, z, s=10, depthshade=True)
    plt.ylim(5, 6)
    plt.show()

    ax.scatter(x, y, z, s=10, depthshade=True)
    plt.show()

    im = plt.scatter(x, y, s=15, c=z, marker='o')#, cmap=plt.cm.get_cmap('RdYlBu'))
    plt.xlabel("Position x [cm]", fontsize=14)
    plt.ylabel("Position y [cm]", fontsize=14)
    plt.colorbar(im, label="Position en z [cm]")#, fontsize=14)
    plt.show()

    im1 = plt.scatter(x, z, s=15, c=y, marker='o')#, cmap=plt.cm.get_cmap('RdYlBu'))
    plt.xlabel("Position x [cm]", fontsize=14)
    plt.ylabel("Position z [cm]", fontsize=14)
    plt.colorbar(im1, label="Position en y [cm]")
    #plt.colorbar.set_label(label="Position en y [cm]")#, size=14)
    plt.show()

    im2 = plt.scatter(y, z, s=15, c=x, marker='o', cmap=plt.cm.get_cmap('RdYlBu'))
    plt.colorbar(im2)
    plt.show()

