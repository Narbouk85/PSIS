import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PSIS_algorithm import PSIS

#For hand.tif.jpg
img = np.array(Image.open('images/hand.tif.jpg'))
deltaPrime = 1/(3*(img.shape[0]*img.shape[1])**2)
g = 30
Q = 32

print("Launched algorithm ")
R = PSIS(img, Q, g, deltaPrime).reshape(img.shape[0], img.shape[1])
print("Algorithm ended")

print(R)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(R)
plt.savefig("results/hand_seg.jpg")

#For cape-original.tif.jpg
img = np.array(Image.open('images/cape-original.tif.jpg'))
deltaPrime = 1/(3*(img.shape[0]*img.shape[1])**2)
g = 14
Q = 32

print("Launched algorithm ")
R = PSIS(img, Q, g, deltaPrime).reshape(img.shape[0], img.shape[1])
print("Algorithm ended")

print(R)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(R)
plt.savefig("results/cape-original_seg.jpg")

#For squirrel-original.tif.jpg
img = np.array(Image.open('images/squirrel-original.tif.jpg'))
deltaPrime = 1/(3*(img.shape[0]*img.shape[1])**2)
g = 23
Q = 32

print("Launched algorithm ")
R = PSIS(img, Q, g, deltaPrime).reshape(img.shape[0], img.shape[1])
print("Algorithm ended")

print(R)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(R)
plt.savefig("results/squirrel-original_seg.jpg")

#For woman-original.tif.jpg
img = np.array(Image.open('images/woman-original.tif.jpg'))
deltaPrime = 1/(3*(img.shape[0]*img.shape[1])**2)
g = 30
Q = 32

print("Launched algorithm ")
R = PSIS(img, Q, g, deltaPrime).reshape(img.shape[0], img.shape[1])
print("Algorithm ended")

print(R)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(R)
plt.savefig("results/woman-original_seg.jpg")
