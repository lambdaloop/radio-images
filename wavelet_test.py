#!/usr/bin/env python2

from PIL import Image
import numpy as np

im = Image.open("bird.jpg") #Can be many different formats.
pix = im.load()
print im.size #Get the width and hight of the image for iterating over
# print pix[x,y] #Get the RGBA Value of the a pixel of an image

width, height = im.size

r = np.array(im.getdata(0)).reshape(height, width)
g = np.array(im.getdata(1)).reshape(height, width)
b = np.array(im.getdata(2)).reshape(height, width)

import matplotlib.pyplot as plt

def plot_image(r, g, b, cmap='gray'):
    plt.clf()
    plt.subplot(3,1,1)
    plt.imshow(r, cmap=cmap)
    plt.subplot(3,1,2)
    plt.imshow(g, cmap=cmap)
    plt.subplot(3,1,3)
    plt.imshow(b, cmap=cmap)
    plt.draw()
    plt.show(block=False)

def ims(x):
    plt.clf()
    plt.imshow(x, cmap='gray')
    plt.draw()
    plt.show()

plot_image(r, g, b)

plt.clf()
plt.imshow(r+g+b, cmap='gray')
plt.draw()

import pywt
import numpy as np

x = r

lev = 4
wav = 'db3'

wp = pywt.WaveletPacket2D(data=x, wavelet=wav, maxlevel=lev, mode='sym')

wps = wp.get_level(lev)

dec = map(lambda x: x.data, wps)
paths = map(lambda x: x.path, wps)
print np.shape(dec[0])

data = np.vstack(dec)
s = np.std(data)

wp2 = pywt.WaveletPacket2D(data=None, wavelet=wav, maxlevel=lev, mode='sym')


thres = 20

res = 0

for p, d in zip(paths, dec):
    dd = np.copy(d)
    dd[abs(d) < thres] = 0
    wp2[p] = dd
    res += np.sum(dd != 0)




print(res)
print(float(data.size) / float(res))

ims(wp2.reconstruct())
