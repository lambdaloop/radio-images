from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pywt
import ctypes
import bitarray
import struct
import util
from bitstring import BitArray

def plot_image(r, g, b, cmap='gray'):
    plt.clf()
    plt.subplot(3,1,1)
    plt.imshow(r, cmap=cmap)
    plt.subplot(3,1,2)
    plt.imshow(g, cmap=cmap)
    plt.subplot(3,1,3)
    plt.imshow(b, cmap=cmap)
    plt.draw()
    plt.show()

def ims(x):
    plt.clf()
    plt.imshow(x, cmap='gray')
    plt.draw()
    plt.show()

    def binary(num):
    return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))

def bitstofloat(bits):
    a = BitArray();
    a.bin = bits;
    return a.float;

def bitarraytostring(bitarray):
    nums = np.array(bitarray);
    output = '';
    for num in nums:
        fixed = bin(num);
        if len(fixed) > 9:
            fixed = fixed.replace('0b', '');
        elif len(fixed) < 9:
            fixed = fixed.replace('b', '0'*(9-len(fixed)));
        else:
            fixed = fixed.replace('b', '');
        output += fixed;
    return output

def compressandencode(name, lev = 4, wav = 'db3', thres = 500):
	"""Outputs Bitarray"""

    im = Image.open("bird.jpg")
    width, height = im.size
    
    print 'Found image with width {0} and height {1}'.format(width, height);
    
    r = np.array(im.getdata(0)).reshape(height, width)
    g = np.array(im.getdata(1)).reshape(height, width)
    b = np.array(im.getdata(2)).reshape(height, width)
    
    transform = np.matrix('.299, .587, .114; -.16874, -.33126, .5; .5, -.41869, -.08131')
    Y = np.zeros(np.shape(r));
    Cb = np.zeros(np.shape(r));
    Cr = np.zeros(np.shape(r));
    
    for i in range(np.shape(r)[0]):
        for j in range(np.shape(r)[1]):
            RGB = np.matrix('{0}; {1}; {2}'.format(r.item((i,j)), g.item((i,j)), b.item((i,j))));
            YCbCr = transform * RGB;
            Y[i,j] = YCbCr.item(0,0);
            Cb[i,j] = YCbCr.item(1,0);       
            Cr[i,j] = YCbCr.item(2,0);
            
    print 'Finished transform to Y, Cb, Cr';
    
    x = Y

    wp = pywt.WaveletPacket2D(data=x, wavelet=wav, maxlevel=lev, mode='sym')

    wps = wp.get_level(lev)

    dec = map(lambda x: x.data, wps)
    
    print 'Got coefficients';
            
    uncompressed = '';
    for d in dec:
        dd = np.float32(d);
        dd[abs(d) < thres] = 0;
        for i in range(np.shape(dd)[0]):
            for j in range(np.shape(dd)[1]):
                uncompressed += binary(dd[i][j]);
    
    drows, dcols = np.shape(dec[0])
    uncompressed = binary(drows) + binary(dcols) + uncompressed;

    compressed = util.compress(np.array(uncompressed));
    
    print 'Compressed to {0} bits'.format(len(compressed));
    
    return compressed;

def decompressanddecode(compressed):
	"""Takes Bitarray"""
	
    uncompressed = util.decompress(compressed);
    drows = int(bitstofloat(bitarraytostring(uncompressed[32*0:32*0+32])));
    dcols = int(bitstofloat(bitarraytostring(uncompressed[32*1:32*1+32])));
    uncompressed = uncompressed[32*2:];
    
    numCoeff = len(uncompressed) / 32 / 256
    lev = 4
    wav = 'db3'
    wp2 = pywt.WaveletPacket2D(data=None, wavelet=wav, maxlevel=lev, mode='sym')

    coeff = np.zeros(len(uncompressed)/32)
    for i in range(len(uncompressed) / 32):
        coeff[i] = bitstofloat(bitarraytostring(uncompressed[32*i:32*i+32]))

    for pindex in range(len(paths)):
        dd = np.zeros((drows, dcols));
        for i in range(drows*dcols):
            dd[int(np.floor(i/dcols))][i%drows] = coeff[drows*dcols*pindex + i];
        wp2[paths[pindex]] = dd;

    ims(wp2.reconstruct())