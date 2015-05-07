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

def squared_mean(img):
    return np.mean(img*img)

def mse(img1, img2):
    return np.mean((img1-img2)**2)

def mse_compr(y, wav='db3',lev=4):
    wp = pywt.WaveletPacket2D(data=x, wavelet=wav, maxlevel=lev, mode='sym')
    wps = wp.get_level(lev)
    dec = map(lambda x: x.data, wps)
    paths = map(lambda x: x.path, wps)
    data = np.vstack(dec)
    s = np.std(data)
    wp2 = pywt.WaveletPacket2D(data=None, wavelet=wav, maxlevel=lev, mode='sym')
    thres = np.sqrt(squared_mean(y))
    res = 0

    uncompressed = '';
    for p, d in zip(paths, dec):
        dd = np.copy(d)
        #if p.count("a") == 0:
        #    rms = np.sqrt(squared_mean(d))
        #    dd[abs(d - rms) < thres] = rms
        #    dd[abs(d + rms) < thres] = -rms
        dd[abs(d) < thres] = 0
        wp2[p] = dd
        res += np.sum(dd != 0)
        flattened = np.ndarray.flatten(dd);
        for coeff in flattened:
            uncompressed += binary(coeff);
    # drows, dcols = np.shape(dec[0])
    # uncompressed = binary(drows) + binary(dcols) + uncompressed;
    compressed = util.compress(bitarray.bitarray(uncompressed));
    ty = wp2.reconstruct()

    return mse(y, ty), len(compressed)


def bitarraytostring(bitarray):
    # nums = np.array(bitarray);
    # output = '';
    # for num in nums:
    #     fixed = bin(num);
    #     if len(fixed) > 9:
    #         fixed = fixed.replace('0b', '');
    #     elif len(fixed) < 9:
    #         fixed = fixed.replace('b', '0'*(9-len(fixed)));
    #     else:
    #         fixed = fixed.replace('b', '');
    #     output += fixed;
    # return output
    return bitarray.to01()

preamble_len = 32*4

def encode_preamble(width, height, drows, dcols):
    preamble = binary(width) + binary(height) + \
               binary(drows) + binary(dcols)

    return preamble

def decode_preamble(uncompressed):
    out = []
    for i in range(4):
        x = bitstofloat(bitarraytostring(uncompressed[(32*i):(32*(i+1))]))
        out.append(x)

    width = int(out[0])
    height = int(out[1])
    drows = int(out[2])
    dcols = int(out[3])

    return (width, height, drows, dcols), uncompressed[preamble_len:]
        
def compressandencode(name, lev = 4, wav = 'db3', thres = 500):
    """Outputs Bitarray"""

    im = Image.open(name);
    width, height = im.size;

    print 'Found image with width {0} and height {1}'.format(width, height);

    r = np.array(im.getdata(0)).reshape(height, width)
    g = np.array(im.getdata(1)).reshape(height, width)
    b = np.array(im.getdata(2)).reshape(height, width)

    transform = np.matrix('.299, .587, .114; -.16874, -.33126, .5; .5, -.41869, -.08131')
    def apply_transform(rgb):
        return np.array(np.dot(transform, rgb))[0]

    rgb = np.dstack((r,g,b))
    ycbcr = np.apply_along_axis(apply_transform, 2, rgb)

    # transform = np.matrix('.299, .587, .114; -.16874, -.33126, .5; .5, -.41869, -.08131')
    # Y = np.zeros(np.shape(r));
    # Cb = np.zeros(np.shape(r));
    # Cr = np.zeros(np.shape(r));

    # for i in range(np.shape(r)[0]):
    #     for j in range(np.shape(r)[1]):
    #         RGB = np.matrix('{0}; {1}; {2}'.format(r.item((i,j)), g.item((i,j)), b.item((i,j))));
    #         YCbCr = transform * RGB;
    #         Y[i,j] = YCbCr.item(0,0);
    #         Cb[i,j] = YCbCr.item(1,0);       
    #         Cr[i,j] = YCbCr.item(2,0);
            
    print 'Finished transform to Y, Cb, Cr';

    x = ycbcr[:,:,0]

    wp = pywt.WaveletPacket2D(data=x, wavelet=wav, maxlevel=lev, mode='sym')

    wps = wp.get_level(lev)

    dec = map(lambda x: x.data, wps)

    print 'Got coefficients';
            
    uncompressed = '';
    for d in dec:
        dd = np.float32(d);
        dd[abs(d) < thres] = 0;

        flattened = np.ndarray.flatten(dd);
        for coeff in flattened:
            uncompressed += binary(coeff);
        """for i in range(np.shape(dd)[0]):
            for j in range(np.shape(dd)[1]):
                uncompressed += binary(dd[i][j]);"""

    drows, dcols = np.shape(dec[0])
    preamble = encode_preamble(width, height, drows, dcols)
    uncompressed =  preamble + uncompressed;

    once = util.compress(bitarray.bitarray(uncompressed));
    twice = util.compress(once.tolist());
    compressed = bitarray.bitarray(binary(len(twice))) + twice
    
    print 'Compressed to {0} bits'.format(len(compressed));

    return compressed;

def ycbcr_to_rgb(ycbcr):
    """Takes in an ycbcr numpy array (700, 700, 3) and returns an rgb numpy array (700, 700, 3)"""
    transform = np.matrix('.299, .587, .114; -.16874, -.33126, .5; .5, -.41869, -.08131')
    inverse = transform.getI()

    def apply_transform(ycbcr):
        return np.array(np.dot(inverse, ycbcr))[0]

    return np.apply_along_axis(apply_transform, 2, ycbcr)



def decompressanddecode(compressed):
    """Takes Bitarray"""

    paths = ['aaaa', 'aaah', 'aaav', 'aaad', 'aaha', 'aahh', 'aahv', 'aahd', 'aava', 'aavh', 'aavv', 'aavd', 'aada', 'aadh', 'aadv', 'aadd',
    'ahaa', 'ahah', 'ahad', 'ahha', 'ahhh', 'ahhv', 'ahhd', 'ahva', 'ahvh', 'ahvv', 'ahvd', 'ahda', 'ahdh', 'ahdv', 'ahdd', 'avaa', 'avah',
    'avav', 'avad', 'avha', 'avhh', 'avhv', 'avhd', 'avva', 'avvh', 'avvv', 'avvd', 'avda', 'avdh', 'avdv', 'avdd', 'adaa', 'adah', 'adav',
    'adad', 'adha', 'adhh', 'adhv', 'adhd', 'adva', 'advh', 'advv', 'advd', 'adda', 'addh', 'addv', 'addd', 'haaa', 'haah', 'haav', 'haad', 
    'haha', 'hahh', 'hahv', 'hahd', 'hava', 'havh', 'havv', 'havd', 'hada', 'hadh', 'hadv', 'hadd', 'hhaa', 'hhah', 'hhav', 'hhad', 'hhha',
    'hhhh', 'hhhv', 'hhhd', 'hhva', 'hhvh', 'hhvv', 'hhvd', 'hhda', 'hhdh', 'hhdv', 'hhdd', 'hvaa', 'hvah', 'hvav', 'hvad', 'hvha', 'hvhh',
    'hvhv', 'hvhd', 'hvva', 'hvvh', 'hvvv', 'hvvd', 'hvda', 'hvdh', 'hvdv', 'hvdd', 'hdaa', 'hdah', 'hdav', 'hdad', 'hdha', 'hdhh', 'hdhv',
    'hdhd', 'hdva', 'hdvh', 'hdvv', 'hdvd', 'hdda', 'hddh', 'hddv', 'hddd', 'vaaa', 'vaah', 'vaav', 'vaad', 'vaha', 'vahh', 'vahv', 'vahd',
    'vava', 'vavh', 'vavv', 'vavd', 'vada', 'vadh', 'vadv', 'vadd', 'vhaa', 'vhah', 'vhav', 'vhad', 'vhha', 'vhhh', 'vhhv', 'vhhd', 'vhva',
    'vhvh', 'vhvv', 'vhvd', 'vhda', 'vhdh', 'vhdv', 'vhdd', 'vvaa', 'vvah', 'vvav', 'vvad', 'vvha', 'vvhh', 'vvhv', 'vvhd', 'vvva', 'vvvh',
    'vvvv', 'vvvd', 'vvda', 'vvdh', 'vvdv', 'vvdd', 'vdaa', 'vdah', 'vdav', 'vdad', 'vdha', 'vdhh', 'vdhv', 'vdhd', 'vdva', 'vdvh', 'vdvv',
    'vdvd', 'vdda', 'vddh', 'vddv', 'vddd', 'daaa', 'daah', 'daav', 'daad', 'daha', 'dahh', 'dahv', 'dahd', 'dava', 'davh', 'davv', 'davd',
    'dada', 'dadh', 'dadv', 'dadd', 'dhaa', 'dhah', 'dhav', 'dhad', 'dhha', 'dhhh', 'dhhv', 'dhhd', 'dhva', 'dhvh', 'dhvv', 'dhvd', 'dhda',
    'dhdh', 'dhdv', 'dhdd', 'dvaa', 'dvah', 'dvav', 'dvad', 'dvha', 'dvhh', 'dvhv', 'dvhd', 'dvva', 'dvvh', 'dvvv', 'dvvd', 'dvda', 'dvdh',
    'dvdv', 'dvdd', 'ddaa', 'ddah', 'ddav', 'ddad', 'ddha', 'ddhh', 'ddhv', 'ddhd', 'ddva', 'ddvh', 'ddvv', 'ddvd', 'ddda', 'dddh', 'dddv',
    'dddd']

    L = int(bitstofloat(bitarraytostring(compressed[0:32])));
    compressed = compressed[32:(L+32)]

    print('decompressing...')
    uncompressed = util.decompress(util.decompress(compressed))

    print('reconstructing...')
    params, uncompressed = decode_preamble(uncompressed)
    width, height, drows, dcols = params
    
    #numCoeff = len(uncompressed) / 32 / 256
    lev = 4
    wav = 'db3'
    wp2 = pywt.WaveletPacket2D(data=None, wavelet=wav, maxlevel=lev, mode='sym')

    coeff = np.zeros(len(uncompressed)/32)
    for i in range(len(uncompressed) / 32):
        coeff[i] = bitstofloat(bitarraytostring(uncompressed[32*i:32*i+32]))

    for pindex in range(len(paths)):
        wp2[paths[pindex]] = np.reshape(coeff[drows*dcols*pindex:drows*dcols*(pindex+1)], (drows, dcols));
    #for pindex in range(len(paths)):
    #    dd = np.zeros((drows, dcols));
    #    for i in range(drows*dcols):
    #       dd[int(np.floor(i/dcols))][i%drows] = coeff[drows*dcols*pindex + i];
    #    wp2[paths[pindex]] = dd;

    imm = wp2.reconstruct()
    # ims(imm)
    return imm[0:height, 0:width]

def encode_cartoon(rgb):
    bytea = bytearray(rgb.flatten())
    bitar = bitarray.bitarray()
    bitar.frombytes(str(bytea))
    return util.compress(bitar)

def decode_cartoon(compressed, height, width):
    uncompressed = util.decompress(compressed)
    rgb = np.array(bytearray(uncompressed))
    return rgb.reshape((height, width, 3))

def downsample(matrix):
    if np.shape(matrix)[0] % 2 != 0:
        matrix = np.vstack((matrix, matrix[-1]));
    if np.shape(matrix)[1] %2 != 0:
        matrix = np.hstack((matrix, np.transpose(np.matrix(matrix[:,-1]))));
    
    output = np.zeros((int(np.shape(matrix)[0])/2, int(np.shape(matrix)[1])/2));
    for i in range(int(np.shape(matrix)[0])/2):
        for j in range(int(np.shape(matrix)[1])/2):
            output[i,j] = (matrix[2*i,2*j] + matrix[2*i+1,2*j] + matrix[2*i,2*j+1] + matrix[2*i+1,2*j+1]) / 4.0;
    return output

def upsample(matrix):
    output = np.zeros((np.shape(matrix)[0]*2, np.shape(matrix)[1]*2));
    for i in range(int(np.shape(matrix)[0])):
        for j in range(int(np.shape(matrix)[1])):
            output[2*i, 2*j] = matrix[i,j];
            output[2*i+1, 2*j] = matrix[i,j];
            output[2*i, 2*j+1] = matrix[i,j];
            output[2*i+1, 2*j+1] = matrix[i,j];
    
    return output;

def Imdownsample(matrix):
    im = Image.fromarray(matrix);
    im = im.resize((np.shape(matrix)[0]/2, np.shape(matrix)[1]/2))
    return np.array(im.getdata()).reshape(np.shape(matrix)[0]/2, np.shape(matrix)[1]/2)

def Imupsample(matrix):
    im = Image.fromarray(matrix);
    im = im.resize((np.shape(matrix)[0]*2, np.shape(matrix)[1]*2))
    return np.array(im.getdata()).reshape(np.shape(matrix)[0]*2, np.shape(matrix)[1]*2)

def autodownsample(matrix, max_pixels):
    """Returns the number of times to downsample the matrix so that it has fewer than max_pixels"""
    n = int(np.ceil(np.log(np.shape(matrix)[0] * np.shape(matrix)[1] / max_pixels) / np.log(4)));
    return n;


