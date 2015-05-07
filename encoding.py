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

def bitstoint(bits):
    a = BitArray();
    a.bin = bits;
    return a.int;

def bitarraytostring(bitarray):
    return bitarray.to01()


def binary_short(num):
    return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!h', num))

def binary_int(num):
    return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!i', num))

# use this as much as possible
def binary_short_byte(num):
    return struct.pack('!h', num)

def byte_to_short(byte):
    return struct.unpack('!h', byte)[0]




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
    return bitarray.to01()


LEVEL = 5
WAVELET = 'db3'

def get_wavelets(data, wavelet=WAVELET, lev=LEVEL, thres=None):

    wp = pywt.WaveletPacket2D(data=data, wavelet=wavelet, maxlevel=lev, mode='sym')

    #wps = sorted(wp.get_level(lev), key=lambda x: x.path)
    wps = wp.get_level(lev)

    dec = list(map(lambda x: x.data, wps))
    paths = list(map(lambda x: x.path, wps))

    if thres == None:
        thres = np.sqrt(np.mean(np.square(data)))

    for d in dec:
        dd = np.float32(d)
        dd[abs(dd) < thres] = 0

    return zip(paths, dec)
        

def encode_wavelets(waves):
    out = []
    
    for path, coefs in waves:
        
        for i in range(coefs.shape[0]):
            for j in range(coefs.shape[1]):
                b = binary_short_byte(np.round(np.clip(coefs[i,j], -2**15, 2**15)))
                out.append(b)

    drows, dcols = coefs.shape
    
    bb = ''.join(out)
    bb = binary_short_byte(drows) + binary_short_byte(dcols) + bb

    a = bitarray.bitarray(endian='big')
    a.frombytes(bb)
    
    return a

def decode_wavelets(uncompressed):
    wp3 = pywt.WaveletPacket2D(data=np.zeros((100,100)), wavelet=wav, maxlevel=lev, mode='sym')
    paths = map(lambda x: x.path, wp3.get_level(5))

    drows = byte_to_short(uncompressed[:16].tobytes())
    dcols = byte_to_short(uncompressed[16:32].tobytes())

    
    uncompressed = uncompressed[32:]
    
    wp2 = pywt.WaveletPacket2D(data=None, wavelet=wav, maxlevel=lev, mode='sym')

    N = 16
    
    coeff = np.zeros(len(uncompressed)/N)
    for i in range(len(uncompressed) /N):
        coeff[i] = byte_to_short(uncompressed[N*i:N*(i+1)].tobytes())

    d = np.zeros((drows, dcols))
    i, j = 0, 0
    pindex = 0
    n = 0
    while pindex < len(paths):
        d[i,j] = coeff[n]
        j += 1
        n += 1
        if j >= dcols:
            j = 0
            i += 1
        if i >= drows:
            i, j = 0, 0
            wp2[paths[pindex]] = np.copy(d)
            pindex += 1

    return wp2


        
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

    mat = np.dstack((r,g,b))
    ycbcr = np.apply_along_axis(apply_transform, 2, mat)
 
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

    bb = binary_short_byte(height) + binary_short_byte(width) + \
         str(bytea)
    bitar.frombytes(bb)
    return util.compress(bitar)

def decode_cartoon(compressed):
    uncompressed = util.decompress(compressed)
    height = byte_to_short(uncompressed[:16].tobytes())
    width = byte_to_short(uncompressed[16:32].tobytes())
    rgb = np.array(bytearray(uncompressed[32:]))
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
