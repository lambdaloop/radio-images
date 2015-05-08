from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pywt
import ctypes
import bitarray
import struct
import util
from bitstring import BitArray
from scipy import interpolate

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

def ims_rgb(x):
    plt.clf()
    plt.imshow(np.clip(x, 0, 255)/256.0, cmap='gray')
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

def binary_int_byte(num):
    return struct.pack('!i', num)

def byte_to_int(byte):
    return struct.unpack('!i', byte)[0]

def binary_float_byte(num):
    return struct.pack('!f', num)

def byte_to_float(byte):
    return struct.unpack('!f', byte)[0]


def squared_mean(img):
    return np.mean(img*img)

def mse(img1, img2):
    return np.mean((img1-img2)**2)

def psnr(im1, im2):
    m = mse(im1, im2)
    return 10.0 * np.log10((256**2) / float(m) )
    

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

def get_wavelets(data, wavelet=WAVELET, lev=LEVEL, thres_scale=1, no_a=True):

    wp = pywt.WaveletPacket2D(data=data, wavelet=wavelet, maxlevel=lev, mode='sym')

    #wps = sorted(wp.get_level(lev), key=lambda x: x.path)
    wps = wp.get_level(lev)

    dec = list(map(lambda x: x.data, wps))
    paths = list(map(lambda x: x.path, wps))

    # if thres == None:
    thres = thres_scale * np.sqrt(np.mean(np.square(data)))

    out_dec = []
    
    for d,p in zip(dec,paths):
        dd = np.float32(d)
        if p.count("a") < lev - 1 or no_a:
            dd[abs(dd) < thres] = 0
        out_dec.append(dd)

    return zip(paths, out_dec)
        

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

    a = bitarray.bitarray('', endian='big')
    a.frombytes(bb)
    
    return a

def decode_wavelets(uncompressed):
    wp3 = pywt.WaveletPacket2D(data=np.zeros((100,100)),
                               wavelet=WAVELET, maxlevel=LEVEL, mode='sym')
    paths = map(lambda x: x.path, wp3.get_level(LEVEL))

    drows = byte_to_short(uncompressed[:16].tobytes())
    dcols = byte_to_short(uncompressed[16:32].tobytes())

    
    uncompressed = uncompressed[32:]
    
    wp2 = pywt.WaveletPacket2D(data=None,
                               wavelet=WAVELET, maxlevel=LEVEL, mode='sym')

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

MAX_PIXELS = 1000000 #1M
CARTOON_PIXELS = 60000 # 50k
BIT_THRESHOLD = 150000 # 150k bits we can send

def compressandencode(name):
    """Outputs Bitarray"""

    im = Image.open(name);
    width, height = im.size;

    # rescale, keep track of rescale factor and send that
    # check if cartoon, send if yes, encode whether cartoon

    # otherwise:
    # convert to Y, Cb, Cr
    # use wavelets to encode each one
    # combine all arrays together
    
    # compress twice, add length at beginning

    # return!

    print 'Found image with width {0} and height {1}'.format(width, height);

    r = np.array(im.getdata(0)).reshape(height, width)
    g = np.array(im.getdata(1)).reshape(height, width)
    b = np.array(im.getdata(2)).reshape(height, width)


    mat = np.dstack((r,g,b))

    n_down = inter_autodownsample(r, MAX_PIXELS)

    n_down_cartoon = inter_autodownsample(r, CARTOON_PIXELS)

    print 'Downsampling {0} times...'.format(n_down_cartoon)

    dmat_cartoon = inter_resample_3d(mat, n_down_cartoon)

    print 'Checking if cartoon...'
    cartoon = encode_cartoon(dmat_cartoon)

    print 'Size is {0} bits'.format(len(cartoon))
    
    if len(cartoon) < BIT_THRESHOLD-144:
        print 'Detected cartoon!'
        is_cartoon = 1
        out = cartoon
        n_down = n_down_cartoon
    else:
        print 'Not a cartoon, continuing...'
        is_cartoon = 0

        print 'Downsampling {0} times...'.format(n_down)
        dmat = inter_resample_3d(mat, n_down)
        
        ycbcr = rgb_to_ycbcr(dmat)
    
        print 'Finished transform to Y, Cb, Cr';

        out = use_wav(ycbcr, False)

        if len(out) > BIT_THRESHOLD-144:
            out = use_wav(ycbcr, True)
            while len(out) > BIT_THRESHOLD-144:
                n_down += len(out) / (BIT_THRESHOLD-144)
                dmat = inter_resample_3d(mat, n_down)
                out = use_wav(ycbcr, True)

        # Y = ycbcr[:,:,0]
        # Cb = ycbcr[:,:,1]
        # Cr = ycbcr[:,:,2]

        # # print 'Downsampling Cb and Cr...'
        # Cb = downsample_n(Cb, 2)
        # Cr = downsample_n(Cr, 2)

        # print 'Encoding Y...'
        # waves = get_wavelets(Y, thres_scale=0.9)
        # eY = encode_wavelets(waves)

        # print 'Encoding Cb...'
        # waves = get_wavelets(Cb, thres_scale=3)
        # eCb = encode_wavelets(waves)

        # print 'Encoding Cr...'
        # waves = get_wavelets(Cr, thres_scale=3)
        # eCr = encode_wavelets(waves)

        # pre = binary_int_byte(len(eY)) + binary_int_byte(len(eCb))

        # a = bitarray.bitarray('',endian='big')
        # a.frombytes(pre)

        # uncompressed = a + eY + eCb + eCr

        # print 'Compressing...'
        # once = util.compress(uncompressed);
        # twice = util.compress(once);
        # out = twice

    preamble = binary_int_byte(len(out)) + binary_float_byte(n_down) + \
               binary_short_byte(is_cartoon) + \
               binary_int_byte(height) + binary_int_byte(width)
    
    a = bitarray.bitarray('',endian='big')
    a.frombytes(preamble)

    compressed = a + out
    
    print 'Compressed to {0} bits'.format(len(compressed));

    return compressed;

def ycbcr_to_rgb(ycbcr):
    """Takes in an ycbcr numpy array (700, 700, 3) and returns an rgb numpy array (700, 700, 3)"""
    transform = np.matrix('.299, .587, .114; -.16874, -.33126, .5; .5, -.41869, -.08131')
    inverse = transform.getI()

    def apply_transform(ycbcr):
        return np.array(np.dot(inverse, ycbcr))[0]

    return np.apply_along_axis(apply_transform, 2, ycbcr)

def rgb_to_ycbcr(rgb):
    """Takes in an ycbcr numpy array (700, 700, 3) and returns an rgb numpy array (700, 700, 3)"""
    transform = np.matrix('.299, .587, .114; -.16874, -.33126, .5; .5, -.41869, -.08131')

    def apply_transform(x):
        return np.array(np.dot(transform, x))[0]

    return np.apply_along_axis(apply_transform, 2, rgb)


def decode_natural(compressed):
    bits = util.decompress(util.decompress(compressed))

    wave_len = byte_to_int(bits[:32].tobytes())
    wave_small_len = byte_to_int(bits[32:64].tobytes())

    data = bits[64:]

    wp_Y = decode_wavelets(data[:wave_len])
    wp_Cb = decode_wavelets(data[wave_len:(wave_len + wave_small_len)])
    wp_Cr = decode_wavelets(
        data[(wave_len + wave_small_len):(wave_len + 2*wave_small_len)])

    Y = wp_Y.reconstruct()
    Cb = wp_Cb.reconstruct()
    Cr = wp_Cr.reconstruct()

    Cb = inter_resample(Cb, 1/2.0)
    Cr = inter_resample(Cr, 1/2.0)

    h = min(Y.shape[0], Cb.shape[0])
    w = min(Y.shape[1], Cb.shape[1])
    
    Y = Y[0:h, 0:w]
    Cb = Cb[0:h, 0:w]
    Cr = Cr[0:h, 0:w]
    
    ycbcr = np.dstack((Y, Cb, Cr))
    rgb = ycbcr_to_rgb(ycbcr)

    return rgb

def decompressanddecode(bits):
    """Takes Bitarray"""

    comp_len = byte_to_int(bits[:32].tobytes())
    n_down = byte_to_float(bits[32:64].tobytes())
    is_cartoon = byte_to_short(bits[64:80].tobytes())
    height = byte_to_int(bits[80:112].tobytes())
    width = byte_to_int(bits[112:144].tobytes())

    compressed = bits[144:(comp_len+144)]
    if is_cartoon:
        print 'Detected cartoon!'
        print 'Decompressing...'
        rgb = decode_cartoon(compressed)
    else:
        rgb = decode_natural(compressed)

    if n_down == 0:
        urgb = rgb
    else:
        urgb = inter_resample_3d(rgb, 1/float(n_down))

    urgb = urgb[0:height, 0:width, :]
    
    return urgb

def use_wav(ycbcr, a_option):

    Y = ycbcr[:,:,0]
    Cb = ycbcr[:,:,1]
    Cr = ycbcr[:,:,2]

    # print 'Downsampling Cb and Cr...'
    Cb = inter_resample(Cb, 2)
    Cr = inter_resample(Cr, 2)

    print 'Encoding Y...'
    waves = get_wavelets(Y, thres_scale=1.0, no_a=a_option)
    eY = encode_wavelets(waves)

    print 'Encoding Cb...'
    waves = get_wavelets(Cb, thres_scale=3)
    eCb = encode_wavelets(waves)

    print 'Encoding Cr...'
    waves = get_wavelets(Cr, thres_scale=3)
    eCr = encode_wavelets(waves)

    pre = binary_int_byte(len(eY)) + binary_int_byte(len(eCb))

    a = bitarray.bitarray('',endian='big')
    a.frombytes(pre)

    uncompressed = a + eY + eCb + eCr

    print 'Compressing...'
    once = util.compress(uncompressed);
    twice = util.compress(once);
    return twice


def encode_cartoon(rgb):
    height, width = rgb[:, :, 0].shape    
    
    bytea = bytearray(np.int8(rgb.flatten()))
    bitar = bitarray.bitarray('')

    bb = binary_int_byte(height) + binary_int_byte(width) + \
         str(bytea)
    bitar.frombytes(bb)

    return util.compress(bitar)

def decode_cartoon(compressed):
    uncompressed = util.decompress(compressed)
    height = byte_to_int(uncompressed[:32].tobytes())
    width = byte_to_int(uncompressed[32:64].tobytes())
    rgb = np.array(bytearray(uncompressed[64:]))
    
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
    # output = np.zeros((np.shape(matrix)[0]*2, np.shape(matrix)[1]*2));
    # for i in range(int(np.shape(matrix)[0])):
    #     for j in range(int(np.shape(matrix)[1])):
    #         output[2*i, 2*j] = matrix[i,j];
    #         output[2*i+1, 2*j] = matrix[i,j];
    #         output[2*i, 2*j+1] = matrix[i,j];
    #         output[2*i+1, 2*j+1] = matrix[i,j];
    
    # return output;
    return matrix.repeat(2, axis=0).repeat(2, axis=1)

def downsample_n(matrix, n):
    if n <= 0:
        return matrix
    
    mat = matrix
    for i in range(n):
        mat = downsample(mat)
    return mat

def upsample_n(matrix, n):
    if n <= 0:
        return matrix

    mat = matrix
    for i in range(n):
        mat = upsample(mat)
    return mat


def downsample_3d_n(matrix, n):
    if n <= 0:
        return matrix

    out = []
    
    for d in range(3):
        mat = matrix[:, :, d]
        for i in range(n):
            mat = downsample(mat)
        out.append(mat)
        
    return np.dstack(out)

def upsample_3d_n(matrix, n):
    if n <= 0:
        return matrix

    out = []
    
    for d in range(3):
        mat = matrix[:, :, d]
        for i in range(n):
            mat = upsample(mat)
        out.append(mat)
        
    return np.dstack(out)


def Imdownsample(matrix):
    im = Image.fromarray(matrix);
    im = im.resize((np.shape(matrix)[0]/2, np.shape(matrix)[1]/2))
    return np.array(im.getdata()).reshape(np.shape(matrix)[0]/2, np.shape(matrix)[1]/2)

def Imupsample(matrix):
    im = Image.fromarray(matrix);
    im = im.resize((np.shape(matrix)[0]*2, np.shape(matrix)[1]*2))
    return np.array(im.getdata()).reshape(np.shape(matrix)[0]*2, np.shape(matrix)[1]*2)

def autodownsample(matrix, max_pixels):
    """Returns the number of times to downsample the matrix so that it has fewer than max_pixels
    in powers of 2."""
    size = np.shape(matrix)[0] * np.shape(matrix)[1]
    if size <= max_pixels:
        return int(0)
    
    n = int(np.ceil(np.log(float(size) / max_pixels) / np.log(4.0)));
    return n;

def inter_autodownsample(matrix, max_pixels):
    """Returns the number of times to downsample the matrix so that it has fewer than max_pixels
    as a float.  Use with inter_resample."""
    size = np.shape(matrix)[0] * np.shape(matrix)[1]
    if size <= max_pixels:
        return int(0)
    
    n = float(size) / max_pixels;
    return np.sqrt(n);


def load_image(name):
    im = Image.open(name);
    width, height = im.size;

    r = np.array(im.getdata(0)).reshape(height, width)
    g = np.array(im.getdata(1)).reshape(height, width)
    b = np.array(im.getdata(2)).reshape(height, width)

    return np.dstack([r, g, b])


def inter_resample(matrix, n):
    """Matrix must be 2D. Returns a 2D matrix resampled depending on n.
    If n>=1, it downsamples matrix by n times.  If n <= 1, it upsamples
    matrix by 1/n times. MAKE SURE n IS A FLOAT!!!"""

    if n == 0:
        return matrix
        
    
    cols = np.arange(0, np.shape(matrix)[1], 1)
    rows = np.arange(0, np.shape(matrix)[0], 1)
    
    f = interpolate.interp2d(cols, rows, matrix)
    
    downrows = np.arange(0, np.shape(matrix)[0], n)
    downcols = np.arange(0, np.shape(matrix)[1], n)

    down = f(downcols, downrows)
             
    return down;
    
def inter_resample_3d(mat, n):
    out = []
    for i in range(3):
        x = inter_resample(mat[:, :, i], n)
        out.append(x)

    return np.dstack(out)
