#!/usr/bin/env python2

import util

from radio import *
from encoding import *
from radio_params import *
from radio_transmit import *

from PIL import Image
import os
import os.path
import glob 

files = glob.glob('../final/*.tiff')
files_comp = glob.glob('../final/*_comp.tiff')

files = set(files) - set(files_comp)

for name in files:
    print(name)
    im = load_image(name)
    # ims_rgb(im)

    x = compressandencode(name)
    imc = decompressanddecode(x)

    path, ext = os.path.splitext(name)
    new_name = path + '_comp' + ext

    immc = np.uint8(np.round(np.clip(imc, 0, 255)))
    
    Image.fromarray(immc).save(new_name)

    p = psnr(im, immc)
    print('PSNR: {0}'.format(p))
    print ''
