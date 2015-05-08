#!/usr/bin/env python2

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import Queue
import threading,time
import sys

import threading,time
import multiprocessing

from rtlsdr import RtlSdr, limit_time
import bitarray

## our own stuff
import util

from radio import *
from encoding import *
from radio_params import *
from radio_transmit import *

record_len = 75

## configure SDR
sdr = RtlSdr()

freq_offset = 83.2e3

sample_rate = 240e3
center_freq = 443.582e6 - freq_offset
#center_freq = 145.442e6 #- freq_offset

sdr.sample_rate = sample_rate  
sdr.center_freq = center_freq   
sdr.gain = 10

## get samples
samples = []

def callback(ss, obj):
    samples.append(ss)
    #obj.cancel_read_async()

print('Ready to record!')
raw_input("Press Enter to record...")
    
print('recording...')
N_samples = sample_rate*0.5 # approximately seconds
sdr.read_samples_async(limit_time(record_len)(callback), N_samples)   # get samples

print('recorded!')

sdr_samples = np.hstack(samples)

print('demodulating...')
ss = smart_demod(sdr_samples, sample_rate, freq_offset)
#del sdr_samples'

print('finding packets...')
packets = findPackets(ss)[0]

print('error correcting packets...')
decodes = map(lambda p: util.correct(decodePacket(p)), packets)

new_x = np.hstack(decodes)
new_x_bits = bitarray.bitarray(new_x.tolist())

print('decompressing...')
imm2 = decompressanddecode(new_x_bits)

ims_rgb(imm2)
