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

name = '../final/calBlue.tiff'

# im = load_image(name)
# ims_rgb(im)

x = compressandencode(name)

ar = np.array(x.tolist())
bit_list = []
for i in xrange(0, len(ar), Nbits_packet_noec):
    b = ar[i:i+Nbits_packet_noec]
    if len(b) < Nbits_packet_noec:
        b = np.append(b, np.zeros(Nbits_packet_noec - len(b)))
    bit_list.append(util.encorrect(b))
    
N_packets = len(bit_list)

print("Number of packets: {0}".format(N_packets))

def printDevNumbers(p):
    N = p.get_device_count()
    for n in range(0,N):
        name = p.get_device_info_by_index(n).get('name')
        print n, name
        
# p = pyaudio.PyAudio()
# printDevNumbers(p)
# p.terminate()

# CHANGE as needed
dusb_in = 5
dusb_out = 5
din = 0
dout = 0

if sys.platform == 'darwin':  # Mac
    s = serial.Serial(port='/dev/tty.SLAB_USBtoUART')
else:                         #windows
    s = serial.Serial(port='/dev/ttyUSB1') ##### CHANGE !!!!!!
s.setDTR(0)

zero_sample = np.zeros(int(44100 * 0.3))

packet = build_packet(bit_list[0])

record_len = int(((len(packet) + between_len) / float(fs))*N_packets) + 3

print('Transmission length: {0} seconds'.format(record_len))


print('Ready to send!')
raw_input("Press Enter to send...")

print('Sending...')

# creates a queue
Qout = Queue.Queue()

# initialize a serial port 
s.setDTR(0)
# create a pyaudio object
p = pyaudio.PyAudio()

t_play = threading.Thread(target = play_audio,   args = (Qout,   p, 44100, dusb_out, s ,0.2 ))

#Qout.put(sig2)
Qout.put("KEYON")
Qout.put(zero_sample)

for i in range(N_packets):
    Qout.put(build_packet(bit_list[i]))
    Qout.put(np.zeros(between_len))

Qout.put(np.zeros(1024))
Qout.put("KEYOFF")
Qout.put(zero_sample)
Qout.put("EOT")

# play audio from Queue 
t_play.start()

    
# must wait for the queue to empty before terminating pyaudio
while not(Qout.empty()) :
    time.sleep(1)
    

p.terminate()# kill a playing thread

print 'Sent!'

