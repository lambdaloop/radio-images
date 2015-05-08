#!/usr/bin/env python2

from radio import *
from radio_params import *
import numpy as np
from scipy import signal

def build_packet(bits):
    msg = afsk(1.0 * bits, f_low, f_high, baud, fs)
    return np.hstack([pulse, np.zeros(wait_len), msg])

def findPackets(sig):
    corr = np.abs(crossCorr(sig, pulse))
    thres = np.max(corr) / 2.0
    
    packets = []
    
    # wait_len_adj = wait_len * float(fs_down) / float(fs)
    # msg_len_adj = msg_len * float(fs_down) / float(fs)
    
    # packet_len_adj = len(pulse_re) + wait_len_adj + msg_len_adj
    packet_len = len(pulse) + wait_len + msg_len
    
    ps = []
    
    for i in range(0, len(corr), int(packet_len)):
        c = corr[i:i+packet_len]
        if sum(c > thres) > 0:
            ps.append(np.argmax(c) + i)
    
    #ps = np.where(corr > thres)[0]
    
    last_flag = -1
    
    for p in ps:
        start, end = int(p + wait_len), int(p + msg_len + wait_len)
        
        if end >= len(sig):
            last_flag = start - len(pulse) - wait_len - 100
            break
        else:
            packets.append(sig[start:end])
    
    return packets, last_flag

def decodePacket(pp, fs_down=fs):
    demod = nc_afsk_demod(pp, f_low, f_high, TBW=1.5, fs=fs_down)
    #demod = demod - np.mean(demod)
    decoded = decode_bits2(np.sign(demod), fs=fs_down, baud=baud)
    return decoded

def get_packets(sig, fs_down=fs):
    sigs = findPackets(sig, fs_down)
    decodes = [decodePacket(s, fs_down) for s in sigs]
    return decodes
    

def demodulate(sdr_samples, sample_rate, ignore=0):
    t = np.arange(len(sdr_samples)) / sample_rate
    demod = np.exp(-2j*pi*freq_offset*t)
    y = sdr_samples * demod

    threshold = 0.04
    y = y[abs(y) > threshold]

    from scipy import signal  #should be imported already?

    sig = angle(y[1:] * conj(y[:-1]))

    h = signal.firwin(256,5000.0,nyq=sample_rate/2.0)
    sigf = signal.fftconvolve(sig, h)
    if ignore > 0:
        sigf = sigf[ignore:-ignore]
    
    
    downsample = 24
    sigfd = sigf[::downsample]

    fs_down = fs
    sigfdd = signal.resample(sigfd, len(sigfd) * float(fs_down) / (sample_rate / float(downsample)))
    
    return sigfdd

def smart_demod(sdr_samples, sample_rate, freq_offset, chunk_len = 10240):
    t = np.arange(len(sdr_samples)) / sample_rate
    demod = np.exp(-2j*pi*freq_offset*t)
    sdr_samples *= demod
    y = sdr_samples

    # take care of noise when not sending data
    threshold = np.min(abs(y))*1.2
    # threshold = 0.025

    # if sum(abs(y) < threshold)/float(len(y)) < 0.5:
    y = y[abs(y) > threshold]

    sig = np.angle(y[1:] * np.conj(y[:-1]))

    h = signal.firwin(256,5000.0,nyq=sample_rate/2.0)
    sigf = signal.fftconvolve(sig, h)

    downsample = 24
    sigfd = sigf[::downsample]

    del sig
    del sigf
    
    result = np.array([])
    pad = 1024
    
    scale = float(fs) / (sample_rate / float(downsample))
    
    out = []
        
    for i in xrange(0, len(sigfd), chunk_len):
        s = sigfd[i:i+chunk_len]
        r = signal.resample(s, len(s) * scale)
        out.append(r)
    
    return np.hstack(out)
        













