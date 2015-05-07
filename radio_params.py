#!/usr/bin/env python2

import util
import numpy as np
from radio import genChirpPulse

fs = 44100  # sampling rate
Ns = 20 # samples per bit
baud = float(fs)/Ns  # symbol rate
# print(baud)

f_low = 1102.5
f_high = 2205

Nbits_packet_noec = 2400
wait_len = 160

Nbits_packet = len(util.encorrect(np.zeros(Nbits_packet_noec)))

between_len = wait_len * 2

msg_len = Ns * Nbits_packet

pulse = genChirpPulse(3200, 500.0, 3500.0, 44100.0)
pulse = pulse.real / 2.0
