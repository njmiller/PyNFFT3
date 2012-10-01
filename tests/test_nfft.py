#!/usr/bin/env python

import numpy as np
import pynfft

nfft1 = pynfft.NFFT.init_1d(20,20)

f = np.array(np.sin(np.arange(20)*2*np.pi/20),dtype=np.complex)

nfft1.f = f

print("f = ", f)
print("nfft1.f = ", nfft1.f)

f = 0

print("f = ", f)
print("nfft1.f = ", nfft1.f)
