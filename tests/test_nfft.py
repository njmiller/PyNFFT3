#!/usr/bin/env python

import numpy as np
import pynfft

print pynfft

nfft1 = pynfft.NFFT.init_1d(20,20)

f_hat = np.array(np.sin(np.arange(20)*2*np.pi/20),dtype=np.complex)

nfft1.f_hat = f_hat

print("f_hat = ", f_hat)
print("nfft1.f_hat = ", nfft1.f_hat)

x = np.arange(20) / 20.0 - 0.49
nfft1.x = x

nfft1.precompute()

nfft1.trafo_1d()

print("nfft1.f = ", nfft1.f)

nfft1.f_hat = np.zeros(20)
print("nfft1.f_hat = ", nfft1.f_hat)

nfft1.adjoint_1d()
print("nfft1.f_hat = ", nfft1.f_hat/20)


