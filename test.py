#!/usr/bin/env python

import numpy as np

import nfft

aaa = nfft.NFFT_2D(3,2,4)

x = np.array([1.0,0.4,1.3,2.5])
y = np.array([0.6,0.2,1.7,4.1])

aaa.set_nodes(x,y)

aaa.show_nodes()

