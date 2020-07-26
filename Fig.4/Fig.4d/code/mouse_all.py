#!/usr/bin/env python3

import os,sys

epss = [2.9451562666948146, 7.067543494682078, 14.988894647507053]

samp0 = int(sys.argv[1])
samp1 = int(sys.argv[2])
gpu = int(sys.argv[3])
epsi = int(sys.argv[4])

eps = epss[epsi]

for i in range(samp0,samp1): 
    os.system('CUDA_VISIBLE_DEVICES=%d ./mouse_single.py %f %d %d &' %(gpu, eps, i, epsi))

    
