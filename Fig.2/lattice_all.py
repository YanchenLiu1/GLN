#!/usr/bin/env python

import os,sys


j = int(sys.argv[1])

a = 5

for i in range(j*a*2,j*a*2+a):
    os.system('CUDA_VISIBLE_DEVICES=0 ./lattice_single.py %d &' %(i))
    
for i in range(j*a*2+a,j*a*2+2*a):
    os.system('CUDA_VISIBLE_DEVICES=1 ./lattice_single.py %d &' %(i))