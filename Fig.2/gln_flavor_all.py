#!/usr/bin/env python

import os,sys

j = int(sys.argv[1])

a = 5

for i in range(j*a*2,j*a*2+a):
    os.system('CUDA_VISIBLE_DEVICES=0 ./gln_flavor_single.py %d &' %(i))
    
for i in range(j*a*2+a,j*a*2+2*a):
    os.system('CUDA_VISIBLE_DEVICES=1 ./gln_flavor_single.py %d &' %(i))
    
#for i in range(j+2*a,j+3*a):
#    os.system('CUDA_VISIBLE_DEVICES=2 ./gln_flavor_single.py %d &' %(i))
    
#for i in range(j+3*a,j+4*a):
#    os.system('CUDA_VISIBLE_DEVICES=3 ./gln_flavor_single.py %d &' %(i))