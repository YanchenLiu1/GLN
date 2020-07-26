#!/usr/bin/env python

import os,sys
from numpy import *

eps = float(sys.argv[1])
eps = 2.9451562666948146 #  7.067543494682078 # 14.988894647507053 # 1.9190347513349437 #  19.9916690966 # 2.9451562666948146   # 6.622696204630026 #  # 24.993334577567865 # 17.990744073751454 #   # 4.966821568814516 #10.32 # 17.87

for i in range(96): 
    os.system('./mouse_perturbation.py %f &' %(eps))
    
