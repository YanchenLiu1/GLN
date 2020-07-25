#!/usr/bin/env python

import os,sys

from numpy import *
from itertools import combinations

import net3d_v10_8 as n3
import net3d_v10_8_no_crossing as n3n

import json
jlo = lambda s: json.load(open(s,'r'))

import gln_v7 as gln
import functions as fc

samp = int(sys.argv[1])
relax = 20

Adj = jlo('./adj_er_k6.json')
Adj = array(Adj)
edgelist = fc.A2E_weighted(Adj)

r = 0.1
k,A = .1, 100
#k,A = 1.0, 100
seg = 20
params = {'links':
       {'k':k,'amplitude':A, 'thickness':r, 
        'Temp0':.0, 'ce':10, 'segs':seg,
       'weighted': 0},
       'nodes':
       {'amplitude' : A, 'radius': r*3, 'weighted':0},
      }

pts = random.rand(len(Adj),3)*len(Adj)**(1.0/3)


n3net = n3.netRads(pts, edgelist,**{'n_radius':0.2, 'links':{'thickness':0.1,'k':1.,'T0':.1,'ce':15}, 'A':10})

n3.iter_converge(n3net, its = relax, max_its=relax, draw = 0,verbose = False)
n3net.gnam = 'tmp-%d'%samp
n3net.save(path='./', tv=0)

n3net = n3n.networkBase(JSON='./tmp-%d.json'%samp, **params)

# n3net = n3n.networkBase(pts,edgelist, **params)
n3n.iter_converge(n3net, its = 500, max_its=1000, draw = 0, verbose = False, tol = 0.001, c0 = 0.02)
n3n.iter_converge(n3net, its = 500, max_its=8000, draw = 0, verbose = False, tol = 0.001, ct = 0.08)
n3net.gnam = 'er_n100_k6_relax%d-%d'%(relax,samp)
n3net.save(path='./', tv=0)

el = n3n.tf.reduce_sum(n3net.links.dp**2).eval()

net = jlo('./er_n100_k6_relax%d-%d.json'%(relax,samp))

network = gln.simplify_v7(Adj,net)
network.find_base_loops(3)
network.get_min_loop_set()
network.cal_normalization_const()
norm = network.new_norm
print norm
network.cal_m2_old()
network.cal_m2_normalized_new(norm = norm)
print network.m2
print samp

open('./ngln_er_n100_k6.txt','a').write('%f\n'%network.m2_normalized)
open('./elastic_er_n100_k6.txt','a').write('%f\n'%el)