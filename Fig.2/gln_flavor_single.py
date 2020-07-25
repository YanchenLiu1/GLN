#!/usr/bin/env python

import os,sys
from numpy import *
from itertools import combinations

import net3d_v10_8 as n3
import net3d_v10_8_no_crossing as n3n
import gln_v7 as gln
import functions as fc

import json
jlo = lambda s: json.load(open(s,'r'))


samp = int(sys.argv[1])
relax = 240

net = jlo('./flavor1.json')
edge_list = [net['links'][i]['end_points'] for i in net['links'].keys()]
Adj = fc.edge_list_2_adj(edge_list)


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

#pts = random.rand(len(Adj),3)*len(Adj)**(1.0/3)
pts = random.rand(len(Adj),3)*len(Adj)**(1.0/3)*1.7

n3net = n3.netRads(pts, edge_list,**{'n_radius':0.05, 'links':{'T0':.0,'ce':15}, 'A':80})

n3.iter_converge(n3net, its = relax, max_its=relax, draw = False,verbose = False)


#n3net = n3n.networkBase(pts,edge_list, **params)
#n3.iter_converge(n3net, its = 300, max_its=300, draw = False,verbose = False)
#n3.iter_converge(n3net, its = 350, max_its=350, draw = False,verbose = False)
#n3.iter_converge(n3net, its = 400, max_its=400, draw = False,verbose = False)
n3net.gnam = 'flavor_relax%d-%d'%(relax,samp)
n3net.save(path='./', tv=0)

n3net = n3n.networkBase(JSON='./flavor_relax%d-%d.json'%(relax,samp), **params)

#n3net = n3n.networkBase(pts,edge_list, **params)
#n3n.iter_converge(n3net, its = 500, max_its=1000, draw = False, verbose = False, tol = 0.001, c0 = 0.03)
#n3n.iter_converge(n3net, its = 1000, max_its=5000, draw = False, verbose = False, tol = 0.001, ct = 0.03)
n3n.iter_converge(n3net, its = 500, max_its=3000, draw = False, verbose = False, tol = 0.001, ct = 0.1)


n3net.gnam = 'flavor_relax%d-%d'%(relax,samp)
n3net.save(path='./', tv=0)
net = jlo('./flavor_relax%d-%d.json'%(relax,samp))
#net = jlo('./tmp-%d.json'%samp)
network = gln.simplify_v7(Adj,net)
#network.simplify_network()

network.find_base_loops(3)
network.get_min_loop_set()
network.cal_normalization_const()
norm = network.new_norm
print norm
network.cal_m2_old()
network.cal_m2_normalized_new(norm = norm)
print network.m2
print samp, 'Done'

#n3net.gnam = '2dlattice_%d'%network.m2
#n3net.save(path='./', tv=0)

#n3net = n3.networkBase(JSON='./tmp-%d.json'%samp, **params)
#n3.iter_converge(n3net, its = 1500, max_its=3000, draw = False, verbose = False)
#n3net.gnam = 'lattice_%d'%network.m2
#n3net.save(path='./', tv=0)

el=n3n.tf.reduce_sum(n3net.links.dp**2).eval()

#vs = []
#for i in range(50):
#    print i,
#    n3net.rebin(rebin=n3net.it_num)
#    n3net.iter_all(it=1,ct=0.0)
#    vs += [n3net.V_link.eval().sum()]
#vs = array(vs)
#rep = vs.mean()
#energy = rep + el
  
#a = sum(zip(*n3net.tv[-50:])[1])/50
#edgelen = a.copy()

#open('./gln_flavor_s3_20seg_new.txt','a').write('%f\n'%network.m2)
#open('./gln_norm_flavor_s3_20seg_new.txt','a').write('%f\n'%network.m2_normalized)
#open('./edge_len_flavor_s3_20seg_new.txt','a').write('%f\n'%edgelen)
#open('./energy_flavor_s3_20seg_new.txt','a').write('%f\n'%energy)
#open('./elastic_flavor_s3_20seg_new.txt','a').write('%f\n'%el)
#open('./repulsion_flavor_s3_20seg_new.txt','a').write('%f\n'%rep)

open('./ngln_flavor.txt','a').write('%f\n'%network.m2_normalized)
open('./elastic_flavor.txt','a').write('%f\n'%el)