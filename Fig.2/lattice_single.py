#!/usr/bin/env python

import os,sys
import net3d_v10_8_long as n3_long
import net3d_v10_8 as n3
import gln_v7 as gln
from numpy import *
import json
jlo = lambda s: json.load(open(s,'r'))
import functions as fc
from itertools import combinations

samp = int(sys.argv[1])
n = 400

_, edge_list = n3.Lattice(n,2)
Adj = fc.edge_list_2_adj(edge_list)
pts = random.rand(len(Adj),3)

n3net = n3_long.netRads(pts, edge_list)
n3net.gnam = 'tmp%d'%(samp)
n3net.save(path='./', tv=0)
net = jlo('./tmp%d.json'%(samp))

loops = set()
for i in range(len(Adj)):
    tmp = where(Adj[i,:]!=0)[0]
    for j,k in combinations(tmp, 2):
        a = set(where(Adj[j,:]!=0)[0])
        b = set(where(Adj[k,:]!=0)[0])
        c = a.intersection(b)
        while c:
            d = c.pop()
            if d!=i:
                loop = frozenset([i,j,k,d])
                loops.add(loop)
                break
loops_list = []
for loop in loops:
    loop = list(loop)
    if Adj[loop[0],loop[1]]!=0:
        tmp = [loop[0],loop[1]]
        if Adj[loop[1],loop[2]]!=0:
            tmp += [loop[2],loop[3],loop[0]]
        else:
            tmp += [loop[3],loop[2],loop[0]]
    else:
        tmp = [loop[0],loop[2]]
        if Adj[loop[2],loop[3]]!=0:
            tmp += [loop[3],loop[1],loop[0]]
        else:
            tmp += [loop[1],loop[3],loop[0]]
    loops_list.append(tmp)
    
    
network = gln.simplify_v7(Adj,net)
network.input_base_loops(loops_list)

network.cal_normalization_const()
norm = network.new_norm
print norm
network.cal_m2_old()
network.cal_m2_normalized_new(norm = norm)
print network.m2
print samp

open('./gln_lattice_rand.txt','a').write('%f\n'%network.m2)
open('./ngln_lattice_rand.txt','a').write('%f\n'%network.m2_normalized)