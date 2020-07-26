#!/usr/bin/env python

import os,sys
from numpy import *
from itertools import combinations

import net3d_v10_8 as n3
import net3d_v10_8_no_crossing as n3n
import gln_v7 as gln
import functions as fc
from scipy import interpolate

import json
jlo = lambda s: json.load(open(s,'r'))

eps = float(sys.argv[1])

net = jlo('./mouse_straight_edges_new.json')

net = fc.add_line_perturbation_net_new(net,eps,seg=2)
for i in net['links']:
    l = array(net['links'][i]['points'])
    tck, u = interpolate.splprep(l.T.tolist(),k=2)
    new = interpolate.splev(linspace(0,1,10), tck)
    net['links'][i]['points'] = array(new).T.tolist()

edge_list = [net['links'][i]['end_points'] for i in net['links']]
Adj = fc.edge_list_2_adj(edge_list)

#json.dump(net, open('mouse_perturbed_eps%f.json'%eps, 'w'))

network = gln.simplify_v7(Adj,net)
network.find_base_loops(40)
# network.get_min_loop_set()
network.cal_normalization_const()
norm = network.new_norm
print norm
network.cal_m2_old()
network.cal_m2_normalized_new(norm = norm)
print network.m2_normalized

#open('./ngln_mouse_perturbed_3seg.json.txt','a').write('%f\n'%network.m2_normalized)
#open('./perturbation_mouse_perturbed_3seg.json.txt','a').write('%f\n'%eps)
#open('./ngln_no_multi_windings_mouse_perturbed_3seg.txt','a').write('%f\n'%network.ngln_no_multi_windings)

open('./ngln_new_mouse_perturbed_eps%f_interpolate.txt'%eps,'a').write('%f\n'%network.m2_normalized)