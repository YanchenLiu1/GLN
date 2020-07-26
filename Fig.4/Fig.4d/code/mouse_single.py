#!/usr/bin/env python3

import os,sys

from numpy import *
from itertools import combinations

import net3d_3_v11_5 as n3

import functions as fc
from scipy import interpolate

import json
jlo = lambda s: json.load(open(s,'r'))

eps = float(sys.argv[1])
samp = int(sys.argv[2])
epsi = int(sys.argv[3])

relaxation_time = [5000,5000,8000]
ct = [0.01,0.01,0.1]

net = jlo('./mouse_straight_edges_new.json')
#net = jlo('./mouse_brain_links_not_reversed-2019-07-05-symmetrized-line_spreads-aggregated_nodes-n26-L472.json')

net = fc.add_line_perturbation_net_new(net,eps,seg=2)
for i in net['links']:
    l = array(net['links'][i]['points'])
    tck, u = interpolate.splprep(l.T.tolist(),k=2)
    new = interpolate.splev(linspace(0,1,10), tck)
    net['links'][i]['points'] = array(new).T.tolist()


json.dump(net, open('../sims/initial/mouse_sim_eps%f_samp%d_interpolate_initial.json'%(eps,samp), 'w'))


params = {
    'links':
                   {'k':1.0,'amplitude':1e2, 'thickness': 0.14067288590231117,  #0.7352812666783554, 
                    'Temp0':.0, 'ce':10, 'segs':80,
                   'weighted': 1,
                   },
}
net = n3.networkBase(JSON='../sims/initial/mouse_sim_eps%f_samp%d_interpolate_initial.json'%(eps,samp), 
                     fixed=True, keep_paths=True, **params)
net.net_name = 'mouse_sim_eps%f_samp%d_interpolate'%(eps,samp)

# n3.iter_converge(net, its = 500, max_its=2000, save_path='../sim-mouse/relax/', draw = 0, verbose = False, tol = 0.00001, c0 = 0.1)
#n3.iter_converge(net, its =300, max_its=6000, save_path='../sim-mouse/relax/', draw = 0, verbose = False, tol = 0.00001, ct = 1.)
#n3.iter_converge(net, its =300, max_its=15000, save_path='../sim-mouse/relax/', draw = 0, verbose = False, tol = 0.00001, ct = 0.8)
# n3.iter_converge(net, its = 500, max_its=15000, save_path='../sims/relax/', draw = 0, verbose = False, tol = 0.00001, ct = 0.1)
n3.iter_converge(net, its = 200, max_its=relaxation_time[epsi], save_path='../sims/relax/', draw = 0, verbose = False, tol = 0.00001, ct = ct[epsi])

net = jlo('../sims/relax/mouse_sim_eps%f_samp%d_interpolate.json'%(eps,samp))
elastic = []
k = net['info']['links']['k']*79
for i in net['links']:
    points = array(net['links'][i]['points'])
    r = net['links'][i]['radius']
    elastic.append(((points[1:]- points[:-1])**2).sum()*r*k)
elastic_energy = sum(elastic)


open('../sims/stats/mouse_sim_relaxed_eps%f_elastic_energy_interpolate_new.txt'%eps,'a').write('%f\n'%elastic_energy)