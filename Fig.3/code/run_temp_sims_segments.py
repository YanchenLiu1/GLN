#!/usr/bin/env python

from temperature_gln import *
from collections import defaultdict

sim_params = {
    'N':100,
    'd':3,
    'T':1.,
    'samples':50,
    'segments': 2,
}
sim_params.update(eval(sys.argv[1]))


n, d, segments, Temperature, samples= (sim_params[k] for k in ['N','d','segments', 'T','samples' ])

pts, edgelist = Lattice(n,d)
Adj = fc.edge_list_2_adj(edgelist)
Adj_list = defaultdict(list)
for i,j in edgelist:
    Adj_list[i].append(j)
    Adj_list[j].append(i)



# file_name = 'Temperature-Lattice-%dD-N%d-T%.3g.csv' %(d,n, Temperature)
file_name = 'Temperature-Lattice-'+ '-'.join([k+'%g' %v for k,v in sim_params.items()]) +'.csv' 

print ('Running: %s\n with params: %s'% (file_name, sim_params))

##################
## Loops ###
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

#############

stats = {k:[] for k in ['GLN','nGLN','Edge_Length','Elastic_Energy']}
net_params = {'n_radius':0.2, 'links':{'thickness':0.1,'k':1.,'T0':0,'ce':15}, 'A':10}

from pandas import DataFrame

for _ in range(samples):
    print _ 
    n3net = netRads(pts, edgelist,**net_params)
    
    n3net.make_net_json()
    n3net.dt = 0
    n3net.ct = 1
    n3net.iter_()
    # To calculate lengths
    
    # Add Thermal noise and segments  
    net = n3net.JSON
    for l in net['links']:
        p = expand(array(net['links'][l]['points']), segments )
        net['links'][l]['points'] = (p + Temperature * random.randn(*p.shape)).tolist()

    network = gln.simplify_v8(Adj_list,net)
    network.input_base_loops(loops_list)
    network.cal_normalization_const()
    norm = network.new_norm
    network.cal_m2_old()
    network.cal_m2_normalized_new(norm = norm)
    stats['GLN'] += [network.m2]
    stats['nGLN'] += [network.m2_normalized]
    
    stats['Edge_Length'] += [n3net.lens()]
    stats['Elastic_Energy'] += [n3net.lens(2)] # using len**2 for elastic energy
    if _ % 100 == 0:
        p = DataFrame(stats)
        p.to_csv(os.path.join( '../sims/temperature/',file_name),index = None)
#     break


p = DataFrame(stats)
p.to_csv(os.path.join( '../sims/temperature/',file_name),index = None)