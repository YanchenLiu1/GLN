
import os,sys

from numpy import *
from itertools import combinations


import json
jlo = lambda s: json.load(open(s,'r'))

import gln_v8 as gln
import functions as fc

def Lattice(n, d=3):
    m = int(n**(1./d))+1
    print( m) 
    pts = array([float32((arange(n)/m**i) % m) for i in range(d)]).T
    edg = []
    for i in range(n):
        for j in range(i+1,n):
            if absolute(pts[i]-pts[j]).sum() == 1:
                edg += [[i,j]]
#     for i in range(n):
    return pts,edg


def expand(pts, n):
    s = pts.shape
    x = linspace(0,s[0]-1,n)
    xp = arange(s[0]) 
    return array([interp(x,xp,i) for i in pts.T]).T



def draw_net_tv(pts,edg ,dims = [0,1], figsize=(10,5), node_col = '#3355ba', link_col = '#33aa55', 
                new=True, tv=True, **kw):
    
    if new: figure(figsize=figsize)
    subplot(aspect = 'equal')

    scatter(*pts.T[dims], zorder=100, c=node_col)
    for l in edg:
        plot(*pts[int0(l)[:2]].T[dims],c=link_col)
    


########################
# Simple Layout, for fast initial layout
########################
POWn = 2
POW_SN = 2

class netSimple:
    def __init__(self,pts, edg, directed = False, **kw):
        self.pts = array(pts)
        self.elist = array(edg)
        self.get_Adj_mat(directed)
        self.params = {
            'A': 200,
            'pow': POWn,
            'n_radius': 1.,
            'k': 10,
        }
        self.params.update(kw)
        
        self.dt = 1e-3
        self.it_num = 0
        self.tv = []
        self.t = 0
        self._init(kw)
        
    def _init(self):
        pass 
    
    def get_Adj_mat(self, directed):
        self.Adj = zeros((len(self.pts), len(self.pts)))
        for l in self.elist:
            w = (l[2] if len(l) > 2 else 1.)
            l0 = tuple(int0(l[:2]))
            self.Adj[l0] =  w
            if not directed:
                self.Adj[l0[::-1]] = w 
            
    def iter_(self):
        self.get_distz()
        self.t += self.dt
        self.tv += [[self.t, self.lens()]]
        self.it_num += 1

        self.compute_forces()
        self.f_max = abs(self.force_mat).max()
        self.update_dt()
        self.pts += self.dt * self.force_mat.sum(0)
        
    def iter_all(self,t, ct = 0.1):
        self.ct = ct
        for i in range(t):
            self.iter_()
            
        
    def update_dt(self):
        """should be done repeatedly, but as it converges exponentially, we'll use it once in every iter."""
        if self.dt > self.ct/self.f_max:
            self.dt /= 2
        elif self.dt < 2*self.ct /self.f_max:
            self.dt *= 2
            
    def compute_forces(self):
        self.comp_F_NN()
        self.comp_F_NL()
        self.force_mat = self.F_NN + self.F_NL
        
    def get_distz(self):
        self.dp = (self.pts - self.pts[:,newaxis])
        self.dpn = linalg.norm(self.dp, axis=-1, keepdims=1)
        
    def lens(self, pow = 1):
        return ((self.dpn[:,:,0] * self.Adj)**pow).sum()/2
        
    def comp_F_NN(self):
        A = self.params['A']
        ex = self.params['pow']
        r = self.params['n_radius']
        self.F_NN = A * self.dp / r * (self.dpn/r)**(ex-2) * exp(-(self.dpn/r)**ex)
        
    def comp_F_NL(self):
        k = self.params['k']
        self.F_NL = - k * self.dp * self.Adj[:,:,newaxis] # weights also determine strength of springs
        

class netRads(netSimple):
    def _init(self,kw):
        self.params['nodes']={
            'radius':self.params['n_radius'],
            'weighted':True,
            'labels':list(range(len(self.pts))),
        } # for compatibility with networkBase
        self.params.setdefault('long_rep', 0.01) 
        self.params['links']= {
            'T0': self.params['nodes']['radius'] / 2.,
            'ce': 100.,
            'weighted':True,
            'thickness': 0.1*self.params['n_radius']
        }
        if 'nodes' in kw: self.params['nodes'].update(kw['nodes'])
        if 'links' in kw: self.params['links'].update(kw['links'])
        
        self.net_name = 'net-n%d-L%d' %(len(self.pts),len(self.elist))
        self.get_degrees()
        self.set_radii()

    def get_degrees(self):
        self.degrees = self.Adj.sum(1)
    
    def set_radii(self):
        d = self.pts.shape[-1]
        rs = lambda x: ((x**(d-1)).sum(-1))**(1./(d-1))+1e-5 # use root sum squared of weights as radii for 3D
        self.degreesRS = r0 = (rs(self.Adj) if self.params['nodes']['weighted'] else array([1]*len(self.Adj)))
        
        self.rad_mat = self.params['nodes']['radius']*(r0+r0[:,newaxis])[:,:,newaxis]
        
    def comp_f_short(self):
        A = self.params['A']
        ex = self.params['pow']
        r = self.rad_mat # self.params['n_radius']
        #self.f_short = A * self.dp / r * (self.dpn/r)**(ex-2) * exp(-(self.dpn/r)**ex)
        # we want forces to get stronger for larger nodes because they also have more elastic forces pulling them 
        self.f_short = A * self.dp * r**POW_SN * (self.dpn/r)**(ex-2) * exp(-(self.dpn/r)**ex)
        # this should automatically solve any issue from strong elastic forces
        
    def comp_f_long(self):
        d = float(self.pts.shape[-1])
        self.f_long = self.params['long_rep'] * self.params['A'] * self.dp / (self.dpn**(d-1) +1e-3)**(d/(d-1))
        
    def comp_thermal_noise(self):
        self.noise = self.params['links']['T0'] * exp(- self.it_num/self.params['links']['ce']) * random.randn(*self.pts.shape)
        
    def iter_(self):
        r = self.params['nodes']['radius']/10.
        self.get_distz()
        self.t += self.dt
        self.tv += [[self.t, self.lens()]]
        self.it_num += 1

        self.compute_forces()
        self.f_max = abs(self.force_mat).max()
        self.update_dt()
        self.pts += r * arcsinh( self.dt * self.force_mat.sum(0) / r) 
        
        
    def iter_all(self,t,ct=0.1, **kw):
        self.ct = ct
        for i in range(t):
            self.iter_()
            self.comp_thermal_noise()
            self.pts+= self.noise
        
    def comp_F_NN(self):
        self.comp_f_short()
        self.comp_f_long()
        self.comp_thermal_noise()
        
        self.F_NN = self.f_short + self.f_long 
        
    def make_links(net):
        #['end_points', 'points', 'radius', 'weight']
        lnks = {}
        nl = net.params['nodes']['labels']
        for k in net.elist:
            k0 = int0(k[:2])
            lnks[str('%s;%s'%(nl[k0[0]],nl[k0[1]]))] = { #str(k0)
                'end_points': k0.tolist(), 
                'points': net.pts[k0].tolist(), 
                'radius': net.params['links']['thickness']*(1 if len(k)<3 else k[2]), 
                'weight': 1
            }
        return lnks

    def make_net_json(net):
        net.JSON = {'scale':1,
                   'links': net.make_links(), #net.make_links(),
                   'nodes': {
                       'positions':net.pts.tolist(),
                       'labels': list(net.params['nodes']['labels'])#range(len(net.pts))
                   },
                   'info': net.params
                   }
        #net.JSON['info']['nodes'].update({'labels':net.JSON['nodes']['labels']})

    def save(net, path, tv=True):    
        fnam = path+'/'+net.net_name
        net.params['links']['thickness'] = .1* net.params['nodes']['radius'] 
        net.make_net_json() #net.net_json()
        json.dump(net.JSON, open(fnam+'.json', 'w'))
        if tv: savetxt(fnam+'-tv.txt',net.tv) 
