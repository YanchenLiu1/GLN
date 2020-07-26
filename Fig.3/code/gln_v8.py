### deleted the network simplification part; do not neet Adj; node can have labels that are not integers ###

from numpy import *
from collections import Counter

def rotate_y(pts,theta):
    R = array([[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]])
    return dot(R,pts.T).T
def rotate_x(pts,theta):
    R = array([[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]])
    return dot(R,pts.T).T
def rotate_z(pts,theta):
    R = array([[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]])
    return dot(R,pts.T).T

class simplify_v8():
    def __init__(self, adj_list, net):
        self.adj_list = adj_list
        self.net = net.copy()
        #self.Deg = sum(Adj,0)
        #self.idx_p = {i:i for i in range(len(self.Deg))}  ## indexing of nodes, position:index
        #self.idx_i = {i:i for i in range(len(self.Deg))}
        #self.Adj_new = Adj
        self.edges = net['links'].copy()
        
        a = list(self.edges.keys())
        
        for edge in a:
            if self.edges[edge]['end_points'][0] <= self.edges[edge]['end_points'][1]:
                if '[%s, %s]'%(self.edges[edge]['end_points'][0],self.edges[edge]['end_points'][1]) not in self.edges:
                    self.edges['[%s, %s]'%(self.edges[edge]['end_points'][0],self.edges[edge]['end_points'][1])] = {0:self.edges[edge]}
                else:
                    self.edges['[%s, %s]'%(self.edges[edge]['end_points'][0],self.edges[edge]['end_points'][1])][len(self.edges['[%s, %s]'%(self.edges[edge]['end_points'][0],self.edges[edge]['end_points'][1])])] = self.edges[edge]
                del self.edges[edge]
            else:
                new_end_points = self.edges[edge]['end_points'][::-1]
                if '[%s, %s]'%(new_end_points[0],new_end_points[1]) not in self.edges:
                    self.edges['[%s, %s]'%(new_end_points[0],new_end_points[1])] = {0:{'end_points': new_end_points, 'points': self.edges[edge]['points'][::-1], 'radius':self.edges[edge]['radius']}}
                else:
                    self.edges['[%s, %s]'%(new_end_points[0],new_end_points[1])][len(self.edges['[%s, %s]'%(new_end_points[0],new_end_points[1])])] = {'end_points': new_end_points, 'points': self.edges[edge]['points'][::-1], 'radius':self.edges[edge]['radius']}
                del self.edges[edge]
            

    def input_base_loops(self,loops):
        self.base_loop = {}
        m = len(loops)
        for i in range(m):
            self.base_loop[i] = {'loop':loops[i]}
            points = []
            for j in range(len(loops[i])-1):
                if loops[i][j]<loops[i][j+1]:
                    points += self.edges['[%s, %s]'%(str(loops[i][j]),str(loops[i][j+1]))][0]['points'][1:]
                else:
                    points += self.edges['[%s, %s]'%(str(loops[i][j+1]),str(loops[i][j]))][0]['points'][::-1][1:]
            points.append(points[0])
            self.base_loop[i]['points'] = points
            
    def input_base_loops_2(self,loops):
        self.base_loop = loops
        
        
        
    
    def find_base_loops(self,num_source): # notice that points does not include same end node twice
#         self.multi_loop = {} # loops comping from multi-edges
        self.base_loop = {}
        #self.base_edges = set() # a set of edges that are not in the mst
        loop_set = set()
        
        ### dealing with multi-edges ###
        for edge in self.edges.values():
#             if tuple(edge[0]['end_points']) not in self.tree_edges:
#                 self.base_edges.add(tuple(edge[0]['end_points']))
            if len(edge) > 1: # multi-edge
                loop_set.add(frozenset(edge[0]['end_points']))
                for i in range(len(edge)): # for each pair of multi_edges
                    for j in range(i+1,len(edge)):
                        seg_1 = edge[i]['points']
                        seg_2 = edge[j]['points'][::-1]
                        loop = seg_1 + seg_2[1:]  # the starting and ending points are the same
#                         self.multi_loop[len(self.multi_loop)] = {'end_points':edge[0]['end_points'], 'points':loop}
#                        self.base_loop[len(self.base_loop)] = {'loop':edge[0]['end_points'], 'points':loop[1:]}
                        self.base_loop[len(self.base_loop)] = {'loop':edge[0]['end_points']+[edge[0]['end_points'][0]], 'points':loop}
                        self.base_loop[len(self.base_loop)-1]['nei'] = {edge[0]['end_points'][0]:[edge[0]['end_points'][1],edge[0]['end_points'][1]], edge[0]['end_points'][1]:[edge[0]['end_points'][0],edge[0]['end_points'][0]]}
#                         self.base_loop[len(self.base_loop)-1]['nei'] = {loop[h]:[loop[h-1],loop[h+1]] for h in range(1,len(loop)-1)}
#                         self.base_loop[len(self.base_loop)-1]['nei'][loop[0]] = [loop[-2],loop[1]]
                        
        
        
#         for edge in self.base_edges:
            
        
#         self.base_edges = list(self.base_edges)
#        N = len(self.idx_i)
        N = len(self.net['nodes']['positions'])
        for n in range(num_source):
#             a = list(self.idx_i.keys())
#             a = a[n:] + a[:n]
#             print a 
            #gnodes=set(self.idx_i.keys())
#            gnodes=set(self.idx_i.keys())
            gnodes=set(self.net['nodes']['labels'])
#             b = gnodes.pop()
#             print b
            root = None
            while gnodes:  # loop over connected components
                if root is None:
#                     root=gnodes.pop()
    #                 root = random.sample(gnodes, 1)[0]
    #                 gnodes.remove(root)
                    a = list(gnodes)
                    if n < len(a):
                        tmpn = int(N*random.rand())
#                         a = a[n+10:] + a[:n+10]
                        a = a[tmpn:] + a[:tmpn]
                        root = a[0]
                    else:
                        root = a[-1]
#                     root = a[0]
#                     print root
                    gnodes.remove(root)
                stack=[root]
                pred={root:root}
                used={root:set()}
                while stack:  # walk the spanning tree finding cycles
                    z=stack.pop()  # use last-in so cycles easier to find
                    zused=used[z]
                    for nbr in self.adj_list[z]:
                    #for nbr in nonzero(self.Adj_new[self.idx_i[z],:])[0]:
                        #nbr = self.idx_p[nbr]
                        if nbr not in used:   # new node
                            pred[nbr]=z
                            stack.append(nbr)
                            used[nbr]=set([z])
                        elif nbr == z:        # self loops
                            if set([z]) not in loop_set:
                                for i in self.edges['[%s, %s]'%(str(z),str(z))].keys():
                                    self.base_loop[len(self.base_loop)] = {'loop':[z, z], 'points':self.edges['[%s, %s]'%(str(z),str(z))][i]['points']}
                                loop_set.add(frozenset([z]))
                            #cycles.append([z])
                        elif nbr not in zused:# found a cycle
                            pn=used[nbr]
                            cycle=[nbr,z]
                            p=pred[z]
                            while p not in pn:
                                cycle.append(p)
                                p=pred[p]
                            cycle.append(p)
                            if set(cycle) not in loop_set:
                                loop_set.add(frozenset(cycle))
                                self.base_loop[len(self.base_loop)] = {'loop':cycle}
                                points = []
                                cycle.append(cycle[0])
                                for i in range(len(cycle)-1):
                                    if cycle[i]<cycle[i+1]:
                                        points += self.edges['[%s, %s]'%(str(cycle[i]),str(cycle[i+1]))][0]['points'][1:]
                                    else:
                                        points += self.edges['[%s, %s]'%(str(cycle[i+1]),str(cycle[i]))][0]['points'][::-1][1:]
                                #points.append(points[0])
                                points = [points[-1]] + points
                                self.base_loop[len(self.base_loop)-1]['points'] = points
                                self.base_loop[len(self.base_loop)-1]['nei'] = {cycle[h]:[cycle[h-1],cycle[h+1]] for h in range(1,len(cycle)-1)}
                                self.base_loop[len(self.base_loop)-1]['nei'][cycle[0]] = [cycle[-2],cycle[1]]

                            #cycles.append(cycle)
                            used[nbr].add(z)
                gnodes-=set(pred)
                #print root
                root=None
                
        self.loop_set = set(frozenset(self.base_loop[i]['loop']) for i in range(len(self.base_loop)))
    

    
    def cal_linking_num(self, lp1, lp2):
        seg1 = [[lp1[i],lp1[i+1]] for i in range(len(lp1)-1)]
        seg2 = [[lp2[i],lp2[i+1]] for i in range(len(lp2)-1)]
        
        total_lk = 0
        for s1 in seg1:   
            k1 = (s1[1][1]-s1[0][1])/(s1[1][0]-s1[0][0])
            b1 = (s1[1][0]*s1[0][1]-s1[0][0]*s1[1][1])/(s1[1][0]-s1[0][0])
            for s2 in seg2:
                k2 = (s2[1][1]-s2[0][1])/(s2[1][0]-s2[0][0])
                b2 = (s2[1][0]*s2[0][1]-s2[0][0]*s2[1][1])/(s2[1][0]-s2[0][0])
                if k1 != k2:
                    x = (b2-b1)/(k1-k2)
#                     y = (k1*b2-k2*b1)/(k1-k2)
                    if ((x-s1[0][0])*(x-s1[1][0]) < 0) and ((x-s2[0][0])*(x-s2[1][0]) < 0): # there's a crossing
                        z1 = (s1[1][2]-s1[0][2])/(s1[1][0]-s1[0][0])*(x-s1[0][0]) + s1[0][2]
                        z2 = (s2[1][2]-s2[0][2])/(s2[1][0]-s2[0][0])*(x-s2[0][0]) + s2[0][2]
                        if z1 > z2:
                            n = array([-(s1[1][1]-s1[0][1]),s1[1][0]-s1[0][0]])
                            r2 = array([s2[1][0]-s2[0][0],s2[1][1]-s2[0][1]])
                            if n.dot(r2)>0:
                                lk = 1
                            else:
                                lk = -1
                        else:
                            n = array([-(s2[1][1]-s2[0][1]),s2[1][0]-s2[0][0]])
                            r2 = array([s1[1][0]-s1[0][0],s1[1][1]-s1[0][1]])
                            if n.dot(r2)>0:
                                lk = 1
                            else:
                                lk = -1
                        total_lk += lk
                        
        return abs(total_lk)/2
    
    def cal_linking_num_array(self, lp1, lp2):
        lp1 = array(lp1)
        lp2 = array(lp2)
#        X11 = lp1[1:,0]
#        X10 = lp1[:-1:,0]
#        Y11 = lp1[1:,1]
#        Y10 = lp1[:-1,1]
#        X21 = lp2[1:,0]
#        X20 = lp2[:-1:,0]
#        Y21 = lp2[1:,1]
#        Y20 = lp2[:-1,1]
        
        k1 = (lp1[1:,2]-lp1[:-1,2])/(lp1[1:,0]-lp1[:-1,0])
        k2 = (lp2[1:,2]-lp2[:-1,2])/(lp2[1:,0]-lp2[:-1,0])
        b1 = (lp1[1:,0]*lp1[:-1,2] - lp1[:-1,0]*lp1[1:,2])/(lp1[1:,0]-lp1[:-1,0])
        b2 = (lp2[1:,0]*lp2[:-1,2] - lp2[:-1,0]*lp2[1:,2])/(lp2[1:,0]-lp2[:-1,0])

        x = (b2[:,newaxis]-b1)/(k1[newaxis]-k2[:,newaxis])
        condition1 = (x-lp1[:-1:,0][newaxis])*(x-lp1[1:,0][newaxis])
        condition2 = (x-lp2[:-1:,0][:,newaxis])*(x-lp2[1:,0][:,newaxis])
        z1 = (lp1[1:,1][newaxis]-lp1[:-1,1][newaxis])/(lp1[1:,0][newaxis]-lp1[:-1,0][newaxis])*(x-lp1[:-1,0][newaxis]) + lp1[:-1,1][newaxis]
        z2 = (lp2[1:,1][:,newaxis]-lp2[:-1,1][:,newaxis])/(lp2[1:,0][:,newaxis]-lp2[:-1,0][:,newaxis])*(x-lp2[:-1,0][:,newaxis]) + lp2[:-1,1][:,newaxis]

        n1 = array([-(lp1[1:,2]-lp1[:-1,2]), (lp1[1:,0]-lp1[:-1,0])])
        n2 = array([-(lp2[1:,2]-lp2[:-1,2]), (lp2[1:,0]-lp2[:-1,0])])
        r1 = array([(lp2[1:,0]-lp2[:-1,0]), (lp2[1:,2]-lp2[:-1,2])])
        r2 = array([(lp1[1:,0]-lp1[:-1,0]), (lp1[1:,2]-lp1[:-1,2])])

        condition3 = -(lp2[1:,0]-lp2[:-1,0])[:,newaxis]*(lp1[1:,2]-lp1[:-1,2])[newaxis]+(lp1[1:,0]-lp1[:-1,0])[newaxis]*(lp2[1:,2]-lp2[:-1,2])[:,newaxis]  # n.dot(r2) for z1 > z2
        mask1 = zeros_like(x)
        mask1[where(k1[newaxis]!=k2[:,newaxis])] = 1.  # k1!=k2
        mask1[where((condition1>0)+(condition2>0))] = 0   # condition1 <0 and condition2 <0

        return abs((((z1>z2)*1 + (z1<z2)*(-1))*((condition3>0)*1 + (condition3<0)*-1)*mask1).sum())/2
        
    
    
    def cal_linking_num_array_new(self, lp1, lp2):
        lp1 = array(lp1)
        lp2 = array(lp2)

        k1 = (lp1[1:,1]-lp1[:-1,1])/(lp1[1:,0]-lp1[:-1,0])
        k2 = (lp2[1:,1]-lp2[:-1,1])/(lp2[1:,0]-lp2[:-1,0])
        b1 = (lp1[1:,0]*lp1[:-1,1] - lp1[:-1,0]*lp1[1:,1])/(lp1[1:,0]-lp1[:-1,0])
        b2 = (lp2[1:,0]*lp2[:-1,1] - lp2[:-1,0]*lp2[1:,1])/(lp2[1:,0]-lp2[:-1,0])

        x = (b2[:,newaxis]-b1[newaxis])/(k1[newaxis]-k2[:,newaxis])
        condition1 = (x-lp1[:-1,0][newaxis])*(x-lp1[1:,0][newaxis])
        condition2 = (x-lp2[:-1,0][:,newaxis])*(x-lp2[1:,0][:,newaxis])
        z1 = (lp1[1:,2][newaxis]-lp1[:-1,2][newaxis])/(lp1[1:,0][newaxis]-lp1[:-1,0][newaxis])*(x-lp1[:-1,0][newaxis]) + lp1[:-1,2][newaxis]
        z2 = (lp2[1:,2][:,newaxis]-lp2[:-1,2][:,newaxis])/(lp2[1:,0][:,newaxis]-lp2[:-1,0][:,newaxis])*(x-lp2[:-1,0][:,newaxis]) + lp2[:-1,2][:,newaxis]
        # z2 = (lp2[1:,1][newaxis]-lp2[:-1,1][newaxis])/(lp2[1:,0][newaxis]-lp2[:-1,0][newaxis])*(x-lp2[:-1,0][newaxis]) + lp2[:-1,1][newaxis]


        n1 = array([-(lp1[1:,1]-lp1[:-1,1]), (lp1[1:,0]-lp1[:-1,0])])
        n2 = array([-(lp2[1:,1]-lp2[:-1,1]), (lp2[1:,0]-lp2[:-1,0])])
        r1 = array([(lp2[1:,0]-lp2[:-1,0]), (lp2[1:,2]-lp2[:-1,1])])
        r2 = array([(lp1[1:,0]-lp1[:-1,0]), (lp1[1:,2]-lp1[:-1,1])])

        condition3 = -(lp2[1:,0]-lp2[:-1,0])[:,newaxis]*(lp1[1:,1]-lp1[:-1,1])[newaxis]+(lp1[1:,0]-lp1[:-1,0])[newaxis]*(lp2[1:,1]-lp2[:-1,1])[:,newaxis]  # n.dot(r2) for z1 > z2
        mask1 = zeros_like(x)
        mask1[where(k1[newaxis]!=k2[:,newaxis])] = 1.  # k1!=k2
        # mask1[where((condition1>=0)+(condition2>=0))] = 0   # condition1 <0 and condition2 <0
        mask1[where(condition1>=0)] = 0
        mask1[where(condition2>=0)] = 0

        return abs((((z1>z2)*1 + (z1<z2)*(-1))*((condition3>0)*1 + (condition3<0)*-1)*mask1).sum())/2

        
    

    def cal_m2_old(self):
        
#         self.new_norm = 0
        self.tangled_loops = {}
        self.m2 = 0
        self.gln_no_multi_windings = 0
        
        for i in self.base_loop.keys():
            for j in range(i+1,len(self.base_loop)):
        #         if j == 459:
                a = array(self.base_loop[i]['points']).copy()
                c = array([a[k+1][0]-a[k][0] for k in range(len(a)-1)])
                if not set(self.base_loop[i]['loop']).intersection(set(self.base_loop[j]['loop'])): # no common nodes
                    b = array(self.base_loop[j]['points']).copy()
                    d = array([b[k+1][0]-b[k][0] for k in range(len(b)-1)])

                    xmax1 = max(a[:,0])
                    xmin1 = min(a[:,0])
                    ymax1 = max(a[:,1])
                    ymin1 = min(a[:,1])
                    zmax1 = max(a[:,2])
                    zmin1 = min(a[:,2])
                    xmax2 = max(b[:,0])
                    xmin2 = min(b[:,0])
                    ymax2 = max(b[:,1])
                    ymin2 = min(b[:,1])
                    zmax2 = max(b[:,2])
                    zmin2 = min(b[:,2])
                    if (xmin1<xmax2 and xmin2<xmax1) and (ymin1<ymax2 and ymin2<ymax1) and (zmin1<zmax2 and zmin2<zmax1):                      

                        if prod((c > 0)+(c < 0)) == 0:
                            a = rotate_x(a,pi*0.01354)
                            a = rotate_y(a,pi*0.01354)
                            a = rotate_z(a,pi*0.02)
                            a = rotate_x(a,pi*0.01354)
                            a = rotate_y(a,pi*0.01354)
                            a = rotate_z(a,pi*0.02)
                            b = rotate_x(b,pi*0.01354)
                            b = rotate_y(b,pi*0.01354)
                            b = rotate_z(b,pi*0.02)
                            b = rotate_x(b,pi*0.01354)
                            b = rotate_y(b,pi*0.01354)
                            b = rotate_z(b,pi*0.02)
                            c = array([a[k+1][0]-a[k][0] for k in range(len(a)-1)])
                            d = array([b[k+1][0]-b[k][0] for k in range(len(b)-1)])
                            if prod((c > 0)+(c < 0)) == 0:
                                a1 = array(self.base_loop[i]['points'])
                                o = where(c==0)[0] + 1
                                mask = ones(len(a1),dtype=bool)
                                mask[o] = False
                                a1 = a1[mask]
                                self.base_loop[i]['points'] = [list(a1[u]) for u in range(len(a1))]
                                a = array(self.base_loop[i]['points']).copy()
                                b = array(self.base_loop[j]['points']).copy()
                                c = array([a[k+1][0]-a[k][0] for k in range(len(a)-1)])
                                if prod((c > 0)+(c < 0)) == 0:
                                    a = rotate_x(a,pi*0.01354)
                                    a = rotate_y(a,pi*0.01354)
                                    a = rotate_z(a,pi*0.02)
                                    a = rotate_x(a,pi*0.01354)
                                    a = rotate_y(a,pi*0.01354)
                                    a = rotate_z(a,pi*0.02)
                                    b = rotate_x(b,pi*0.01354)
                                    b = rotate_y(b,pi*0.01354)
                                    b = rotate_z(b,pi*0.02)
                                    b = rotate_x(b,pi*0.01354)
                                    b = rotate_y(b,pi*0.01354)
                                    b = rotate_z(b,pi*0.02)
                                    #c = array([a[k+1][0]-a[k][0] for k in range(len(a)-1)])
                                    d = array([b[k+1][0]-b[k][0] for k in range(len(b)-1)])
        #                    if prod((d > 0)+(d < 0)) == 0:
        #                        b = rotate_y(b,pi*0.0142653)
        #                        b = rotate_x(b,pi*0.051533)
        #                        b = rotate_z(b,pi*0.023524)
        #                        b = rotate_y(b,pi*0.0142653)
        #                        b = rotate_x(b,pi*0.051533)
        #                        b = rotate_z(b,pi*0.023524)
        #                        d = array([b[k+1][0]-b[k][0] for k in range(len(b)-1)])
                            if prod((d > 0)+(d < 0)) == 0:
                                a2 = array(self.base_loop[j]['points'])
                                o = where(d==0)[0] + 1
                                mask = ones(len(a2),dtype=bool)
                                mask[o] = False
                                a2 = a2[mask]
                                self.base_loop[j]['points'] = [list(a2[u]) for u in range(len(a2))]
                                a = array(self.base_loop[i]['points']).copy()
                                b = array(self.base_loop[j]['points'])
                                c = array([a[k+1][0]-a[k][0] for k in range(len(a)-1)])
                                if prod((c > 0)+(c < 0)) == 0:
                                    a = rotate_x(a,pi*0.01354)
                                    a = rotate_y(a,pi*0.01354)
                                    a = rotate_z(a,pi*0.02)
                                    a = rotate_x(a,pi*0.01354)
                                    a = rotate_y(a,pi*0.01354)
                                    a = rotate_z(a,pi*0.02)
                                    b = rotate_x(b,pi*0.01354)
                                    b = rotate_y(b,pi*0.01354)
                                    b = rotate_z(b,pi*0.02)
                                    b = rotate_x(b,pi*0.01354)
                                    b = rotate_y(b,pi*0.01354)
                                    b = rotate_z(b,pi*0.02)
                        elif prod((d > 0)+(d < 0)) == 0:
                            a = rotate_x(a,pi*0.01354)
                            a = rotate_y(a,pi*0.01354)
                            a = rotate_z(a,pi*0.02)
                            a = rotate_x(a,pi*0.01354)
                            a = rotate_y(a,pi*0.01354)
                            a = rotate_z(a,pi*0.02)
                            b = rotate_x(b,pi*0.01354)
                            b = rotate_y(b,pi*0.01354)
                            b = rotate_z(b,pi*0.02)
                            b = rotate_x(b,pi*0.01354)
                            b = rotate_y(b,pi*0.01354)
                            b = rotate_z(b,pi*0.02)
                            c = array([a[k+1][0]-a[k][0] for k in range(len(a)-1)])
                            d = array([b[k+1][0]-b[k][0] for k in range(len(b)-1)])
                            if prod((d > 0)+(d < 0)) == 0:                       
                                a2 = array(self.base_loop[j]['points'])
                                o = where(d==0)[0] + 1
                                mask = ones(len(a2),dtype=bool)
                                mask[o] = False
                                a2 = a2[mask]
                                self.base_loop[j]['points'] = [list(a2[u]) for u in range(len(a2))]
                                a = array(self.base_loop[i]['points']).copy()
                                b = array(self.base_loop[j]['points'])
        #                        d = array([b[k+1][0]-b[k][0] for k in range(len(b)-1)])
        #                            if prod((d > 0)+(d < 0)) == 0:
        #                                b = rotate_x(b,pi*0.01354)
        #                                b = rotate_y(b,pi*0.01354)
        #                                b = rotate_z(b,pi*0.02)
        #                                b = rotate_x(b,pi*0.01354)
        #                                b = rotate_y(b,pi*0.01354)


                        m2 =  self.cal_linking_num_array_new(a,b)
                    
        
                        #m2 = self.cal_linking_num(a,b)
                        self.m2 += m2
                        #self.gln_no_multi_windings += bool(m2)
                        if m2!=0:
                            self.gln_no_multi_windings +=1
                            self.tangled_loops[len(self.tangled_loops)] = {'loop1':i,'loop2':j,'ln':m2}


    def loop_subtraction(self, i, j):
        comm = set(self.base_loop[i]['loop']).intersection(set(self.base_loop[j]['loop']))
        lp1 = self.base_loop[i]['loop']
        lp2 = self.base_loop[j]['loop']
        
        dict_1 = self.base_loop[i]['nei']
        dict_2 = self.base_loop[j]['nei']
        
        edge_set_1 = set(frozenset((dict_1[b][0],b)) for b in comm).union(set(frozenset((dict_1[b][1],b)) for b in comm))
        edge_set_2 = set(frozenset((dict_2[b][0],b)) for b in comm).union(set(frozenset((dict_2[b][1],b)) for b in comm))
                            
        comm_edge = edge_set_1.intersection(edge_set_2)
        
        max_id = [i,j][where(array([len(lp1),len(lp2)])==max([len(lp1),len(lp2)]))[0][0]]
        
#         if len(lp1)>len(lp2):
#             max_id = i
#         else:
#             max_id = j

#         my_dict = {lp1[k]:lp1[k+1] for k in range(len(lp1)-1)}
        my_dict = set((lp1[k],lp1[k+1]) for k in range(len(lp1)-1))
#         print 'my_dict for the first loop:', my_dict
#         print 'the first loop:', lp1    
#         print 'the second loop:', lp2
        
        edge = list(comm_edge.pop())
        b = edge[0]
        d = edge[1]
        if dict_2[b][0]==d:
            if (b,d) in my_dict:
                flag = 1
            else:
                flag = 0
        else:
            if (d,b) in my_dict:
                flag = 1
            else:
                flag = 0
                
                
        while comm_edge:
            edge = list(comm_edge.pop())
            if dict_2[edge[0]][0]==edge[1]:
                if (edge[0],edge[1]) in my_dict:
                    if flag == 0:
                        return 0
                else:
                    if flag == 1:
                        return 0
            else:
                if (edge[1],edge[0]) in my_dict:
                    if flag == 0:
                        return 0
                else:
                    if flag == 1:
                        return 0

        c = b
        while d!=b:
#             print 'b=%d, d=%d'%(b,d), d!=b
            d = dict_2[c][flag]
#             print 'd=', d
            #if (d,c) in my_dict.items():
            if (d,c) in my_dict:
                #del my_dict[d]
                my_dict.remove((d,c))
#                 print 'removed:',(d,c)
            else:
#                 my_dict_new[c] = d
                my_dict.add((c,d))
#                 print 'not remove:',(d,c)
            c = d
#         print 'my_dict_new:', my_dict_new
#         print 'my_dict after deletion of mutual edges:', my_dict
        
        
            
#         for u,v in my_dict_new.items():
#             #my_dict[u] = v
#             my_dict.add((u,v))
#         print 'my_dict after combining with my_dict_new:', my_dict
        
        flag_1 = 0
#         print lp1,lp2
#         print my_dict
        
        my_list = [item[0] for item in my_dict] + [item[1] for item in my_dict]
        a = Counter(my_list)
        source_nodes = [u for u,v in a.items() if v==4]
#         num_of_new_loop = len(source_nodes)+1
        
        new_loops = []
        if len(source_nodes)>0:
            for node in source_nodes:
                while [item[1] for item in my_dict if item[0]==node]:
                    new_loop = [node]
                    old = node
                    b = -11
                    while b!=node:
                        search = [item[1] for item in my_dict if item[0]==old]
#                         for m in search:
#                             if m not in new_loop:
#                                 break
                        b = search[0]
                        new_loop.append(b)
                        my_dict.remove((old,b))
                        old = b
                    a = Counter(new_loop)
                    h = [u for u,v in a.items() if v==2]
                    if len(h)>=2:
                        h.remove(node)
                        new_loop_tmp = [new_loop]
                        h = [p for p in new_loop if p in h]
                        h = h[:len(h)/2]
                        self.new_loop = new_loop
                        self.original_h =(array(h)).copy()
                        while h:
                            index = [p for p in range(len(new_loop_tmp[-1])) if new_loop_tmp[-1][p]==h[0]]
                            new_loop1 = new_loop_tmp[-1][index[0]:index[1]] + [new_loop_tmp[-1][index[0]]]
                            new_loop2 = new_loop_tmp[-1][:index[0]] + new_loop_tmp[-1][index[1]:]
                            self.new_loop1 = new_loop1
                            self.new_loop2 = new_loop2
                            self.long_loop = new_loop_tmp[-1]
                            del h[0]
                            del new_loop_tmp[-1]
                            new_loop_tmp.append(new_loop2) 
                            new_loop_tmp.append(new_loop1)
                        new_loops += new_loop_tmp
                    else:
                        new_loops.append(new_loop)
#                 new_loop1 = [node]
#                 new_loop2 = [node]
    #             search = [item[1] for item in my_dict if item[0]==node]
    #             b = search[0]
#                 old = node
#                 b = -11
#                 while b!=node:
#                     search = [item[1] for item in my_dict if item[0]==old]
#                     b = search[0]
#                     new_loop1.append(b)
#                     my_dict.remove((old,b))
#                     old = b
#                 old = node
#                 b = -11
#                 while b!=node:
#                     search = [item[1] for item in my_dict if item[0]==old]
#                     b = search[0]
#                     new_loop2.append(b)
#                     my_dict.remove((old,b))
#                     old = b 
                    
#                 new_loops.append(new_loop1)
#                 new_loops.append(new_loop2)
                

        while my_dict:
            new_loop = []
            a = my_dict.pop()
            new_loop += list(a)
            b = a[1]
            old = b
            while b!=a[0]:
                search = [item[1] for item in my_dict if item[0]==old]
                b = search[0]
                new_loop.append(b)
                my_dict.remove((old,b))
                old = b
            new_loops.append(new_loop)
        
        for new_loop in new_loops:
            if len(new_loop) < max(len(lp1),len(lp2)):
                flag_1 = 1
                
                if set(new_loop) not in self.loop_set:
#                         print 'plus one loop:',max(self.base_loop_copy.keys())+1
                        self.base_loop_copy[max(self.base_loop_copy.keys())+1] = {'loop':new_loop}
                        self.base_loop_copy[max(self.base_loop_copy.keys())]['nei'] = {new_loop[h]:[new_loop[h-1],new_loop[h+1]] for h in range(1,len(new_loop)-1)}
                        self.base_loop_copy[max(self.base_loop_copy.keys())]['nei'][new_loop[0]] = [new_loop[-2],new_loop[1]]

    #                     print new_loop
                        points = []
                        for i in range(len(new_loop)-1):
                            if new_loop[i]<new_loop[i+1]:
                                points += self.edges['[%s, %s]'%(str(new_loop[i]),str(new_loop[i+1]))][0]['points'][1:]
                            else:
                                points += self.edges['[%s, %s]'%(str(new_loop[i+1]),str(new_loop[i]))][0]['points'][::-1][1:]
                        points.append(points[0])

                        self.base_loop_copy[max(self.base_loop_copy.keys())]['points'] = points

                        self.loop_set.add(frozenset(new_loop))
                    
        if flag_1 == 1:
            if set(self.base_loop[max_id]['loop']) in self.loop_set:
                self.loop_set.remove(frozenset(self.base_loop[max_id]['loop']))
            if max_id in self.base_loop_copy.keys():
#                 print 'minus one loop:',max_id
                del self.base_loop_copy[max_id]
#             self.base_loop_copy = dict((i,self.base_loop_copy[i]) if i<max_id else (i,self.base_loop_copy[i+1]) 
#                               for i in range(len(self.base_loop_copy)))
        return flag_1
            


    def get_min_loop_set(self):
#         num = len(self.base_loop)
#         new_num = 0
        k = 0
        flag = 1
        self.base_loop_copy = self.base_loop.copy()
        while flag>0:
            flag = 0
            k += 1
            print k
            
            for i in range(len(self.base_loop)):
                for j in range(i+1,len(self.base_loop)):
                    if not (i in self.base_loop_copy.keys() and j in self.base_loop_copy.keys()):
                        continue
                    a = list(set(self.base_loop[i]['loop']).intersection(set(self.base_loop[j]['loop'])))
                    if not (len(self.base_loop[i]['loop']) == 4 and len(self.base_loop[j]['loop']) == 4):
                        if len(a) >= 2:
                            edge_set_1 = set(frozenset((self.base_loop[i]['nei'][b][0],b)) for b in a).union(set(frozenset((self.base_loop[i]['nei'][b][1],b)) for b in a))
                            edge_set_2 = set(frozenset((self.base_loop[j]['nei'][b][0],b)) for b in a).union(set(frozenset((self.base_loop[j]['nei'][b][1],b)) for b in a))
                            #if set(self.base_loop[i]['nei'][a[0]]).intersection(set(a)) and set(self.base_loop[j]['nei'][a[0]]).intersection(set(a)):
                            #if set(nei_list_1).intersection(a) and set(nei_list_2).intersection(a):
                            if edge_set_1.intersection(edge_set_2):  
                                if len(edge_set_1.intersection(edge_set_2)) >= 2 or len(a)>=3:
                                    flag += self.loop_subtraction(i,j)
            tmp = list(self.base_loop_copy.keys())
            self.base_loop_copy = dict((i,self.base_loop_copy[tmp[i]]) for i in range(len(self.base_loop_copy)))
            self.base_loop = self.base_loop_copy.copy()
#             new_num = len(self.base_loop)

                
    
    
    def cal_normalization_const(self):
        self.new_norm = 0
        for i in self.base_loop.keys():
            for j in range(i+1,len(self.base_loop)):
                if not set(self.base_loop[i]['loop']).intersection(set(self.base_loop[j]['loop'])): # no common nodes
                    self.new_norm += 1
        
    
    def cal_m2_normalized(self):
        
        l = len(self.base_loop)
        if l>=2:
            self.m2_normalized = 2*float(self.m2)/l/(l-1)
        else:
            self.m2_normalized = 0
            
    def cal_m2_normalized_new(self, norm = None):
        
        if not norm:
            if self.new_norm>=1:
                self.m2_normalized = float(self.m2)/self.new_norm
                self.ngln_no_multi_windings = float(self.gln_no_multi_windings)/self.new_norm
            else:
                self.m2_normalized = 0
        else:
            if norm >= 1:
                self.m2_normalized = float(self.m2)/norm
                self.ngln_no_multi_windings = float(self.gln_no_multi_windings)/norm
            else:
                self.m2_normalized = 0