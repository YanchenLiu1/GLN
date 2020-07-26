from numpy import *
from itertools import combinations
from numpy.random import choice
from scipy.sparse import dok_matrix

def edge_list_2_adj(edge_list):
    N = max(max(list(zip(*edge_list))[0]),max(list(zip(*edge_list))[1]))+1
    Adj = zeros((N,N))
    for edge in edge_list:
        if edge[0]!=edge[1]:
            Adj[edge[0],edge[1]] += 1
            Adj[edge[1],edge[0]] += 1
        else:
            Adj[edge[1],edge[0]] += 1
    return Adj

def gen_ER(n,p):
    Adj = zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            if random.rand()<p:
                Adj[i,j] = 1
                Adj[j,i] = 1
    return Adj

def A2E_weighted(A):
    E = []
    row, col = where(triu(A))
    for i,j in zip(*(row,col)):
        E.append([i,j,A[i,j]])
    E = array(E)
    return E

def A2E(A):
    E = []
    row, col = where(triu(A))
    for i,j in zip(*(row,col)):
        E.append([i,j])
#     E = array(E)
    return E

def add_line_seg(Adj,seg=2):
    E = A2E(Adj)
    E_new = E[:]
    N = len(Adj)
    N_new = N+1
    for e in E:
        E_new.remove(e)
        edge = [e[0]] + [N_new+j for j in range(seg-1)] + [e[1]]
        N_new += seg-1
        for i in range(len(edge)-1):
            E_new.append([edge[i],edge[i+1]])
    Adj_new = edge_list_2_adj(E_new)
    return Adj_new

def add_line_seg_net(net,seg=2,factor=1,xf=1,yf=1,zf=1,center=array([0.0,.0,.0])):
    if factor != 1:
        for _,edge in net['links'].items():
            pts = random.rand(seg-1,3)*factor - 0.5*factor + center
            edge['points'] = [edge['points'][0]] + [list(pts[i]) for i in range(len(pts))] + [edge['points'][1]]
    else:
        a = array([xf,yf,zf])
        for _,edge in net['links'].items():
            pts = random.rand(seg-1,3)*a - 0.5*a + center
            edge['points'] = [edge['points'][0]] + [list(pts[i]) for i in range(len(pts))] + [edge['points'][1]]
    return net

def add_line_perturbation_net_weird(net,eps,seg=2): ## assume straight links
    for i in net['links']:
        points = array(net['links'][i]['points'])
        if sum((points[0]-points[-1])**2)>0:
            #rand = random.rand(seg-1,3)
            x = points[0,0] - points[-1,0]
            y = points[0,1] - points[-1,1]
            z = points[0,2] - points[-1,2]
            a1 = (random.rand()-0.5)*2*eps
            a2 = (random.rand()-0.5)*2*eps
            while 4*a1**2*x**2*y**2-4*(y**2+z**2)*(a1**2*(x**2+z**2)-eps**2*z**2) < 0:
                a1 = (random.rand()-0.5)*2*eps
            while 4*a2**2*x**2*y**2-4*(y**2+z**2)*(a2**2*(x**2+z**2)-eps**2*z**2) < 0:
                a2 = (random.rand()-0.5)*2*eps
            b11 = (-2*a1*x*y+sqrt(4*a1**2*x**2*y**2-4*(y**2+z**2)*(a1**2*(x**2+z**2)-eps**2*z**2)))/2/(y**2+z**2)
            b12 = (-2*a1*x*y-sqrt(4*a1**2*x**2*y**2-4*(y**2+z**2)*(a1**2*(x**2+z**2)-eps**2*z**2)))/2/(y**2+z**2)
            b21 = (-2*a2*x*y+sqrt(4*a2**2*x**2*y**2-4*(y**2+z**2)*(a2**2*(x**2+z**2)-eps**2*z**2)))/2/(y**2+z**2)
            b22 = (-2*a2*x*y-sqrt(4*a2**2*x**2*y**2-4*(y**2+z**2)*(a2**2*(x**2+z**2)-eps**2*z**2)))/2/(y**2+z**2)
            if eps**2-a1**2-b11**2 >= 0:
                new_point1 = points[0,:] + (points[0,:] + points[1,:])/3 + array([a1,b11,sqrt(eps**2-a1**2-b11**2)])
            else:
                new_point1 = points[0,:] + (points[0,:] + points[1,:])/3 + array([a1,b12,sqrt(eps**2-a1**2-b12**2)])
            if eps**2-a2**2-b21**2 >= 0:
                new_point2 = points[0,:] + (points[0,:] + points[1,:])/3*2 + array([a2,b21,sqrt(eps**2-a2**2-b21**2)])
            else:
                new_point2 = points[0,:] + (points[0,:] + points[1,:])/3*2 + array([a2,b22,sqrt(eps**2-a2**2-b22**2)])
        else:
            rand1 = (random.rand(1,3)-0.5)*2
            rand2 = (random.rand(1,3)-0.5)*2
            new_point1 = points[0,:] + (points[0,:] + points[1,:])/3 + (rand1)/sqrt(sum((rand1)**2))*eps
            new_point1 = points[0,:] + (points[0,:] + points[1,:])/3*2 + (rand2)/sqrt(sum((rand2)**2))*eps
        #new_point = (points[0,:] + points[1,:])/2 + (rand-0.5)/sqrt(sum((rand-0.5)**2))*eps
        #net['links'][i]['points'] = [net['links'][i]['points'][0], new_point.tolist()[0], net['links'][i]['points'][-1]]
        net['links'][i]['points'] = [net['links'][i]['points'][0], new_point1.tolist()[0],new_point2.tolist()[0], net['links'][i]['points'][-1]]
    return net


def add_line_perturbation_net(net,eps,seg=2): ## assume straight links
    for i in net['links']:
        points = array(net['links'][i]['points'])
        rand = random.rand(seg-1,3)
        #rand = random.normal(0,1,(seg-1,3))
        #new_point = (points[0,:] + points[1,:])/2 + (rand-0.5)/sqrt(sum((rand-0.5)**2))*eps
        new_point = []
        factor = random.normal(0,eps)
        intervel = (points[-1]-points[0])/seg
        center = array([(points[0] + 0.5*intervel + j*intervel).tolist() for j in range(seg-1)])
        new_point = [list(rand[j]/sqrt(sum(rand[j]**2))*factor + center[j]) for j in range(seg-1)]
        net['links'][i]['points'] = [net['links'][i]['points'][0]] + [list(new_point[j]) for j in range(len(new_point))] + [net['links'][i]['points'][-1]]
    return net

def add_line_perturbation_net_new(net,eps,seg=2): ## assume straight links
    for i in net['links']:
        points = array(net['links'][i]['points'])
        #rand = random.rand(seg-1,3)
        rand = random.normal(0,eps,(seg-1,3))
        #new_point = (points[0,:] + points[1,:])/2 + (rand-0.5)/sqrt(sum((rand-0.5)**2))*eps
        new_point = []
        #factor = random.normal(0,eps)
        intervel = (points[-1]-points[0])/seg
        center = array([(points[0] + j*intervel).tolist() for j in range(1,seg)])
        new_point = rand + center
        net['links'][i]['points'] = [net['links'][i]['points'][0]] + [list(new_point[j]) for j in range(len(new_point))] + [net['links'][i]['points'][-1]]
    return net

def generate_ba_model(N,m, m0=2):    
    if m>m0:
        print('Error! m cannot be larger than m0!')
        raise
    
    A = dok_matrix((N,N), dtype = int)    
    degrees = zeros((N,))
    nodes = arange(N)
    
    init_nodes = list(range(m0))
    source_nodes,target_nodes = zip(*list(combinations(init_nodes,2)))
    A[source_nodes,target_nodes] = 1
    degrees[init_nodes] = m0-1.0
        
    Nt = m0
    while Nt<N:
        target_nodes = choice(nodes[:Nt],size=m,replace=False,p=degrees[:Nt]/degrees[:Nt].sum())
        A[[Nt]*m,target_nodes] = 1
        degrees[target_nodes] += 1
        degrees[Nt] += m
        Nt+=1

    return (A+A.T).toarray()

### function to generate sbm adjacency matrix ###
def get_adj_sbm(N, q, cin, cout, sizelist = None):
    '''
    input:
        N: number of nodes
        q: number of groups
        cin: probability of connection between nodes within one group
        cout: probability of connection between nodes in different groups
    output:
        A: adjacency matrix
    '''
    
    n = N//q
    # make a random NxN matrix (entries between 0 and 1)
    r = random.rand(N,N)
    # symmetrize
    x = arange(N)
    y = x > x[:,newaxis]
    upper = y*r
    r = upper + upper.T
    # brute force: make a matrix of zeros
    A = zeros((N,N))
    if not sizelist: 
        for i in range(q-1):
            for j in range(q-1):
                if i==j:
                    A[i*n:(i+1)*n,j*n:(j+1)*n] = (r[i*n:(i+1)*n,j*n:(j+1)*n] < cin)
                else:
                    A[i*n:(i+1)*n,j*n:(j+1)*n] = (r[i*n:(i+1)*n,j*n:(j+1)*n] < cout)
            A[i*n:(i+1)*n,(q-1)*n:] = (r[i*n:(i+1)*n,(q-1)*n:] < cout)

        for j in range(q-1):
            A[(q-1)*n:,j*n:(j+1)*n] = (r[(q-1)*n:,j*n:(j+1)*n] < cout)

        A[(q-1)*n:,(q-1)*n:] = (r[(q-1)*n:,(q-1)*n:] < cin)
    else:
        c=[0]+list(cumsum(sizelist))
        for i in range(q):
            for j in range(q):
                if i==j:
                    A[c[i]:(c[i]+sizelist[i]),c[i]:(c[i]+sizelist[i])]=(r[c[i]:(c[i]+sizelist[i]),c[i]:(c[i]+sizelist[i])] < cin)
                else:
                    A[c[i]:(c[i]+sizelist[i]),c[j]:(c[j]+sizelist[j])]=(r[c[i]:(c[i]+sizelist[i]),c[j]:(c[j]+sizelist[j])] < cout)
    # remove diagonal 
    A = A - diag(A.diagonal())
    
    return A

def make_tri(n_tri):
    n = n_tri*3
    adj = zeros((n,n))
    for i in range(n_tri):
        adj[i*3,i*3+1] = 1
        adj[i*3,i*3+2] = 1
        adj[i*3+1,i*3] = 1
        adj[i*3+1,i*3+2] = 1
        adj[i*3+2,i*3] = 1
        adj[i*3+2,i*3+1] = 1
        
    return adj

def get_binning(data, num_bins = 15, is_pmf = True, log_binning = False, threshold = 0, low=None, up=None):
    
    # Let's filter out the isolated nodes
    values = list(filter(lambda x:x>threshold, data))
    if len(values)!=len(data):
        print("%s isolated nodes have been removed" % (len(data)-len(values)))
    
    # We need to define the support of our distribution
    if not low:
        lower_bound = min(values)
        upper_bound = max(values)
    else:
        lower_bound = low
        upper_bound = up

    if log_binning:
        lower_bound = log10(lower_bound)
        upper_bound = log10(upper_bound)
        
        # Log binning
        bin_edges = logspace(lower_bound, upper_bound, num_bins+1, base = 10)
        
    else:

        # Linear Binning
        bin_edges = linspace(lower_bound, upper_bound, num_bins+1)
    
    if is_pmf:
        y, __ = histogram(values,bins = bin_edges, density = False)
        p = (y+0.00000000001)/y.sum()
        #print p
    else:
        p, __ = histogram(values, bins = bin_edges, density = True)
        
        #print(p.sum())
        #print((p*np.diff(bin_edges)).sum())
    
    # Now, we need to compute for each y the value of x
    x = bin_edges[1:] - diff(bin_edges)/2 # centering x at the midpoint of the bin
    
    x = x[p>0]
    p = p[p>0]
    
    return x,p
    
def make_n_clique(a,n_c):
    n = n_c*a
    adj = zeros((n,n))
    for i in range(n_c):        
        for j in range(a):
            for k in range(j+1,a):
                adj[i*a+j,i*a+k] = 1
                adj[i*a+k,i*a+j] = 1

        
    return adj

def make_n_polygon(a,n_pol):
    n = n_pol*a
    adj = zeros((n,n))
    for i in range(n_pol):
        adj[i*a,i*a+1] = 1
        adj[i*a,i*a+a-1] = 1
        
        adj[i*a+a-1,i*a] = 1
        adj[i*a+a-1,i*a+a-2] = 1
        for j in range(1,a-1):
            adj[i*a+j,i*a+j-1] = 1
            adj[i*a+j,i*a+j+1] = 1

        
    return adj