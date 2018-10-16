
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
#matplotlib inline

from pprint import pprint 

# create a function that maps transition probability dataframe 
# to markov edges and weights

def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx,col)] = Q.loc[idx,col]
    return edges

def viterbi(pi, a, b, obs):
    
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]
    
    # init blank path
    path = np.zeros(T)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))
  
    # init delta and phi 
    print(delta[:,0])
    print(pi)
    print(b[:,obs[0]])
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    print('\nStart Walk Forward\n')    
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]] 
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
    
    # find optimal path
    print('-'*50)
    print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    #p('init path\n    t={} path[{}-1]={}\n'.format(T-1, T, path[T-1])) #LPW
    for t in range(T-2, -1, -1): 
        path[t] = phi[int(path[t+1]), [t+1]]
        #p(' '*4 + 't={t}, path[{t}+1]={path}, [{t}+1]={i}'.format(t=t, path=path[t+1], i=[t+1])) #LPW
        print('path[{}] = {}'.format(t, path[t]))
        
    return path, delta, phi



hidden_states = ['A+', 'C+', 'G+', 'T+','A- ', 'C-', 'G-', 'T-']
pi = [0.05,0.03, 0.03, 0.03, 0.22,0.22,0.22,0.2]
state_space = pd.Series(pi, index=hidden_states, name='states')
print(state_space)
print('\n', state_space.sum())
print()
p = .48
q = .52

a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.18*p, 0.274*p, 0.426*p, 0.12*p, (1-p)/4, (1-p)/4, (1-p)/4, (1-p)/4]
a_df.loc[hidden_states[1]] = [0.171*p, 0.368*p, 0.274*p, 0.188*p, (1-p)/4, (1-p)/4, (1-p)/4, (1-p)/4]
a_df.loc[hidden_states[2]] = [0.161*p, 0.339*p, 0.375*p, 0.125*p, (1-p)/4, (1-p)/4, (1-p)/4, (1-p)/4]
a_df.loc[hidden_states[3]] = [0.079*p, 0.355*p, 0.384*p, 0.182*p, (1-p)/4, (1-p)/4, (1-p)/4, (1-p)/4]
a_df.loc[hidden_states[4]] = [(1-q)/4, (1-q)/4, (1-q)/4, (1-q)/4, 0.3*q, 0.205*q, 0.285*q, 0.21*q]
a_df.loc[hidden_states[5]] = [(1-q)/4, (1-q)/4, (1-q)/4, (1-q)/4, 0.322*q, 0.298*q, 0.078*q, 0.302*q]
a_df.loc[hidden_states[6]] = [(1-q)/4, (1-q)/4, (1-q)/4, (1-q)/4, 0.248*q, 0.246*q, 0.298*q, 0.208*q]
a_df.loc[hidden_states[7]] = [(1-q)/4, (1-q)/4, (1-q)/4, (1-q)/4, 0.177*q, 0.239*q, 0.292*q, 0.292*q]

print(a_df)

a = a_df.values
print('\n', a, a.shape, '\n')
print(a_df.sum(axis=1))
print()
# create matrix of observation (emission) probabilities
# b or beta = observation probabilities given state
# matrix is size (M x O) where M is number of states (A+,A-,...etc)
# and O is number of different possible observations

observable_states = ['A', 'C', 'G','T']
print()
print("observable_states:\n", observable_states)
print("hidden_states:\n", hidden_states)
print()


b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states[0]] = [1, 0, 0,0]
b_df.loc[hidden_states[1]] = [0, 1, 0,0]
b_df.loc[hidden_states[2]] = [0, 0, 1,0]
b_df.loc[hidden_states[3]] = [0, 0, 0,1]
b_df.loc[hidden_states[4]] = [1, 0, 0,0]
b_df.loc[hidden_states[5]] = [0, 1, 0,0]
b_df.loc[hidden_states[6]] = [0, 0, 1,0]
b_df.loc[hidden_states[7]] = [0, 0, 0,1]

print(b_df)

b = b_df.values
print('\n', b, b.shape, '\n')
print(b_df.sum(axis=1))
print()


# Graphing begin

# Now we create the graph edges and the graph object. 
# create graph edges and weights

hide_edges_wts = _get_markov_edges(a_df)
pprint(hide_edges_wts)

emit_edges_wts = _get_markov_edges(b_df)
pprint(emit_edges_wts)
print()

# create graph object
G = nx.MultiDiGraph()

# nodes correspond to states
G.add_nodes_from(hidden_states)
#print(f'Nodes:\n{G.nodes()}\n')
print('Nodes:\n', G.nodes(), '\n')

# edges represent hidden probabilities
for k, v in hide_edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

# edges represent emission probabilities
for k, v in emit_edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
    
print('Edges:')
pprint(G.edges(data=True))    

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='graphviz-2.38/release/bin/neato')
nx.draw_networkx(G, pos)

# create edge labels for jupyter plot but is not necessary
emit_edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G , pos, edge_labels=emit_edge_labels)
nx.drawing.nx_pydot.write_dot(G, 'CpGIslands.dot')
# In Windows: dot -Tps filename.dot -o outfile.ps

print()
#Graphing end

obs_map = {'A':0, 'C':1, 'G':2, 'T':3}
lower_obs_map = {'a':0, 'c':1, 'g':2, 't':3}

mFile = open('dna_seq.txt')
mObs = []

with mFile as f:
    for line in f:
        for c in line.rstrip():
            mObs.append(lower_obs_map[c])

#print(mObs)
obs = np.array(mObs)

inv_obs_map = dict((v,k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]

print( pd.DataFrame(np.column_stack([obs, obs_seq]), 
                columns=['Obs_code', 'Obs_seq']) )
print()


path, delta, phi = viterbi(pi, a, b, obs)
print('\nsingle best state path: \n', path)
print('delta:\n', delta)
print('phi:\n', phi)
print()

# Let's take a look at the result. 
state_map = {0:'I', 1:'I', 2:'I' , 3:'I', 4:'N', 5:'N', 6:'N', 7:'N'}
state_path = [state_map[v] for v in path]

print()
print("RESULT:")
print(pd.DataFrame()
 .assign(Observation=obs_seq)
 .assign(Best_Path=state_path))

