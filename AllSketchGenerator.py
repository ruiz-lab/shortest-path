import numpy as np
from itertools import chain
import time

from Graphs import connectedErdosRenyiGraph
from ShortestPathAlgorithms import shortestDistances_networkx, shortestDistances_GNN, seedSets

def generate_ErdosRenyi_randomSeeds(num_graphs,k_max,n,lbd):
    all_graph_info = []
    all_random_seeds = []
    for N in range(num_graphs):
        while True:
            try:
                graph_info = connectedErdosRenyiGraph(n,lbd/n)
                break
            except:
                pass
        all_graph_info.append(graph_info)
        random_seeds = []
        for k in range(k_max):
            random_seeds.append(seedSets(graph_info,k+1))
        all_random_seeds.append(random_seeds)
        print(N)
    return all_graph_info,all_random_seeds

def get_distance_matrices(graph_info,seeds,which_cuda,model1=None,model2=None,criterion_type='mse',method = 'networkx'):
    
    assert criterion_type in ['mse','l2','l1'], 'Invalid criterion type.'
    assert method in ['networkx','GNN'], 'Invalid method.'
    
    if model1 == None:
        if method != 'networkx':
            raise AssertionError('Invalid inputs.')
        if graph_info[1]:
            y_pred_source_seeds,dur1 = shortestDistances_networkx(graph_info, sources=seeds)
            y_pred_target_seeds,dur2 = shortestDistances_networkx(graph_info, targets=seeds)
            dur = dur1+dur2
            return [y_pred_source_seeds.T,y_pred_target_seeds],dur
        else:
            y_pred,dur = shortestDistances_networkx(graph_info, sources=seeds)
            return [y_pred],dur
    else:
        if graph_info[1]:
            y_pred_source_seeds,dur1 = shortestDistances_GNN(which_cuda,model1,criterion_type,graph_info,seeds)
            y_pred_target_seeds,dur2 = shortestDistances_GNN(which_cuda,model2,criterion_type,graph_info,seeds)
            dur = dur1+dur2
            return [y_pred_source_seeds,y_pred_target_seeds],dur
        else:
            y_pred,dur = shortestDistances_GNN(which_cuda,model1,criterion_type,graph_info,seeds)
            return [y_pred],dur
        
def get_all_sketches(distance_matrices,seedSet,seeds):

    index_dict = dict(zip(seeds,range(len(seeds))))
    nodes = list(range(distance_matrices[0].shape[0]))
    #print(len(distance_matrices))

    if len(distance_matrices) == 1:
        dists = distance_matrices[0]
        SarmaSketches = dict()
        BourgainSketches = dict()
        time1 = time.time()
        for node in nodes:
            all_selected_dists = [[dists[node][index_dict.get(i)] for i in S] for S in seedSet]
            indices_list = [[index for index, value in enumerate(l) if value == min(l)] for l in all_selected_dists]
            SarmaSketches[node] = np.array([(S[i],min(selected_dists)) for S,indices,selected_dists in list(zip(seedSet,indices_list,all_selected_dists)) for i in indices])
        time2 = time.time()
        for node in nodes:
            all_selected_dists = [[dists[node][index_dict.get(i)] for i in S] for S in seedSet]
            BourgainSketches[node] = np.array([min(l) for l in all_selected_dists])
        time3 = time.time()
        return [SarmaSketches],[BourgainSketches],time2-time1,time3-time2
    else:
        dist_source_seeds = distance_matrices[0]
        dist_target_seeds = distance_matrices[1]
        SarmaSketches_source = dict()
        BourgainSketches_source = dict()
        SarmaSketches_target = dict()
        BourgainSketches_target = dict()
        for node in nodes:
            all_selected_dists = [[dist_source_seeds[node][index_dict.get(i)] for i in S] for S in seedSet]
            indices_list = [[index for index, value in enumerate(l) if value == min(l)] for l in all_selected_dists]
            SarmaSketches_target[node] = np.array([(S[i],min(selected_dists)) for S,indices,selected_dists in list(zip(seedSet,indices_list,all_selected_dists)) for i in indices])
            all_selected_dists = [[dist_target_seeds[node][index_dict.get(i)] for i in S] for S in seedSet]
            indices_list = [[index for index, value in enumerate(l) if value == min(l)] for l in all_selected_dists]
            SarmaSketches_source[node] = np.array([(S[i],min(selected_dists)) for S,indices,selected_dists in list(zip(seedSet,indices_list,all_selected_dists)) for i in indices])
        time2 = time.time()
        for node in nodes:
            all_selected_dists = [[dist_source_seeds[node][index_dict.get(i)] for i in S] for S in seedSet]
            BourgainSketches_target[node] = np.array([min(l) for l in all_selected_dists])
            all_selected_dists = [[dist_target_seeds[node][index_dict.get(i)] for i in S] for S in seedSet]
            BourgainSketches_source[node] = np.array([min(l) for l in all_selected_dists])
        time3 = time.time()
        return [SarmaSketches_source,SarmaSketches_target],[BourgainSketches_source,BourgainSketches_target],time2-time1,time3-time2

def time_all_sketches_all_graphs(samples,which_cuda,model1=None,model2=None,criterion_type='mse',method='networkx'):
    
    assert criterion_type in ['mse','l2','l1'], 'Invalid criterion type.'
    assert method in ['networkx','GNN'], 'Invalid method.'

    all_graph_info = samples[0]
    all_seedSets = samples[1]    
    num_graphs = len(all_graph_info)
    k_max = len(all_seedSets[0])
    all_dur_dist = np.zeros((num_graphs,k_max))
    all_dur_Sarma = np.zeros((num_graphs,k_max))
    all_dur_Bourgain = np.zeros((num_graphs,k_max))
    for N in range(num_graphs):
        if N % 10 == 0:
            print(str(N/num_graphs*100)+'% complete',num_graphs)
        graph_info = all_graph_info[N]
        seedSets = all_seedSets[N]
        for k in range(k_max):
            seedSet = seedSets[k]
            seeds = list(chain(*seedSet))
            seeds.sort()
            seeds = list(set(seeds))
            distance_matrices,dur_dist = get_distance_matrices(graph_info,seeds,which_cuda,model1,model2,criterion_type,method)
            SarmaSketches, BourgainSketches, dur_Sarma, dur_Bourgain = get_all_sketches(distance_matrices,seedSet,seeds)
            all_dur_dist[N,k] = dur_dist
            all_dur_Sarma[N,k] = dur_Sarma
            all_dur_Bourgain[N,k] = dur_Bourgain

    return all_dur_dist,all_dur_Sarma,all_dur_Bourgain