import networkx as nx
import numpy as np
from collections import deque
import torch
import time
from Models import predict

def shortestDistances_networkx(graph_info,sources=None,targets=None):
    time1 = time.time()
    G = graph_info[0]
    num_nodes = len(G.nodes())
    if sources == None and targets == None:
        dist = np.zeros((num_nodes,num_nodes))
        if 'weight' in G.graph:
            shortest_path_lengths = nx.shortest_path_length(G,weight="weight")
        else:
            shortest_path_lengths = nx.shortest_path_length(G)
        for source_node, lengths in shortest_path_lengths:
            keys_list = [n for n in range(num_nodes) if n not in lengths.keys()]
            values_list = np.ones(len(keys_list))*np.inf
            lengths.update(dict(zip(keys_list, values_list)))
            lengths = dict(sorted(lengths.items(), key=lambda item: item[0]))
            dist[int(source_node)] = list(lengths.values())
    elif targets == None:
        dist = np.zeroes((len(sources),num_nodes))
        for i in range(len(sources)):
            if 'weight' in G.graph:
                shortest_path_lengths = nx.shortest_path_length(G,source=sources[i],weight="weight")
            else:
                shortest_path_lengths = nx.shortest_path_length(G,source=sources[i])
            keys_list = [n for n in range(num_nodes) if n not in shortest_path_lengths.keys()]
            values_list = np.ones(len(keys_list))*np.inf
            shortest_path_lengths.update(dict(zip(keys_list, values_list)))
            shortest_path_lengths = dict(sorted(shortest_path_lengths.items(), key=lambda item: item[0]))
            dist[i] = np.array(list(shortest_path_lengths.values()))
    elif sources == None:
        dist = np.zeroes((num_nodes,len(targets)))
        for i in range(len(targets)):
            if 'weight' in G.graph:
                shortest_path_lengths = nx.shortest_path_length(G,target=targets[i],weight="weight")
            else:
                shortest_path_lengths = nx.shortest_path_length(G,target=targets[i])
            keys_list = [n for n in range(num_nodes) if n not in shortest_path_lengths.keys()]
            values_list = np.ones(len(keys_list))*np.inf
            shortest_path_lengths.update(dict(zip(keys_list, values_list)))
            shortest_path_lengths = dict(sorted(shortest_path_lengths.items(), key=lambda item: item[0]))
            dist[:,i] = np.array(list(shortest_path_lengths.values()))
    elif len(sources) == 1 and len(targets) == 1:
        try:
            if 'weight' in G.graph:
                dist = nx.shortest_path_length(G,source=sources,target=targets,weight="weight")
            else:
                dist = nx.shortest_path_length(G,source=sources,target=targets)
        except:
            dist = np.inf
    else:
        if len(sources) <= len(targets):
            dist,_ = shortestDistances_networkx(graph_info,sources=sources)
            dist = dist[:,targets]
        else:
            dist,_ = shortestDistances_networkx(graph_info,targets=targets)
            dist = dist[sources]
    time2 = time.time()
    return dist,time2-time1

def shortestDistances_GNN(model,criterion_type,graph_info,seeds=None):
    time1 = time.time()
    gpu_bool = torch.cuda.is_available()
    G = graph_info[0]
    num_nodes = len(G.nodes())
    if seeds == None:
        seeds = G.nodes()
    num_seeds = len(seeds)
    samples_edge_index = [torch.tensor(np.array(list(G.edges())).T).to(torch.int64)]
    if graph_info[2]: ## if weighted
        samples_weights = [torch.tensor(list(nx.get_edge_attributes(G,'weight').values())).to(torch.float32)]
    else:
        samples_weights = []
    one_hot_seeds = np.zeros((num_nodes,num_seeds))
    for i in range(num_seeds):
        one_hot_seeds[seeds[i],i] = 1
    one_hot_seeds = torch.tensor(one_hot_seeds.astype(np.float32), requires_grad=True)
    r = model.out_channels
    if r == 1:
        y_pred = predict(gpu_bool, model, criterion_type, [one_hot_seeds], samples_edge_index, samples_weights)[0]
    else:
        samples_x = []
        if num_seeds % r != 0:
            n_extra = int(np.ceil(num_seeds/r))*r - num_seeds
            other_nodes = [node for node in range(num_nodes) if node not in seeds]
            if len(other_nodes) < n_extra:
                nodes = seeds + other_nodes + seeds[:(n_extra-len(other_nodes))]
            else:
                nodes = seeds + other_nodes[:n_extra]
        else:
            nodes = seeds
        list_of_seeds = [nodes[i:i + r] for i in range(0, len(nodes), r)]
        for s in list_of_seeds:
            x = np.zeros((num_nodes,r))
            for i in range(r):
                x[s[i],i] = 1
            samples_x.append(torch.tensor(x.astype(np.float32), requires_grad=True))
        y_pred = predict(gpu_bool, model, criterion_type, samples_x, samples_edge_index, samples_weights)
        y_pred = np.concatenate(y_pred, axis=1)[:,:num_seeds]
    # my_dict = {}
    # for i in range(num_seeds):
    #     my_dict[seeds[i]] = y_pred[:,i]
    time2 = time.time()
    return y_pred.T,time2-time1 # transpose if model predicts distances from seeds to all nodes

def closestSeed_fromSource(G, source, target_set):
    # Works for weighted & unweighted, directed & undirected graphs 
    # Doesn't allow self-directed nodes
    closest_point, smallest_weight = None, float('inf')
    target_set = [x for x in target_set if x != source]
    if len(target_set) > 0:
        visited = set()  # To keep track of visited vertices (vertices in paths stored in queue)
        queue = deque([(source, 0, [])])  # Initialize the queue with the starting vertex, its path, and the total weight
        storage = deque() # To store nodes removed from queue
        while queue:
            vertex, weight, path = queue.popleft()  # Dequeue a vertex, its path, and the total weight
            storage.append((vertex, weight))
            if vertex not in target_set:
                for neighbor in G.neighbors(vertex):
                    if 'weight' in G.graph:
                        edge_weight = G.get_edge_data(vertex, neighbor)['weight'] 
                    else:
                        edge_weight = 1
                    if neighbor not in path and neighbor != vertex and edge_weight != 0:
                        new_weight = weight + edge_weight
                        if neighbor not in visited:
                            queue.append((neighbor, new_weight, path + [vertex]))
                            visited.add(neighbor)
                        else: # Replace the path from 'start' to 'neighbor' in queue or storage with a shorter one if found
                            weight_info = [(index,item[1]) for index,item in enumerate(list(queue)) if item[0] == neighbor]
                            if len(weight_info) != 0:
                                for index, w in weight_info:
                                    if new_weight < w:
                                        del queue[index]
                                        queue.append((neighbor, new_weight, path + [vertex]))
                            else:
                                weights = [item[1] for item in storage if item[0] == neighbor]
                                if len(weights) > 0 and new_weight < np.min(weights):
                                    queue.append((neighbor, new_weight, path + [vertex]))
            else:
                if closest_point == None:
                    closest_point, smallest_weight = vertex, weight
                else:
                    if weight < smallest_weight:
                        closest_point, smallest_weight = vertex, weight
                k = len(queue)
                for i in range(k):
                    if smallest_weight <= queue[k-1-i][1]:
                        queue.remove(queue[k-1-i]) # Remove all paths in queue which are longer than the current shortest from 'start' to a seed 
    return closest_point, smallest_weight

def closestSeed_toTarget(G, target, source_set): 
    # Works for weighted & unweighted, directed & undirected graphs 
    # Doesn't allow self-directed nodes
    closest_point, smallest_weight = None, float('inf')
    source_set = [x for x in source_set if x != target]
    if len(source_set) > 0:
        visited = set()  # To keep track of visited vertices (vertices in paths stored in queue)
        queue = deque()  # Initialize the queue with the starting vertex, its path, and the total weight
        for i in range(len(source_set)):
            queue.append((source_set[i], 0, []))
        storage = deque()

        while queue:
            vertex, weight, path = queue.popleft()  # Dequeue a vertex, its path, and the total weight
            storage.append((vertex, weight))
            if vertex != target:
                for neighbor in G.neighbors(vertex):
                    if 'weight' in G.graph:
                        edge_weight = G.get_edge_data(vertex, neighbor)['weight'] 
                    else:
                        edge_weight = 1
                    if neighbor not in path and neighbor not in source_set and neighbor != vertex and edge_weight != 0:
                        new_weight = weight + edge_weight
                        if neighbor not in visited:
                            queue.append((neighbor, new_weight, path + [vertex]))
                            visited.add(neighbor)
                        else: # Replace the path from a start node to 'neighbor' in queue with a shorter one if found
                            weight_info = [(index,item[1]) for index,item in enumerate(list(queue)) if item[0] == neighbor]
                            if len(weight_info) != 0:
                                for index, w in weight_info:
                                    if new_weight < w:
                                        del queue[index]
                                        queue.append((neighbor, new_weight, path + [vertex]))
                            else:
                                weights = [item[1] for item in storage if item[0] == neighbor]
                                if len(weights) > 0 and new_weight < np.min(weights):
                                    queue.append((neighbor, new_weight, path + [vertex]))            
            else:
                if closest_point == None:
                    closest_point, smallest_weight = vertex, weight
                else:
                    if weight < smallest_weight:
                        closest_point, smallest_weight = vertex, weight
                k = len(queue)
                for i in range(k):
                    if smallest_weight <= queue[k-1-i][1]:
                        queue.remove(queue[k-1-i]) # Remove all paths in queue which are longer than the current shortest from 'start' to a seed
    return closest_point, smallest_weight

###### Offine Sketch ######

def sampleSet(graph_info,K):
    G = graph_info[0]
    support = [node for node, degree in G.degree() if degree >= 1]
    if len(support) == 0:
        return []
    r = int(np.floor(np.sqrt(len(G.nodes()))))
    sample_sizes = 2**np.array(range(r+1))
    sampled = [np.random.choice(support,size=sum(sample_sizes),replace=False) for i in range(K)]
    sublists = []
    for nodes in sampled:
        start = 0
        for size in sample_sizes:
            sublists.append(nodes[start:start+size])
            start += size
    return sublists

def offlineSketch(sample_set,dist=None,graph_info=None,source=None,target=None):
    if sample_set == []:
        raise ValueError('Cannot sample a sufficent number of nodes because graph is strongly disconnected.')
    if dist == None:
        if source == None and target == None:
            raise ValueError('Invalid inputs.')
        elif graph_info == None:
            raise ValueError('Invalid inputs.')
    if dist != None:
        return np.array([(idx,dist[idx]) for S in sample_set for idx in S if dist[idx] == min([dist[i] for i in S])])
    elif source != None:
        return np.array([closestSeed_fromSource(graph_info[0],source,S) for S in sample_set])
    elif target != None:
        return np.array([closestSeed_toTarget(graph_info[0],target,S) for S in sample_set])
    else:
        raise ValueError('Invalid inputs.')

###### Online ######
    
def online_Sarma(sketch_u,sketch_v):
    dist = np.inf
    if sketch_u.shape[0] != 0 and sketch_v.shape[0] != 0:
        common_nodes = [w for w in sketch_u[:,0] if w in sketch_v[:,0]]
        for w in common_nodes:
            dist = min(sketch_u[sketch_u[:, 0] == w][0,1] + sketch_v[sketch_v[:, 0] == w][0,1],dist) 
    return dist

def online_Bourgain(d_u_S,d_v_S,d_S_u = None,d_S_v = None):
    if d_S_u == None and d_S_v == None:
        to_remove = [idx for idx,val in enumerate(list(zip(d_u_S,d_v_S))) if val[0] == float('inf') or val[1] == float('inf')]
        d_u_S = np.array([val for idx, val in enumerate(d_u_S) if idx not in to_remove])
        d_v_S = np.array([val for idx, val in enumerate(d_v_S) if idx not in to_remove])
        return np.max(np.abs(d_u_S-d_v_S))
    elif d_S_u != None and d_S_v != None:
        to_remove = [idx for idx,val in enumerate(list(zip(d_u_S,d_v_S))) if val[0] == float('inf') or val[1] == float('inf')]
        d_u_S = np.array([val for idx, val in enumerate(d_u_S) if idx not in to_remove])
        d_v_S = np.array([val for idx, val in enumerate(d_v_S) if idx not in to_remove])
        to_remove = [idx for idx,val in enumerate(list(zip(d_S_u,d_S_v))) if val[0] == float('inf') or val[1] == float('inf')]
        d_S_u = np.array([val for idx, val in enumerate(d_S_u) if idx not in to_remove])
        d_S_v = np.array([val for idx, val in enumerate(d_S_v) if idx not in to_remove])
        return max([0,np.max(d_S_v-d_S_u),np.max(d_u_S-d_v_S)])
    else:
        raise ValueError('Invalid inputs.')

###### Shortest Distances ######
    
def shortestDistances_Sarma(graph_info,sample_set,sources=None,targets=None,model=None,criterion_type=None,method='networkx'):
    time1 = time.time()
    G = graph_info[0]
    num_nodes = len(G.nodes())
    if sources == None and targets == None:
        if method == 'networkx':
            dist = shortestDistances_networkx(graph_info)
        elif method == 'GNN':
            dist = shortestDistances_GNN(model,criterion_type,graph_info)
        all_sketch_u = []
        if method == 'BFS':
            for u in range(num_nodes):
                all_sketch_u.append(offlineSketch(sample_set,graph_info,source=u))
        else:
            for u in range(num_nodes):
                all_sketch_u.append(offlineSketch(sample_set,dist=dist[u]))
        if graph_info[1]: # if directed
            all_sketch_v = []
            if method == 'BFS':
                for v in range(num_nodes):
                    all_sketch_v.append(offlineSketch(sample_set,graph_info,target=v))
            else:
                for v in range(num_nodes):
                    all_sketch_v.append(offlineSketch(sample_set,dist=dist[:,v]))
        else:
            all_sketch_v = all_sketch_u
        time2 = time.time()
        dist_Sarma = np.ones((num_nodes,num_nodes))*np.inf
        if graph_info[1]: ## if directed
            for u in range(num_nodes):
                for v in range(num_nodes):
                    if u == v:
                        dist_Sarma[u,v] = 0
                    else:
                        dist_Sarma[u,v] = online_Sarma(all_sketch_u[u],all_sketch_v[v])
        else:
            for u in range(num_nodes):
                dist_Sarma[u,u] = 0
                for v in range(u+1,num_nodes):
                    dist_Sarma[u,v] = online_Sarma(all_sketch_u[u],all_sketch_v[v])
                    dist_Sarma[v,u] = dist_Sarma[u,v]
        time3 = time.time()
    elif targets == None:
        if method == 'networkx':
            dist = shortestDistances_networkx(graph_info)
        elif method == 'GNN':
            dist = shortestDistances_GNN(model,criterion_type,graph_info)
        all_sketch_v = []
        if method == 'BFS':
            for v in range(num_nodes):
                all_sketch_v.append(offlineSketch(sample_set,graph_info,target=v))
        else:
            for v in range(num_nodes):
                all_sketch_v.append(offlineSketch(sample_set,dist=dist[:,v]))
        if graph_info[1]: # if directed
            all_sketch_u = []
            if method == 'BFS':
                for u in sources:
                    all_sketch_u.append(offlineSketch(sample_set,graph_info,source=u))
            else:
                for u in sources:
                    all_sketch_u.append(offlineSketch(sample_set,dist=dist[u]))
        else:
            all_sketch_u = [all_sketch_v[u] for u in sources]
        time2 = time.time()
        dist_Sarma = np.ones((len(sources),num_nodes))*np.inf
        for i in range(len(sources)):
            for v in range(num_nodes):
                if u == v:
                    dist_Sarma[i,v] = 0
                else:
                    dist_Sarma[i,v] = online_Sarma(all_sketch_u[i],all_sketch_v[v])
        time3 = time.time()
    elif sources == None:
        if method == 'networkx':
            dist = shortestDistances_networkx(graph_info)
        elif method == 'GNN':
            dist = shortestDistances_GNN(model,criterion_type,graph_info)
        all_sketch_u = []
        if method == 'BFS':
            for u in range(num_nodes):
                all_sketch_u.append(offlineSketch(sample_set,graph_info,source=u))
        else:
            for u in range(num_nodes):
                all_sketch_u.append(offlineSketch(sample_set,dist=dist[u]))
        if graph_info[1]: # if directed
            all_sketch_v = []
            if method == 'BFS':
                for v in targets:
                    all_sketch_v.append(offlineSketch(sample_set,graph_info,target=v))
            else:
                for v in targets:
                    all_sketch_v.append(offlineSketch(sample_set,dist=dist[:,v]))
        else:
            all_sketch_v = [all_sketch_u[v] for v in targets]
        time2 = time.time()
        dist_Sarma = np.ones((num_nodes,len(targets)))*np.inf
        for i in range(len(targets)):
            for u in range(num_nodes):
                if u == v:
                    dist_Sarma[u,i] = 0
                else:
                    dist_Sarma[u,i] = online_Sarma(all_sketch_u[u],all_sketch_v[i])
        time3 = time.time()
    elif len(sources) == 1 and len(targets) == 1:
        u = sources[0]
        v = targets[0]
        if u != v:
            if method == 'networkx':
                sketch_u = offlineSketch(sample_set,shortestDistances_networkx(G,source=u))
                sketch_v = offlineSketch(sample_set,shortestDistances_networkx(G,target=v))
            elif method == 'BFS':
                sketch_u = offlineSketch(sample_set,graph_info,source=u)
                sketch_v = offlineSketch(sample_set,graph_info,target=v)
            else:
                if graph_info[1]: ## if directed
                    dist = shortestDistances_GNN(model,criterion_type,graph_info)
                    sketch_u = offlineSketch(sample_set,dist[u])
                    sketch_v = offlineSketch(sample_set,dist[:,v])
                else:
                    dist = shortestDistances_GNN(model,criterion_type,graph_info,[u,v])
                    sketch_u = offlineSketch(sample_set,dist[0])
                    sketch_v = offlineSketch(sample_set,dist[1])
            time2 = time.time()
            dist_Sarma = np.inf
            if sketch_u.shape[0] != 0 and sketch_v.shape[0] != 0:
                dist_Sarma = online_Sarma(sketch_u,sketch_v)
            time3 = time.time()
        else:
            return 0,0,0
    else:
        if len(sources) <= len(targets):
            dist_Sarma,d31,d21 = shortestDistances_Sarma(graph_info,sample_set,sources=sources,model=model,criterion_type=criterion_type,method=method)
            dist_Sarma = dist_Sarma[:,targets]
        else:
            dist_Sarma,d31,d21 = shortestDistances_Sarma(graph_info,sample_set,targets=targets,model=model,criterion_type=criterion_type,method=method)
            dist_Sarma = dist_Sarma[sources]
        return dist_Sarma,d31,d21
    return dist_Sarma,dist_Sarma,time3-time1,time2-time1

def shortestDistances_Bourgain(graph_info,sample_set,sources=None,targets=None,model=None,criterion_type=None,method='networkx'):
    time1 = time.time()
    G = graph_info[0]
    num_nodes = len(G.nodes())
    if sources == None and targets == None:
        if method == 'networkx':
            dist = shortestDistances_networkx(graph_info)
        elif method == 'GNN':
            dist = shortestDistances_GNN(model,criterion_type,graph_info)
        if graph_info[1]: # if directed
            all_d_u_S = []
            all_d_S_u = []
            if method == 'BFS':
                for u in range(num_nodes):
                    all_d_u_S.append(offlineSketch(sample_set,graph_info,source=u)[:,1])
                    all_d_S_u.append(offlineSketch(sample_set,graph_info,target=u)[:,1])
            else:
                for u in range(num_nodes):
                    all_d_u_S.append(offlineSketch(sample_set,dist=dist[u])[:,1])
                    all_d_S_u.append(offlineSketch(sample_set,dist=dist[:,u])[:,1])
        else:
            all_d_u_S = []
            if method == 'BFS':
                for u in range(num_nodes):
                    all_d_u_S.append(offlineSketch(sample_set,graph_info,source=u)[:,1])
            else:
                for u in range(num_nodes):
                    all_d_u_S.append(offlineSketch(sample_set,dist=dist[u])[:,1])
            all_d_S_u = [None]*num_nodes
        time2 = time.time()
        dist_Bourgain = np.ones((num_nodes,num_nodes))*np.inf
        if graph_info[1]: ## if directed
            for u in range(num_nodes):
                for v in range(num_nodes):
                    if u == v:
                        dist_Bourgain[u,v] = 0
                    else:
                        dist_Bourgain[u,v] = online_Bourgain(all_d_u_S[u],all_d_u_S[v],all_d_S_u[u],all_d_S_u[v])
        else:
            for u in range(num_nodes):
                dist_Bourgain[u,u] = 0
                for v in range(u+1,num_nodes):
                    dist_Bourgain[u,v] = online_Bourgain(all_d_u_S[u],all_d_v_S[v],all_d_S_u[u],all_d_S_v[v])
                    dist_Bourgain[v,u] = dist_Bourgain[u,v]
        time3 = time.time()
    elif targets == None:
        if method == 'networkx':
            dist = shortestDistances_networkx(graph_info)
        elif method == 'GNN':
            dist = shortestDistances_GNN(model,criterion_type,graph_info)
        if graph_info[1]: # if directed
            all_d_v_S = []
            all_d_S_v = []
            if method == 'BFS':
                for v in range(num_nodes):
                    all_d_v_S.append(offlineSketch(sample_set,graph_info,source=v)[:,1])
                    all_d_S_v.append(offlineSketch(sample_set,graph_info,target=v)[:,1])
            else:
                for v in range(num_nodes):
                    all_d_v_S.append(offlineSketch(sample_set,dist=dist[v])[:,1])
                    all_d_S_v.append(offlineSketch(sample_set,dist=dist[:,v])[:,1])
            all_d_u_S = [all_d_v_S[u] for u in sources]
            all_d_S_u = [all_d_S_v[u] for u in sources]
        else:
            all_d_v_S = []
            if method == 'BFS':
                for v in range(num_nodes):
                    all_d_v_S.append(offlineSketch(sample_set,graph_info,source=v)[:,1])
            else:
                for v in range(num_nodes):
                    all_d_v_S.append(offlineSketch(sample_set,dist=dist[v])[:,1])
            all_d_u_S = [all_d_v_S[u] for u in sources]
            all_d_S_v = [None]*num_nodes
            all_d_S_u = [None]*len(sources)
        time2 = time.time()
        dist_Bourgain = np.ones((len(sources),num_nodes))*np.inf
        for i in range(len(sources)):
            for v in range(num_nodes):
                if u == v:
                    dist_Bourgain[i,v] = 0
                else:
                    dist_Bourgain[i,v] = online_Bourgain(all_d_u_S[i],all_d_v_S[v],all_d_S_u[i],all_d_S_v[v])
        time3 = time.time()
    elif sources == None:
        if method == 'networkx':
            dist = shortestDistances_networkx(graph_info)
        elif method == 'GNN':
            dist = shortestDistances_GNN(model,criterion_type,graph_info)
        if graph_info[1]: # if directed
            all_d_u_S = []
            all_d_S_u = []
            if method == 'BFS':
                for u in range(num_nodes):
                    all_d_u_S.append(offlineSketch(sample_set,graph_info,source=u)[:,1])
                    all_d_S_u.append(offlineSketch(sample_set,graph_info,target=u)[:,1])
            else:
                for u in range(num_nodes):
                    all_d_u_S.append(offlineSketch(sample_set,dist=dist[u])[:,1])
                    all_d_S_u.append(offlineSketch(sample_set,dist=dist[:,u])[:,1])
            all_d_v_S = [all_d_u_S[v] for v in targets]
            all_d_S_v = [all_d_S_u[v] for v in targets]
        else:
            all_d_u_S = []
            if method == 'BFS':
                for u in range(num_nodes):
                    all_d_u_S.append(offlineSketch(sample_set,graph_info,source=u)[:,1])
            else:
                for v in range(num_nodes):
                    all_d_u_S.append(offlineSketch(sample_set,dist=dist[u])[:,1])
            all_d_v_S = [all_d_u_S[v] for v in targets]
            all_d_S_u = [None]*num_nodes
            all_d_S_v = [None]*len(targets)
        time2 = time.time()
        dist_Bourgain = np.ones((num_nodes,len(targets)))*np.inf
        for i in range(len(targets)):
            for u in range(num_nodes):
                if u == v:
                    dist_Bourgain[u,i] = 0
                else:
                    dist_Bourgain[u,i] = online_Bourgain(all_d_u_S[u],all_d_v_S[i],all_d_S_u[u],all_d_S_v[i])
        time3 = time.time()
    elif len(sources) == 1 and len(targets) == 1:
        u = sources[0]
        v = targets[0]
        if u != v:
            if graph_info[1]: # if directed
                if method == 'networkx':
                    d_u_S = offlineSketch(sample_set,shortestDistances_networkx(G,source=u))[:,1]
                    d_v_S = offlineSketch(sample_set,shortestDistances_networkx(G,source=v))[:,1]
                    d_S_u = offlineSketch(sample_set,shortestDistances_networkx(G,target=u))[:,1]
                    d_S_v = offlineSketch(sample_set,shortestDistances_networkx(G,target=v))[:,1]
                elif method == 'BFS':
                    d_u_S = offlineSketch(sample_set,graph_info,source=u)[:,1]
                    d_v_S = offlineSketch(sample_set,graph_info,source=v)[:,1]
                    d_S_u = offlineSketch(sample_set,graph_info,target=u)[:,1]
                    d_S_v = offlineSketch(sample_set,graph_info,target=v)[:,1]
                else:
                    dist = shortestDistances_GNN(model,criterion_type,graph_info)
                    d_u_S = offlineSketch(sample_set,dist[u])[:,1]
                    d_v_S = offlineSketch(sample_set,dist[v])[:,1]
                    d_S_u = offlineSketch(sample_set,dist[:,u])[:,1]
                    d_S_v = offlineSketch(sample_set,dist[:,v])[:,1]
            else:
                if method == 'networkx':
                    d_u_S = offlineSketch(sample_set,shortestDistances_networkx(G,source=u))[:,1]
                    d_v_S = offlineSketch(sample_set,shortestDistances_networkx(G,source=v))[:,1]
                elif method == 'BFS':
                    d_u_S = offlineSketch(sample_set,graph_info,source=u)[:,1]
                    d_v_S = offlineSketch(sample_set,graph_info,source=v)[:,1]
                else:
                    dist = shortestDistances_GNN(model,criterion_type,graph_info,[u,v])
                    d_u_S = offlineSketch(sample_set,dist[0])[:,1]
                    d_v_S = offlineSketch(sample_set,dist[1])[:,1]
                d_S_u = None
                d_S_v = None
            time2 = time.time()
            dist_Bourgain = online_Bourgain(d_u_S,d_v_S,d_S_u,d_S_v)
            time3 = time.time()
        else:
            return 0,0,0
    else:
        if len(sources) <= len(targets):
            dist_Bourgain,d31,d21 = shortestDistances_Bourgain(graph_info,sample_set,sources=sources,model=model,criterion_type=criterion_type,method=method)
            dist_Bourgain = dist_Bourgain[:,targets]
        else:
            dist_Bourgain,d31,d21 = shortestDistances_Bourgain(graph_info,sample_set,targets=targets,model=model,criterion_type=criterion_type,method=method)
            dist_Bourgain = dist_Bourgain[sources]
        return dist_Bourgain,d31,d21
    return dist_Bourgain,dist_Bourgain,time3-time1,time2-time1