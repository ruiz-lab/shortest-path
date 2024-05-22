import networkx as nx
import numpy as np
from itertools import chain
from collections import deque
import torch
import time

def predict(gpu_bool,which_cuda,model,criterion_type,samples_x,samples_edge_index=[],samples_weights=[]):

    selected_cuda = 'cuda:'+str(which_cuda)

    if len(samples_edge_index) > 0:
        if len(samples_edge_index) == 1:
            flag = True
        else:
            flag = False
    
    y_pred = []
    if gpu_bool:
        model = model.to(selected_cuda)
    model.eval()
    with torch.no_grad():
        if model.out_channels == 1:
            if model.name == 'mlp':
                for x in samples_x:
                    if gpu_bool:
                        x = x.to(selected_cuda)
                    pred_all = []
                    for j in range(x.shape[1]):
                        out = model(x[:,j].reshape(len(x[:,j]),1))  # Perform a single forward pass.
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        elif criterion_type in ['mse','l2','l1']:
                            pred = out.squeeze()
                            #####
                            pred = pred*(1-x)
                            #####
                        else:
                            pred = torch.round(out.squeeze())
                        pred_all.append(pred.cpu())
                    y_pred.append(np.array(pred_all).T)
            elif len(samples_weights) == 0:
                if flag:
                    edge_index = samples_edge_index[-1]
                    if gpu_bool:
                        edge_index = edge_index.to(selected_cuda)
                    for x in samples_x:
                        if gpu_bool:
                            x = x.to(selected_cuda)
                        pred_all = []
                        for j in range(x.shape[1]):
                            out = model(x[:,j].reshape(len(x[:,j]),1),edge_index)  # Perform a single forward pass.
                            if criterion_type in ['bce','ce','multimargin']:
                                pred = out.argmax(dim=1) #  Use the class with highest probability.
                            elif criterion_type in ['mse','l2','l1']:
                                pred = out.squeeze()
                                #####
                                pred = pred*(1-x)
                                #####
                            else:
                                pred = torch.round(out.squeeze())
                            pred_all.append(pred.cpu())
                        y_pred.append(np.array(pred_all).T)
                else:
                    for x,edge_index in list(zip(samples_x,samples_edge_index)):
                        if gpu_bool:
                            x = x.to(selected_cuda)
                            edge_index = edge_index.to(selected_cuda)
                        pred_all = []
                        for j in range(x.shape[1]):
                            out = model(x[:,j].reshape(len(x[:,j]),1),edge_index)  # Perform a single forward pass.
                            if criterion_type in ['bce','ce','multimargin']:
                                pred = out.argmax(dim=1) #  Use the class with highest probability.
                            elif criterion_type in ['mse','l2','l1']:
                                pred = out.squeeze()
                                #####
                                pred = pred*(1-x)
                                #####
                            else:
                                pred = torch.round(out.squeeze())
                            pred_all.append(pred.cpu())
                        y_pred.append(np.array(pred_all).T)
            else:
                if flag:
                    edge_index = samples_edge_index[-1]
                    weights = samples_weights[-1]
                    if gpu_bool:
                        edge_index = edge_index.to(selected_cuda)
                        weights = weights.to(selected_cuda)
                    for x in samples_x:
                        if gpu_bool:
                            x = x.to(selected_cuda)
                        pred_all = []
                        for j in range(x.shape[1]):
                            out = model(x[:,j].reshape(len(x[:,j]),1),edge_index,weights)  # Perform a single forward pass.
                            if criterion_type in ['bce','ce','multimargin']:
                                pred = out.argmax(dim=1) #  Use the class with highest probability.
                            elif criterion_type in ['mse','l2','l1']:
                                pred = out.squeeze()
                                #####
                                pred = pred*(1-x)
                                #####
                            else:
                                pred = torch.round(out.squeeze())
                            pred_all.append(pred.cpu())
                        y_pred.append(np.array(pred_all).T)
                else:
                    for x,edge_index,weights in list(zip(samples_x,samples_edge_index,samples_weights)):
                        if gpu_bool:
                            x = x.to(selected_cuda)
                            edge_index = edge_index.to(selected_cuda)
                            weights = weights.to(selected_cuda)
                        pred_all = []
                        for j in range(x.shape[1]):
                            out = model(x[:,j].reshape(len(x[:,j]),1),edge_index,weights)  # Perform a single forward pass.
                            if criterion_type in ['bce','ce','multimargin']:
                                pred = out.argmax(dim=1) #  Use the class with highest probability.
                            elif criterion_type in ['mse','l2','l1']:
                                pred = out.squeeze()
                                #####
                                pred = pred*(1-x)
                                #####
                            else:
                                pred = torch.round(out.squeeze())
                            pred_all.append(pred.cpu())
                        y_pred.append(np.array(pred_all).T)
        else:
            if model.name == 'mlp':
                for x in samples_x:
                    if gpu_bool:
                        x = x.to(selected_cuda)
                    out = model(x)  # Perform a single forward pass.
                    if criterion_type in ['bce','ce','multimargin']:
                        pred = out.argmax(dim=1) #  Use the class with highest probability.
                    elif criterion_type in ['mse','l2','l1']:
                        pred = out.squeeze()
                        #####
                        pred = pred*(1-x)
                        #####
                    else:
                        pred = torch.round(out.squeeze())
                    y_pred.append(pred.cpu())
            elif len(samples_weights) == 0:
                if flag:
                    edge_index = samples_edge_index[-1]
                    if gpu_bool:
                        edge_index = edge_index.to(selected_cuda)
                    for x in samples_x:
                        if gpu_bool:
                            x = x.to(selected_cuda)
                        out = model(x,edge_index)  # Perform a single forward pass.
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        elif criterion_type in ['mse','l2','l1']:
                            pred = out.squeeze()
                            #####
                            pred = pred*(1-x)
                            #####
                        else:
                            pred = torch.round(out.squeeze())
                        y_pred.append(pred.cpu())
                else:
                    for x,edge_index in list(zip(samples_x,samples_edge_index)):
                        if gpu_bool:
                            x = x.to(selected_cuda)
                            edge_index = edge_index.to(selected_cuda)
                        out = model(x,edge_index)  # Perform a single forward pass.
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        elif criterion_type in ['mse','l2','l1']:
                            pred = out.squeeze()
                            #####
                            pred = pred*(1-x)
                            #####
                        else:
                            pred = torch.round(out.squeeze())
                        y_pred.append(pred.cpu())
            else:
                if flag:
                    edge_index = samples_edge_index[-1]
                    weights = samples_weights[-1]
                    if gpu_bool:
                        edge_index = edge_index.to(selected_cuda)
                        weights = weights.to(selected_cuda)
                    for x in samples_x:
                        if gpu_bool:
                            x = x.to(selected_cuda)
                        out = model(x,edge_index,weights)  # Perform a single forward pass.
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        elif criterion_type in ['mse','l2','l1']:
                            pred = out.squeeze()
                            #####
                            pred = pred*(1-x)
                            #####
                        else:
                            pred = torch.round(out.squeeze())
                        y_pred.append(pred.cpu())
                else:
                    for x,edge_index,weights in list(zip(samples_x,samples_edge_index,samples_weights)):
                        if gpu_bool:
                            x = x.to(selected_cuda)
                            edge_index = edge_index.to(selected_cuda)
                            weights = weights.to(selected_cuda)
                        out = model(x,edge_index,weights)  # Perform a single forward pass.
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        elif criterion_type in ['mse','l2','l1']:
                            pred = out.squeeze()
                            #####
                            pred = pred*(1-x)
                            #####
                        else:
                            pred = torch.round(out.squeeze())
                        y_pred.append(pred.cpu())
    model = model.to('cpu')
    return y_pred

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
        dist = np.zeros((len(sources),num_nodes))
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
        dist = np.zeros((num_nodes,len(targets)))
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
                dist = nx.shortest_path_length(G,source=sources[0],target=targets[0],weight="weight")
            else:
                dist = nx.shortest_path_length(G,source=sources[0],target=targets[0])
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
    return np.squeeze(dist),time2-time1

def shortestDistances_GNN(which_cuda,model,criterion_type,graph_info,seeds=None):
    time1 = time.time()
    gpu_bool = torch.cuda.is_available()
    G = graph_info[0]
    num_nodes = len(G.nodes())
    if seeds == None:
        seeds = list(range(num_nodes))
    num_seeds = len(seeds)
    samples_edge_index = [torch.tensor(np.array(G.edges()).astype(np.int64).T)]
    if graph_info[2]: # if weighted
        samples_weights = [torch.tensor(np.array(nx.get_edge_attributes(G,'weight').values()).astype(np.float32))]
    else:
        samples_weights = []
    r = model.out_channels
    if r == 1:
        one_hot_seeds = np.zeros((num_nodes,num_seeds))
        for i in range(num_seeds):
            one_hot_seeds[seeds[i],i] = 1
        one_hot_seeds = torch.tensor(one_hot_seeds.astype(np.float32), requires_grad=True)
        y_pred = predict(gpu_bool,which_cuda,model, criterion_type, [one_hot_seeds], samples_edge_index, samples_weights)[0]
    else:
        samples_x = []
        if num_seeds % r != 0:
            n_extra = int(np.ceil(num_seeds/r))*r - num_seeds
            nodes = seeds + seeds[:n_extra]
            while len(nodes) % r != 0:
                n_extra = int(np.ceil(len(nodes)/r))*r - len(nodes)
                nodes = nodes + nodes[:n_extra]
        else:
            nodes = seeds
        list_of_seeds = [nodes[i:i + r] for i in range(0, len(nodes), r)]
        for s in list_of_seeds:
            x = np.zeros((num_nodes,r))
            for i in range(r):
                x[s[i],i] = 1
            samples_x.append(torch.tensor(x.astype(np.float32), requires_grad=True))
        y_pred = predict(gpu_bool,which_cuda,model, criterion_type, samples_x, samples_edge_index, samples_weights)
        y_pred = np.squeeze(np.concatenate(y_pred, axis=1)[:,:num_seeds])
    # my_dict = {}
    # for i in range(num_seeds):
    #     my_dict[seeds[i]] = y_pred[:,i]
    time2 = time.time()
    return y_pred,time2-time1

def closestSeed_fromSource(G, source, target_set):
    # Works for weighted & unweighted, directed & undirected graphs 
    # Doesn't allow self-directed nodes
    closest_point, smallest_weight = None, float('inf')
    #target_set = [x for x in target_set if x != source]
    if source in target_set:
        return source,0
    if len(target_set) > 0:
        visited = set()  # To keep track of visited vertices (vertices in paths stored in queue)
        queue = deque([(source, 0, [source])])  # Initialize the queue with the starting vertex, total weight, and its path 
        storage = deque() # To store nodes removed from queue
        while queue:
            vertex, weight, path = queue.popleft()  # Dequeue a vertex, total weight, and its path
            storage.append((vertex, weight))
            if vertex not in target_set:
                for neighbor in G.successors(vertex):
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
    #source_set = [x for x in source_set if x != target]
    if target in source_set:
        return target,0
    if len(source_set) > 0:
        visited = set()  # To keep track of visited vertices (vertices in paths stored in queue)
        queue = deque()  # Initialize the queue with the starting vertex, its path, and the total weight
        for i in range(len(source_set)):
            queue.append((source_set[i], 0, [source_set[i]]))
        storage = deque()
        while queue:
            vertex, weight, path = queue.popleft()  # Dequeue a vertex, its path, and the total weight
            storage.append((vertex, weight))
            if vertex != target:
                for neighbor in G.successors(vertex):
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
                        closest_point, smallest_weight = path[0], weight
                    else:
                        if weight < smallest_weight:
                            closest_point, smallest_weight = path[0], weight
                    k = len(queue)
                    for i in range(k):
                        if smallest_weight <= queue[k-1-i][1]:
                            queue.remove(queue[k-1-i]) # Remove all paths in queue which are longer than the current shortest from 'start' to a seed
        return closest_point, smallest_weight

###### Offine Sketch ######

def seedSets(graph_info,k):
    G = graph_info[0]
    support = [node for node, degree in G.degree() if degree > 1]
    num_nodes = len(support)
    if num_nodes == 0:
        return []
    r = int(np.floor(np.log(len(G.nodes()))))
    seedSets = []
    for i in range(r+1):
        size = 2**i
        if size <= num_nodes:
            seedSets = seedSets+[list(np.random.choice(support,size=size,replace=False)) for i in range(k)]
        else:
            seedSets = seedSets+[support]
            break
    return seedSets
    
def revise_seedSets(graph_info,k,function):
    G = graph_info[0]
    support = [node for node, degree in G.degree() if degree > 1]
    num_nodes = len(support)
    if num_nodes == 0:
        return []
    r = int(np.floor(np.log(len(G.nodes()))))
    num_nodes_to_select = k*(np.sum(2**(np.array(range(r+1)))))
    if graph_info[2]:
        try:
            scores = function(G,weight='weight')
        except:
            try:
                scores = function(G,distance='weight')
            except:
                scores = function(G)
    else:
        scores = function(G)
    scores = {key: value for key, value in scores.items() if key in support}
    sorted_nodes = sorted(scores, key=scores.get, reverse=True)
    sorted_scores = [scores[node] for node in sorted_nodes]
    if num_nodes_to_select < len(sorted_nodes):
        sorted_nodes = sorted_nodes[:num_nodes_to_select]
        sorted_scores = sorted_scores[:num_nodes_to_select]
    sorted_scores = [val+0.000001 for val in sorted_scores]
    sorted_scores = [b/sum(sorted_scores) for b in sorted_scores]
    seedSets = []
    for i in range(r+1):
        size = 2**i
        if size <= num_nodes_to_select:
            seedSets = seedSets+[list(np.random.choice(sorted_nodes,size=size,replace=False,p=sorted_scores)) for i in range(k)]
        else:
            seedSets = seedSets+[sorted_nodes]
            break
    return seedSets

def offlineSketches(graph_info,seedSet,nodes=None,which_sketch='source',sketch_type='both',dists=None,which_cuda=None,model=None,criterion_type=None):
    
    assert which_sketch in ['source','target'], 'Invalid sketch data.'
    assert sketch_type in ['both','Bourgain','Sarma'], 'Invalid sketch type.'
    
    seeds = list(chain(*seedSet))
    seeds.sort()
    seeds = list(set(seeds))
    if isinstance(dists, np.ndarray):
        if which_sketch == 'source':
            dists = dists[:,seeds]
        else:
            dists = dists[seeds,:].T
    elif model != None and criterion_type != None:
        assert which_cuda != None, 'Missing the cuda device.'
        dists,_ = shortestDistances_GNN(which_cuda,model,criterion_type,graph_info,seeds)
    else:
        if which_sketch == 'source':
            dists,_ = shortestDistances_networkx(graph_info,targets=seeds)
        else:
            dists,_ = shortestDistances_networkx(graph_info,sources=seeds)
            dists = dists.T
    index_dict = dict(zip(seeds,range(len(seeds))))
    SarmaSketches = dict()
    BourgainSketches = dict()
    if not isinstance(nodes,list):
        nodes = list(range(dists.shape[0]))
    for node in nodes:
        all_selected_dists = [[dists[node][index_dict.get(i)] for i in S] for S in seedSet]
        if sketch_type != 'Bourgain':
            indices_list = [[index for index, value in enumerate(l) if value == min(l)] for l in all_selected_dists]
            SarmaSketches[node] = np.array([(S[i],min(selected_dists)) for S,indices,selected_dists in list(zip(seedSet,indices_list,all_selected_dists)) for i in indices])
        if sketch_type != 'Sarma':
            BourgainSketches[node] = np.array([min(l) for l in all_selected_dists])
    
    if sketch_type == 'Bourgain':
        return BourgainSketches
    elif sketch_type == 'Sarma':
        return SarmaSketches
    else:
        return BourgainSketches,SarmaSketches

###### Online ######

def online_Bourgain(d_u_S,d_v_S,d_S_u=None,d_S_v=None):
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
    
def online_Sarma(sketch_u,sketch_v):
    dist = np.inf
    if sketch_u.shape[0] != 0 and sketch_v.shape[0] != 0:
        common_nodes = [w for w in sketch_u[:,0] if w in sketch_v[:,0]]
        for w in common_nodes:
            dist = min(sketch_u[sketch_u[:, 0] == w][0,1] + sketch_v[sketch_v[:, 0] == w][0,1],dist) 
    return dist