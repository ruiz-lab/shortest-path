import networkx as nx
import numpy as np

from ShortestPathAlgorithms import shortestDistances_GNN, seedSets, revise_seedSets, offlineSketches, online_Bourgain, online_Sarma

def reviseSeedSets(input,revise_random_seeds=False,centralities=['degree','closeness','betweenness','harmonic','laplacian','pagerank']):
    
    if len(input) == 2 or len(input) == 4:
        all_revisedSeeds = [None]*6
    else:
        all_revisedSeeds = input[2]
        
    k_max = len(input[1][0])
    all_random_seeds = []
    all_seeds_by_degree = []
    all_seeds_by_closeness = []
    all_seeds_by_betweenness = []
    # all_seeds_by_harmonic = []
    # all_seeds_by_laplacian = []
    all_seeds_by_pagerank = []
    num_graphs = len(input[0])
    for i in range(num_graphs):
        if i % 10 == 0:
            print(str(i/num_graphs*100)+'% complete')
        graph_info = input[0][i]
        random_seeds = []
        seeds_by_degree = []
        seeds_by_closeness = []
        seeds_by_betweenness = []
        # seeds_by_harmonic = []
        # seeds_by_laplacian = []
        seeds_by_pagerank = []
        for k in range(1,k_max+1):
            if revise_random_seeds:
                Sarma = seedSets(graph_info,k)
                random_seeds.append(Sarma)
            if 'degree' in centralities:
                #print('Calculating degree centrality...')
                degree = revise_seedSets(graph_info,k,nx.degree_centrality)
                seeds_by_degree.append(degree)
            if 'closeness' in centralities:
                #print('Calculating closeness centrality...')
                closeness = revise_seedSets(graph_info,k,nx.closeness_centrality)
                seeds_by_closeness.append(closeness)
            if 'betweenness' in centralities:
                #print('Calculating betweenness centrality...')
                betweenness = revise_seedSets(graph_info,k,nx.betweenness_centrality)
                seeds_by_betweenness.append(betweenness)
            # if 'harmonic' in centralities:
            #     #print('Calculating harmonic centrality...')
            #     harmonic = revise_seedSets(graph_info,k,nx.harmonic_centrality)
            #     seeds_by_harmonic.append(harmonic)
            # if 'laplacian' in centralities:
            #     #print('Calculating laplacian centrality...')
            #     laplacian = revise_seedSets(graph_info,k,nx.laplacian_centrality)
            #     seeds_by_laplacian.append(laplacian)
            if 'pagerank' in centralities:
                #print('Calculating pagerank...')
                pagerank = revise_seedSets(graph_info,k,nx.pagerank)
                seeds_by_pagerank.append(pagerank)
        if revise_random_seeds:
            all_random_seeds.append(random_seeds)
        all_seeds_by_degree.append(seeds_by_degree)
        all_seeds_by_closeness.append(seeds_by_closeness)
        all_seeds_by_betweenness.append(seeds_by_betweenness)
        # all_seeds_by_harmonic.append(seeds_by_harmonic)
        # all_seeds_by_laplacian.append(seeds_by_laplacian)
        all_seeds_by_pagerank.append(seeds_by_pagerank)
    
    if 'degree' in centralities:
        all_revisedSeeds[0] = all_seeds_by_degree
    if 'closeness' in centralities:
        all_revisedSeeds[1] = all_seeds_by_closeness
    if 'betweenness' in centralities:
        all_revisedSeeds[2] = all_seeds_by_betweenness
    # if 'harmonic' in centralities:
    #     all_revisedSeeds[3] = all_seeds_by_harmonic
    # if 'laplacian' in centralities:
    #     all_revisedSeeds[4] = all_seeds_by_laplacian
    if 'pagerank' in centralities:
        all_revisedSeeds[5] = all_seeds_by_pagerank
    if revise_random_seeds:
        if len(input) == 2:
            return input[0],all_random_seeds,all_revisedSeeds
        elif len(input) == 4:
            return input[0],all_random_seeds,all_revisedSeeds,input[2],input[3]
        else:
            return input[0],all_random_seeds,all_revisedSeeds,input[3],input[4]
    else:
        if len(input) == 2:
            return input[0],input[1],all_revisedSeeds
        elif len(input) == 4:
            return input[0],input[1],all_revisedSeeds,input[2],input[3]
        else:
            return input[0],input[1],all_revisedSeeds,input[3],input[4]

def revise_random_dists(input):
    all_selected_pairs = []
    all_selected_dists = []
    for N in range(len(input[0])):
        n = len(input[0][N][0].nodes())
        nodes = range(n)
        num_pairs = int(2559/1999*n-2800/1999)
        pairs = []
        dists = []
        while len(pairs) < num_pairs:
            l1 = np.random.choice(nodes,size=num_pairs,replace=True)
            l2 = np.random.choice(nodes,size=num_pairs,replace=True)
            new_pairs = [(u,v) if u<=v else (v,u) for (u,v) in list(zip(l1,l2))]
            pairs = list(set(pairs+new_pairs))
        pairs = pairs[:num_pairs]
        for u,v in pairs:
            dists.append(nx.shortest_path_length(input[0][N][0],source=u,target=v))
        all_selected_pairs.append(pairs[:num_pairs])
        all_selected_dists.append(dists)
    if len(input) == 2 or len(input) == 4:
        return input[0],input[1],all_selected_pairs,all_selected_dists
    elif len(input) == 3 or len(input) == 5:
        return input[0],input[1],input[2],all_selected_pairs,all_selected_dists

# def test_transferability(which_cuda,model1,model2,criterion_type,samples,seed_metric,GNN_only=True):
    
#     assert criterion_type in ['mse','l2','l1'], 'Invalid criterion type.'

#     all_graph_info = samples[0]
#     all_selected_dists = samples[4]
#     all_pairs = samples[3]
#     if seed_metric == 'random':
#         all_seedSets = samples[1]
#     elif seed_metric == 'degree':
#         all_seedSets = samples[2][0]
#     elif seed_metric == 'closeness':
#         all_seedSets = samples[2][1]
#     elif seed_metric == 'betweenness':
#         all_seedSets = samples[2][2]
#     elif seed_metric == 'harmonic':
#         all_seedSets = samples[2][3]
#     elif seed_metric == 'laplacian':
#         all_seedSets = samples[2][4]
#     elif seed_metric == 'pagerank':
#         all_seedSets = samples[2][5]
#     else:
#         raise AssertionError('Invalid seed metric.')
    
#     all_actual = []
#     all_pred = []
#     num_graphs = len(all_graph_info)
#     num_graphs = 50

#     if GNN_only:
#         for N in range(num_graphs):
#             if N % 10 == 0:
#                 print(str(N/num_graphs*100)+'% complete')
#             graph_info = all_graph_info[N]
#             selected_actual = all_selected_dists[N]
#             seedSets = all_seedSets[N]
#             pairs = all_pairs[N]
#             k_max = len(seedSets)
#             pred_GNN1 = shortestDistances_GNN(which_cuda,model1,criterion_type,graph_info)[0]
#             pred_GNN2 = shortestDistances_GNN(which_cuda,model2,criterion_type,graph_info)[0]
#             pred = np.zeros((6,k_max,len(pairs)))
#             all_nodes = np.array(pairs).flatten()
#             for k in range(k_max):
#                 BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1,nodes=all_nodes)
#                 BourgainDists_GNN1 = []
#                 SarmaDists_GNN1 = []
#                 BourgainSketches_GNN2_uS,SarmaSketches_GNN2_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN2,nodes=all_nodes)
#                 BourgainDists_GNN2 = []
#                 SarmaDists_GNN2 = []
#                 if k == 0:
#                     for u,v in pairs:
#                         BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
#                         SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_uS[v]))
#                         BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v]))
#                         SarmaDists_GNN2.append(online_Sarma(SarmaSketches_GNN2_uS[u],SarmaSketches_GNN2_uS[v]))
#                 else:
#                     for u,v in pairs:
#                         BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
#                         SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_uS[v]))
#                         BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v]))
#                         SarmaDists_GNN2.append(online_Sarma(SarmaSketches_GNN2_uS[u],SarmaSketches_GNN2_uS[v]))
#                 pred[2,k] = BourgainDists_GNN1
#                 pred[3,k] = SarmaDists_GNN1
#                 pred[4,k] = BourgainDists_GNN2
#                 pred[5,k] = SarmaDists_GNN2
#             pred = np.where(pred == float('inf'), len(graph_info[0].nodes()), pred)
#             all_actual.append(selected_actual)
#             all_pred.append(pred)
#     else:
#         for N in range(num_graphs):
#             if N % 10 == 0:
#                 print(str(N/num_graphs*100)+'% complete')
#             graph_info = all_graph_info[N]
#             selected_actual = all_selected_dists[N]
#             seedSets = all_seedSets[N]
#             pairs = all_pairs[N]
#             k_max = len(seedSets)
#             pred_GNN1 = shortestDistances_GNN(which_cuda,model1,criterion_type,graph_info)[0]
#             pred_GNN2 = shortestDistances_GNN(which_cuda,model2,criterion_type,graph_info)[0]
#             pred = np.zeros((6,k_max,len(pairs)))
#             all_nodes = np.array(pairs).flatten()
#             for k in range(k_max):
#                 BourgainSketches_networkx_uS,SarmaSketches_networkx_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',nodes=all_nodes)
#                 BourgainDists_networkx = []
#                 SarmaDists_networkx = []
#                 BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1,nodes=all_nodes)
#                 BourgainDists_GNN1 = []
#                 SarmaDists_GNN1 = []
#                 BourgainSketches_GNN2_uS,SarmaSketches_GNN2_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN2,nodes=all_nodes)
#                 BourgainDists_GNN2 = []
#                 SarmaDists_GNN2 = []
#                 if k == 0:
#                     for u,v in pairs:
#                         BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v]))
#                         SarmaDists_networkx.append(online_Sarma(SarmaSketches_networkx_uS[u],SarmaSketches_networkx_uS[v]))
#                         BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
#                         SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_uS[v]))
#                         BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v]))
#                         SarmaDists_GNN2.append(online_Sarma(SarmaSketches_GNN2_uS[u],SarmaSketches_GNN2_uS[v]))
#                 else:
#                     for u,v in pairs:
#                         BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v]))
#                         SarmaDists_networkx.append(online_Sarma(SarmaSketches_networkx_uS[u],SarmaSketches_networkx_uS[v]))
#                         BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
#                         SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_uS[v]))
#                         BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v]))
#                         SarmaDists_GNN2.append(online_Sarma(SarmaSketches_GNN2_uS[u],SarmaSketches_GNN2_uS[v]))
#                 pred[0,k] = BourgainDists_networkx
#                 pred[1,k] = SarmaDists_networkx
#                 pred[2,k] = BourgainDists_GNN1
#                 pred[3,k] = SarmaDists_GNN1
#                 pred[4,k] = BourgainDists_GNN2
#                 pred[5,k] = SarmaDists_GNN2
#         pred = np.where(pred == float('inf'), len(graph_info[0].nodes()), pred)
#         all_actual.append(selected_actual)
#         all_pred.append(pred)

#     return all_actual,all_pred

def test_transferability(which_cuda,model1,model2,criterion_type,samples,seed_metric,GNN_only=True):
    
    assert criterion_type in ['mse','l2','l1'], 'Invalid criterion type.'

    all_graph_info = samples[0]
    all_selected_dists = samples[4]
    all_pairs = samples[3]
    if seed_metric == 'random':
        all_seedSets = samples[1]
    elif seed_metric == 'degree':
        all_seedSets = samples[2][0]
    elif seed_metric == 'closeness':
        all_seedSets = samples[2][1]
    elif seed_metric == 'betweenness':
        all_seedSets = samples[2][2]
    elif seed_metric == 'harmonic':
        all_seedSets = samples[2][3]
    elif seed_metric == 'laplacian':
        all_seedSets = samples[2][4]
    elif seed_metric == 'pagerank':
        all_seedSets = samples[2][5]
    else:
        raise AssertionError('Invalid seed metric.')
    
    all_actual = []
    all_pred = []
    num_graphs = len(all_graph_info)

    if GNN_only:
        for N in range(num_graphs):
            if N % 10 == 0:
                print(str(N/num_graphs*100)+'% complete')
            graph_info = all_graph_info[N]
            selected_actual = all_selected_dists[N]
            seedSets = all_seedSets[N]
            pairs = all_pairs[N]
            k_max = len(seedSets)
            pred_GNN1 = shortestDistances_GNN(which_cuda,model1,criterion_type,graph_info)[0]
            pred_GNN2 = shortestDistances_GNN(which_cuda,model2,criterion_type,graph_info)[0]
            pred = np.zeros((6,k_max,len(pairs)))
            all_nodes = np.array(pairs).flatten()
            for k in range(k_max):
                BourgainSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN1,nodes=all_nodes)
                BourgainDists_GNN1 = []
                BourgainSketches_GNN2_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN2,nodes=all_nodes)
                BourgainDists_GNN2 = []
                if k == 0:
                    for u,v in pairs:
                        BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
                        BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v]))
                else:
                    for u,v in pairs:
                        BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
                        BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v]))
                pred[2,k] = BourgainDists_GNN1
                pred[4,k] = BourgainDists_GNN2
            pred = np.where(pred == float('inf'), len(graph_info[0].nodes()), pred)
            all_actual.append(selected_actual)
            all_pred.append(pred)
    else:
        for N in range(num_graphs):
            if N % 10 == 0:
                print(str(N/num_graphs*100)+'% complete')
            graph_info = all_graph_info[N]
            selected_actual = all_selected_dists[N]
            seedSets = all_seedSets[N]
            pairs = all_pairs[N]
            k_max = len(seedSets)
            pred_GNN1 = shortestDistances_GNN(which_cuda,model1,criterion_type,graph_info)[0]
            pred_GNN2 = shortestDistances_GNN(which_cuda,model2,criterion_type,graph_info)[0]
            pred = np.zeros((6,k_max,len(pairs)))
            all_nodes = np.array(pairs).flatten()
            for k in range(k_max):
                BourgainSketches_networkx_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',nodes=all_nodes)
                BourgainDists_networkx = []
                BourgainSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN1,nodes=all_nodes)
                BourgainDists_GNN1 = []
                BourgainSketches_GNN2_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN2,nodes=all_nodes)
                BourgainDists_GNN2 = []
                if k == 0:
                    for u,v in pairs:
                        BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v]))
                        BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
                        BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v]))
                else:
                    for u,v in pairs:
                        BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v]))
                        BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
                        BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v]))
                pred[0,k] = BourgainDists_networkx
                pred[2,k] = BourgainDists_GNN1
                pred[4,k] = BourgainDists_GNN2
            pred = np.where(pred == float('inf'), len(graph_info[0].nodes()), pred)
            all_actual.append(selected_actual)
            all_pred.append(pred)

    return all_actual,all_pred