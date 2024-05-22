import networkx as nx
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from Graphs import dRegularGraph,connectedErdosRenyiGraph
from ShortestPathAlgorithms import shortestDistances_networkx, shortestDistances_GNN, seedSets, revise_seedSets, offlineSketches, online_Bourgain, online_Sarma
from Models import predict_allBatches

def evaluate_inner(model_name1,model_name2,model_name3,model_name4,title,dir,data_name,y_actual,y_pred1,y_pred2,y_pred3,y_pred4):
    
    y_actual = np.array(list(chain(*list(chain(*y_actual)))))
    y_pred1 = np.array(list(chain(*list(chain(*y_pred1)))))
    diff_pred1 = y_actual - y_pred1
    if y_pred2 != None:
        y_pred2 = np.array(list(chain(*list(chain(*y_pred2)))))
        diff_pred2 = y_actual - y_pred2
    else:
        diff_pred2 = None
    if y_pred3 != None:
        y_pred3 = np.array(list(chain(*list(chain(*y_pred3)))))
        diff_pred3 = y_actual - y_pred3
    else:
        diff_pred3 = None
    if y_pred4 != None:
        y_pred4 = np.array(list(chain(*list(chain(*y_pred4)))))
        diff_pred4 = y_actual - y_pred4
    else:
        diff_pred4 = None

    text = data_name+', '+title
    mse1 = round(mean_squared_error(y_actual,y_pred1),4)
    with open(dir+'/mse.txt', 'a') as file:
        file.write(text+': '+model_name1+' MSE = '+str(mse1)+'\n')
    if isinstance(y_pred2, np.ndarray):
        mse2 = round(mean_squared_error(y_actual,y_pred2),4)
        with open(dir+'/mse.txt', 'a') as file:
            file.write(text+': '+model_name2+' MSE = '+str(mse2)+'\n')
    if isinstance(y_pred3, np.ndarray):
        mse3 = round(mean_squared_error(y_actual,y_pred3),4)
        with open(dir+'/mse.txt', 'a') as file:
            file.write(text+': '+model_name3+' MSE = '+str(mse3)+'\n')
    if isinstance(y_pred4, np.ndarray):
        mse4 = round(mean_squared_error(y_actual,y_pred4),4)
        with open(dir+'/mse.txt', 'a') as file:
            file.write(text+': '+model_name4+' MSE = '+str(mse4)+'\n')

    plt.plot([0,np.max(y_actual)],[0,np.max(y_actual)],'k',label='y = x')
    data = {'Category': y_actual,'Values': y_pred1}
    df = pd.DataFrame(data)
    mean_by_category = df.groupby('Category')['Values'].mean().reset_index()
    categories = mean_by_category['Category'].tolist()
    means = mean_by_category['Values'].tolist()
    plt.plot(categories, means, color='#1f77b4', label=model_name1+' Fitted Line, MSE = '+str(mse1), alpha = 0.75)
    if len(y_actual) > 50000:
        plt.scatter(y_actual[:50000], y_pred1[:50000], color='#1f77b4', alpha = 0.005)
    else:
        plt.scatter(y_actual, y_pred1, color='#1f77b4', alpha = 0.005)
    if isinstance(y_pred2, np.ndarray):
        data = {'Category': y_actual,'Values': y_pred2}
        df = pd.DataFrame(data)
        mean_by_category = df.groupby('Category')['Values'].mean().reset_index()
        categories = mean_by_category['Category'].tolist()
        means = mean_by_category['Values'].tolist()
        plt.plot(categories, means, color='#ff7f0e', label=model_name2+' Fitted Line, MSE = '+str(mse2), alpha = 0.75)
        if len(y_actual) > 50000:
            plt.scatter(y_actual[:50000], y_pred2[:50000], color='#ff7f0e', alpha = 0.005)
        else:
            plt.scatter(y_actual, y_pred2, color='#ff7f0e', alpha = 0.005)
    if isinstance(y_pred3, np.ndarray):
        data = {'Category': y_actual,'Values': y_pred3}
        df = pd.DataFrame(data)
        mean_by_category = df.groupby('Category')['Values'].mean().reset_index()
        categories = mean_by_category['Category'].tolist()
        means = mean_by_category['Values'].tolist()
        plt.plot(categories, means, color='#2ca02c', label=model_name3+' Fitted Line, MSE = '+str(mse3), alpha = 0.75)
        if len(y_actual) > 50000:
            plt.scatter(y_actual[:50000], y_pred3[:50000], color='#2ca02c', alpha = 0.005)
        else:
            plt.scatter(y_actual, y_pred3, color='#2ca02c', alpha = 0.005)
    if isinstance(y_pred4, np.ndarray):
        data = {'Category': y_actual,'Values': y_pred4}
        df = pd.DataFrame(data)
        mean_by_category = df.groupby('Category')['Values'].mean().reset_index()
        categories = mean_by_category['Category'].tolist()
        means = mean_by_category['Values'].tolist()
        plt.plot(categories, means, color='#d62728', label=model_name4+' Fitted Line, MSE = '+str(mse4), alpha = 0.75)
        if len(y_actual) > 50000:
            plt.scatter(y_actual[:50000], y_pred4[:50000], color='#d62728', alpha = 0.005)
        else:
            plt.scatter(y_actual, y_pred4, color='#d62728', alpha = 0.005)
    plt.xlabel('Actual Distance')
    plt.ylabel('Predicted Distance')
    plt.title(text)
    plt.legend()
    plt.savefig(dir+'/'+data_name+'_'+title+'_predscatter.png')
    #plt.close('all')
    plt.show()

    return diff_pred1,diff_pred2,diff_pred3,diff_pred4
    
def evaluate(title,dir,which_cuda,model1,model2,model3,model4,criterion_type,samples):

    assert criterion_type in ['ce','bce','bcelogits','mse','l2','l1','multimargin','mse-mse'], 'Criterion type not yet defined.'

    print('Evaluating model performance...')
    y_pred_train1,y_pred_val1,y_pred_test1 = predict_allBatches(which_cuda,model1,criterion_type,samples)
    if model2 != None:
        y_pred_train2,y_pred_val2,y_pred_test2 = predict_allBatches(which_cuda,model2,criterion_type,samples)
        name2 = model2.name
    else:
        y_pred_train2 = None
        y_pred_val2 = None
        y_pred_test2 = None
        name2 = None
    if model3 != None:
        y_pred_train3,y_pred_val3,y_pred_test3 = predict_allBatches(which_cuda,model3,criterion_type,samples)
        name3 = model3.name
    else:
        y_pred_train3 = None
        y_pred_val3 = None
        y_pred_test3 = None
        name3 = None
    if model4 != None:
        y_pred_train4,y_pred_val4,y_pred_test4 = predict_allBatches(which_cuda,model4,criterion_type,samples)
        name4 = model4.name
    else:
        y_pred_train4 = None
        y_pred_val4 = None
        y_pred_test4 = None
        name4 = None
    diff_pred1_train,diff_pred2_train,diff_pred3_train,diff_pred4_train = evaluate_inner(model1.name,name2,name3,name4,title,dir,'Training Data',samples[0][1],y_pred_train1,y_pred_train2,y_pred_train3,y_pred_train4)
    diff_pred1_val,diff_pred2_val,diff_pred3_val,diff_pred4_val = evaluate_inner(model1.name,name2,name3,name4,title,dir,'Validation Data',samples[1][1],y_pred_val1,y_pred_val2,y_pred_val3,y_pred_val4)
    diff_pred1_test,diff_pred2_test,diff_pred3_test,diff_pred4_test = evaluate_inner(model1.name,name2,name3,name4,title,dir,'Test Data',samples[2][1],y_pred_test1,y_pred_test2,y_pred_test3,y_pred_test4)

    if len(diff_pred1_train) > 100000:
        diff_pred1_train = diff_pred1_train[:100000]
        diff_pred2_train = diff_pred2_train[:100000]
        diff_pred3_train = diff_pred3_train[:100000]
        diff_pred4_train = diff_pred4_train[:100000]
    if len(diff_pred1_val) > 100000:
        diff_pred1_val = diff_pred1_val[:100000]
        diff_pred2_val = diff_pred2_val[:100000]
        diff_pred3_val = diff_pred3_val[:100000]
        diff_pred4_val = diff_pred4_val[:100000]
    if len(diff_pred1_test) > 100000:
        diff_pred1_test = diff_pred1_test[:100000]
        diff_pred2_test = diff_pred2_test[:100000]
        diff_pred3_test = diff_pred3_test[:100000]
        diff_pred4_test = diff_pred4_test[:100000]
        
    sns.kdeplot(diff_pred1_train, fill=True, label='Training Data', alpha=0.2)
    sns.kdeplot(diff_pred1_val, fill=True, label='Validation Data', alpha=0.2)
    sns.kdeplot(diff_pred1_test, fill=True, label='Test Data', alpha=0.2)
    plt.xlabel('Actual Distance - Predicted Distance')
    plt.ylabel('Density')
    plt.title(model1.name+': '+title)
    plt.legend()
    plt.savefig(dir+'/'+model1.name+'_'+title+'_errordensity.png')
    plt.close('all')

    if model2 != None:
        sns.kdeplot(diff_pred2_train, fill=True, label='Training Data', alpha=0.2)
        sns.kdeplot(diff_pred2_val, fill=True, label='Validation Data', alpha=0.2)
        sns.kdeplot(diff_pred2_test, fill=True, label='Test Data', alpha=0.2)
        plt.xlabel('Actual Distance - Predicted Distance')
        plt.ylabel('Density')
        plt.title(name2+': '+title)
        plt.legend()
        plt.savefig(dir+'/'+name2+'_'+title+'_errordensity.png') 
        plt.close('all')

    if model3 != None:
        sns.kdeplot(diff_pred3_train, fill=True, label='Training Data', alpha=0.2)
        sns.kdeplot(diff_pred3_val, fill=True, label='Validation Data', alpha=0.2)
        sns.kdeplot(diff_pred3_test, fill=True, label='Test Data', alpha=0.2)
        plt.xlabel('Actual Distance - Predicted Distance')
        plt.ylabel('Density')
        plt.title(name3+': '+title)
        plt.legend()
        plt.savefig(dir+'/'+name3+'_'+title+'_errordensity.png') 
        plt.close('all')

    if model4 != None:
        sns.kdeplot(diff_pred4_train, fill=True, label='Training Data', alpha=0.2)
        sns.kdeplot(diff_pred4_val, fill=True, label='Validation Data', alpha=0.2)
        sns.kdeplot(diff_pred4_test, fill=True, label='Test Data', alpha=0.2)
        plt.xlabel('Actual Distance - Predicted Distance')
        plt.ylabel('Density')
        plt.title(name4+': '+title)
        plt.legend()
        plt.savefig(dir+'/'+name4+'_'+title+'_errordensity.png') 
        plt.close('all')

def generateSamples_evaluation(num_graphs,k_max,function,*args,**kwargs):
    
    print('Generating data for evaluation...')
    min_graph_size = np.inf
    max_graph_size = 0
    min_num_edges = np.inf
    max_num_edges = 0
    all_graph_info = []
    all_selected_pairs = []
    all_random_seeds = []
    num_all_nodes = 0
    num_all_edges = 0
    G,directed,weighted = function(*args,**kwargs)
    n = len(G.nodes())
    r = int(np.floor(np.sqrt(n)))
    if n % 2 == 0:
        sizes = list(np.random.choice(range(int(np.ceil(r/2)),int(n/2)+1), size=num_graphs, replace=True)*2)
    else:
        sizes = list(np.random.choice(range(int(np.ceil(r/2)),int(np.ceil(n/2))), size=num_graphs, replace=True)*2)
    
    n_rejected1 = 0
    n_rejected2 = 0
    while len(sizes) > 0:
        try:
            G,directed,weighted = function(*args,**kwargs)
            largest_component = max(nx.strongly_connected_components(G), key=len)
            num_nodes = len(largest_component)
            if num_nodes in sizes:
                sizes.remove(num_nodes)
                G = G.subgraph(largest_component)
                G = nx.relabel_nodes(G, {node: index for index, node in enumerate(G.nodes())})
                graph_info = G,directed,weighted
                all_graph_info.append(graph_info)
                random_seeds = []
                for k in range(1,k_max+1):
                    Sarma = seedSets(graph_info,k)
                    random_seeds.append(Sarma)
                all_random_seeds.append(random_seeds)
                nodes = range(num_nodes)
                pairs = []
                if directed:
                    num_pairs = int(2559/1999*num_nodes-2800/1999) ## 5 pairs for n = 5 and 10000 pairs for n = 12800, linear regression
                    while len(pairs) < num_pairs:
                        l1 = np.random.choice(nodes,size=num_pairs,replace=True)
                        l2 = np.random.choice(nodes,size=num_pairs,replace=True)
                        pairs = list(set(pairs+list(zip(l1,l2))))
                    num_edges = len([(u,v) for (u,v) in G.edges() if u != v])
                else:
                    num_pairs = int(2559/1999*num_nodes-2800/1999) ## 5 pairs for n = 5 and 10000 pairs for n = 12800, linear regression
                    while len(pairs) < num_pairs:
                        l1 = np.random.choice(nodes,size=num_pairs,replace=True)
                        l2 = np.random.choice(nodes,size=num_pairs,replace=True)
                        new_pairs = [(u,v) if u<=v else (v,u) for (u,v) in list(zip(l1,l2))]
                        pairs = list(set(pairs+new_pairs))
                    num_edges = len([(u,v) for (u,v) in G.edges() if v > u])
                all_selected_pairs.append(pairs[:num_pairs])
                num_all_nodes += num_nodes
                num_all_edges += num_edges
                min_graph_size = min(min_graph_size,num_nodes)
                max_graph_size = max(max_graph_size,num_nodes)
                min_num_edges = min(min_num_edges,num_edges)
                max_num_edges = max(max_num_edges,num_edges)
            else:
                n_rejected2 += 1
        except:
            n_rejected1 += 1
        if n_rejected1 + n_rejected2 >= 10000:
            print('Total number of graphs generated:',len(all_graph_info))
            print('Number of graphs rejected in loop 1 because of undefined errors:',n_rejected1)
            print('Number of graphs rejected in loop 1 because the largest component has insufficient size:',n_rejected2)
            print('Stuck in loop 1.')
            break

    num_remaining = len(sizes)
    min_size = min(sizes)
    print('Graph size threshold for loop 2:',min_size)
    n_rejected1 = 0
    n_rejected2 = 0
    k = 0
    while k < num_remaining:
        try:
            G,directed,weighted = function(*args,**kwargs)
            largest_component = max(nx.strongly_connected_components(G), key=len)
            num_nodes = len(largest_component)
            if num_nodes >= min_size:
                G = G.subgraph(largest_component)
                G = nx.relabel_nodes(G, {node: index for index, node in enumerate(G.nodes())})
                graph_info = G,directed,weighted
                all_graph_info.append(graph_info)
                random_seeds = []
                for i in range(1,k_max+1):
                    Sarma = seedSets(graph_info,i)
                    random_seeds.append(Sarma)
                all_random_seeds.append(random_seeds)
                nodes = range(num_nodes)
                pairs = []
                if directed:
                    num_pairs = int(2559/1999*num_nodes-2800/1999) ## 5 pairs for n = 5 and 10000 pairs for n = 12800, linear regression
                    while len(pairs) < num_pairs:
                        l1 = np.random.choice(nodes,size=num_pairs,replace=True)
                        l2 = np.random.choice(nodes,size=num_pairs,replace=True)
                        pairs = list(set(pairs+list(zip(l1,l2))))
                    num_edges = len([(u,v) for (u,v) in G.edges() if u != v])
                else:
                    num_pairs = int(2559/1999*num_nodes-2800/1999) ## 5 pairs for n = 5 and 10000 pairs for n = 12800, linear regression
                    while len(pairs) < num_pairs:
                        l1 = np.random.choice(nodes,size=num_pairs,replace=True)
                        l2 = np.random.choice(nodes,size=num_pairs,replace=True)
                        new_pairs = [(u,v) if u<=v else (v,u) for (u,v) in list(zip(l1,l2))]
                        pairs = list(set(pairs+new_pairs))
                    num_edges = len([(u,v) for (u,v) in G.edges() if v > u])
                all_selected_pairs.append(pairs[:num_pairs])
                num_all_nodes += num_nodes
                num_all_edges += num_edges
                min_graph_size = min(min_graph_size,num_nodes)
                max_graph_size = max(max_graph_size,num_nodes)
                min_num_edges = min(min_num_edges,num_edges)
                max_num_edges = max(max_num_edges,num_edges)
            else:
                n_rejected2 += 1
        except:
            n_rejected1 += 1
        if n_rejected1 + n_rejected2 >= 10000:
            print('Total number of graphs generated:',len(all_graph_info))
            print('Number of graphs rejected in loop 2 because of undefined errors:',n_rejected1)
            print('Number of graphs rejected in loop 2 because the largest component has insufficient size:',n_rejected2)
            raise ValueError('Stuck in loop 2.')

    graph_size_info = [min_graph_size,max_graph_size,num_all_nodes/num_graphs]
    edges_info = [min_num_edges,max_num_edges,num_all_edges/num_graphs]
    print('Graph size (min, max, mean):',graph_size_info)
    print('Number of edges (min, max, mean):',edges_info)

    all_dists = []
    all_durs = []
    for graph_info in all_graph_info:
        dists,durs = shortestDistances_networkx(graph_info)
        all_dists.append(dists)
        all_durs.append(durs)
        
    return all_graph_info,[all_dists,np.mean(np.array(all_durs))],all_random_seeds,[None]*6,all_selected_pairs,graph_size_info,edges_info

def generateSamples_evaluation_dRegular(num_graphs,k_max,n,lbd):
    
    print('Generating data for evaluation...')
    min_graph_size = np.inf
    max_graph_size = 0
    min_num_edges = np.inf
    max_num_edges = 0
    all_graph_info = []
    all_selected_pairs = []
    all_random_seeds = []
    num_all_nodes = 0
    num_all_edges = 0
    r = int(np.floor(np.sqrt(n)))
    if n % 2 == 0:
        sizes = list(np.random.choice(range(max(int(np.ceil(r/2)),lbd),int(n/2)+1), size=num_graphs, replace=True)*2)
    else:
        sizes = list(np.random.choice(range(max(int(np.ceil(r/2)),lbd),int(np.ceil(n/2))), size=num_graphs, replace=True)*2)
    print(sizes)

    n_rejected1 = 0
    n_rejected2 = 0
    for num_nodes in sizes:
        try:
            graph_info = dRegularGraph(num_nodes,lbd)
            while not nx.is_strongly_connected(graph_info[0]):
                graph_info = dRegularGraph(num_nodes,lbd)
                n_rejected2 += 1
                if n_rejected1 + n_rejected2 >= 10000:
                    print('Total number of graphs generated:',len(all_graph_info))
                    print('Number of graphs rejected because of undefined errors:',n_rejected1)
                    print('Number of graphs rejected because the largest component has insufficient size:',n_rejected2)
                    raise ValueError('Possibly stuck in infinite loop.')
            all_graph_info.append(graph_info)
            random_seeds = []
            for k in range(1,k_max+1):
                Sarma = seedSets(graph_info,k)
                random_seeds.append(Sarma)
            all_random_seeds.append(random_seeds)
            nodes = range(num_nodes)
            pairs = []
            num_pairs = int(2559/1999*num_nodes-2800/1999) ## 5 pairs for n = 5 and 10000 pairs for n = 12800, linear regression
            while len(pairs) < num_pairs:
                l1 = np.random.choice(nodes,size=num_pairs,replace=True)
                l2 = np.random.choice(nodes,size=num_pairs,replace=True)
                new_pairs = [(u,v) if u<=v else (v,u) for (u,v) in list(zip(l1,l2))]
                pairs = list(set(pairs+new_pairs))
            all_selected_pairs.append(pairs[:num_pairs])
            num_all_nodes += num_nodes
            num_edges = len([(u,v) for (u,v) in graph_info[0].edges() if v > u])
            num_all_edges += num_edges
            min_graph_size = min(min_graph_size,num_nodes)
            max_graph_size = max(max_graph_size,num_nodes)
            min_num_edges = min(min_num_edges,num_edges)
            max_num_edges = max(max_num_edges,num_edges)
        except:
            n_rejected1 += 1
        if n_rejected1 + n_rejected2 >= 10000:
            print('Total number of graphs generated:',len(all_graph_info))
            print('Number of graphs rejected because of undefined errors:',n_rejected1)
            print('Number of graphs rejected because the largest component has insufficient size:',n_rejected2)
            raise ValueError('Possibly stuck in infinite loop.')

    graph_size_info = [min_graph_size,max_graph_size,num_all_nodes/num_graphs]
    edges_info = [min_num_edges,max_num_edges,num_all_edges/num_graphs]
    print('Graph size (min, max, mean):',graph_size_info)
    print('Number of edges (min, max, mean):',edges_info)
    
    all_dists = []
    all_durs = []
    for graph_info in all_graph_info:
        dists,durs = shortestDistances_networkx(graph_info)
        all_dists.append(dists)
        all_durs.append(durs)

    return all_graph_info,[all_dists,np.mean(np.array(all_durs))],all_random_seeds,[None]*6,all_selected_pairs,graph_size_info,edges_info

def generateSamples_evaluation_ErdosRenyi(num_graphs,k_max,n,lbd):
    
    print('Generating data for evaluation...')
    min_graph_size = np.inf
    max_graph_size = 0
    min_num_edges = np.inf
    max_num_edges = 0
    all_graph_info = []
    all_selected_pairs = []
    all_random_seeds = []
    num_all_nodes = 0
    num_all_edges = 0
    r = int(np.floor(np.sqrt(n)))
    
    while len(all_graph_info) < num_graphs:
        sizes = list(np.random.choice(range(r,n+1), size=num_graphs, replace=True))
        print(sizes)
        for num_nodes in sizes:
            try:
                graph_info = connectedErdosRenyiGraph(num_nodes,lbd/num_nodes)
                all_graph_info.append(graph_info)
                random_seeds = []
                for k in range(1,k_max+1):
                    Sarma = seedSets(graph_info,k)
                    random_seeds.append(Sarma)
                all_random_seeds.append(random_seeds)
                nodes = range(num_nodes)
                pairs = []
                num_pairs = int(2559/1999*num_nodes-2800/1999) ## 5 pairs for n = 5 and 10000 pairs for n = 12800, linear regression
                while len(pairs) < num_pairs:
                    l1 = np.random.choice(nodes,size=num_pairs,replace=True)
                    l2 = np.random.choice(nodes,size=num_pairs,replace=True)
                    new_pairs = [(u,v) if u<=v else (v,u) for (u,v) in list(zip(l1,l2))]
                    pairs = list(set(pairs+new_pairs))
                all_selected_pairs.append(pairs[:num_pairs])
                num_all_nodes += num_nodes
                num_edges = len([(u,v) for (u,v) in graph_info[0].edges() if v > u])
                num_all_edges += num_edges
                min_graph_size = min(min_graph_size,num_nodes)
                max_graph_size = max(max_graph_size,num_nodes)
                min_num_edges = min(min_num_edges,num_edges)
                max_num_edges = max(max_num_edges,num_edges)
                if len(all_graph_info) == num_graphs:
                    break
            except:
                pass

    graph_size_info = [min_graph_size,max_graph_size,num_all_nodes/num_graphs]
    edges_info = [min_num_edges,max_num_edges,num_all_edges/num_graphs]
    print('Graph size (min, max, mean):',graph_size_info)
    print('Number of edges (min, max, mean):',edges_info)
    
    all_dists = []
    all_durs = []
    for graph_info in all_graph_info:
        dists,durs = shortestDistances_networkx(graph_info)
        all_dists.append(dists)
        all_durs.append(durs)

    return all_graph_info,[all_dists,np.mean(np.array(all_durs))],all_random_seeds,[None]*6,all_selected_pairs,graph_size_info,edges_info

def revisePairs(input):
    all_selected_pairs = []
    for graph_info in input[0]:
        num_nodes = len(graph_info[0].nodes())
        nodes = range(num_nodes)
        pairs = []
        if graph_info[1]: # if directed
            num_pairs = int(2559/1999*num_nodes-2800/1999) ## 5 pairs for n = 5 and 10000 pairs for n = 12800, linear regression
            while len(pairs) < num_pairs:
                l1 = np.random.choice(nodes,size=num_pairs,replace=True)
                l2 = np.random.choice(nodes,size=num_pairs,replace=True)
                pairs = list(set(pairs+list(zip(l1,l2))))
        else:
            num_pairs = int(2559/1999*num_nodes-2800/1999) ## 5 pairs for n = 5 and 10000 pairs for n = 12800, linear regression
            while len(pairs) < num_pairs:
                l1 = np.random.choice(nodes,size=num_pairs,replace=True)
                l2 = np.random.choice(nodes,size=num_pairs,replace=True)
                new_pairs = [(u,v) if u<=v else (v,u) for (u,v) in list(zip(l1,l2))]
                pairs = list(set(pairs+new_pairs))
        all_selected_pairs.append(pairs[:num_pairs])
    return input[0],input[1],input[2],input[3],all_selected_pairs,input[5],input[6]

def reviseSeedSets(input,revise_random_seeds=False,centralities=['degree','closeness','betweenness','harmonic','laplacian','pagerank']):
    
    if len(input[3]) != 6:
        all_revisedSeeds = [None]*6
    else:
        all_revisedSeeds = input[3]
        
    k_max = len(input[2][0])
    all_random_seeds = []
    all_seeds_by_degree = []
    all_seeds_by_closeness = []
    all_seeds_by_betweenness = []
    all_seeds_by_harmonic = []
    all_seeds_by_laplacian = []
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
        seeds_by_harmonic = []
        seeds_by_laplacian = []
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
            if 'harmonic' in centralities:
                #print('Calculating harmonic centrality...')
                harmonic = revise_seedSets(graph_info,k,nx.harmonic_centrality)
                seeds_by_harmonic.append(harmonic)
            if 'laplacian' in centralities:
                #print('Calculating laplacian centrality...')
                laplacian = revise_seedSets(graph_info,k,nx.laplacian_centrality)
                seeds_by_laplacian.append(laplacian)
            if 'pagerank' in centralities:
                #print('Calculating pagerank...')
                pagerank = revise_seedSets(graph_info,k,nx.pagerank)
                seeds_by_pagerank.append(pagerank)
        if revise_random_seeds:
            all_random_seeds.append(random_seeds)
        all_seeds_by_degree.append(seeds_by_degree)
        all_seeds_by_closeness.append(seeds_by_closeness)
        all_seeds_by_betweenness.append(seeds_by_betweenness)
        all_seeds_by_harmonic.append(seeds_by_harmonic)
        all_seeds_by_laplacian.append(seeds_by_laplacian)
        all_seeds_by_pagerank.append(seeds_by_pagerank)
    
    if 'degree' in centralities:
        all_revisedSeeds[0] = all_seeds_by_degree
    if 'closeness' in centralities:
        all_revisedSeeds[1] = all_seeds_by_closeness
    if 'betweenness' in centralities:
        all_revisedSeeds[2] = all_seeds_by_betweenness
    if 'harmonic' in centralities:
        all_revisedSeeds[3] = all_seeds_by_harmonic
    if 'laplacian' in centralities:
        all_revisedSeeds[4] = all_seeds_by_laplacian
    if 'pagerank' in centralities:
        all_revisedSeeds[5] = all_seeds_by_pagerank
    if revise_random_seeds:
        return input[0],input[1],all_random_seeds,all_revisedSeeds,input[4],input[5],input[6]
    else:
        return input[0],input[1],input[2],all_revisedSeeds,input[4],input[5],input[6]

def evaluate_all_distances(which_cuda,node_to_seed_model1,node_to_seed_model2,criterion_type,samples,seed_metric='random',GNN_only=True,seed_to_node_model1=None,seed_to_node_model2=None):

    assert criterion_type in ['mse','l2','l1'], 'Invalid criterion type.'

    all_graph_info = samples[0]
    all_actual_dists = samples[1][0]
    if seed_metric == 'random':
        all_seedSets = samples[2]
    elif seed_metric == 'degree':
        all_seedSets = samples[3][0]
    elif seed_metric == 'closeness':
        all_seedSets = samples[3][1]
    elif seed_metric == 'betweenness':
        all_seedSets = samples[3][2]
    elif seed_metric == 'harmonic':
        all_seedSets = samples[3][3]
    elif seed_metric == 'laplacian':
        all_seedSets = samples[3][4]
    elif seed_metric == 'pagerank':
        all_seedSets = samples[3][5]
    else:
        raise AssertionError('Invalid seed metric.')

    all_GNN_preds = []
    all_pred = []
    num_graphs = len(all_graph_info)

    if GNN_only:

        if node_to_seed_model2 != None:
            for N in range(num_graphs):
                if N % 10 == 0:
                    print(str(N/num_graphs*100)+'% complete')
                graph_info = all_graph_info[N]
                actual = all_actual_dists[N]
                seedSets = all_seedSets[N]
                k_max = len(seedSets)
                GNN_preds = []
                pred_GNN1,dur_GNN = shortestDistances_GNN(which_cuda,node_to_seed_model1,criterion_type,graph_info)
                data = pred_GNN1.ravel(),dur_GNN
                GNN_preds.append(data)
                pred_GNN2,dur_GNN = shortestDistances_GNN(which_cuda,node_to_seed_model2,criterion_type,graph_info)
                data = pred_GNN2.ravel(),dur_GNN
                GNN_preds.append(data)
                pred = np.zeros((6,k_max,actual.shape[0],actual.shape[1]))
                if graph_info[1]:
                    pred_GNN3,dur_GNN = shortestDistances_GNN(which_cuda,seed_to_node_model1,criterion_type,graph_info)
                    pred_GNN3 = pred_GNN3.T
                    data = pred_GNN3.ravel(),dur_GNN
                    GNN_preds.append(data)
                    pred_GNN4,dur_GNN = shortestDistances_GNN(which_cuda,seed_to_node_model2,criterion_type,graph_info)
                    pred_GNN4 = pred_GNN4.T
                    data = pred_GNN4.ravel(),dur_GNN
                    GNN_preds.append(data)
                    for k in range(k_max):
                        BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1)
                        BourgainSketches_GNN2_uS,SarmaSketches_GNN2_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN2)
                        BourgainSketches_GNN1_Su,SarmaSketches_GNN1_Su = offlineSketches(graph_info,seedSets[k],which_sketch='target',sketch_type='both',dists=pred_GNN3)
                        BourgainSketches_GNN2_Su,SarmaSketches_GNN2_Su = offlineSketches(graph_info,seedSets[k],which_sketch='target',sketch_type='both',dists=pred_GNN4)
                        BourgainDists_GNN1 = np.zeros_like(actual)
                        SarmaDists_GNN1 = np.zeros_like(actual)
                        BourgainDists_GNN2 = np.zeros_like(actual)
                        SarmaDists_GNN2 = np.zeros_like(actual)
                        for u in range(actual.shape[0]):
                            for v in range(actual.shape[1]):
                                if u != v:
                                    BourgainDists_GNN1[u,v] = online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v])
                                    SarmaDists_GNN1[u,v] = online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_Su[v])
                                    BourgainDists_GNN2[u,v] = online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v],BourgainSketches_GNN2_Su[u],BourgainSketches_GNN2_Su[v])
                                    SarmaDists_GNN2[u,v] = online_Sarma(SarmaSketches_GNN2_uS[u],SarmaSketches_GNN2_Su[v])
                else:
                    GNN_preds.append(None)
                    GNN_preds.append(None)
                    for k in range(k_max):
                        BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1)
                        BourgainSketches_GNN2_uS,SarmaSketches_GNN2_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN2)
                        BourgainDists_GNN1 = np.zeros_like(actual)
                        SarmaDists_GNN1 = np.zeros_like(actual)
                        BourgainDists_GNN2 = np.zeros_like(actual)
                        SarmaDists_GNN2 = np.zeros_like(actual)
                        for u in range(actual.shape[0]):
                            for v in range(u+1,actual.shape[1]):
                                BourgainDists_GNN1[u,v] = online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v])
                                SarmaDists_GNN1[u,v] = online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_uS[v])
                                BourgainDists_GNN2[u,v] = online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v])
                                SarmaDists_GNN2[u,v] = online_Sarma(SarmaSketches_GNN2_uS[u],SarmaSketches_GNN2_uS[v])
                                BourgainDists_GNN1[v,u] = BourgainDists_GNN1[u,v]
                                SarmaDists_GNN1[v,u] = SarmaDists_GNN1[u,v]
                                BourgainDists_GNN2[v,u] = BourgainDists_GNN2[u,v]
                                SarmaDists_GNN2[v,u] = SarmaDists_GNN2[u,v]
                    pred[2,k] = BourgainDists_GNN1
                    pred[3,k] = SarmaDists_GNN1
                    pred[4,k] = BourgainDists_GNN2
                    pred[5,k] = SarmaDists_GNN2
                pred = np.where(pred == float('inf'), len(graph_info[0].nodes()), pred)
                pred = pred.reshape(6,k_max,actual.shape[0]*actual.shape[1])
                all_GNN_preds.append(GNN_preds)
                all_pred.append(pred)
                        
        else:
            for N in range(num_graphs):
                if N % 10 == 0:
                    print(str(N/num_graphs*100)+'% complete')
                graph_info = all_graph_info[N]
                actual = all_actual_dists[N]
                seedSets = all_seedSets[N]
                k_max = len(seedSets)
                GNN_preds = []
                pred_GNN1,dur_GNN = shortestDistances_GNN(which_cuda,node_to_seed_model1,criterion_type,graph_info)
                data = pred_GNN1.ravel(),dur_GNN
                GNN_preds.append(data)
                GNN_preds.append(None)
                pred = np.zeros((6,k_max,actual.shape[0],actual.shape[1]))
                if graph_info[1]:
                    pred_GNN3,dur_GNN = shortestDistances_GNN(which_cuda,seed_to_node_model1,criterion_type,graph_info)
                    pred_GNN3 = pred_GNN3.T
                    data = pred_GNN3.ravel(),dur_GNN
                    GNN_preds.append(data)
                    GNN_preds.append(None)
                    for k in range(k_max):
                        BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1)
                        BourgainSketches_GNN1_Su,SarmaSketches_GNN1_Su = offlineSketches(graph_info,seedSets[k],which_sketch='target',sketch_type='both',dists=pred_GNN3)
                        BourgainDists_GNN1 = np.zeros_like(actual)
                        SarmaDists_GNN1 = np.zeros_like(actual)
                        for u in range(actual.shape[0]):
                            for v in range(actual.shape[1]):
                                if u != v:
                                    BourgainDists_GNN1[u,v] = online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v])
                                    SarmaDists_GNN1[u,v] = online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_Su[v])
                else:
                    GNN_preds.append(None)
                    GNN_preds.append(None)
                    for k in range(k_max):
                        BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1)
                        BourgainDists_GNN1 = np.zeros_like(actual)
                        SarmaDists_GNN1 = np.zeros_like(actual)
                        for u in range(actual.shape[0]):
                            for v in range(u+1,actual.shape[1]):
                                BourgainDists_GNN1[u,v] = online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v])
                                SarmaDists_GNN1[u,v] = online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_uS[v])
                                BourgainDists_GNN1[v,u] = BourgainDists_GNN1[u,v]
                                SarmaDists_GNN1[v,u] = SarmaDists_GNN1[u,v]
                    pred[2,k] = BourgainDists_GNN1
                    pred[3,k] = SarmaDists_GNN1
                pred = np.where(pred == float('inf'), len(graph_info[0].nodes()), pred)
                pred = pred.reshape(6,k_max,actual.shape[0]*actual.shape[1])
                all_GNN_preds.append(GNN_preds)
                all_pred.append(pred)

    else:

        if node_to_seed_model2 != None:
            for N in range(num_graphs):
                if N % 10 == 0:
                    print(str(N/num_graphs*100)+'% complete')
                graph_info = all_graph_info[N]
                actual = all_actual_dists[N]
                seedSets = all_seedSets[N]
                k_max = len(seedSets)
                GNN_preds = []
                pred_GNN1,dur_GNN = shortestDistances_GNN(which_cuda,node_to_seed_model1,criterion_type,graph_info)
                data = pred_GNN1.ravel(),dur_GNN
                GNN_preds.append(data)
                pred_GNN2,dur_GNN = shortestDistances_GNN(which_cuda,node_to_seed_model2,criterion_type,graph_info)
                data = pred_GNN2.ravel(),dur_GNN
                GNN_preds.append(data)
                pred = np.zeros((6,k_max,actual.shape[0],actual.shape[1]))
                if graph_info[1]:
                    pred_GNN3,dur_GNN = shortestDistances_GNN(which_cuda,seed_to_node_model1,criterion_type,graph_info)
                    pred_GNN3 = pred_GNN3.T
                    data = pred_GNN3.ravel(),dur_GNN
                    GNN_preds.append(data)
                    pred_GNN4,dur_GNN = shortestDistances_GNN(which_cuda,seed_to_node_model2,criterion_type,graph_info)
                    pred_GNN4 = pred_GNN4.T
                    data = pred_GNN4.ravel(),dur_GNN
                    GNN_preds.append(data)
                    for k in range(k_max):
                        BourgainSketches_networkx_uS,SarmaSketches_networkx_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=actual)
                        BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1)
                        BourgainSketches_GNN2_uS,SarmaSketches_GNN2_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN2)
                        BourgainSketches_networkx_Su,SarmaSketches_networkx_Su = offlineSketches(graph_info,seedSets[k],which_sketch='target',sketch_type='both',dists=actual)
                        BourgainSketches_GNN1_Su,SarmaSketches_GNN1_Su = offlineSketches(graph_info,seedSets[k],which_sketch='target',sketch_type='both',dists=pred_GNN3)
                        BourgainSketches_GNN2_Su,SarmaSketches_GNN2_Su = offlineSketches(graph_info,seedSets[k],which_sketch='target',sketch_type='both',dists=pred_GNN4)
                        BourgainDists_networkx = np.zeros_like(actual)
                        SarmaDists_networkx = np.zeros_like(actual)
                        BourgainDists_GNN1 = np.zeros_like(actual)
                        SarmaDists_GNN1 = np.zeros_like(actual)
                        BourgainDists_GNN2 = np.zeros_like(actual)
                        SarmaDists_GNN2 = np.zeros_like(actual)
                        for u in range(actual.shape[0]):
                            for v in range(actual.shape[1]):
                                if u != v:
                                    BourgainDists_networkx[u,v] = online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v],BourgainSketches_networkx_Su[u],BourgainSketches_networkx_Su[v])
                                    SarmaDists_networkx[u,v] = online_Sarma(SarmaSketches_networkx_uS[u],SarmaSketches_networkx_Su[v])
                                    BourgainDists_GNN1[u,v] = online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v])
                                    SarmaDists_GNN1[u,v] = online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_Su[v])
                                    BourgainDists_GNN2[u,v] = online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v],BourgainSketches_GNN2_Su[u],BourgainSketches_GNN2_Su[v])
                                    SarmaDists_GNN2[u,v] = online_Sarma(SarmaSketches_GNN2_uS[u],SarmaSketches_GNN2_Su[v])
                else:
                    GNN_preds.append(None)
                    GNN_preds.append(None)
                    for k in range(k_max):
                        BourgainSketches_networkx_uS,SarmaSketches_networkx_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=actual)
                        BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1)
                        BourgainSketches_GNN2_uS,SarmaSketches_GNN2_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN2)
                        BourgainDists_networkx = np.zeros_like(actual)
                        SarmaDists_networkx = np.zeros_like(actual)
                        BourgainDists_GNN1 = np.zeros_like(actual)
                        SarmaDists_GNN1 = np.zeros_like(actual)
                        BourgainDists_GNN2 = np.zeros_like(actual)
                        SarmaDists_GNN2 = np.zeros_like(actual)
                        for u in range(actual.shape[0]):
                            for v in range(u+1,actual.shape[1]):
                                BourgainDists_networkx[u,v] = online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v])
                                SarmaDists_networkx[u,v] = online_Sarma(SarmaSketches_networkx_uS[u],SarmaSketches_networkx_uS[v])
                                BourgainDists_GNN1[u,v] = online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v])
                                SarmaDists_GNN1[u,v] = online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_uS[v])
                                BourgainDists_GNN2[u,v] = online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v])
                                SarmaDists_GNN2[u,v] = online_Sarma(SarmaSketches_GNN2_uS[u],SarmaSketches_GNN2_uS[v])
                                BourgainDists_networkx[v,u] = BourgainDists_networkx[u,v]
                                SarmaDists_networkx[v,u] = SarmaDists_networkx[u,v]
                                BourgainDists_GNN1[v,u] = BourgainDists_GNN1[u,v]
                                SarmaDists_GNN1[v,u] = SarmaDists_GNN1[u,v]
                                BourgainDists_GNN2[v,u] = BourgainDists_GNN2[u,v]
                                SarmaDists_GNN2[v,u] = SarmaDists_GNN2[u,v]
                    pred[0,k] = BourgainDists_networkx
                    pred[1,k] = SarmaDists_networkx
                    pred[2,k] = BourgainDists_GNN1
                    pred[3,k] = SarmaDists_GNN1
                    pred[4,k] = BourgainDists_GNN2
                    pred[5,k] = SarmaDists_GNN2
                pred = np.where(pred == float('inf'), len(graph_info[0].nodes()), pred)
                pred = pred.reshape(6,k_max,actual.shape[0]*actual.shape[1])
                all_GNN_preds.append(GNN_preds)
                all_pred.append(pred)
                        
        else:
            for N in range(num_graphs):
                if N % 10 == 0:
                    print(str(N/num_graphs*100)+'% complete')
                graph_info = all_graph_info[N]
                actual = all_actual_dists[N]
                seedSets = all_seedSets[N]
                k_max = len(seedSets)
                GNN_preds = []
                pred_GNN1,dur_GNN = shortestDistances_GNN(which_cuda,node_to_seed_model1,criterion_type,graph_info)
                data = pred_GNN1.ravel(),dur_GNN
                GNN_preds.append(data)
                GNN_preds.append(None)
                pred = np.zeros((6,k_max,actual.shape[0],actual.shape[1]))
                if graph_info[1]:
                    pred_GNN3,dur_GNN = shortestDistances_GNN(which_cuda,seed_to_node_model1,criterion_type,graph_info)
                    pred_GNN3 = pred_GNN3.T
                    data = pred_GNN3.ravel(),dur_GNN
                    GNN_preds.append(data)
                    GNN_preds.append(None)
                    for k in range(k_max):
                        BourgainSketches_networkx_uS,SarmaSketches_networkx_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=actual)
                        BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1)
                        BourgainSketches_networkx_Su,SarmaSketches_networkx_Su = offlineSketches(graph_info,seedSets[k],which_sketch='target',sketch_type='both',dists=actual)
                        BourgainSketches_GNN1_Su,SarmaSketches_GNN1_Su = offlineSketches(graph_info,seedSets[k],which_sketch='target',sketch_type='both',dists=pred_GNN3)
                        BourgainDists_networkx = np.zeros_like(actual)
                        SarmaDists_networkx = np.zeros_like(actual)
                        BourgainDists_GNN1 = np.zeros_like(actual)
                        SarmaDists_GNN1 = np.zeros_like(actual)
                        for u in range(actual.shape[0]):
                            for v in range(actual.shape[1]):
                                if u != v:
                                    BourgainDists_networkx[u,v] = online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v],BourgainSketches_networkx_Su[u],BourgainSketches_networkx_Su[v])
                                    SarmaDists_networkx[u,v] = online_Sarma(SarmaSketches_networkx_uS[u],SarmaSketches_networkx_Su[v])
                                    BourgainDists_GNN1[u,v] = online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v])
                                    SarmaDists_GNN1[u,v] = online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_Su[v])
                else:
                    GNN_preds.append(None)
                    GNN_preds.append(None)
                    for k in range(k_max):
                        BourgainSketches_networkx_uS,SarmaSketches_networkx_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=actual)
                        BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1)
                        BourgainDists_networkx = np.zeros_like(actual)
                        SarmaDists_networkx = np.zeros_like(actual)
                        BourgainDists_GNN1 = np.zeros_like(actual)
                        SarmaDists_GNN1 = np.zeros_like(actual)
                        for u in range(actual.shape[0]):
                            for v in range(u+1,actual.shape[1]):
                                BourgainDists_networkx[u,v] = online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v])
                                SarmaDists_networkx[u,v] = online_Sarma(SarmaSketches_networkx_uS[u],SarmaSketches_networkx_uS[v])
                                BourgainDists_GNN1[u,v] = online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v])
                                SarmaDists_GNN1[u,v] = online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_uS[v])
                                BourgainDists_networkx[v,u] = BourgainDists_networkx[u,v]
                                SarmaDists_networkx[v,u] = SarmaDists_networkx[u,v]
                                BourgainDists_GNN1[v,u] = BourgainDists_GNN1[u,v]
                                SarmaDists_GNN1[v,u] = SarmaDists_GNN1[u,v]
                    pred[0,k] = BourgainDists_networkx
                    pred[1,k] = SarmaDists_networkx
                    pred[2,k] = BourgainDists_GNN1
                    pred[3,k] = SarmaDists_GNN1
                pred = np.where(pred == float('inf'), len(graph_info[0].nodes()), pred)
                pred = pred.reshape(6,k_max,actual.shape[0]*actual.shape[1])
                all_GNN_preds.append(GNN_preds)
                all_pred.append(pred)

    return None,all_GNN_preds,all_pred

# def evaluate_random_distances(which_cuda,node_to_seed_model1,node_to_seed_model2,criterion_type,samples,seed_metric='random',GNN_only=True,seed_to_node_model1=None,seed_to_node_model2=None):

#     assert criterion_type in ['mse','l2','l1'], 'Invalid criterion type.'

#     all_graph_info = samples[0]
#     all_actual_dists = samples[1][0]
#     if seed_metric == 'random':
#         all_seedSets = samples[2]
#     elif seed_metric == 'degree':
#         all_seedSets = samples[3][0]
#     elif seed_metric == 'closeness':
#         all_seedSets = samples[3][1]
#     elif seed_metric == 'betweenness':
#         all_seedSets = samples[3][2]
#     elif seed_metric == 'harmonic':
#         all_seedSets = samples[3][3]
#     elif seed_metric == 'laplacian':
#         all_seedSets = samples[3][4]
#     elif seed_metric == 'pagerank':
#         all_seedSets = samples[3][5]
#     else:
#         raise AssertionError('Invalid seed metric.')
#     all_pairs = samples[4]

#     all_actual = []
#     all_GNN_preds = []
#     all_pred = []
#     num_graphs = len(all_graph_info)

#     if GNN_only:

#         if node_to_seed_model2 != None:
#             for N in range(num_graphs):
#                 if N % 10 == 0:
#                     print(str(N/num_graphs*100)+'% complete')
#                 graph_info = all_graph_info[N]
#                 actual = all_actual_dists[N]
#                 seedSets = all_seedSets[N]
#                 pairs = all_pairs[N]
#                 k_max = len(seedSets)
#                 GNN_preds = []
#                 pred_GNN1 = shortestDistances_GNN(which_cuda,node_to_seed_model1,criterion_type,graph_info)[0]
#                 pred_GNN2 = shortestDistances_GNN(which_cuda,node_to_seed_model2,criterion_type,graph_info)[0]
#                 pred = np.zeros((6,k_max,len(pairs)))
#                 if graph_info[1]:
#                     pred_GNN3 = shortestDistances_GNN(which_cuda,seed_to_node_model1,criterion_type,graph_info)[0].T
#                     pred_GNN4 = shortestDistances_GNN(which_cuda,seed_to_node_model2,criterion_type,graph_info)[0].T
#                     selected_actual = []
#                     selected_pred_GNN1 = []
#                     selected_pred_GNN2 = []
#                     selected_pred_GNN3 = []
#                     selected_pred_GNN4 = []
#                     sources = list(set(np.array(pairs)[:,0]))
#                     targets = list(set(np.array(pairs)[:,1]))
#                     for k in range(k_max):
#                         BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1,nodes=sources)
#                         BourgainSketches_GNN2_uS,SarmaSketches_GNN2_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN2,nodes=sources)
#                         BourgainSketches_GNN1_Su,SarmaSketches_GNN1_Su = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN3,nodes=targets)
#                         BourgainSketches_GNN2_Su,SarmaSketches_GNN2_Su = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN4,nodes=targets)
#                         BourgainDists_GNN1 = []
#                         SarmaDists_GNN1 = []
#                         BourgainDists_GNN2 = []
#                         SarmaDists_GNN2 = []
#                         if k == 0:
#                             for u,v in pairs:
#                                 selected_actual.append(actual[u,v])
#                                 selected_pred_GNN1.append(pred_GNN1[u,v])
#                                 selected_pred_GNN2.append(pred_GNN2[u,v])
#                                 selected_pred_GNN3.append(pred_GNN3[u,v])
#                                 selected_pred_GNN4.append(pred_GNN4[u,v])
#                                 BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v]))
#                                 SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_Su[v]))
#                                 BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v],BourgainSketches_GNN2_Su[u],BourgainSketches_GNN2_Su[v]))
#                                 SarmaDists_GNN2.append(online_Sarma(SarmaSketches_GNN2_uS[u],SarmaSketches_GNN2_Su[v]))
#                         else:
#                             for u,v in pairs:
#                                 BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v]))
#                                 SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_Su[v]))
#                                 BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v],BourgainSketches_GNN2_Su[u],BourgainSketches_GNN2_Su[v]))
#                                 SarmaDists_GNN2.append(online_Sarma(SarmaSketches_GNN2_uS[u],SarmaSketches_GNN2_Su[v]))
#                         pred[2,k] = BourgainDists_GNN1
#                         pred[3,k] = SarmaDists_GNN1
#                         pred[4,k] = BourgainDists_GNN2
#                         pred[5,k] = SarmaDists_GNN2
#                     data = selected_pred_GNN1,None
#                     GNN_preds.append(data)
#                     data = selected_pred_GNN2,None
#                     GNN_preds.append(data)
#                     data = selected_pred_GNN3,None
#                     GNN_preds.append(data)
#                     data = selected_pred_GNN4,None
#                     GNN_preds.append(data)
#                 else:
#                     selected_actual = []
#                     selected_pred_GNN1 = []
#                     selected_pred_GNN2 = []
#                     all_nodes = np.array(pairs).flatten()
#                     for k in range(k_max):
#                         BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1,nodes=all_nodes)
#                         BourgainSketches_GNN2_uS,SarmaSketches_GNN2_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN2,nodes=all_nodes)
#                         BourgainDists_GNN1 = []
#                         SarmaDists_GNN1 = []
#                         BourgainDists_GNN2 = []
#                         SarmaDists_GNN2 = []
#                         if k == 0:
#                             for u,v in pairs:
#                                 selected_actual.append(actual[u,v])
#                                 selected_pred_GNN1.append(pred_GNN1[u,v])
#                                 selected_pred_GNN2.append(pred_GNN2[u,v])
#                                 BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
#                                 SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_uS[v]))
#                                 BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v]))
#                                 SarmaDists_GNN2.append(online_Sarma(SarmaSketches_GNN2_uS[u],SarmaSketches_GNN2_uS[v]))
#                         else:
#                             for u,v in pairs:
#                                 BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
#                                 SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_uS[v]))
#                                 BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v]))
#                                 SarmaDists_GNN2.append(online_Sarma(SarmaSketches_GNN2_uS[u],SarmaSketches_GNN2_uS[v]))
#                         pred[2,k] = BourgainDists_GNN1
#                         pred[3,k] = SarmaDists_GNN1
#                         pred[4,k] = BourgainDists_GNN2
#                         pred[5,k] = SarmaDists_GNN2
#                     data = selected_pred_GNN1,None
#                     GNN_preds.append(data)
#                     data = selected_pred_GNN2,None
#                     GNN_preds.append(data)
#                     GNN_preds.append(None)
#                     GNN_preds.append(None)
#                 pred = np.where(pred == float('inf'), len(graph_info[0].nodes()), pred)
#                 all_actual.append(selected_actual)
#                 all_GNN_preds.append(GNN_preds)
#                 all_pred.append(pred)

#         else:
#             for N in range(num_graphs):
#                 if N % 10 == 0:
#                     print(str(N/num_graphs*100)+'% complete')
#                 graph_info = all_graph_info[N]
#                 actual = all_actual_dists[N]
#                 seedSets = all_seedSets[N]
#                 pairs = all_pairs[N]
#                 k_max = len(seedSets)
#                 GNN_preds = []
#                 pred_GNN1 = shortestDistances_GNN(which_cuda,node_to_seed_model1,criterion_type,graph_info)[0]
#                 pred = np.zeros((6,k_max,len(pairs)))
#                 if graph_info[1]:
#                     pred_GNN3 = shortestDistances_GNN(which_cuda,seed_to_node_model1,criterion_type,graph_info)[0].T
#                     selected_actual = []
#                     selected_pred_GNN1 = []
#                     selected_pred_GNN3 = []
#                     sources = list(set(np.array(pairs)[:,0]))
#                     targets = list(set(np.array(pairs)[:,1]))
#                     for k in range(k_max):
#                         BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1,nodes=sources)
#                         BourgainSketches_GNN1_Su,SarmaSketches_GNN1_Su = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN3,nodes=targets)
#                         BourgainDists_GNN1 = []
#                         SarmaDists_GNN1 = []
#                         if k == 0:
#                             for u,v in pairs:
#                                 selected_actual.append(actual[u,v])
#                                 selected_pred_GNN1.append(pred_GNN1[u,v])
#                                 selected_pred_GNN3.append(pred_GNN3[u,v])
#                                 BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v]))
#                                 SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_Su[v]))
#                         else:
#                             for u,v in pairs:
#                                 BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v]))
#                                 SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_Su[v]))
#                         pred[2,k] = BourgainDists_GNN1
#                         pred[3,k] = SarmaDists_GNN1
#                     data = selected_pred_GNN1,None
#                     GNN_preds.append(data)
#                     GNN_preds.append(None)
#                     data = selected_pred_GNN3,None
#                     GNN_preds.append(data)
#                     GNN_preds.append(None)
#                 else:
#                     selected_actual = []
#                     selected_pred_GNN1 = []
#                     all_nodes = np.array(pairs).flatten()
#                     for k in range(k_max):
#                         BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1,nodes=all_nodes)
#                         BourgainDists_GNN1 = []
#                         SarmaDists_GNN1 = []
#                         if k == 0:
#                             for u,v in pairs:
#                                 selected_actual.append(actual[u,v])
#                                 selected_pred_GNN1.append(pred_GNN1[u,v])
#                                 BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
#                                 SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_uS[v]))
#                         else:
#                             for u,v in pairs:
#                                 BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
#                                 SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_uS[v]))
#                         pred[2,k] = BourgainDists_GNN1
#                         pred[3,k] = SarmaDists_GNN1
#                     data = selected_pred_GNN1,None
#                     GNN_preds.append(data)
#                     GNN_preds.append(None)
#                     GNN_preds.append(None)
#                     GNN_preds.append(None)
#                 pred = np.where(pred == float('inf'), len(graph_info[0].nodes()), pred)
#                 all_actual.append(selected_actual)
#                 all_GNN_preds.append(GNN_preds)
#                 all_pred.append(pred)

#     else:

#         if node_to_seed_model2 != None:
#             for N in range(num_graphs):
#                 if N % 10 == 0:
#                     print(str(N/num_graphs*100)+'% complete')
#                 graph_info = all_graph_info[N]
#                 actual = all_actual_dists[N]
#                 seedSets = all_seedSets[N]
#                 pairs = all_pairs[N]
#                 k_max = len(seedSets)
#                 GNN_preds = []
#                 pred_GNN1 = shortestDistances_GNN(which_cuda,node_to_seed_model1,criterion_type,graph_info)[0]
#                 pred_GNN2 = shortestDistances_GNN(which_cuda,node_to_seed_model2,criterion_type,graph_info)[0]
#                 pred = np.zeros((6,k_max,len(pairs)))
#                 if graph_info[1]:
#                     pred_GNN3 = shortestDistances_GNN(which_cuda,seed_to_node_model1,criterion_type,graph_info)[0].T
#                     pred_GNN4 = shortestDistances_GNN(which_cuda,seed_to_node_model2,criterion_type,graph_info)[0].T
#                     selected_actual = []
#                     selected_pred_GNN1 = []
#                     selected_pred_GNN2 = []
#                     selected_pred_GNN3 = []
#                     selected_pred_GNN4 = []
#                     sources = list(set(np.array(pairs)[:,0]))
#                     targets = list(set(np.array(pairs)[:,1]))
#                     for k in range(k_max):
#                         BourgainSketches_networkx_uS,SarmaSketches_networkx_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=actual,nodes=sources)
#                         BourgainSketches_networkx_Su,SarmaSketches_networkx_Su = offlineSketches(graph_info,seedSets[k],which_sketch='target',sketch_type='both',dists=actual,nodes=targets)
#                         BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1,nodes=sources)
#                         BourgainSketches_GNN2_uS,SarmaSketches_GNN2_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN2,nodes=sources)
#                         BourgainSketches_GNN1_Su,SarmaSketches_GNN1_Su = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN3,nodes=targets)
#                         BourgainSketches_GNN2_Su,SarmaSketches_GNN2_Su = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN4,nodes=targets)
#                         BourgainDists_networkx = []
#                         SarmaDists_networkx = []
#                         BourgainDists_GNN1 = []
#                         SarmaDists_GNN1 = []
#                         BourgainDists_GNN2 = []
#                         SarmaDists_GNN2 = []
#                         if k == 0:
#                             for u,v in pairs:
#                                 selected_actual.append(actual[u,v])
#                                 selected_pred_GNN1.append(pred_GNN1[u,v])
#                                 selected_pred_GNN2.append(pred_GNN2[u,v])
#                                 selected_pred_GNN3.append(pred_GNN3[u,v])
#                                 selected_pred_GNN4.append(pred_GNN4[u,v])
#                                 BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v],BourgainSketches_networkx_Su[u],BourgainSketches_networkx_Su[v]))
#                                 SarmaDists_networkx.append(online_Sarma(SarmaSketches_networkx_uS[u],SarmaSketches_networkx_Su[v]))
#                                 BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v]))
#                                 SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_Su[v]))
#                                 BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v],BourgainSketches_GNN2_Su[u],BourgainSketches_GNN2_Su[v]))
#                                 SarmaDists_GNN2.append(online_Sarma(SarmaSketches_GNN2_uS[u],SarmaSketches_GNN2_Su[v]))
#                         else:
#                             for u,v in pairs:
#                                 BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v],BourgainSketches_networkx_Su[u],BourgainSketches_networkx_Su[v]))
#                                 SarmaDists_networkx.append(online_Sarma(SarmaSketches_networkx_uS[u],SarmaSketches_networkx_Su[v]))
#                                 BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v]))
#                                 SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_Su[v]))
#                                 BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v],BourgainSketches_GNN2_Su[u],BourgainSketches_GNN2_Su[v]))
#                                 SarmaDists_GNN2.append(online_Sarma(SarmaSketches_GNN2_uS[u],SarmaSketches_GNN2_Su[v]))
#                         pred[0,k] = BourgainDists_networkx
#                         pred[1,k] = SarmaDists_networkx
#                         pred[2,k] = BourgainDists_GNN1
#                         pred[3,k] = SarmaDists_GNN1
#                         pred[4,k] = BourgainDists_GNN2
#                         pred[5,k] = SarmaDists_GNN2
#                     data = selected_pred_GNN1,None
#                     GNN_preds.append(data)
#                     data = selected_pred_GNN2,None
#                     GNN_preds.append(data)
#                     data = selected_pred_GNN3,None
#                     GNN_preds.append(data)
#                     data = selected_pred_GNN4,None
#                     GNN_preds.append(data)
#                 else:
#                     selected_actual = []
#                     selected_pred_GNN1 = []
#                     selected_pred_GNN2 = []
#                     all_nodes = np.array(pairs).flatten()
#                     for k in range(k_max):
#                         BourgainSketches_networkx_uS,SarmaSketches_networkx_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=actual,nodes=all_nodes)
#                         BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1,nodes=all_nodes)
#                         BourgainSketches_GNN2_uS,SarmaSketches_GNN2_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN2,nodes=all_nodes)
#                         BourgainDists_networkx = []
#                         SarmaDists_networkx = []
#                         BourgainDists_GNN1 = []
#                         SarmaDists_GNN1 = []
#                         BourgainDists_GNN2 = []
#                         SarmaDists_GNN2 = []
#                         if k == 0:
#                             for u,v in pairs:
#                                 selected_actual.append(actual[u,v])
#                                 selected_pred_GNN1.append(pred_GNN1[u,v])
#                                 selected_pred_GNN2.append(pred_GNN2[u,v])
#                                 BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v]))
#                                 SarmaDists_networkx.append(online_Sarma(SarmaSketches_networkx_uS[u],SarmaSketches_networkx_uS[v]))
#                                 BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
#                                 SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_uS[v]))
#                                 BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v]))
#                                 SarmaDists_GNN2.append(online_Sarma(SarmaSketches_GNN2_uS[u],SarmaSketches_GNN2_uS[v]))
#                         else:
#                             for u,v in pairs:
#                                 BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v]))
#                                 SarmaDists_networkx.append(online_Sarma(SarmaSketches_networkx_uS[u],SarmaSketches_networkx_uS[v]))
#                                 BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
#                                 SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_uS[v]))
#                                 BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v]))
#                                 SarmaDists_GNN2.append(online_Sarma(SarmaSketches_GNN2_uS[u],SarmaSketches_GNN2_uS[v]))
#                         pred[0,k] = BourgainDists_networkx
#                         pred[1,k] = SarmaDists_networkx
#                         pred[2,k] = BourgainDists_GNN1
#                         pred[3,k] = SarmaDists_GNN1
#                         pred[4,k] = BourgainDists_GNN2
#                         pred[5,k] = SarmaDists_GNN2
#                     data = selected_pred_GNN1,None
#                     GNN_preds.append(data)
#                     data = selected_pred_GNN2,None
#                     GNN_preds.append(data)
#                     GNN_preds.append(None)
#                     GNN_preds.append(None)
#                 pred = np.where(pred == float('inf'), len(graph_info[0].nodes()), pred)
#                 all_actual.append(selected_actual)
#                 all_GNN_preds.append(GNN_preds)
#                 all_pred.append(pred)

#         else:
#             for N in range(num_graphs):
#                 if N % 10 == 0:
#                     print(str(N/num_graphs*100)+'% complete')
#                 graph_info = all_graph_info[N]
#                 actual = all_actual_dists[N]
#                 seedSets = all_seedSets[N]
#                 pairs = all_pairs[N]
#                 k_max = len(seedSets)
#                 GNN_preds = []
#                 pred_GNN1 = shortestDistances_GNN(which_cuda,node_to_seed_model1,criterion_type,graph_info)[0]
#                 pred = np.zeros((6,k_max,len(pairs)))
#                 if graph_info[1]:
#                     pred_GNN3 = shortestDistances_GNN(which_cuda,seed_to_node_model1,criterion_type,graph_info)[0].T
#                     selected_actual = []
#                     selected_pred_GNN1 = []
#                     selected_pred_GNN3 = []
#                     sources = list(set(np.array(pairs)[:,0]))
#                     targets = list(set(np.array(pairs)[:,1]))
#                     for k in range(k_max):
#                         BourgainSketches_networkx_uS,SarmaSketches_networkx_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=actual,nodes=sources)
#                         BourgainSketches_networkx_Su,SarmaSketches_networkx_Su = offlineSketches(graph_info,seedSets[k],which_sketch='target',sketch_type='both',dists=actual,nodes=targets)
#                         BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1,nodes=sources)
#                         BourgainSketches_GNN1_Su,SarmaSketches_GNN1_Su = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN3,nodes=targets)
#                         BourgainDists_networkx = []
#                         SarmaDists_networkx = []
#                         BourgainDists_GNN1 = []
#                         SarmaDists_GNN1 = []
#                         if k == 0:
#                             for u,v in pairs:
#                                 selected_actual.append(actual[u,v])
#                                 selected_pred_GNN1.append(pred_GNN1[u,v])
#                                 selected_pred_GNN3.append(pred_GNN3[u,v])
#                                 BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v],BourgainSketches_networkx_Su[u],BourgainSketches_networkx_Su[v]))
#                                 SarmaDists_networkx.append(online_Sarma(SarmaSketches_networkx_uS[u],SarmaSketches_networkx_Su[v]))
#                                 BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v]))
#                                 SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_Su[v]))
#                         else:
#                             for u,v in pairs:
#                                 BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v],BourgainSketches_networkx_Su[u],BourgainSketches_networkx_Su[v]))
#                                 SarmaDists_networkx.append(online_Sarma(SarmaSketches_networkx_uS[u],SarmaSketches_networkx_Su[v]))
#                                 BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v]))
#                                 SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_Su[v]))
#                         pred[0,k] = BourgainDists_networkx
#                         pred[1,k] = SarmaDists_networkx
#                         pred[2,k] = BourgainDists_GNN1
#                         pred[3,k] = SarmaDists_GNN1
#                     data = selected_pred_GNN1,None
#                     GNN_preds.append(data)
#                     GNN_preds.append(None)
#                     data = selected_pred_GNN3,None
#                     GNN_preds.append(data)
#                     GNN_preds.append(None)
#                 else:
#                     selected_actual = []
#                     selected_pred_GNN1 = []
#                     all_nodes = np.array(pairs).flatten()
#                     for k in range(k_max):
#                         BourgainSketches_networkx_uS,SarmaSketches_networkx_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=actual,nodes=all_nodes)
#                         BourgainSketches_GNN1_uS,SarmaSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='both',dists=pred_GNN1,nodes=all_nodes)
#                         BourgainDists_networkx = []
#                         SarmaDists_networkx = []
#                         BourgainDists_GNN1 = []
#                         SarmaDists_GNN1 = []
#                         if k == 0:
#                             for u,v in pairs:
#                                 selected_actual.append(actual[u,v])
#                                 selected_pred_GNN1.append(pred_GNN1[u,v])
#                                 BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v]))
#                                 SarmaDists_networkx.append(online_Sarma(SarmaSketches_networkx_uS[u],SarmaSketches_networkx_uS[v]))
#                                 BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
#                                 SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_uS[v]))
#                         else:
#                             for u,v in pairs:
#                                 BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v]))
#                                 SarmaDists_networkx.append(online_Sarma(SarmaSketches_networkx_uS[u],SarmaSketches_networkx_uS[v]))
#                                 BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
#                                 SarmaDists_GNN1.append(online_Sarma(SarmaSketches_GNN1_uS[u],SarmaSketches_GNN1_uS[v]))
#                         pred[0,k] = BourgainDists_networkx
#                         pred[1,k] = SarmaDists_networkx
#                         pred[2,k] = BourgainDists_GNN1
#                         pred[3,k] = SarmaDists_GNN1
#                     data = selected_pred_GNN1,None
#                     GNN_preds.append(data)
#                     GNN_preds.append(None)
#                     GNN_preds.append(None)
#                     GNN_preds.append(None)
#                 pred = np.where(pred == float('inf'), len(graph_info[0].nodes()), pred)
#                 all_actual.append(selected_actual)
#                 all_GNN_preds.append(GNN_preds)
#                 all_pred.append(pred)

#     return all_actual,all_GNN_preds,all_pred

def evaluate_random_distances(which_cuda,node_to_seed_model1,node_to_seed_model2,criterion_type,samples,seed_metric='random',GNN_only=True,seed_to_node_model1=None,seed_to_node_model2=None):

    assert criterion_type in ['mse','l2','l1'], 'Invalid criterion type.'

    all_graph_info = samples[0]
    all_actual_dists = samples[1][0]
    if seed_metric == 'random':
        all_seedSets = samples[2]
    elif seed_metric == 'degree':
        all_seedSets = samples[3][0]
    elif seed_metric == 'closeness':
        all_seedSets = samples[3][1]
    elif seed_metric == 'betweenness':
        all_seedSets = samples[3][2]
    elif seed_metric == 'harmonic':
        all_seedSets = samples[3][3]
    elif seed_metric == 'laplacian':
        all_seedSets = samples[3][4]
    elif seed_metric == 'pagerank':
        all_seedSets = samples[3][5]
    else:
        raise AssertionError('Invalid seed metric.')
    all_pairs = samples[4]

    all_actual = []
    all_GNN_preds = []
    all_pred = []
    num_graphs = len(all_graph_info)

    if GNN_only:

        if node_to_seed_model2 != None:
            for N in range(num_graphs):
                if N % 10 == 0:
                    print(str(N/num_graphs*100)+'% complete')
                graph_info = all_graph_info[N]
                actual = all_actual_dists[N]
                seedSets = all_seedSets[N]
                pairs = all_pairs[N]
                k_max = len(seedSets)
                GNN_preds = []
                pred_GNN1 = shortestDistances_GNN(which_cuda,node_to_seed_model1,criterion_type,graph_info)[0]
                pred_GNN2 = shortestDistances_GNN(which_cuda,node_to_seed_model2,criterion_type,graph_info)[0]
                pred = np.zeros((6,k_max,len(pairs)))
                if graph_info[1]:
                    pred_GNN3 = shortestDistances_GNN(which_cuda,seed_to_node_model1,criterion_type,graph_info)[0].T
                    pred_GNN4 = shortestDistances_GNN(which_cuda,seed_to_node_model2,criterion_type,graph_info)[0].T
                    selected_actual = []
                    selected_pred_GNN1 = []
                    selected_pred_GNN2 = []
                    selected_pred_GNN3 = []
                    selected_pred_GNN4 = []
                    sources = list(set(np.array(pairs)[:,0]))
                    targets = list(set(np.array(pairs)[:,1]))
                    for k in range(k_max):
                        BourgainSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN1,nodes=sources)
                        BourgainSketches_GNN2_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN2,nodes=sources)
                        BourgainSketches_GNN1_Su = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN3,nodes=targets)
                        BourgainSketches_GNN2_Su = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN4,nodes=targets)
                        BourgainDists_GNN1 = []
                        BourgainDists_GNN2 = []
                        if k == 0:
                            for u,v in pairs:
                                selected_actual.append(actual[u,v])
                                selected_pred_GNN1.append(pred_GNN1[u,v])
                                selected_pred_GNN2.append(pred_GNN2[u,v])
                                selected_pred_GNN3.append(pred_GNN3[u,v])
                                selected_pred_GNN4.append(pred_GNN4[u,v])
                                BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v]))
                                BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v],BourgainSketches_GNN2_Su[u],BourgainSketches_GNN2_Su[v]))
                        else:
                            for u,v in pairs:
                                BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v]))
                                BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v],BourgainSketches_GNN2_Su[u],BourgainSketches_GNN2_Su[v]))
                        pred[2,k] = BourgainDists_GNN1
                        pred[4,k] = BourgainDists_GNN2
                    data = selected_pred_GNN1,None
                    GNN_preds.append(data)
                    data = selected_pred_GNN2,None
                    GNN_preds.append(data)
                    data = selected_pred_GNN3,None
                    GNN_preds.append(data)
                    data = selected_pred_GNN4,None
                    GNN_preds.append(data)
                else:
                    selected_actual = []
                    selected_pred_GNN1 = []
                    selected_pred_GNN2 = []
                    all_nodes = np.array(pairs).flatten()
                    for k in range(k_max):
                        BourgainSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN1,nodes=all_nodes)
                        BourgainSketches_GNN2_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN2,nodes=all_nodes)
                        BourgainDists_GNN1 = []
                        BourgainDists_GNN2 = []
                        if k == 0:
                            for u,v in pairs:
                                selected_actual.append(actual[u,v])
                                selected_pred_GNN1.append(pred_GNN1[u,v])
                                selected_pred_GNN2.append(pred_GNN2[u,v])
                                BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
                                BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v]))
                        else:
                            for u,v in pairs:
                                BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
                                BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v]))
                        pred[2,k] = BourgainDists_GNN1
                        pred[4,k] = BourgainDists_GNN2
                    data = selected_pred_GNN1,None
                    GNN_preds.append(data)
                    data = selected_pred_GNN2,None
                    GNN_preds.append(data)
                    GNN_preds.append(None)
                    GNN_preds.append(None)
                pred = np.where(pred == float('inf'), len(graph_info[0].nodes()), pred)
                all_actual.append(selected_actual)
                all_GNN_preds.append(GNN_preds)
                all_pred.append(pred)

        else:
            for N in range(num_graphs):
                if N % 10 == 0:
                    print(str(N/num_graphs*100)+'% complete')
                graph_info = all_graph_info[N]
                actual = all_actual_dists[N]
                seedSets = all_seedSets[N]
                pairs = all_pairs[N]
                k_max = len(seedSets)
                GNN_preds = []
                pred_GNN1 = shortestDistances_GNN(which_cuda,node_to_seed_model1,criterion_type,graph_info)[0]
                pred = np.zeros((6,k_max,len(pairs)))
                if graph_info[1]:
                    pred_GNN3 = shortestDistances_GNN(which_cuda,seed_to_node_model1,criterion_type,graph_info)[0].T
                    selected_actual = []
                    selected_pred_GNN1 = []
                    selected_pred_GNN3 = []
                    sources = list(set(np.array(pairs)[:,0]))
                    targets = list(set(np.array(pairs)[:,1]))
                    for k in range(k_max):
                        BourgainSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN1,nodes=sources)
                        BourgainSketches_GNN1_Su = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN3,nodes=targets)
                        BourgainDists_GNN1 = []
                        if k == 0:
                            for u,v in pairs:
                                selected_actual.append(actual[u,v])
                                selected_pred_GNN1.append(pred_GNN1[u,v])
                                selected_pred_GNN3.append(pred_GNN3[u,v])
                                BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v]))
                        else:
                            for u,v in pairs:
                                BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v]))
                        pred[2,k] = BourgainDists_GNN1
                    data = selected_pred_GNN1,None
                    GNN_preds.append(data)
                    GNN_preds.append(None)
                    data = selected_pred_GNN3,None
                    GNN_preds.append(data)
                    GNN_preds.append(None)
                else:
                    selected_actual = []
                    selected_pred_GNN1 = []
                    all_nodes = np.array(pairs).flatten()
                    for k in range(k_max):
                        BourgainSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN1,nodes=all_nodes)
                        BourgainDists_GNN1 = []
                        if k == 0:
                            for u,v in pairs:
                                selected_actual.append(actual[u,v])
                                selected_pred_GNN1.append(pred_GNN1[u,v])
                                BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
                        else:
                            for u,v in pairs:
                                BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
                        pred[2,k] = BourgainDists_GNN1
                    data = selected_pred_GNN1,None
                    GNN_preds.append(data)
                    GNN_preds.append(None)
                    GNN_preds.append(None)
                    GNN_preds.append(None)
                pred = np.where(pred == float('inf'), len(graph_info[0].nodes()), pred)
                all_actual.append(selected_actual)
                all_GNN_preds.append(GNN_preds)
                all_pred.append(pred)

    else:

        if node_to_seed_model2 != None:
            for N in range(num_graphs):
                if N % 10 == 0:
                    print(str(N/num_graphs*100)+'% complete')
                graph_info = all_graph_info[N]
                actual = all_actual_dists[N]
                seedSets = all_seedSets[N]
                pairs = all_pairs[N]
                k_max = len(seedSets)
                GNN_preds = []
                pred_GNN1 = shortestDistances_GNN(which_cuda,node_to_seed_model1,criterion_type,graph_info)[0]
                pred_GNN2 = shortestDistances_GNN(which_cuda,node_to_seed_model2,criterion_type,graph_info)[0]
                pred = np.zeros((6,k_max,len(pairs)))
                if graph_info[1]:
                    pred_GNN3 = shortestDistances_GNN(which_cuda,seed_to_node_model1,criterion_type,graph_info)[0].T
                    pred_GNN4 = shortestDistances_GNN(which_cuda,seed_to_node_model2,criterion_type,graph_info)[0].T
                    selected_actual = []
                    selected_pred_GNN1 = []
                    selected_pred_GNN2 = []
                    selected_pred_GNN3 = []
                    selected_pred_GNN4 = []
                    sources = list(set(np.array(pairs)[:,0]))
                    targets = list(set(np.array(pairs)[:,1]))
                    for k in range(k_max):
                        BourgainSketches_networkx_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=actual,nodes=sources)
                        BourgainSketches_networkx_Su = offlineSketches(graph_info,seedSets[k],which_sketch='target',sketch_type='Bourgain',dists=actual,nodes=targets)
                        BourgainSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN1,nodes=sources)
                        BourgainSketches_GNN2_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN2,nodes=sources)
                        BourgainSketches_GNN1_Su = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN3,nodes=targets)
                        BourgainSketches_GNN2_Su = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN4,nodes=targets)
                        BourgainDists_networkx = []
                        BourgainDists_GNN1 = []
                        BourgainDists_GNN2 = []
                        if k == 0:
                            for u,v in pairs:
                                selected_actual.append(actual[u,v])
                                selected_pred_GNN1.append(pred_GNN1[u,v])
                                selected_pred_GNN2.append(pred_GNN2[u,v])
                                selected_pred_GNN3.append(pred_GNN3[u,v])
                                selected_pred_GNN4.append(pred_GNN4[u,v])
                                BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v],BourgainSketches_networkx_Su[u],BourgainSketches_networkx_Su[v]))
                                BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v]))
                                BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v],BourgainSketches_GNN2_Su[u],BourgainSketches_GNN2_Su[v]))
                        else:
                            for u,v in pairs:
                                BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v],BourgainSketches_networkx_Su[u],BourgainSketches_networkx_Su[v]))
                                BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v]))
                                BourgainDists_GNN2.append(online_Bourgain(BourgainSketches_GNN2_uS[u],BourgainSketches_GNN2_uS[v],BourgainSketches_GNN2_Su[u],BourgainSketches_GNN2_Su[v]))
                        pred[0,k] = BourgainDists_networkx
                        pred[2,k] = BourgainDists_GNN1
                        pred[4,k] = BourgainDists_GNN2
                    data = selected_pred_GNN1,None
                    GNN_preds.append(data)
                    data = selected_pred_GNN2,None
                    GNN_preds.append(data)
                    data = selected_pred_GNN3,None
                    GNN_preds.append(data)
                    data = selected_pred_GNN4,None
                    GNN_preds.append(data)
                else:
                    selected_actual = []
                    selected_pred_GNN1 = []
                    selected_pred_GNN2 = []
                    all_nodes = np.array(pairs).flatten()
                    for k in range(k_max):
                        BourgainSketches_networkx_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=actual,nodes=all_nodes)
                        BourgainSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN1,nodes=all_nodes)
                        BourgainSketches_GNN2_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN2,nodes=all_nodes)
                        BourgainDists_networkx = []
                        BourgainDists_GNN1 = []
                        BourgainDists_GNN2 = []
                        if k == 0:
                            for u,v in pairs:
                                selected_actual.append(actual[u,v])
                                selected_pred_GNN1.append(pred_GNN1[u,v])
                                selected_pred_GNN2.append(pred_GNN2[u,v])
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
                    data = selected_pred_GNN1,None
                    GNN_preds.append(data)
                    data = selected_pred_GNN2,None
                    GNN_preds.append(data)
                    GNN_preds.append(None)
                    GNN_preds.append(None)
                pred = np.where(pred == float('inf'), len(graph_info[0].nodes()), pred)
                all_actual.append(selected_actual)
                all_GNN_preds.append(GNN_preds)
                all_pred.append(pred)

        else:
            for N in range(num_graphs):
                if N % 10 == 0:
                    print(str(N/num_graphs*100)+'% complete')
                graph_info = all_graph_info[N]
                actual = all_actual_dists[N]
                seedSets = all_seedSets[N]
                pairs = all_pairs[N]
                k_max = len(seedSets)
                GNN_preds = []
                pred_GNN1 = shortestDistances_GNN(which_cuda,node_to_seed_model1,criterion_type,graph_info)[0]
                pred = np.zeros((6,k_max,len(pairs)))
                if graph_info[1]:
                    pred_GNN3 = shortestDistances_GNN(which_cuda,seed_to_node_model1,criterion_type,graph_info)[0].T
                    selected_actual = []
                    selected_pred_GNN1 = []
                    selected_pred_GNN3 = []
                    sources = list(set(np.array(pairs)[:,0]))
                    targets = list(set(np.array(pairs)[:,1]))
                    for k in range(k_max):
                        BourgainSketches_networkx_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=actual,nodes=sources)
                        BourgainSketches_networkx_Su = offlineSketches(graph_info,seedSets[k],which_sketch='target',sketch_type='Bourgain',dists=actual,nodes=targets)
                        BourgainSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN1,nodes=sources)
                        BourgainSketches_GNN1_Su = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN3,nodes=targets)
                        BourgainDists_networkx = []
                        BourgainDists_GNN1 = []
                        if k == 0:
                            for u,v in pairs:
                                selected_actual.append(actual[u,v])
                                selected_pred_GNN1.append(pred_GNN1[u,v])
                                selected_pred_GNN3.append(pred_GNN3[u,v])
                                BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v],BourgainSketches_networkx_Su[u],BourgainSketches_networkx_Su[v]))
                                BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v]))
                        else:
                            for u,v in pairs:
                                BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v],BourgainSketches_networkx_Su[u],BourgainSketches_networkx_Su[v]))
                                BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v],BourgainSketches_GNN1_Su[u],BourgainSketches_GNN1_Su[v]))
                        pred[0,k] = BourgainDists_networkx
                        pred[2,k] = BourgainDists_GNN1
                    data = selected_pred_GNN1,None
                    GNN_preds.append(data)
                    GNN_preds.append(None)
                    data = selected_pred_GNN3,None
                    GNN_preds.append(data)
                    GNN_preds.append(None)
                else:
                    selected_actual = []
                    selected_pred_GNN1 = []
                    all_nodes = np.array(pairs).flatten()
                    for k in range(k_max):
                        BourgainSketches_networkx_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=actual,nodes=all_nodes)
                        BourgainSketches_GNN1_uS = offlineSketches(graph_info,seedSets[k],which_sketch='source',sketch_type='Bourgain',dists=pred_GNN1,nodes=all_nodes)
                        BourgainDists_networkx = []
                        BourgainDists_GNN1 = []
                        if k == 0:
                            for u,v in pairs:
                                selected_actual.append(actual[u,v])
                                selected_pred_GNN1.append(pred_GNN1[u,v])
                                BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v]))
                                BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
                        else:
                            for u,v in pairs:
                                BourgainDists_networkx.append(online_Bourgain(BourgainSketches_networkx_uS[u],BourgainSketches_networkx_uS[v]))
                                BourgainDists_GNN1.append(online_Bourgain(BourgainSketches_GNN1_uS[u],BourgainSketches_GNN1_uS[v]))
                        pred[0,k] = BourgainDists_networkx
                        pred[2,k] = BourgainDists_GNN1
                    data = selected_pred_GNN1,None
                    GNN_preds.append(data)
                    GNN_preds.append(None)
                    GNN_preds.append(None)
                    GNN_preds.append(None)
                pred = np.where(pred == float('inf'), len(graph_info[0].nodes()), pred)
                all_actual.append(selected_actual)
                all_GNN_preds.append(GNN_preds)
                all_pred.append(pred)

    return all_actual,all_GNN_preds,all_pred