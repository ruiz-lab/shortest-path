import numpy as np
from itertools import chain
from sklearn.metrics import confusion_matrix
import torch
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle

from Graphs import dRegularGraph,connectedErdosRenyiGraph
from ShortestPathAlgorithms import shortestDistances_networkx

class MLP(torch.nn.Module):

    def __init__(self, in_channels, all_hidden_channels, out_channels, activation):
        assert activation in ['sigmoid', 'softmax', 'relu', None], 'Invalid activation function.'
        super(MLP, self).__init__()
        self.name = 'MLP'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = torch.nn.ModuleList()
        widths = [in_channels]+all_hidden_channels+[out_channels]
        for i in range(len(widths)-1):
            self.layers.append(torch.nn.Linear(widths[i], widths[i+1]))
        if activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = torch.nn.Softmax(dim=1)
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        else:
            self.activation = None

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1,len(self.layers)):
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.layers[i](x)
        if self.activation != None:
            x = self.activation(x)
        return x

class GCN(torch.nn.Module):
    
    def __init__(self, in_channels, all_hidden_channels, out_channels, activation, aggr='sum'):
        assert activation in ['sigmoid', 'softmax', 'relu', None], 'Invalid activation function.'
        super(GCN, self).__init__()
        self.name = 'GCN'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = torch.nn.ModuleList()
        widths = [in_channels]+all_hidden_channels+[out_channels]
        for i in range(len(widths)-1):
            self.layers.append(GCNConv(widths[i], widths[i+1],aggr=aggr))
        if activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = torch.nn.Softmax(dim=1)
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        else:
            self.activation = None

    def forward(self, x, edge_index, weights=None):
        if weights == None:
            x = self.layers[0](x,edge_index)
            for i in range(1,len(self.layers)):
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.layers[i](x,edge_index)
        else:
            x = self.layers[0](x,edge_index,weights)
            for i in range(1,len(self.layers)):
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.layers[i](x,edge_index,weights)
        if self.activation != None:
            x = self.activation(x)
        return x

class SAGE(torch.nn.Module):
    
    def __init__(self, in_channels, all_hidden_channels, out_channels, activation, aggr='sum'):
        assert activation in ['sigmoid', 'softmax', 'relu', None], 'Invalid activation function.'
        super(SAGE, self).__init__()
        self.name = 'SAGE'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = torch.nn.ModuleList()
        widths = [in_channels]+all_hidden_channels+[out_channels]
        for i in range(len(widths)-1):
            self.layers.append(SAGEConv(widths[i], widths[i+1],aggr=aggr))
        if activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = torch.nn.Softmax(dim=1)
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        else:
            self.activation = None

    def forward(self, x, edge_index, weights=None):
        x = self.layers[0](x,edge_index)
        for i in range(1,len(self.layers)):
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.layers[i](x,edge_index)
        if self.activation != None:
            x = self.activation(x)
        return x

class GIN(torch.nn.Module):
    
    def __init__(self, in_channels, all_hidden_channels, out_channels, activation, aggr='sum'):
        assert activation in ['sigmoid', 'softmax', 'relu', None], 'Invalid activation function.'
        super(GIN, self).__init__()
        self.name = 'GIN'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = torch.nn.ModuleList()
        widths = [in_channels]+all_hidden_channels+[out_channels]
        for i in range(len(widths)-1):
            self.layers.append(GINConv(torch.nn.Linear(widths[i], widths[i+1]),aggr=aggr))
        if activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = torch.nn.Softmax(dim=1)
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        else:
            self.activation = None

    def forward(self, x, edge_index, weights=None):
        x = self.layers[0](x,edge_index)
        for i in range(1,len(self.layers)):
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.layers[i](x,edge_index)
        if self.activation != None:
            x = self.activation(x)
        return x

class GAT(torch.nn.Module):
    
    def __init__(self, in_channels, all_hidden_channels, out_channels, activation, aggr='sum'):
        assert activation in ['sigmoid', 'softmax', 'relu', None], 'Invalid activation function.'
        super(GAT, self).__init__()
        self.name = 'GAT'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = torch.nn.ModuleList()
        widths = [in_channels]+all_hidden_channels+[out_channels]
        for i in range(len(widths)-1):
            self.layers.append(GATConv(widths[i], widths[i+1],aggr=aggr))
        if activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = torch.nn.Softmax(dim=1)
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        else:
            self.activation = None

    def forward(self, x, edge_index, weights=None):
        x = self.layers[0](x,edge_index)
        for i in range(1,len(self.layers)):
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.layers[i](x,edge_index)
        if self.activation != None:
            x = self.activation(x)
        return x

def build(in_channels,all_hidden_channels,out_channels,model,criterion_type,optimizer_type,scheduler_type=None,activation=None):

    if isinstance(model, str):
        model = model.upper()
        if isinstance(activation, str):
            activation = activation.lower()
        if model == 'MLP':
            model = MLP(in_channels,all_hidden_channels,out_channels,activation)
        elif model == 'GCN':
            model = GCN(in_channels,all_hidden_channels,out_channels,activation)
        elif model == 'SAGE':
            model = SAGE(in_channels,all_hidden_channels,out_channels,activation)
        elif model == 'GIN':
            model = GIN(in_channels,all_hidden_channels,out_channels,activation)
        elif model == 'GAT':
            model = GAT(in_channels,all_hidden_channels,out_channels,activation)
        else:
            raise AssertionError(
                'Model type not yet defined.'
            )
    
    criterion_type = criterion_type.lower()
    if criterion_type == 'ce':
        criterion = [torch.nn.CrossEntropyLoss()]
    elif criterion_type == 'bce':
        criterion = [torch.nn.BCELoss()]
    elif criterion_type == 'bcelogits':
        criterion = [torch.nn.BCEWithLogitsLoss()]
    elif criterion_type in ['mse','l2']:
        criterion = [torch.nn.MSELoss()]
    elif criterion_type == 'l1':
        criterion = [torch.nn.L1Loss()]
    elif criterion_type == 'multimargin': # cuda crashed (same for focal loss)
        criterion = [torch.nn.MultiMarginLoss()]
    elif criterion_type == 'mse-mse':
        criterion1 = torch.nn.MSELoss()
        criterion2 = torch.nn.MSELoss()
        criterion = [criterion1,criterion2]
    else:
        raise AssertionError(
            'Criterion type not yet defined.'
        )
    
    optimizer_type = optimizer_type.lower()
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
    elif optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, weight_decay=0.0001, alpha=0.99, eps=1e-8, momentum=0.9)
    elif optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.0001, lr_decay=0.0001)
    else:
        raise AssertionError(
            'Optimizer type not yet defined.'
        )
    
    if scheduler_type == None:
        scheduler = None
    else:
        scheduler_type = scheduler_type.lower()
        if scheduler_type == 'step':
            scheduler = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)]
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=False)]
        elif scheduler_type == 'exponential':
            scheduler = [torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)]
        elif scheduler_type == 'cosine':
            scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)]
        elif scheduler_type == 'cyclic':
            if optimizer_type in ['sgd','rmsprop']:
                scheduler = [torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, cycle_momentum=True)]
            else:
                scheduler = [torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, cycle_momentum=False)]
        elif scheduler_type == 'cyclic-cosine':
            cycle_epochs = 10
            if optimizer_type in ['sgd','rmsprop']:
                scheduler_cyclic = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=cycle_epochs, cycle_momentum=True)
            else:
                scheduler_cyclic = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=cycle_epochs, cycle_momentum=False)
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cycle_epochs * 2)
            scheduler = [scheduler_cyclic,scheduler_cosine]
        else:
            raise AssertionError(
                'Scheduler type not yet defined.'
            )
    
    return model,criterion,optimizer,scheduler

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

def predict_oneGraph(gpu_bool,which_cuda,model,criterion_type,x_full,edge_index=[],weights=[]):
    r = model.out_channels
    num_nodes = x_full.shape[0]
    nodes = list(range(num_nodes))
    samples_x = []
    if num_nodes % r != 0:
        n_extra = int(np.ceil(num_nodes/r))*r - num_nodes
        nodes = nodes + nodes[:n_extra]
    list_of_seeds = [nodes[i:i + r] for i in range(0, len(nodes), r)]
    for s in list_of_seeds:
        samples_x.append(x_full[:,s])
    y_pred = predict(gpu_bool,which_cuda,model,criterion_type, samples_x, edge_index, weights)
    y_pred = np.squeeze(np.concatenate(y_pred, axis=1)[:,:num_nodes])
    return y_pred

def predict_allBatches(which_cuda,model,criterion_type,samples):

    x_train = [torch.tensor(arr, requires_grad=True) for arr in samples[0][0]]
    x_val = [torch.tensor(arr, requires_grad=True) for arr in samples[1][0]]
    x_test = [torch.tensor(arr, requires_grad=True) for arr in samples[2][0]]
    edge_index_train = [torch.tensor(arr) for arr in samples[0][2]]
    edge_index_val = [torch.tensor(arr) for arr in samples[1][2]]
    edge_index_test = [torch.tensor(arr) for arr in samples[2][2]]
    weights_train = [torch.tensor(arr) for arr in samples[0][3]]
    weights_val = [torch.tensor(arr) for arr in samples[1][3]]
    weights_test = [torch.tensor(arr) for arr in samples[2][3]]

    gpu_bool = torch.cuda.is_available()
    y_pred_train = []
    y_pred_val = []
    y_pred_test = []
    if len(samples[0][2]) > 0:
        if len(samples[0][3]) > 0:
            for x_full,edge_index,weights in list(zip(x_train,edge_index_train,weights_train)):
                y_pred_train.append(predict_oneGraph(gpu_bool,which_cuda,model,criterion_type,x_full,[edge_index],[weights]))
            for x_full,edge_index,weights in list(zip(x_val,edge_index_val,weights_val)):
                y_pred_val.append(predict_oneGraph(gpu_bool,which_cuda,model,criterion_type,x_full,[edge_index],[weights]))
            for x_full,edge_index,weights in list(zip(x_test,edge_index_test,weights_test)):
                y_pred_test.append(predict_oneGraph(gpu_bool,which_cuda,model,criterion_type,x_full,[edge_index],[weights]))
        else:
            for x_full,edge_index in list(zip(x_train,edge_index_train)):
                y_pred_train.append(predict_oneGraph(gpu_bool,which_cuda,model,criterion_type,x_full,[edge_index]))
            for x_full,edge_index in list(zip(x_val,edge_index_val)):
                y_pred_val.append(predict_oneGraph(gpu_bool,which_cuda,model,criterion_type,x_full,[edge_index]))
            for x_full,edge_index in list(zip(x_test,edge_index_test)):
                y_pred_test.append(predict_oneGraph(gpu_bool,which_cuda,model,criterion_type,x_full,[edge_index]))
    else:
        for x_full in x_train:
            y_pred_train.append(predict_oneGraph(gpu_bool,which_cuda,model,criterion_type,x_full))
        for x_full in x_val:
            y_pred_val.append(predict_oneGraph(gpu_bool,which_cuda,model,criterion_type,x_full))
        for x_full in x_test:
            y_pred_test.append(predict_oneGraph(gpu_bool,which_cuda,model,criterion_type,x_full))
    
    return y_pred_train,y_pred_val,y_pred_test

def test(gpu_bool,which_cuda,model,criterion,criterion_type,samples_x,samples_y,samples_edge_index=[],samples_weights=[]):
    
    selected_cuda = 'cuda:'+str(which_cuda)

    if len(samples_edge_index) > 0:
        if len(samples_edge_index) == 1:
            flag = True
        else:
            flag = False
    
    r = model.in_channels
    t_loss = 0
    total_samples = 0
    y_pred = []
    model.eval()
    with torch.no_grad():
        if r == 1:
            if len(samples_edge_index) == 0:
                for x,y in list(zip(samples_x,samples_y)):
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                    pred_all = []
                    for j in range(x.shape[1]):
                        out = model(x[:,j].reshape(len(x[:,j]),1))  # Perform a single forward pass.
                        #####
                        if criterion_type in ['mse','l2','l1']:
                            out = out.squeeze()*(1-x[:,j].reshape(len(x[:,j]),1)).squeeze()
                        #####
                        t_loss += criterion[0](out.squeeze(), y[:,j])
                        total_samples += 1
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        else:
                            pred = out.squeeze()
                        pred_all.append(pred.cpu())
                    y_pred.append(np.array(pred_all).T)
            elif len(samples_weights) == 0:
                if flag:
                    edge_index = samples_edge_index[-1]
                    if gpu_bool:
                        edge_index = edge_index.to(selected_cuda)
                    for x,y in list(zip(samples_x,samples_y)):
                        if gpu_bool:
                            x = x.to(selected_cuda)
                            y = y.to(selected_cuda)
                        pred_all = []
                        for j in range(x.shape[1]):
                            out = model(x[:,j].reshape(len(x[:,j]),1),edge_index)  # Perform a single forward pass.
                            #####
                            if criterion_type in ['mse','l2','l1']:
                                out = out.squeeze()*(1-x[:,j].reshape(len(x[:,j]),1)).squeeze()
                            #####
                            t_loss += criterion[0](out.squeeze(), y[:,j])
                            total_samples += 1
                            if criterion_type in ['bce','ce','multimargin']:
                                pred = out.argmax(dim=1) #  Use the class with highest probability.
                            else:
                                pred = out.squeeze()
                            pred_all.append(pred.cpu())
                        y_pred.append(np.array(pred_all).T)
                else:
                    for x,y,edge_index in list(zip(samples_x,samples_y,samples_edge_index)):
                        if gpu_bool:
                            x = x.to(selected_cuda)
                            y = y.to(selected_cuda)
                            edge_index = edge_index.to(selected_cuda)
                        pred_all = []
                        for j in range(x.shape[1]):
                            out = model(x[:,j].reshape(len(x[:,j]),1),edge_index)  # Perform a single forward pass.
                            #####
                            if criterion_type in ['mse','l2','l1']:
                                out = out.squeeze()*(1-x[:,j].reshape(len(x[:,j]),1)).squeeze()
                            #####
                            t_loss += criterion[0](out.squeeze(), y[:,j])
                            total_samples += 1
                            if criterion_type in ['bce','ce','multimargin']:
                                pred = out.argmax(dim=1) #  Use the class with highest probability.
                            else:
                                pred = out.squeeze()
                            pred_all.append(pred.cpu())
                        y_pred.append(np.array(pred_all).T)
            else:
                if flag:
                    edge_index = samples_edge_index[-1]
                    weights = samples_weights[-1]
                    if gpu_bool:
                        edge_index = edge_index.to(selected_cuda)
                        weights = weights.to(selected_cuda)
                    for x,y in list(zip(samples_x,samples_y)):
                        if gpu_bool:
                            x = x.to(selected_cuda)
                            y = y.to(selected_cuda)
                        pred_all = []
                        for j in range(x.shape[1]):
                            out = model(x[:,j].reshape(len(x[:,j]),1),edge_index,weights)  # Perform a single forward pass.
                            #####
                            if criterion_type in ['mse','l2','l1']:
                                out = out.squeeze()*(1-x[:,j].reshape(len(x[:,j]),1)).squeeze()
                            #####
                            t_loss += criterion[0](out.squeeze(), y[:,j])
                            total_samples += 1
                            if criterion_type in ['bce','ce','multimargin']:
                                pred = out.argmax(dim=1) #  Use the class with highest probability.
                            else:
                                pred = out.squeeze()
                            pred_all.append(pred.cpu())
                        y_pred.append(np.array(pred_all).T)
                else:
                    for x,y,edge_index,weights in list(zip(samples_x,samples_y,samples_edge_index,samples_weights)):
                        if gpu_bool:
                            x = x.to(selected_cuda)
                            y = y.to(selected_cuda)
                            edge_index = edge_index.to(selected_cuda)
                            weights = weights.to(selected_cuda)
                        pred_all = []
                        for j in range(x.shape[1]):
                            out = model(x[:,j].reshape(len(x[:,j]),1),edge_index,weights)  # Perform a single forward pass.
                            #####
                            if criterion_type in ['mse','l2','l1']:
                                out = out.squeeze()*(1-x[:,j].reshape(len(x[:,j]),1)).squeeze()
                            #####
                            t_loss += criterion[0](out.squeeze(), y[:,j])
                            total_samples += 1
                            if criterion_type in ['bce','ce','multimargin']:
                                pred = out.argmax(dim=1) #  Use the class with highest probability.
                            else:
                                pred = out.squeeze()
                            pred_all.append(pred.cpu())
                        y_pred.append(np.array(pred_all).T)
        else:
            if len(samples_edge_index) == 0:
                for x,y in list(zip(samples_x,samples_y)):
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    y = y[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                    out = model(x)  # Perform a single forward pass.
                    #####
                    if criterion_type in ['mse','l2','l1']:
                        out = out.squeeze()*(1-x)
                    #####
                    if criterion_type in ['bce']:
                        t_loss += criterion[0](out, torch.stack((1-y, y)).T)
                    elif criterion_type == 'mse-mse':
                        t_loss += 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                    elif criterion_type in ['ce','multimargin']:
                        t_loss += criterion[0](out, y) # classification
                    else:
                        t_loss += criterion[0](out.squeeze(), y)
                    total_samples += 1
                    if criterion_type in ['bce','ce','multimargin']:
                        pred = out.argmax(dim=1) #  Use the class with highest probability.
                    else:
                        pred = out.squeeze()
                    y_pred.append(pred.cpu())
            elif len(samples_weights) == 0:
                if flag:
                    edge_index = samples_edge_index[-1]
                    if gpu_bool:
                        edge_index = edge_index.to(selected_cuda)
                    for x,y in list(zip(samples_x,samples_y)):
                        seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                        x = x[:,seeds]
                        y = y[:,seeds]
                        if gpu_bool:
                            x = x.to(selected_cuda)
                            y = y.to(selected_cuda)
                        out = model(x,edge_index)  # Perform a single forward pass.
                        #####
                        if criterion_type in ['mse','l2','l1']:
                            out = out.squeeze()*(1-x)
                        #####
                        if criterion_type in ['bce']:
                            t_loss += criterion[0](out, torch.stack((1-y, y)).T)
                        elif criterion_type == 'mse-mse':
                            t_loss += 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                        elif criterion_type in ['ce','multimargin']:
                            t_loss += criterion[0](out, y) # classification
                        else:
                            t_loss += criterion[0](out.squeeze(), y)
                        total_samples += 1
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        else:
                            pred = out.squeeze()
                        y_pred.append(pred.cpu())
                else:
                    for x,y,edge_index in list(zip(samples_x,samples_y,samples_edge_index)):
                        seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                        x = x[:,seeds]
                        y = y[:,seeds]
                        if gpu_bool:
                            x = x.to(selected_cuda)
                            y = y.to(selected_cuda)
                            edge_index = edge_index.to(selected_cuda)
                        out = model(x,edge_index)  # Perform a single forward pass.
                        #####
                        if criterion_type in ['mse','l2','l1']:
                            out = out.squeeze()*(1-x)
                        #####
                        if criterion_type in ['bce']:
                            t_loss += criterion[0](out, torch.stack((1-y, y)).T)
                        elif criterion_type == 'mse-mse':
                            t_loss += 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                        elif criterion_type in ['ce','multimargin']:
                            t_loss += criterion[0](out, y) # classification
                        else:
                            t_loss += criterion[0](out.squeeze(), y)
                        total_samples += 1
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        else:
                            pred = out.squeeze()
                        y_pred.append(pred.cpu())
            else:
                if flag:
                    edge_index = samples_edge_index[-1]
                    weights = samples_weights[-1]
                    if gpu_bool:
                        edge_index = edge_index.to(selected_cuda)
                        weights = weights.to(selected_cuda)
                    for x,y in list(zip(samples_x,samples_y)):
                        seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                        x = x[:,seeds]
                        y = y[:,seeds]
                        if gpu_bool:
                            x = x.to(selected_cuda)
                            y = y.to(selected_cuda)
                        out = model(x,edge_index,weights)  # Perform a single forward pass.
                        #####
                        if criterion_type in ['mse','l2','l1']:
                            out = out.squeeze()*(1-x)
                        #####
                        if criterion_type in ['bce']:
                            t_loss += criterion[0](out, torch.stack((1-y, y)).T)
                        elif criterion_type == 'mse-mse':
                            t_loss += 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                        elif criterion_type in ['ce','multimargin']:
                            t_loss += criterion[0](out, y) # classification
                        else:
                            t_loss += criterion[0](out.squeeze(), y)
                        total_samples += 1
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        else:
                            pred = out.squeeze()
                        y_pred.append(pred.cpu())
                else:
                    for x,y,edge_index,weights in list(zip(samples_x,samples_y,samples_edge_index,samples_weights)):
                        seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                        x = x[:,seeds]
                        y = y[:,seeds]
                        if gpu_bool:
                            x = x.to(selected_cuda)
                            y = y.to(selected_cuda)
                            edge_index = edge_index.to(selected_cuda)
                            weights = weights.to(selected_cuda)
                        out = model(x,edge_index,weights)  # Perform a single forward pass.
                        #####
                        if criterion_type in ['mse','l2','l1']:
                            out = out.squeeze()*(1-x)
                        #####
                        if criterion_type in ['bce']:
                            t_loss += criterion[0](out, torch.stack((1-y, y)).T)
                        elif criterion_type == 'mse-mse':
                            t_loss += 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                        elif criterion_type in ['ce','multimargin']:
                            t_loss += criterion[0](out, y) # classification
                        else:
                            t_loss += criterion[0](out.squeeze(), y)
                        total_samples += 1
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        else:
                            pred = out.squeeze()
                        y_pred.append(pred.cpu())

    t_loss = t_loss.cpu()/total_samples
    if criterion_type in ['mse','l2','l1','mse-mse']:
        return t_loss, None, None, None
    else:
        y_true = np.array([y.cpu() for y in samples_y])
        y_pred = np.array(y_pred)
        t_accuracy = sum(y_true == y_pred)/len(y_true)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        t_sensitivity = tp / (tp + fn)
        t_specificity = tn / (tn + fp)
        return t_loss, t_accuracy, t_sensitivity, t_specificity

def train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,edge_index_train=[],edge_index_val=[],weights_train=[],weights_val=[],mask_list=None,balanced_masks=False):
    
    selected_cuda = 'cuda:'+str(which_cuda)

    if len(edge_index_train) > 0:
        if len(edge_index_train) == 1:
            flag = True
        else:
            flag = False

    if mask_list == None:
        mask_list = [None]*len(x_train)
    
    r = model.in_channels
    model.train()
    if r == 1:
        if len(edge_index_train) == 0:
            for x,y,mask in list(zip(x_train,y_train,mask_list)):
                optimizer.zero_grad()  # Clear gradients.
                if gpu_bool:
                    x = x.to(selected_cuda)
                    y = y.to(selected_cuda)
                for j in range(x.shape[1]):
                    out = model(x[:,j].reshape(len(x[:,j]),1))  # Perform a single forward pass.
                    #####
                    if criterion_type in ['mse','l2','l1']:
                        out = out.squeeze()*(1-x[:,j].reshape(len(x[:,j]),1)).squeeze()
                    #####
                    if isinstance(mask, np.ndarray):
                        t_loss = criterion[0]((out.squeeze())[mask[:,j]], y[mask[:,j],j])
                    elif balanced_masks:
                        if criterion_type in ['l1','l2','mse']:
                            out_flattened = (out.squeeze())[mask[:,j]]
                            y_flattened = y[mask[:,j],j]
                            storage = {}
                            min_len = r
                            for dist in torch.unique(y_flattened):
                                indices = torch.squeeze(torch.nonzero(y_flattened == dist),dim=1)
                                indices = indices[torch.randperm(indices.size(0))]
                                storage[dist] = indices
                                min_len = min(min_len,indices.size(0))
                            mask = [False]*len(y_flattened)
                            for dist, indices in storage.items():
                                mask = [True if idx in indices[:min_len] else val for idx,val in enumerate(mask)]
                            t_loss = criterion[0](out_flattened[mask], y_flattened[mask])
                        else:
                            raise AssertionError('Criterion type is not compatible with self-generated balanced masks.')
                    else:
                        t_loss = criterion[0](out.squeeze(), y[:,j])
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
        elif len(weights_train) == 0:
            if flag:
                edge_index = edge_index_train[-1]
                if gpu_bool:
                    edge_index = edge_index.to(selected_cuda)
                for x,y,mask in list(zip(x_train,y_train,mask_list)):
                    optimizer.zero_grad()  # Clear gradients.
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                    for j in range(x.shape[1]):
                        out = model(x[:,j].reshape(len(x[:,j]),1),edge_index)  # Perform a single forward pass
                        #####
                        if criterion_type in ['mse','l2','l1']:
                            out = out.squeeze()*(1-x[:,j].reshape(len(x[:,j]),1)).squeeze()
                        #####
                        if isinstance(mask, np.ndarray):
                            t_loss = criterion[0]((out.squeeze())[mask[:,j]], y[mask[:,j],j])
                        elif balanced_masks:
                            if criterion_type in ['l1','l2','mse']:
                                out_flattened = (out.squeeze())[mask[:,j]]
                                y_flattened = y[mask[:,j],j]
                                storage = {}
                                min_len = r
                                for dist in torch.unique(y_flattened):
                                    indices = torch.squeeze(torch.nonzero(y_flattened == dist),dim=1)
                                    indices = indices[torch.randperm(indices.size(0))]
                                    storage[dist] = indices
                                    min_len = min(min_len,indices.size(0))
                                mask = [False]*len(y_flattened)
                                for dist, indices in storage.items():
                                    mask = [True if idx in indices[:min_len] else val for idx,val in enumerate(mask)]
                                t_loss = criterion[0](out_flattened[mask], y_flattened[mask])
                            else:
                                raise AssertionError('Criterion type is not compatible with self-generated balanced masks.')
                        else:
                            t_loss = criterion[0](out.squeeze(), y[:,j])
                        t_loss.backward()  # Derive gradients
                        optimizer.step()  # Update parameters based on gradients.
            else:
                for x,y,edge_index,mask in list(zip(x_train,y_train,edge_index_train,mask_list)):
                    optimizer.zero_grad()  # Clear gradients.
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                        edge_index = edge_index.to(selected_cuda)
                    for j in range(x.shape[1]):
                        out = model(x[:,j].reshape(len(x[:,j]),1),edge_index)  # Perform a single forward pass
                        #####
                        if criterion_type in ['mse','l2','l1']:
                            out = out.squeeze()*(1-x[:,j].reshape(len(x[:,j]),1)).squeeze()
                        #####
                        if isinstance(mask, np.ndarray):
                            t_loss = criterion[0]((out.squeeze())[mask[:,j]], y[mask[:,j],j])
                        elif balanced_masks:
                            if criterion_type in ['l1','l2','mse']:
                                out_flattened = (out.squeeze())[mask[:,j]]
                                y_flattened = y[mask[:,j],j]
                                storage = {}
                                min_len = r
                                for dist in torch.unique(y_flattened):
                                    indices = torch.squeeze(torch.nonzero(y_flattened == dist),dim=1)
                                    indices = indices[torch.randperm(indices.size(0))]
                                    storage[dist] = indices
                                    min_len = min(min_len,indices.size(0))
                                mask = [False]*len(y_flattened)
                                for dist, indices in storage.items():
                                    mask = [True if idx in indices[:min_len] else val for idx,val in enumerate(mask)]
                                t_loss = criterion[0](out_flattened[mask], y_flattened[mask])
                            else:
                                raise AssertionError('Criterion type is not compatible with self-generated balanced masks.')
                        else:
                            t_loss = criterion[0](out.squeeze(), y[:,j])
                        t_loss.backward()  # Derive gradients
                        optimizer.step()  # Update parameters based on gradients.
        else:
            if flag:
                edge_index = edge_index_train[-1]
                weights = weights_train[-1]
                if gpu_bool:
                    edge_index = edge_index.to(selected_cuda)
                    weights = weights.to(selected_cuda)
                for x,y,mask in list(zip(x_train,y_train,mask_list)):
                    optimizer.zero_grad()  # Clear gradients.
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                    for j in range(x.shape[1]):
                        out = model(x[:,j].reshape(len(x[:,j]),1),edge_index,weights)  # Perform a single forward pass.
                        #####
                        if criterion_type in ['mse','l2','l1']:
                            out = out.squeeze()*(1-x[:,j].reshape(len(x[:,j]),1)).squeeze()
                        #####
                        if isinstance(mask, np.ndarray):
                            t_loss = criterion[0]((out.squeeze())[mask[:,j]], y[mask[:,j],j])
                        elif balanced_masks:
                            if criterion_type in ['l1','l2','mse']:
                                out_flattened = (out.squeeze())[mask[:,j]]
                                y_flattened = y[mask[:,j],j]
                                storage = {}
                                min_len = r
                                for dist in torch.unique(y_flattened):
                                    indices = torch.squeeze(torch.nonzero(y_flattened == dist),dim=1)
                                    indices = indices[torch.randperm(indices.size(0))]
                                    storage[dist] = indices
                                    min_len = min(min_len,indices.size(0))
                                mask = [False]*len(y_flattened)
                                for dist, indices in storage.items():
                                    mask = [True if idx in indices[:min_len] else val for idx,val in enumerate(mask)]
                                t_loss = criterion[0](out_flattened[mask], y_flattened[mask])
                            else:
                                raise AssertionError('Criterion type is not compatible with self-generated balanced masks.')
                        else:
                            t_loss = criterion[0](out.squeeze(), y[:,j])
                        t_loss.backward()  # Derive gradients
                        optimizer.step()  # Update parameters based on gradients.
            else:
                for x,y,edge_index,weights,mask in list(zip(x_train,y_train,edge_index_train,weights_train,mask_list)):
                    optimizer.zero_grad()  # Clear gradients.
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                        edge_index = edge_index.to(selected_cuda)
                        weights = weights.to(selected_cuda)
                    for j in range(x.shape[1]):
                        out = model(x[:,j].reshape(len(x[:,j]),1),edge_index,weights)  # Perform a single forward pass.
                        #####
                        if criterion_type in ['mse','l2','l1']:
                            out = out.squeeze()*(1-x[:,j].reshape(len(x[:,j]),1)).squeeze()
                        #####
                        if isinstance(mask, np.ndarray):
                            t_loss = criterion[0]((out.squeeze())[mask[:,j]], y[mask[:,j],j])
                        elif balanced_masks:
                            if criterion_type in ['l1','l2','mse']:
                                out_flattened = (out.squeeze())[mask[:,j]]
                                y_flattened = y[mask[:,j],j]
                                storage = {}
                                min_len = r
                                for dist in torch.unique(y_flattened):
                                    indices = torch.squeeze(torch.nonzero(y_flattened == dist),dim=1)
                                    indices = indices[torch.randperm(indices.size(0))]
                                    storage[dist] = indices
                                    min_len = min(min_len,indices.size(0))
                                mask = [False]*len(y_flattened)
                                for dist, indices in storage.items():
                                    mask = [True if idx in indices[:min_len] else val for idx,val in enumerate(mask)]
                                t_loss = criterion[0](out_flattened[mask], y_flattened[mask])
                            else:
                                raise AssertionError('Criterion type is not compatible with self-generated balanced masks.')
                        else:
                            t_loss = criterion[0](out.squeeze(), y[:,j])
                        t_loss.backward()  # Derive gradients
                        optimizer.step()  # Update parameters based on gradients.
    else:
        if len(edge_index_train) == 0:
            for x,y,mask in list(zip(x_train,y_train,mask_list)):
                optimizer.zero_grad()  # Clear gradients.
                seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                x = x[:,seeds]
                y = y[:,seeds]
                if gpu_bool:
                    x = x.to(selected_cuda)
                    y = y.to(selected_cuda)
                out = model(x)  # Perform a single forward pass.
                #####
                if criterion_type in ['mse','l2','l1']:
                    out = out.squeeze()*(1-x)
                #####
                if isinstance(mask, np.ndarray):
                    mask = mask[:,seeds]
                    if criterion_type == 'bce':
                        t_loss = criterion[0](out[mask], (torch.stack((1-y, y)).T)[mask])
                    elif criterion_type == 'mse-mse':
                        t_loss = 100*criterion[0](out[mask, ::2], y[mask, ::2]) + criterion[1](out[mask, 1::2], y[mask, 1::2])
                    elif criterion_type in ['ce','multimargin']:
                        t_loss = criterion[0](out[mask], y[mask]) # classification
                    elif criterion_type in ['l1','l2','mse']:
                        out_flattened = out.squeeze().flatten()
                        y_flattened = y.flatten()
                        mask = mask.flatten()
                        t_loss = criterion[0](out_flattened[mask], y_flattened[mask])
                    else:
                        t_loss = criterion[0]((out.squeeze())[mask], y[mask])
                elif balanced_masks:
                    if criterion_type in ['l1','l2','mse']:
                        out_flattened = out.squeeze().flatten()
                        y_flattened = y.flatten()
                        storage = {}
                        min_len = r
                        for dist in torch.unique(y_flattened):
                            indices = torch.squeeze(torch.nonzero(y_flattened == dist),dim=1)
                            indices = indices[torch.randperm(indices.size(0))]
                            storage[dist] = indices
                            min_len = min(min_len,indices.size(0))
                        mask = [False]*len(y_flattened)
                        for dist, indices in storage.items():
                            mask = [True if idx in indices[:min_len] else val for idx,val in enumerate(mask)]
                        t_loss = criterion[0](out_flattened[mask], y_flattened[mask])
                    else:
                        raise AssertionError('Criterion type is not compatible with self-generated balanced masks.')
                else:
                    if criterion_type == 'bce':
                        t_loss = criterion[0](out, torch.stack((1-y, y)).T)
                    elif criterion_type == 'mse-mse':
                        t_loss = 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                    elif criterion_type in ['ce','multimargin']:
                        t_loss = criterion[0](out, y) # classification
                    else:
                        t_loss = criterion[0](out.squeeze(), y)
                t_loss.backward()  # Derive gradients
                optimizer.step()  # Update parameters based on gradients.
        elif len(weights_train) == 0:
            if flag:
                edge_index = edge_index_train[-1]
                if gpu_bool:
                    edge_index = edge_index.to(selected_cuda)
                for x,y,mask in list(zip(x_train,y_train,mask_list)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    y = y[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                    out = model(x,edge_index)  # Perform a single forward pass
                    #####
                    if criterion_type in ['mse','l2','l1']:
                        out = out.squeeze()*(1-x)
                    #####
                    if isinstance(mask, np.ndarray):
                        mask = mask[:,seeds]
                        if criterion_type == 'bce':
                            t_loss = criterion[0](out[mask], (torch.stack((1-y, y)).T)[mask])
                        elif criterion_type == 'mse-mse':
                            t_loss = 100*criterion[0](out[mask, ::2], y[mask, ::2]) + criterion[1](out[mask, 1::2], y[mask, 1::2])
                        elif criterion_type in ['ce','multimargin']:
                            t_loss = criterion[0](out[mask], y[mask]) # classification
                        elif criterion_type in ['l1','l2','mse']:
                            out_flattened = out.squeeze().flatten()
                            y_flattened = y.flatten()
                            mask = mask.flatten()
                            t_loss = criterion[0](out_flattened[mask], y_flattened[mask])
                        else:
                            t_loss = criterion[0]((out.squeeze())[mask], y[mask])
                    elif balanced_masks:
                        if criterion_type in ['l1','l2','mse']:
                            out_flattened = out.squeeze().flatten()
                            y_flattened = y.flatten()
                            storage = {}
                            min_len = r
                            for dist in torch.unique(y_flattened):
                                indices = torch.squeeze(torch.nonzero(y_flattened == dist),dim=1)
                                indices = indices[torch.randperm(indices.size(0))]
                                storage[dist] = indices
                                min_len = min(min_len,indices.size(0))
                            mask = [False]*len(y_flattened)
                            for dist, indices in storage.items():
                                mask = [True if idx in indices[:min_len] else val for idx,val in enumerate(mask)]
                            t_loss = criterion[0](out_flattened[mask], y_flattened[mask])
                        else:
                            raise AssertionError('Criterion type is not compatible with self-generated balanced masks.')
                    else:
                        if criterion_type == 'bce':
                            t_loss = criterion[0](out, torch.stack((1-y, y)).T)
                        elif criterion_type == 'mse-mse':
                            t_loss = 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                        elif criterion_type in ['ce','multimargin']:
                            t_loss = criterion[0](out, y) # classification
                        else:
                            t_loss = criterion[0](out.squeeze(), y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
            else:
                for x,y,edge_index,mask in list(zip(x_train,y_train,edge_index_train,mask_list)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    y = y[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                        edge_index = edge_index.to(selected_cuda)
                    out = model(x,edge_index)  # Perform a single forward pass
                    #####
                    if criterion_type in ['mse','l2','l1']:
                        out = out.squeeze()*(1-x)
                    #####
                    if isinstance(mask, np.ndarray):
                        mask = mask[:,seeds]
                        if criterion_type == 'bce':
                            t_loss = criterion[0](out[mask], (torch.stack((1-y, y)).T)[mask])
                        elif criterion_type == 'mse-mse':
                            t_loss = 100*criterion[0](out[mask, ::2], y[mask, ::2]) + criterion[1](out[mask, 1::2], y[mask, 1::2])
                        elif criterion_type in ['ce','multimargin']:
                            t_loss = criterion[0](out[mask], y[mask]) # classification
                        elif criterion_type in ['l1','l2','mse']:
                            out_flattened = out.squeeze().flatten()
                            y_flattened = y.flatten()
                            mask = mask.flatten()
                            t_loss = criterion[0](out_flattened[mask], y_flattened[mask])
                        else:
                            t_loss = criterion[0]((out.squeeze())[mask], y[mask])
                    elif balanced_masks:
                        if criterion_type in ['l1','l2','mse']:
                            out_flattened = out.squeeze().flatten()
                            y_flattened = y.flatten()
                            storage = {}
                            min_len = r
                            for dist in torch.unique(y_flattened):
                                indices = torch.squeeze(torch.nonzero(y_flattened == dist),dim=1)
                                indices = indices[torch.randperm(indices.size(0))]
                                storage[dist] = indices
                                min_len = min(min_len,indices.size(0))
                            mask = [False]*len(y_flattened)
                            for dist, indices in storage.items():
                                mask = [True if idx in indices[:min_len] else val for idx,val in enumerate(mask)]
                            t_loss = criterion[0](out_flattened[mask], y_flattened[mask])
                        else:
                            raise AssertionError('Criterion type is not compatible with self-generated balanced masks.')
                    else:
                        if criterion_type == 'bce':
                            t_loss = criterion[0](out, torch.stack((1-y, y)).T)
                        elif criterion_type == 'mse-mse':
                            t_loss = 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                        elif criterion_type in ['ce','multimargin']:
                            t_loss = criterion[0](out, y) # classification
                        else:
                            t_loss = criterion[0](out.squeeze(), y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
        else:
            if flag:
                edge_index = edge_index_train[-1]
                weights = weights_train[-1]
                if gpu_bool:
                    edge_index = edge_index.to(selected_cuda)
                    weights = weights.to(selected_cuda)
                for x,y,mask in list(zip(x_train,y_train,mask_list)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    y = y[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                    out = model(x,edge_index,weights)  # Perform a single forward pass.
                    #####
                    if criterion_type in ['mse','l2','l1']:
                        out = out.squeeze()*(1-x)
                    #####
                    if isinstance(mask, np.ndarray):
                        mask = mask[:,seeds]
                        if criterion_type == 'bce':
                            t_loss = criterion[0](out[mask], (torch.stack((1-y, y)).T)[mask])
                        elif criterion_type == 'mse-mse':
                            t_loss = 100*criterion[0](out[mask, ::2], y[mask, ::2]) + criterion[1](out[mask, 1::2], y[mask, 1::2])
                        elif criterion_type in ['ce','multimargin']:
                            t_loss = criterion[0](out[mask], y[mask]) # classification
                        elif criterion_type in ['l1','l2','mse']:
                            out_flattened = out.squeeze().flatten()
                            y_flattened = y.flatten()
                            mask = mask.flatten()
                            t_loss = criterion[0](out_flattened[mask], y_flattened[mask])
                        else:
                            t_loss = criterion[0]((out.squeeze())[mask], y[mask])
                    elif balanced_masks:
                        if criterion_type in ['l1','l2','mse']:
                            out_flattened = out.squeeze().flatten()
                            y_flattened = y.flatten()
                            storage = {}
                            min_len = r
                            for dist in torch.unique(y_flattened):
                                indices = torch.squeeze(torch.nonzero(y_flattened == dist),dim=1)
                                indices = indices[torch.randperm(indices.size(0))]
                                storage[dist] = indices
                                min_len = min(min_len,indices.size(0))
                            mask = [False]*len(y_flattened)
                            for dist, indices in storage.items():
                                mask = [True if idx in indices[:min_len] else val for idx,val in enumerate(mask)]
                            t_loss = criterion[0](out_flattened[mask], y_flattened[mask])
                        else:
                            raise AssertionError('Criterion type is not compatible with self-generated balanced masks.')
                    else:
                        if criterion_type == 'bce':
                            t_loss = criterion[0](out, torch.stack((1-y, y)).T)
                        elif criterion_type == 'mse-mse':
                            t_loss = 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                        elif criterion_type in ['ce','multimargin']:
                            t_loss = criterion[0](out, y) # classification
                        else:
                            t_loss = criterion[0](out.squeeze(), y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
            else:
                for x,y,edge_index,weights,mask in list(zip(x_train,y_train,edge_index_train,weights_train,mask_list)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    y = y[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                        edge_index = edge_index.to(selected_cuda)
                        weights = weights.to(selected_cuda)
                    out = model(x,edge_index,weights)  # Perform a single forward pass.
                    #####
                    if criterion_type in ['mse','l2','l1']:
                        out = out.squeeze()*(1-x)
                    #####
                    if isinstance(mask, np.ndarray):
                        mask = mask[:,seeds]
                        if criterion_type == 'bce':
                            t_loss = criterion[0](out[mask], (torch.stack((1-y, y)).T)[mask])
                        elif criterion_type == 'mse-mse':
                            t_loss = 100*criterion[0](out[mask, ::2], y[mask, ::2]) + criterion[1](out[mask, 1::2], y[mask, 1::2])
                        elif criterion_type in ['ce','multimargin']:
                            t_loss = criterion[0](out[mask], y[mask]) # classification
                        elif criterion_type in ['l1','l2','mse']:
                            out_flattened = out.squeeze().flatten()
                            y_flattened = y.flatten()
                            mask = mask.flatten()
                            t_loss = criterion[0](out_flattened[mask], y_flattened[mask])
                        else:
                            t_loss = criterion[0]((out.squeeze())[mask], y[mask])
                    elif balanced_masks:
                        if criterion_type in ['l1','l2','mse']:
                            out_flattened = out.squeeze().flatten()
                            y_flattened = y.flatten()
                            storage = {}
                            min_len = r
                            for dist in torch.unique(y_flattened):
                                indices = torch.squeeze(torch.nonzero(y_flattened == dist),dim=1)
                                indices = indices[torch.randperm(indices.size(0))]
                                storage[dist] = indices
                                min_len = min(min_len,indices.size(0))
                            mask = [False]*len(y_flattened)
                            for dist, indices in storage.items():
                                mask = [True if idx in indices[:min_len] else val for idx,val in enumerate(mask)]
                            t_loss = criterion[0](out_flattened[mask], y_flattened[mask])
                        else:
                            raise AssertionError('Criterion type is not compatible with self-generated balanced masks.')
                    else:
                        if criterion_type == 'bce':
                            t_loss = criterion[0](out, torch.stack((1-y, y)).T)
                        elif criterion_type == 'mse-mse':
                            t_loss = 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                        elif criterion_type in ['ce','multimargin']:
                            t_loss = criterion[0](out, y) # classification
                        else:
                            t_loss = criterion[0](out.squeeze(), y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.

    train_loss, train_accuracy, train_sensitivity, train_specificity = test(gpu_bool,which_cuda,model,criterion,criterion_type,x_train,y_train,edge_index_train,weights_train)
    v_loss, v_accuracy, v_sensitivity, v_specificity = test(gpu_bool,which_cuda,model,criterion,criterion_type,x_val,y_val,edge_index_val,weights_val)

    if scheduler_type in ['step','exponential','cyclic','cosine']:
        scheduler[0].step()
    elif scheduler_type == 'reduce_on_plateau': 
        scheduler[0].step(v_loss)
    elif scheduler_type == 'cyclic-cosine':
        scheduler[0].step()
        scheduler[1].step()
    return train_loss,train_accuracy,train_sensitivity,train_specificity,v_loss,v_accuracy,v_sensitivity,v_specificity

def train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,edge_index_train=[],edge_index_val=[],weights_train=[],weights_val=[],eval='Bourgain'):
    
    assert eval in ['Bourgain','Sarma','Bourgain+Sarma','all'], 'Invalid data for evaluation.'
    assert criterion_type in ['mse','l2','l1'], 'Criterion type is not compatible with evaluated data.'

    selected_cuda = 'cuda:'+str(which_cuda)

    if len(edge_index_train) > 0:
        if len(edge_index_train) == 1:
            flag = True
        else:
            flag = False
    
    r = model.in_channels
    model.train()
    if eval == 'Bourgain':
        if len(edge_index_train) == 0:
            for x,y in list(zip(x_train,y_train)):
                optimizer.zero_grad()  # Clear gradients.
                seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                x = x[:,seeds]
                if gpu_bool:
                    x = x.to(selected_cuda)
                    y = y.to(selected_cuda)
                out = model(x)  # Perform a single forward pass.
                out = out.squeeze()*(1-x)
                all_out = torch.zeros(r, y.shape[0], y.shape[1])
                for i in range(r):
                    col = out[:,i].reshape(-1,1)
                    all_out[i,:,:] = torch.abs(col-col.T)
                max_values,_ = torch.max(all_out, dim=0, keepdim=False)
                if gpu_bool:
                    max_values = max_values.to(selected_cuda)
                t_loss = criterion[0](max_values, y)
                t_loss.backward()  # Derive gradients
                optimizer.step()  # Update parameters based on gradients.
        elif len(weights_train) == 0:
            if flag:
                edge_index = edge_index_train[-1]
                if gpu_bool:
                    edge_index = edge_index.to(selected_cuda)
                for x,y in list(zip(x_train,y_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                    out = model(x,edge_index)  # Perform a single forward pass
                    out = out.squeeze()*(1-x)
                    all_out = torch.zeros(r, y.shape[0], y.shape[1])
                    for i in range(r):
                        col = out[:,i].reshape(-1,1)
                        all_out[i,:,:] = torch.abs(col-col.T)
                    max_values,_ = torch.max(all_out, dim=0, keepdim=False)
                    if gpu_bool:
                        max_values = max_values.to(selected_cuda)
                    t_loss = criterion[0](max_values, y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
            else:
                for x,y,edge_index in list(zip(x_train,y_train,edge_index_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                        edge_index = edge_index.to(selected_cuda)
                    out = model(x,edge_index)  # Perform a single forward pass
                    out = out.squeeze()*(1-x)
                    all_out = torch.zeros(r, y.shape[0], y.shape[1])
                    for i in range(r):
                        col = out[:,i].reshape(-1,1)
                        all_out[i,:,:] = torch.abs(col-col.T)
                    max_values,_ = torch.max(all_out, dim=0, keepdim=False)
                    if gpu_bool:
                        max_values = max_values.to(selected_cuda)
                    t_loss = criterion[0](max_values, y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
        else:
            if flag:
                edge_index = edge_index_train[-1]
                weights = weights_train[-1]
                if gpu_bool:
                    edge_index = edge_index.to(selected_cuda)
                    weights = weights.to(selected_cuda)
                for x,y in list(zip(x_train,y_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                    out = model(x,edge_index,weights)  # Perform a single forward pass.
                    out = out.squeeze()*(1-x)
                    all_out = torch.zeros(r, y.shape[0], y.shape[1])
                    for i in range(r):
                        col = out[:,i].reshape(-1,1)
                        all_out[i,:,:] = torch.abs(col-col.T)
                    max_values,_ = torch.max(all_out, dim=0, keepdim=False)
                    if gpu_bool:
                        max_values = max_values.to(selected_cuda)
                    t_loss = criterion[0](max_values, y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
            else:
                for x,y,edge_index,weights in list(zip(x_train,y_train,edge_index_train,weights_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                        edge_index = edge_index.to(selected_cuda)
                        weights = weights.to(selected_cuda)
                    out = model(x,edge_index,weights)  # Perform a single forward pass.
                    out = out.squeeze()*(1-x)
                    all_out = torch.zeros(r, y.shape[0], y.shape[1])
                    for i in range(r):
                        col = out[:,i].reshape(-1,1)
                        all_out[i,:,:] = torch.abs(col-col.T)
                    max_values,_ = torch.max(all_out, dim=0, keepdim=False)
                    if gpu_bool:
                        max_values = max_values.to(selected_cuda)
                    t_loss = criterion[0](max_values, y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
    elif eval == 'Sarma':
        if len(edge_index_train) == 0:
            for x,y in list(zip(x_train,y_train)):
                optimizer.zero_grad()  # Clear gradients.
                seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                x = x[:,seeds]
                if gpu_bool:
                    x = x.to(selected_cuda)
                    y = y.to(selected_cuda)
                out = model(x)  # Perform a single forward pass.
                out = out.squeeze()*(1-x)
                all_out = torch.zeros(r,y.shape[0], y.shape[1])
                for i in range(r):
                    col = out[:,i].reshape(-1,1)
                    all_out[i,:,:] = col+col.T
                min_values,_ = torch.min(all_out, dim=0, keepdim=False)
                if gpu_bool:
                    min_values = min_values.to(selected_cuda)
                t_loss = criterion[0](min_values, y)
                t_loss.backward()  # Derive gradients
                optimizer.step()  # Update parameters based on gradients.
        elif len(weights_train) == 0:
            if flag:
                edge_index = edge_index_train[-1]
                if gpu_bool:
                    edge_index = edge_index.to(selected_cuda)
                for x,y in list(zip(x_train,y_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                    out = model(x,edge_index)  # Perform a single forward pass
                    out = out.squeeze()*(1-x)
                    all_out = torch.zeros(r,y.shape[0], y.shape[1])
                    for i in range(r):
                        col = out[:,i].reshape(-1,1)
                        all_out[i,:,:] = col+col.T
                    min_values,_ = torch.min(all_out, dim=0, keepdim=False)
                    if gpu_bool:
                        min_values = min_values.to(selected_cuda)
                    t_loss = criterion[0](min_values, y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
            else:
                for x,y,edge_index in list(zip(x_train,y_train,edge_index_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                        edge_index = edge_index.to(selected_cuda)
                    out = model(x,edge_index)  # Perform a single forward pass
                    out = out.squeeze()*(1-x)
                    all_out = torch.zeros(r,y.shape[0], y.shape[1])
                    for i in range(r):
                        col = out[:,i].reshape(-1,1)
                        # print((col+col.T).shape)
                        # print(y.shape)
                        # print(all_out[i,:,:].shape)
                        # print(all_out.shape)
                        all_out[i,:,:] = col+col.T
                    min_values,_ = torch.min(all_out, dim=0, keepdim=False)
                    if gpu_bool:
                        min_values = min_values.to(selected_cuda)
                    t_loss = criterion[0](min_values, y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
        else:
            if flag:
                edge_index = edge_index_train[-1]
                weights = weights_train[-1]
                if gpu_bool:
                    edge_index = edge_index.to(selected_cuda)
                    weights = weights.to(selected_cuda)
                for x,y in list(zip(x_train,y_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                    out = model(x,edge_index,weights)  # Perform a single forward pass.
                    out = out.squeeze()*(1-x)
                    all_out = torch.zeros(r,y.shape[0], y.shape[1])
                    for i in range(r):
                        col = out[:,i].reshape(-1,1)
                        all_out[i,:,:] = col+col.T
                    min_values,_ = torch.min(all_out, dim=0, keepdim=False)
                    if gpu_bool:
                        min_values = min_values.to(selected_cuda)
                    t_loss = criterion[0](min_values, y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
            else:
                for x,y,edge_index,weights in list(zip(x_train,y_train,edge_index_train,weights_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                        edge_index = edge_index.to(selected_cuda)
                        weights = weights.to(selected_cuda)
                    out = model(x,edge_index,weights)  # Perform a single forward pass.
                    out = out.squeeze()*(1-x)
                    all_out = torch.zeros(r,y.shape[0], y.shape[1])
                    for i in range(r):
                        col = out[:,i].reshape(-1,1)
                        all_out[i,:,:] = col+col.T
                    min_values,_ = torch.min(all_out, dim=0, keepdim=False)
                    if gpu_bool:
                        min_values = min_values.to(selected_cuda)
                    t_loss = criterion[0](min_values, y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
    elif eval == 'Bourgain+Sarma':
        if len(edge_index_train) == 0:
            for x,y in list(zip(x_train,y_train)):
                optimizer.zero_grad()  # Clear gradients.
                seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                x = x[:,seeds]
                if gpu_bool:
                    x = x.to(selected_cuda)
                    y = y.to(selected_cuda)
                out = model(x)  # Perform a single forward pass.
                out = out.squeeze()*(1-x)
                all_out_Bourgain = torch.zeros(r,y.shape[0], y.shape[1])
                all_out_Sarma = torch.zeros(r,y.shape[0], y.shape[1])
                for i in range(r):
                    col = out[:,i].reshape(-1,1)
                    all_out_Bourgain[i,:,:] = torch.abs(col-col.T)
                    all_out_Sarma[i,:,:] = col+col.T
                max_values,_ = torch.max(all_out_Bourgain, dim=0, keepdim=False)
                min_values,_ = torch.min(all_out_Sarma, dim=0, keepdim=False)
                if gpu_bool:
                    max_values = max_values.to(selected_cuda)
                    min_values = min_values.to(selected_cuda)
                t_loss = criterion[0](max_values, y) + criterion[0](min_values, y)
                t_loss.backward()  # Derive gradients
                optimizer.step()  # Update parameters based on gradients.
        elif len(weights_train) == 0:
            if flag:
                edge_index = edge_index_train[-1]
                if gpu_bool:
                    edge_index = edge_index.to(selected_cuda)
                for x,y in list(zip(x_train,y_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                    out = model(x,edge_index)  # Perform a single forward pass
                    out = out.squeeze()*(1-x)
                    all_out_Bourgain = torch.zeros(r,y.shape[0], y.shape[1])
                    all_out_Sarma = torch.zeros(r,y.shape[0], y.shape[1])
                    for i in range(r):
                        col = out[:,i].reshape(-1,1)
                        all_out_Bourgain[i,:,:] = torch.abs(col-col.T)
                        all_out_Sarma[i,:,:] = col+col.T
                    max_values,_ = torch.max(all_out_Bourgain, dim=0, keepdim=False)
                    min_values,_ = torch.min(all_out_Sarma, dim=0, keepdim=False)
                    if gpu_bool:
                        max_values = max_values.to(selected_cuda)
                        min_values = min_values.to(selected_cuda)
                    t_loss = criterion[0](max_values, y) + criterion[0](min_values, y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
            else:
                for x,y,edge_index in list(zip(x_train,y_train,edge_index_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                        edge_index = edge_index.to(selected_cuda)
                    out = model(x,edge_index)  # Perform a single forward pass
                    out = out.squeeze()*(1-x)
                    all_out_Bourgain = torch.zeros(r,y.shape[0], y.shape[1])
                    all_out_Sarma = torch.zeros(r,y.shape[0], y.shape[1])
                    for i in range(r):
                        col = out[:,i].reshape(-1,1)
                        all_out_Bourgain[i,:,:] = torch.abs(col-col.T)
                        all_out_Sarma[i,:,:] = col+col.T
                    max_values,_ = torch.max(all_out_Bourgain, dim=0, keepdim=False)
                    min_values,_ = torch.min(all_out_Sarma, dim=0, keepdim=False)
                    if gpu_bool:
                        max_values = max_values.to(selected_cuda)
                        min_values = min_values.to(selected_cuda)
                    t_loss = criterion[0](max_values, y) + criterion[0](min_values, y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
        else:
            if flag:
                edge_index = edge_index_train[-1]
                weights = weights_train[-1]
                if gpu_bool:
                    edge_index = edge_index.to(selected_cuda)
                    weights = weights.to(selected_cuda)
                for x,y in list(zip(x_train,y_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                    out = model(x,edge_index,weights)  # Perform a single forward pass.
                    out = out.squeeze()*(1-x)
                    all_out_Bourgain = torch.zeros(r,y.shape[0], y.shape[1])
                    all_out_Sarma = torch.zeros(r,y.shape[0], y.shape[1])
                    for i in range(r):
                        col = out[:,i].reshape(-1,1)
                        all_out_Bourgain[i,:,:] = torch.abs(col-col.T)
                        all_out_Sarma[i,:,:] = col+col.T
                    max_values,_ = torch.max(all_out_Bourgain, dim=0, keepdim=False)
                    min_values,_ = torch.min(all_out_Sarma, dim=0, keepdim=False)
                    if gpu_bool:
                        max_values = max_values.to(selected_cuda)
                        min_values = min_values.to(selected_cuda)
                    t_loss = criterion[0](max_values, y) + criterion[0](min_values, y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
            else:
                for x,y,edge_index,weights in list(zip(x_train,y_train,edge_index_train,weights_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                        edge_index = edge_index.to(selected_cuda)
                        weights = weights.to(selected_cuda)
                    out = model(x,edge_index,weights)  # Perform a single forward pass.
                    out = out.squeeze()*(1-x)
                    all_out_Bourgain = torch.zeros(r,y.shape[0], y.shape[1])
                    all_out_Sarma = torch.zeros(r,y.shape[0], y.shape[1])
                    for i in range(r):
                        col = out[:,i].reshape(-1,1)
                        all_out_Bourgain[i,:,:] = torch.abs(col-col.T)
                        all_out_Sarma[i,:,:] = col+col.T
                    max_values,_ = torch.max(all_out_Bourgain, dim=0, keepdim=False)
                    min_values,_ = torch.min(all_out_Sarma, dim=0, keepdim=False)
                    if gpu_bool:
                        max_values = max_values.to(selected_cuda)
                        min_values = min_values.to(selected_cuda)
                    t_loss = criterion[0](max_values, y) + criterion[0](min_values, y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
    else:
        if len(edge_index_train) == 0:
            for x,y in list(zip(x_train,y_train)):
                optimizer.zero_grad()  # Clear gradients.
                seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                x = x[:,seeds]
                if gpu_bool:
                    x = x.to(selected_cuda)
                    y = y.to(selected_cuda)
                out = model(x)  # Perform a single forward pass.
                out = out.squeeze()*(1-x)
                all_out_Bourgain = torch.zeros(r,y.shape[0], y.shape[1])
                all_out_Sarma = torch.zeros(r,y.shape[0], y.shape[1])
                for i in range(r):
                    col = out[:,i].reshape(-1,1)
                    all_out_Bourgain[i,:,:] = torch.abs(col-col.T)
                    all_out_Sarma[i,:,:] = col+col.T
                max_values,_ = torch.max(all_out_Bourgain, dim=0, keepdim=False)
                min_values,_ = torch.min(all_out_Sarma, dim=0, keepdim=False)
                if gpu_bool:
                    max_values = max_values.to(selected_cuda)
                    min_values = min_values.to(selected_cuda)
                t_loss = criterion[0](max_values, y) + criterion[0](min_values, y) + criterion[0](out, y[:,seeds])
                t_loss.backward()  # Derive gradients
                optimizer.step()  # Update parameters based on gradients.
        elif len(weights_train) == 0:
            if flag:
                edge_index = edge_index_train[-1]
                if gpu_bool:
                    edge_index = edge_index.to(selected_cuda)
                for x,y in list(zip(x_train,y_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                    out = model(x,edge_index)  # Perform a single forward pass
                    out = out.squeeze()*(1-x)
                    all_out_Bourgain = torch.zeros(r,y.shape[0], y.shape[1])
                    all_out_Sarma = torch.zeros(r,y.shape[0], y.shape[1])
                    for i in range(r):
                        col = out[:,i].reshape(-1,1)
                        all_out_Bourgain[i,:,:] = torch.abs(col-col.T)
                        all_out_Sarma[i,:,:] = col+col.T
                    max_values,_ = torch.max(all_out_Bourgain, dim=0, keepdim=False)
                    min_values,_ = torch.min(all_out_Sarma, dim=0, keepdim=False)
                    if gpu_bool:
                        max_values = max_values.to(selected_cuda)
                        min_values = min_values.to(selected_cuda)
                    t_loss = criterion[0](max_values, y) + criterion[0](min_values, y) + criterion[0](out, y[:,seeds])
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
            else:
                for x,y,edge_index in list(zip(x_train,y_train,edge_index_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                        edge_index = edge_index.to(selected_cuda)
                    out = model(x,edge_index)  # Perform a single forward pass
                    out = out.squeeze()*(1-x)
                    all_out_Bourgain = torch.zeros(r,y.shape[0], y.shape[1])
                    all_out_Sarma = torch.zeros(r,y.shape[0], y.shape[1])
                    for i in range(r):
                        col = out[:,i].reshape(-1,1)
                        all_out_Bourgain[i,:,:] = torch.abs(col-col.T)
                        all_out_Sarma[i,:,:] = col+col.T
                    max_values,_ = torch.max(all_out_Bourgain, dim=0, keepdim=False)
                    min_values,_ = torch.min(all_out_Sarma, dim=0, keepdim=False)
                    if gpu_bool:
                        max_values = max_values.to(selected_cuda)
                        min_values = min_values.to(selected_cuda)
                    t_loss = criterion[0](max_values, y) + criterion[0](min_values, y) + criterion[0](out, y[:,seeds])
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
        else:
            if flag:
                edge_index = edge_index_train[-1]
                weights = weights_train[-1]
                if gpu_bool:
                    edge_index = edge_index.to(selected_cuda)
                    weights = weights.to(selected_cuda)
                for x,y in list(zip(x_train,y_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                    out = model(x,edge_index,weights)  # Perform a single forward pass.
                    out = out.squeeze()*(1-x)
                    all_out_Bourgain = torch.zeros(r,y.shape[0], y.shape[1])
                    all_out_Sarma = torch.zeros(r,y.shape[0], y.shape[1])
                    for i in range(r):
                        col = out[:,i].reshape(-1,1)
                        all_out_Bourgain[i,:,:] = torch.abs(col-col.T)
                        all_out_Sarma[i,:,:] = col+col.T
                    max_values,_ = torch.max(all_out_Bourgain, dim=0, keepdim=False)
                    min_values,_ = torch.min(all_out_Sarma, dim=0, keepdim=False)
                    if gpu_bool:
                        max_values = max_values.to(selected_cuda)
                        min_values = min_values.to(selected_cuda)
                    t_loss = criterion[0](max_values, y) + criterion[0](min_values, y) + criterion[0](out, y[:,seeds])
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
            else:
                for x,y,edge_index,weights in list(zip(x_train,y_train,edge_index_train,weights_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    seeds = np.random.choice(range(x.shape[1]),size=r,replace=False)
                    x = x[:,seeds]
                    if gpu_bool:
                        x = x.to(selected_cuda)
                        y = y.to(selected_cuda)
                        edge_index = edge_index.to(selected_cuda)
                        weights = weights.to(selected_cuda)
                    out = model(x,edge_index,weights)  # Perform a single forward pass.
                    out = out.squeeze()*(1-x)
                    all_out_Bourgain = torch.zeros(r,y.shape[0], y.shape[1])
                    all_out_Sarma = torch.zeros(r,y.shape[0], y.shape[1])
                    for i in range(r):
                        col = out[:,i].reshape(-1,1)
                        all_out_Bourgain[i,:,:] = torch.abs(col-col.T)
                        all_out_Sarma[i,:,:] = col+col.T
                    max_values,_ = torch.max(all_out_Bourgain, dim=0, keepdim=False)
                    min_values,_ = torch.min(all_out_Sarma, dim=0, keepdim=False)
                    if gpu_bool:
                        max_values = max_values.to(selected_cuda)
                        min_values = min_values.to(selected_cuda)
                    t_loss = criterion[0](max_values, y) + criterion[0](min_values, y) + criterion[0](out, y[:,seeds])
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.

    train_loss, train_accuracy, train_sensitivity, train_specificity = test(gpu_bool,which_cuda,model,criterion,criterion_type,x_train,y_train,edge_index_train,weights_train)
    v_loss, v_accuracy, v_sensitivity, v_specificity = test(gpu_bool,which_cuda,model,criterion,criterion_type,x_val,y_val,edge_index_val,weights_val)

    if scheduler_type in ['step','exponential','cyclic','cosine']:
        scheduler[0].step()
    elif scheduler_type == 'reduce_on_plateau': 
        scheduler[0].step(v_loss)
    elif scheduler_type == 'cyclic-cosine':
        scheduler[0].step()
        scheduler[1].step()
    return train_loss,train_accuracy,train_sensitivity,train_specificity,v_loss,v_accuracy,v_sensitivity,v_specificity

def run(model_dir,dir,title,samples,all_hidden_channels,model_type,criterion_type,optimizer_type,scheduler_type,activation,num_epochs=100,early_stopping_patience=None,training_switch=-1,save_model=True,which_cuda=0):

    assert training_switch in range(-1,15), 'Invalid training switch.'

    gpu_bool = torch.cuda.is_available()
    selected_cuda = 'cuda:'+str(which_cuda)

    x_train = [torch.tensor(arr, requires_grad=True) for arr in samples[0][0]]
    x_val = [torch.tensor(arr, requires_grad=True) for arr in samples[1][0]]
    x_test = [torch.tensor(arr, requires_grad=True) for arr in samples[2][0]]
    y_train = [torch.tensor(arr) for arr in samples[0][1]]
    y_val = [torch.tensor(arr) for arr in samples[1][1]]
    y_test = [torch.tensor(arr) for arr in samples[2][1]]
    edge_index_train = [torch.tensor(arr) for arr in samples[0][2]]
    edge_index_val = [torch.tensor(arr) for arr in samples[1][2]]
    edge_index_test = [torch.tensor(arr) for arr in samples[2][2]]
    weights_train = [torch.tensor(arr) for arr in samples[0][3]]
    weights_val = [torch.tensor(arr) for arr in samples[1][3]]
    weights_test = [torch.tensor(arr) for arr in samples[2][3]]
    if training_switch in range(2,7):
        mask_list1 = samples[0][4][0]
        mask_list2 = [arr == False for arr in mask_list1]

    n = int(np.floor(np.sqrt(int(title.replace(',', '').split()[2]))))
    in_channels = n
    out_channels = n
    model,criterion,optimizer,scheduler = build(in_channels,all_hidden_channels,out_channels,model_type,criterion_type,optimizer_type,scheduler_type,activation)
    file = model_dir+'/'+model.name+'_out'+str(out_channels)+'_'+title+'.pth'
    file_old = model_dir+'/'+model.name+'_out'+str(out_channels)+'.pth'
    if os.path.exists(file):
        model.load_state_dict(torch.load(file))
    elif os.path.exists(file_old):
        model.load_state_dict(torch.load(file_old))
    print(model)

    if os.path.exists(model_dir+'/status_'+title+'.pkl'):
        try:
            with open(model_dir+'/status_'+title+'.pkl', 'rb') as file:
                status = pickle.load(file)
                if model_type == 'GCN':
                    done = status[0]
                elif model_type == 'SAGE':
                    done = status[1]
                elif model_type == 'GAT':
                    done = status[2]
                else:
                    done = status[3]
        except:
            status = False,False,False,False
            done = False
    else:
        status = False,False,False,False
        done = False

    #print(status)

    if not done:

        if gpu_bool:
            model = model.to(selected_cuda)

        train_loss = []
        train_acc = []
        train_sen = []
        train_spec = []
        val_loss = []
        val_acc = []
        val_sen = []
        val_spec = []

        if early_stopping_patience != None:
            best_val_loss = float('inf')
            best_epoch = 0
            no_improvement_count = 0

        p = model_dir+'/training_data_'+title+'.pkl'
        if os.path.exists(p):
            try:
                model.load_state_dict(torch.load(model_dir+'/best_model_'+model.name+'_'+title+'.pth'))
                with open(p, 'rb') as file:
                    data = pickle.load(file)
                    train_loss = data[0]
                    val_loss = data[1]
                    train_acc = data[2]
                    val_acc = data[3]
                    train_sen = data[4]
                    val_sen = data[5]
                    train_spec = data[6]
                    val_spec = data[7]
                    early_stopping_patience = data[8]
                    best_val_loss = data[9]
                    best_epoch = data[10]
                    no_improvement_count = data[11]
            except:
                pass

        for epoch in range(max(1,best_epoch+1), num_epochs+1):
            if training_switch == -1:
                if model_type == 'MLP':
                    t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
                else:
                    t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
            elif training_switch == 0:
                if model_type == 'MLP':
                    t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,balanced_masks=True)
                else:
                    t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,balanced_masks=True)
            elif training_switch == 1:
                if epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,balanced_masks=True)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,balanced_masks=True)
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
            elif training_switch == 2:
                if epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list1)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list1)
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list2)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list2)
            elif training_switch == 3:
                if epoch % 3 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list1)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list1)
                elif epoch % 3 == 1:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list2)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list2)
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
            elif training_switch == 4:
                if epoch % 3 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list1)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list1)
                elif epoch % 3 == 1:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list2)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list2)
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,balanced_masks=True)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,balanced_masks=True)
            elif training_switch == 5:
                if epoch % 4 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list1)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list1)
                elif epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list2)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list2)
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
            elif training_switch == 6:
                if epoch % 4 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list1)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list1)
                elif epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list2)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list2)
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,balanced_masks=True)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,balanced_masks=True)
            elif training_switch == 7:
                if epoch % 3 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Bourgain')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Bourgain')
                elif epoch % 3 == 1:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Sarma')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Sarma')
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
            elif training_switch == 8:
                if epoch % 3 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Bourgain')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Bourgain')
                elif epoch % 3 == 1:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Sarma')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Sarma')
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,balanced_masks=True)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,balanced_masks=True)
            elif training_switch == 9:
                if epoch % 4 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Bourgain')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Bourgain')
                elif epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Sarma')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Sarma')
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
            elif training_switch == 10:
                if epoch % 4 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Bourgain')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Bourgain')
                elif epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Sarma')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Sarma')
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,balanced_masks=True)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,balanced_masks=True)
            elif training_switch == 11:
                if epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Bourgain+Sarma')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Bourgain+Sarma')
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
            elif training_switch == 12:
                if epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Bourgain+Sarma')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Bourgain+Sarma')
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,balanced_masks=True)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,balanced_masks=True)
            elif training_switch == 13:
                if epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='all')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='all')
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
            elif training_switch == 14:
                if model_type == 'MLP':
                    t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='all')
                else:
                    t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='all')

            train_loss.append(t_loss)
            train_acc.append(t_acc)
            train_sen.append(t_sen)
            train_spec.append(t_spec)
            val_loss.append(v_loss)
            val_acc.append(v_acc)
            val_sen.append(v_sen)
            val_spec.append(v_spec)
            if epoch % 25 == 0:
                if criterion_type in ['mse','l2','mse-mse']:
                    print(f'Epoch: {epoch:03d}, Training Loss (MSE): {t_loss:.4f}, Validation Loss (MSE): {v_loss:.4f}')
                elif criterion_type == 'l1':
                    print(f'Epoch: {epoch:03d}, Training Loss (MAE): {t_loss:.4f}, Validation Loss (MAE): {v_loss:.4f}')
                else:
                    print(f'Epoch: {epoch:03d}, Training Loss: {t_loss:.4f}, Training Accuracy: {t_acc:.4f}, Training Sensitivity: {t_sen:.4f}, Training Specificity: {t_spec:.4f}, Validation Loss: {v_loss:.4f}, Validation Accuracy: {v_acc:.4f}, Validation Sensitivity: {v_sen:.4f}, Validation Specificity: {v_spec:.4f}')

            if early_stopping_patience != None:
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_epoch = epoch
                    no_improvement_count = 0
                    torch.save(model.state_dict(), model_dir+'/best_model_'+model.name+'_'+title+'.pth')
                    with open(p, 'wb') as file:
                        data = train_loss,val_loss,train_acc,val_acc,train_sen,val_sen,train_spec,val_spec,early_stopping_patience,best_val_loss,best_epoch,no_improvement_count
                        pickle.dump(data, file)
                else:
                    no_improvement_count += 1
                if no_improvement_count >= early_stopping_patience:
                    model.load_state_dict(torch.load(model_dir+'/best_model_'+model.name+'_'+title+'.pth'))
                    if gpu_bool:
                        model = model.to(selected_cuda)
                    break

        if model_type == 'MLP':
            test_loss, test_acc, test_sen, test_spec = test(gpu_bool,which_cuda,model,criterion,criterion_type,x_test, y_test)
        else:
            test_loss, test_acc, test_sen, test_spec = test(gpu_bool,which_cuda,model,criterion,criterion_type,x_test, y_test, edge_index_test, weights_test)
        
        if early_stopping_patience == None:
            if criterion_type in ['mse','l2','mse-mse']:
                print(f'Test Loss (MSE): {test_loss:03f}')
            elif criterion_type == 'l1':
                print(f'Test Loss (MAE): {test_loss:03f}')
            else:
                print(f'Test Loss: {test_loss:03f}, Test Accuracy: {test_acc:03f}, Test Sensitivity: {test_sen:03f}, Test Specificity: {test_spec:03f}')
        else:
            if criterion_type in ['mse','l2','mse-mse']:
                print(f'Best Epoch: {best_epoch:03d}, Test Loss (MSE): {test_loss:03f}')
            elif criterion_type == 'l1':
                print(f'Best Epoch: {best_epoch:03d}, Test Loss (MAE): {test_loss:03f}')
            else:
                print(f'Best Epoch: {best_epoch:03d}, Test Loss: {test_loss:03f}, Test Accuracy: {test_acc:03f}, Test Sensitivity: {test_sen:03f}, Test Specificity: {test_spec:03f}')

        x = range(1, len(train_loss)+1)
        plt.plot(x, train_loss, label = 'Training Loss', alpha = 0.75)
        plt.plot(x, val_loss, label = 'Validation Loss', alpha = 0.75)
        if criterion_type in ['ce','bce','bcelogits','multimargin']:
            plt.plot(x, train_acc, label = 'Training Accuracy', alpha = 0.75)
            plt.plot(x, val_acc, label = 'Validation Accuracy', alpha = 0.75)
            plt.plot(x, train_sen, label = 'Training Sensitivity', alpha = 0.75)
            plt.plot(x, val_sen, label = 'Validation Sensitivity', alpha = 0.75)
            plt.plot(x, train_spec, label = 'Training Specificity', alpha = 0.75)
            plt.plot(x, val_spec, label = 'Validation Specificity', alpha = 0.75)
        plt.xlabel('Epoch')
        plt.ylabel('')
        plt.title(model.name+': '+title)
        plt.legend()
        plt.savefig(dir+'/'+model.name+'_'+title+'_training.png')
        plt.close('all')
        #plt.show()

        model = model.to('cpu')
        if save_model:
            torch.save(model.state_dict(), model_dir+'/'+model.name+'_out'+str(out_channels)+'_'+title+'.pth')
            torch.save(model.state_dict(), model_dir+'/'+model.name+'_out'+str(out_channels)+'.pth')
        
        if os.path.exists(p):
            os.remove(p)
        
        with open(model_dir+'/status_'+title+'.pkl', 'wb') as file:
            if model_type == 'GCN':
                status = True, status[1], status[2], status[3]
                pickle.dump(status, file)
            elif model_type == 'SAGE':
                status = status[0], True, status[2], status[3]
                pickle.dump(status, file)
            elif model_type == 'GAT':
                status = status[0], status[1], True, status[3]
                pickle.dump(status, file)
            else:
                status = status[0], status[1], status[2], True
                pickle.dump(status, file)

    return model

def run_out1(model_dir,dir,title,samples,all_hidden_channels,model_type,criterion_type,optimizer_type,scheduler_type,activation,num_epochs=100,early_stopping_patience=None,training_switch=-1,save_model=True,which_cuda=0):

    assert training_switch in range(-1,15), 'Invalid training switch.'
    
    gpu_bool = torch.cuda.is_available()
    selected_cuda = 'cuda:'+str(which_cuda)

    x_train = [torch.tensor(arr, requires_grad=True) for arr in samples[0][0]]
    x_val = [torch.tensor(arr, requires_grad=True) for arr in samples[1][0]]
    x_test = [torch.tensor(arr, requires_grad=True) for arr in samples[2][0]]
    y_train = [torch.tensor(arr) for arr in samples[0][1]]
    y_val = [torch.tensor(arr) for arr in samples[1][1]]
    y_test = [torch.tensor(arr) for arr in samples[2][1]]
    edge_index_train = [torch.tensor(arr) for arr in samples[0][2]]
    edge_index_val = [torch.tensor(arr) for arr in samples[1][2]]
    edge_index_test = [torch.tensor(arr) for arr in samples[2][2]]
    weights_train = [torch.tensor(arr) for arr in samples[0][3]]
    weights_val = [torch.tensor(arr) for arr in samples[1][3]]
    weights_test = [torch.tensor(arr) for arr in samples[2][3]]
    if training_switch in range(2,7):
        mask_list1 = samples[0][4][0]
        mask_list2 = [arr == False for arr in mask_list1]

    n = int(np.floor(np.sqrt(int(title.replace(',', '').split()[2]))))
    in_channels = n
    out_channels = n
    model,criterion,optimizer,scheduler = build(in_channels,all_hidden_channels,out_channels,model_type,criterion_type,optimizer_type,scheduler_type,activation)
    file = model_dir+'/'+model.name+'_out'+str(out_channels)+'_'+title+'.pth'
    file_old = model_dir+'/'+model.name+'_out'+str(out_channels)+'.pth'
    if os.path.exists(file):
        model.load_state_dict(torch.load(file))
    elif os.path.exists(file_old):
        model.load_state_dict(torch.load(file_old))
    print(model)

    if os.path.exists(model_dir+'/status_out1_'+title+'.pkl'):
        try:
            with open(model_dir+'/status_out1_'+title+'.pkl', 'rb') as file:
                status = pickle.load(file)
                if model_type == 'GCN':
                    done = status[0]
                elif model_type == 'SAGE':
                    done = status[1]
                elif model_type == 'GAT':
                    done = status[2]
                else:
                    done = status[3]
        except:
            status = False,False,False,False
            done = False
    else:
        status = False,False,False,False
        done = False

    if not done:

        if gpu_bool:
            model = model.to(selected_cuda)

        train_loss = []
        train_acc = []
        train_sen = []
        train_spec = []
        val_loss = []
        val_acc = []
        val_sen = []
        val_spec = []

        if early_stopping_patience != None:
            best_val_loss = float('inf')
            best_epoch = 0
            no_improvement_count = 0

        p = model_dir+'/training_data_out1_'+title+'.pkl'
        if os.path.exists(p):
            with open(p, 'rb') as file:
                data = pickle.load(file)
                train_loss = data[0]
                val_loss = data[1]
                train_acc = data[2]
                val_acc = data[3]
                train_sen = data[4]
                val_sen = data[5]
                train_spec = data[6]
                val_spec = data[7]
                early_stopping_patience = data[8]
                best_val_loss = data[9]
                best_epoch = data[10]
                no_improvement_count = data[11]
                model.load_state_dict(torch.load(model_dir+'/best_model_'+model.name+'_'+title+'.pth'))

        for epoch in range(max(1,best_epoch+1), num_epochs+1):
            if training_switch == -1:
                if model_type == 'MLP':
                    t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
                else:
                    t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
            elif training_switch == 0:
                if model_type == 'MLP':
                    t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,balanced_masks=True)
                else:
                    t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,balanced_masks=True)
            elif training_switch == 1:
                if epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,balanced_masks=True)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,balanced_masks=True)
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
            elif training_switch == 2:
                if epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list1)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list1)
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list2)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list2)
            elif training_switch == 3:
                if epoch % 3 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list1)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list1)
                elif epoch % 3 == 1:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list2)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list2)
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
            elif training_switch == 4:
                if epoch % 3 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list1)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list1)
                elif epoch % 3 == 1:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list2)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list2)
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,balanced_masks=True)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,balanced_masks=True)
            elif training_switch == 5:
                if epoch % 4 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list1)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list1)
                elif epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list2)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list2)
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
            elif training_switch == 6:
                if epoch % 4 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list1)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list1)
                elif epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,mask_list=mask_list2)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,mask_list=mask_list2)
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,balanced_masks=True)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,balanced_masks=True)
            elif training_switch == 7:
                if epoch % 3 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Bourgain')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Bourgain')
                elif epoch % 3 == 1:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Sarma')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Sarma')
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
            elif training_switch == 8:
                if epoch % 3 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Bourgain')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Bourgain')
                elif epoch % 3 == 1:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Sarma')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Sarma')
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,balanced_masks=True)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,balanced_masks=True)
            elif training_switch == 9:
                if epoch % 4 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Bourgain')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Bourgain')
                elif epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Sarma')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Sarma')
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
            elif training_switch == 10:
                if epoch % 4 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Bourgain')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Bourgain')
                elif epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Sarma')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Sarma')
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,balanced_masks=True)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,balanced_masks=True)
            elif training_switch == 11:
                if epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Bourgain+Sarma')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Bourgain+Sarma')
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
            elif training_switch == 12:
                if epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='Bourgain+Sarma')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='Bourgain+Sarma')
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,balanced_masks=True)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,balanced_masks=True)
            elif training_switch == 13:
                if epoch % 2 == 0:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='all')
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='all')
                else:
                    if model_type == 'MLP':
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
                    else:
                        t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
            elif training_switch == 14:
                if model_type == 'MLP':
                    t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,eval='all')
                else:
                    t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train_Bourgain_Sarma(gpu_bool,which_cuda,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val,eval='all')

            train_loss.append(t_loss)
            train_acc.append(t_acc)
            train_sen.append(t_sen)
            train_spec.append(t_spec)
            val_loss.append(v_loss)
            val_acc.append(v_acc)
            val_sen.append(v_sen)
            val_spec.append(v_spec)
            if epoch % 25 == 0:
                if criterion_type in ['mse','l2','mse-mse']:
                    print(f'Epoch: {epoch:03d}, Training Loss (MSE): {t_loss:.4f}, Validation Loss (MSE): {v_loss:.4f}')
                elif criterion_type == 'l1':
                    print(f'Epoch: {epoch:03d}, Training Loss (MAE): {t_loss:.4f}, Validation Loss (MAE): {v_loss:.4f}')
                else:
                    print(f'Epoch: {epoch:03d}, Training Loss: {t_loss:.4f}, Training Accuracy: {t_acc:.4f}, Training Sensitivity: {t_sen:.4f}, Training Specificity: {t_spec:.4f}, Validation Loss: {v_loss:.4f}, Validation Accuracy: {v_acc:.4f}, Validation Sensitivity: {v_sen:.4f}, Validation Specificity: {v_spec:.4f}')

            if early_stopping_patience != None:
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_epoch = epoch
                    no_improvement_count = 0
                    torch.save(model.state_dict(), model_dir+'/best_model_'+model.name+'_'+title+'.pth')
                    with open(p, 'wb') as file:
                        data = train_loss,val_loss,train_acc,val_acc,train_sen,val_sen,train_spec,val_spec,early_stopping_patience,best_val_loss,best_epoch,no_improvement_count
                        pickle.dump(data, file)
                else:
                    no_improvement_count += 1
                if no_improvement_count >= early_stopping_patience:
                    model.load_state_dict(torch.load(model_dir+'/best_model_'+model.name+'_'+title+'.pth'))
                    if gpu_bool:
                        model = model.to(selected_cuda)
                    break

        if model_type == 'MLP':
            test_loss, test_acc, test_sen, test_spec = test(gpu_bool,which_cuda,model,criterion,criterion_type,x_test, y_test)
        else:
            test_loss, test_acc, test_sen, test_spec = test(gpu_bool,which_cuda,model,criterion,criterion_type,x_test, y_test, edge_index_test, weights_test)
        
        if early_stopping_patience == None:
            if criterion_type in ['mse','l2','mse-mse']:
                print(f'Test Loss (MSE): {test_loss:03f}')
            elif criterion_type == 'l1':
                print(f'Test Loss (MAE): {test_loss:03f}')
            else:
                print(f'Test Loss: {test_loss:03f}, Test Accuracy: {test_acc:03f}, Test Sensitivity: {test_sen:03f}, Test Specificity: {test_spec:03f}')
        else:
            if criterion_type in ['mse','l2','mse-mse']:
                print(f'Best Epoch: {best_epoch:03d}, Test Loss (MSE): {test_loss:03f}')
            elif criterion_type == 'l1':
                print(f'Best Epoch: {best_epoch:03d}, Test Loss (MAE): {test_loss:03f}')
            else:
                print(f'Best Epoch: {best_epoch:03d}, Test Loss: {test_loss:03f}, Test Accuracy: {test_acc:03f}, Test Sensitivity: {test_sen:03f}, Test Specificity: {test_spec:03f}')

        x = range(1, len(train_loss)+1)
        plt.plot(x, train_loss, label = 'Training Loss', alpha = 0.75)
        plt.plot(x, val_loss, label = 'Validation Loss', alpha = 0.75)
        if criterion_type in ['ce','bce','bcelogits','multimargin']:
            plt.plot(x, train_acc, label = 'Training Accuracy', alpha = 0.75)
            plt.plot(x, val_acc, label = 'Validation Accuracy', alpha = 0.75)
            plt.plot(x, train_sen, label = 'Training Sensitivity', alpha = 0.75)
            plt.plot(x, val_sen, label = 'Validation Sensitivity', alpha = 0.75)
            plt.plot(x, train_spec, label = 'Training Specificity', alpha = 0.75)
            plt.plot(x, val_spec, label = 'Validation Specificity', alpha = 0.75)
        plt.xlabel('Epoch')
        plt.ylabel('')
        plt.title(model.name+': '+title)
        plt.legend()
        plt.savefig(dir+'/'+model.name+'_'+title+'_training.png')
        plt.close('all')

        model = model.to('cpu')
        if save_model:
            torch.save(model.state_dict(), model_dir+'/'+model.name+'_out'+str(out_channels)+'_'+title+'.pth')
            torch.save(model.state_dict(), model_dir+'/'+model.name+'_out'+str(out_channels)+'.pth')
    
        if os.path.exists(p):
            os.remove(p)
        
        with open(model_dir+'/status_out1_'+title+'.pkl', 'wb') as file:
            if model_type == 'GCN':
                status = True, status[1], status[2], status[3]
                pickle.dump(status, file)
            elif model_type == 'SAGE':
                status = status[0], True, status[2], status[3]
                pickle.dump(status, file)
            elif model_type == 'GAT':
                status = status[0], status[1], True, status[3]
                pickle.dump(status, file)
            else:
                status = status[0], status[1], status[2], True
                pickle.dump(status, file)

    return model

def generateSamples_training_inner(sample_dir,title,data_name,num_graphs,function,*args,**kwargs):

    min_graph_size = np.inf
    max_graph_size = 0
    min_num_edges = np.inf
    max_num_edges = 0
    samples_x = []
    samples_y = []
    samples_edge_index = []
    samples_weights = []
    if data_name == 'Training Data':
        mask_list = []
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
                G = G.subgraph(largest_component)
                G = nx.relabel_nodes(G, {node: index for index, node in enumerate(G.nodes())})
                graph_info = G,directed,weighted
                sizes.remove(num_nodes)
                num_all_nodes += num_nodes
                if directed:
                    num_edges = len([(u,v) for (u,v) in G.edges() if u != v])
                else:
                    num_edges = len([(u,v) for (u,v) in G.edges() if v > u])
                num_all_edges += num_edges
                min_graph_size = min(min_graph_size,num_nodes)
                max_graph_size = max(max_graph_size,num_nodes)
                min_num_edges = min(min_num_edges,num_edges)
                max_num_edges = max(max_num_edges,num_edges)
                x = np.eye(num_nodes)
                y,_ = shortestDistances_networkx(graph_info)
                samples_x.append(x.astype(np.float32))
                samples_y.append(y.astype(np.float32))
                samples_edge_index.append(np.array(G.edges()).astype(np.int64).T)
                if weighted: 
                    samples_weights.append(np.array(nx.get_edge_attributes(G,'weight').values()).astype(np.float32))
                if data_name == 'Training Data':
                    mask_list.append(y > np.max(y)/2)
            else:
                n_rejected2 += 1
        except:
            n_rejected1 += 1
        if n_rejected1 + n_rejected2 >= 10000:
            print('Total number of graphs generated:',len(samples_x))
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
                k += 1
                num_all_nodes += num_nodes
                if directed:
                    num_edges = len([(u,v) for (u,v) in G.edges() if u != v])
                else:
                    num_edges = len([(u,v) for (u,v) in G.edges() if v > u])
                num_all_edges += num_edges
                min_graph_size = min(min_graph_size,num_nodes)
                max_graph_size = max(max_graph_size,num_nodes)
                min_num_edges = min(min_num_edges,num_edges)
                max_num_edges = max(max_num_edges,num_edges)
                x = np.eye(num_nodes)
                y,_ = shortestDistances_networkx(graph_info)
                samples_x.append(x.astype(np.float32))
                samples_y.append(y.astype(np.float32))
                samples_edge_index.append(np.array(G.edges()).astype(np.int64).T)
                if weighted: 
                    samples_weights.append(np.array(nx.get_edge_attributes(G,'weight').values()).astype(np.float32))
                if data_name == 'Training Data':
                    mask_list.append(y > np.max(y)/2)
            else:
                n_rejected2 += 1
        except:
            n_rejected1 += 1
        if n_rejected1 + n_rejected2 >= 10000:
            print('Total number of graphs generated:',len(samples_x))
            print('Number of graphs rejected in loop 2 because of undefined errors:',n_rejected1)
            print('Number of graphs rejected in loop 2 because the largest component has insufficient size:',n_rejected2)
            raise ValueError('Stuck in loop 2.')

    y_flattened = np.array(list(chain(*list(chain(*samples_y)))))
    plt.hist(y_flattened, edgecolor='white', alpha=0.4, label='All Data')
    if data_name == 'Training Data':
        balanced_masks = np.array(list(chain(*list(chain(*mask_list)))))
        data = y_flattened[balanced_masks]
        plt.hist(data, bins=int(np.max(data)-np.min(data)+1), edgecolor='white', alpha=0.4, label='Filtered Data ('+str(np.sum(balanced_masks))+' Pairs)')
        plt.legend()
    plt.xlabel('Actual Distance')
    plt.ylabel('Density')
    plt.title(data_name+': '+title)
    plt.savefig(sample_dir+'/'+data_name+', '+title+'_actualdensity.png')
    #plt.close('all')
    plt.show()

    graph_size_info = [min_graph_size,max_graph_size,num_all_nodes/num_graphs]
    edges_info = [min_num_edges,max_num_edges,num_all_edges/num_graphs]
    print('Graph size (min, max, mean):',graph_size_info)
    print('Number of edges (min, max, mean):',edges_info)

    if data_name == 'Training Data':
        if len(mask_list) == 0:
            return samples_x,samples_y,samples_edge_index,samples_weights,[None],graph_size_info,edges_info
        else:
            return samples_x,samples_y,samples_edge_index,samples_weights,[mask_list],graph_size_info,edges_info
    else:
        return samples_x,samples_y,samples_edge_index,samples_weights,None,graph_size_info,edges_info

def generateSamples_training(sample_dir,title,n_train,n_val,n_test,function,*args,**kwargs):
    print('Generating training data...')
    train = generateSamples_training_inner(sample_dir,title,'Training Data',n_train,function,*args,**kwargs)
    print('Generating validation data...')
    val = generateSamples_training_inner(sample_dir,title,'Validation Data',n_val,function,*args,**kwargs)
    print('Generating test data...')
    test = generateSamples_training_inner(sample_dir,title,'Test Data',n_test,function,*args,**kwargs)
    return train, val, test

def generateSamples_training_inner_dRegular(sample_dir,title,data_name,num_graphs,n,lbd):

    min_graph_size = np.inf
    max_graph_size = 0
    min_num_edges = np.inf
    max_num_edges = 0
    samples_x = []
    samples_y = []
    samples_edge_index = []
    samples_weights = []
    if data_name == 'Training Data':
        mask_list = []
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
                    print('Total number of graphs generated:',len(samples_x))
                    print('Number of graphs rejected because of undefined errors:',n_rejected1)
                    print('Number of graphs rejected because the largest component has insufficient size:',n_rejected2)
                    raise ValueError('Possibly stuck in infinite loop.')
            G = graph_info[0]
            num_all_nodes += num_nodes
            num_edges = len([(u,v) for (u,v) in G.edges() if v > u])
            num_all_edges += num_edges
            min_graph_size = min(min_graph_size,num_nodes)
            max_graph_size = max(max_graph_size,num_nodes)
            min_num_edges = min(min_num_edges,num_edges)
            max_num_edges = max(max_num_edges,num_edges)
            x = np.eye(num_nodes)
            y,_ = shortestDistances_networkx(graph_info)
            samples_x.append(x.astype(np.float32))
            samples_y.append(y.astype(np.float32))
            samples_edge_index.append(np.array(G.edges()).astype(np.int64).T)
            if graph_info[2]: 
                samples_weights.append(np.array(nx.get_edge_attributes(G,'weight').values()).astype(np.float32))
            if data_name == 'Training Data':
                mask_list.append(y > np.max(y)/2)
        except:
            n_rejected1 += 1
        if n_rejected1 + n_rejected2 >= 10000:
            print('Total number of graphs generated:',len(samples_x))
            print('Number of graphs rejected because of undefined errors:',n_rejected1)
            print('Number of graphs rejected because the largest component has insufficient size:',n_rejected2)
            raise ValueError('Possibly stuck in infinite loop.')

    y_flattened = np.array(list(chain(*list(chain(*samples_y)))))
    plt.hist(y_flattened, edgecolor='white', alpha=0.4, label='All Data', bins=int(np.max(y_flattened)-np.min(y_flattened)+1))
    plt.xlabel('Actual Distance')
    plt.ylabel('Density')
    plt.title(data_name+': '+title)
    plt.savefig(sample_dir+'/'+data_name+', '+title+'_actualdensity.png')
    #plt.close('all')
    plt.show()

    graph_size_info = [min_graph_size,max_graph_size,num_all_nodes/num_graphs]
    edges_info = [min_num_edges,max_num_edges,num_all_edges/num_graphs]
    print('Graph size (min, max, mean):',graph_size_info)
    print('Number of edges (min, max, mean):',edges_info)

    if data_name == 'Training Data':
        if len(mask_list) == 0:
            return samples_x,samples_y,samples_edge_index,samples_weights,[None],graph_size_info,edges_info
        else:
            return samples_x,samples_y,samples_edge_index,samples_weights,[mask_list],graph_size_info,edges_info
    else:
        return samples_x,samples_y,samples_edge_index,samples_weights,None,graph_size_info,edges_info

def generateSamples_training_dRegular(sample_dir,title,n_train,n_val,n_test,n,lbd):
    print('Generating training data...')
    train = generateSamples_training_inner_dRegular(sample_dir,title,'Training Data',n_train,n,lbd)
    print('Generating validation data...')
    val = generateSamples_training_inner_dRegular(sample_dir,title,'Validation Data',n_val,n,lbd)
    print('Generating test data...')
    test = generateSamples_training_inner_dRegular(sample_dir,title,'Test Data',n_test,n,lbd)
    return train, val, test

def generateSamples_training_inner_ErdosRenyi(sample_dir,title,data_name,num_graphs,n,lbd):

    min_graph_size = np.inf
    max_graph_size = 0
    min_num_edges = np.inf
    max_num_edges = 0
    samples_x = []
    samples_y = []
    samples_edge_index = []
    samples_weights = []
    if data_name == 'Training Data':
        mask_list = []
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
            graph_info = connectedErdosRenyiGraph(num_nodes,lbd/num_nodes)
            while not nx.is_strongly_connected(graph_info[0]):
                graph_info = connectedErdosRenyiGraph(num_nodes,lbd/num_nodes)
                n_rejected2 += 1
                if n_rejected1 + n_rejected2 >= 10000:
                    print('Total number of graphs generated:',len(samples_x))
                    print('Number of graphs rejected because of undefined errors:',n_rejected1)
                    print('Number of graphs rejected because the largest component has insufficient size:',n_rejected2)
                    raise ValueError('Possibly stuck in infinite loop.')
            G = graph_info[0]
            num_all_nodes += num_nodes
            num_edges = len([(u,v) for (u,v) in G.edges() if v > u])
            num_all_edges += num_edges
            min_graph_size = min(min_graph_size,num_nodes)
            max_graph_size = max(max_graph_size,num_nodes)
            min_num_edges = min(min_num_edges,num_edges)
            max_num_edges = max(max_num_edges,num_edges)
            x = np.eye(num_nodes)
            y,_ = shortestDistances_networkx(graph_info)
            samples_x.append(x.astype(np.float32))
            samples_y.append(y.astype(np.float32))
            samples_edge_index.append(np.array(G.edges()).astype(np.int64).T)
            if graph_info[2]: 
                samples_weights.append(np.array(nx.get_edge_attributes(G,'weight').values()).astype(np.float32))
            if data_name == 'Training Data':
                mask_list.append(y > np.max(y)/2)
        except:
            n_rejected1 += 1
        if n_rejected1 + n_rejected2 >= 10000:
            print('Total number of graphs generated:',len(samples_x))
            print('Number of graphs rejected because of undefined errors:',n_rejected1)
            print('Number of graphs rejected because the largest component has insufficient size:',n_rejected2)
            raise ValueError('Possibly stuck in infinite loop.')

    y_flattened = np.array(list(chain(*list(chain(*samples_y)))))
    plt.hist(y_flattened, edgecolor='white', alpha=0.4, label='All Data', bins=int(np.max(y_flattened)-np.min(y_flattened)+1))
    plt.xlabel('Actual Distance')
    plt.ylabel('Density')
    plt.title(data_name+': '+title)
    plt.savefig(sample_dir+'/'+data_name+', '+title+'_actualdensity.png')
    #plt.close('all')
    plt.show()

    graph_size_info = [min_graph_size,max_graph_size,num_all_nodes/num_graphs]
    edges_info = [min_num_edges,max_num_edges,num_all_edges/num_graphs]
    print('Graph size (min, max, mean):',graph_size_info)
    print('Number of edges (min, max, mean):',edges_info)

    if data_name == 'Training Data':
        if len(mask_list) == 0:
            return samples_x,samples_y,samples_edge_index,samples_weights,[None],graph_size_info,edges_info
        else:
            return samples_x,samples_y,samples_edge_index,samples_weights,[mask_list],graph_size_info,edges_info
    else:
        return samples_x,samples_y,samples_edge_index,samples_weights,None,graph_size_info,edges_info

def generateSamples_training_ErdosRenyi(sample_dir,title,n_train,n_val,n_test,n,lbd):
    print('Generating training data...')
    train = generateSamples_training_inner_ErdosRenyi(sample_dir,title,'Training Data',n_train,n,lbd)
    print('Generating validation data...')
    val = generateSamples_training_inner_ErdosRenyi(sample_dir,title,'Validation Data',n_val,n,lbd)
    print('Generating test data...')
    test = generateSamples_training_inner_ErdosRenyi(sample_dir,title,'Test Data',n_test,n,lbd)
    return train, val, test

def appendNewMasks(samples):
    train = samples[0]
    mask_lists = train[4]
    new_masks = []
    for y in train[1]:
        new_masks.append(y > np.max(y)/2)
    mask_lists.append(new_masks)
    train = train[0],train[1],train[2],train[3],mask_lists,train[5],train[6]
    return train, samples[1], samples[2]

def correctTrainingStats(samples):
    train_edges = samples[0][6]
    train_edges = [train_edges[0]/2,train_edges[1]/2,train_edges[2]/2]
    val_edges = samples[1][6]
    val_edges = [val_edges[0]/2,val_edges[1]/2,val_edges[2]/2]
    test_edges = samples[2][6]
    test_edges = [test_edges[0]/2,test_edges[1]/2,test_edges[2]/2]
    train = samples[0][0],samples[0][1],samples[0][2],samples[0][3],samples[0][4],samples[0][5],train_edges
    val = samples[1][0],samples[1][1],samples[1][2],samples[1][3],samples[1][4],samples[1][5],val_edges
    test = samples[2][0],samples[2][1],samples[2][2],samples[2][3],samples[2][4],samples[2][5],test_edges
    return train, val, test