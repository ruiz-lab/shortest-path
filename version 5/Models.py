import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, hidden_channels3, hidden_channels4, hidden_channels5, out_channels, sigmoid = False, reLU = False):
        super(MLP, self).__init__()
        self.name = 'mlp'
        #self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels1)
        self.lin2 = torch.nn.Linear(hidden_channels1, hidden_channels2)
        self.lin3 = torch.nn.Linear(hidden_channels2, hidden_channels3)
        self.lin4 = torch.nn.Linear(hidden_channels3, hidden_channels4)
        self.lin5 = torch.nn.Linear(hidden_channels4, hidden_channels5)
        self.lin6 = torch.nn.Linear(hidden_channels5, out_channels)
        if sigmoid:
            self.sigmoid = torch.nn.Sigmoid()
        else:
            self.sigmoid = None
        if reLU:
            self.reLU = torch.nn.ReLU()
        else:
            self.reLU = None

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin4(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin5(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin6(x)
        if self.sigmoid != None:
            x = self.sigmoid(x)
        if self.reLU != None:
            x = self.reLU(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, hidden_channels3, hidden_channels4, out_channels, sigmoid = False, softmax = False, reLU = False):
        super(GCN, self).__init__()
        self.name = 'gcn'
        #self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GCNConv(in_channels, hidden_channels1)
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2)
        self.conv3 = GCNConv(hidden_channels2, hidden_channels3)
        self.conv4 = GCNConv(hidden_channels3, hidden_channels4)
        self.conv5 = GCNConv(hidden_channels4, out_channels)
        if sigmoid:
            self.sigmoid = torch.nn.Sigmoid()
        else:
            self.sigmoid = None
        if softmax:
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.softmax = None
        if reLU:
            self.reLU = torch.nn.ReLU()
        else:
            self.reLU = None

    def forward(self, x, edge_index, weights=None):
        if weights == None:
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv3(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv4(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv5(x, edge_index)
        else:
            x = self.conv1(x, edge_index, edge_weight=weights)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index, edge_weight=weights)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv3(x, edge_index, edge_weight=weights)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv4(x, edge_index, edge_weight=weights)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv5(x, edge_index, edge_weight=weights)
        if self.sigmoid != None:
            x = self.sigmoid(x)
        if self.softmax != None:
            x = self.softmax(x)
        if self.reLU != None:
            x = self.reLU(x)
        return x

def build(in_channels,out_channels,model,criterion_type,optimizer_type,scheduler_type = None):
    
    if isinstance(model, str):
        hidden_channels1 = 128
        hidden_channels2 = 64
        hidden_channels3 = 32
        hidden_channels4 = 8
        if model == 'mlp':
            if criterion_type == 'bce':
                model = MLP(in_channels,hidden_channels1,hidden_channels2,hidden_channels3,hidden_channels4,out_channels,sigmoid = True)
            elif criterion_type == 'ce':
                model = MLP(in_channels,hidden_channels1,hidden_channels2,hidden_channels3,hidden_channels4,out_channels,softmax = True)
            elif criterion_type in ['mse','l2','l1']: 
                model = MLP(in_channels,hidden_channels1,hidden_channels2,hidden_channels3,hidden_channels4,out_channels,reLU = True)
            else:
                model = MLP(in_channels,hidden_channels1,hidden_channels2,hidden_channels3,hidden_channels4,out_channels)
        elif model == 'gcn':
            if criterion_type == 'bce':
                model = MLP(in_channels,hidden_channels1,hidden_channels2,hidden_channels3,hidden_channels4,out_channels,sigmoid = True)
            elif criterion_type == 'ce':
                model = MLP(in_channels,hidden_channels1,hidden_channels2,hidden_channels3,hidden_channels4,out_channels,softmax = True)
            elif criterion_type in ['mse','l2','l1']: 
                model = GCN(in_channels,hidden_channels1,hidden_channels2,hidden_channels3,hidden_channels4,out_channels,reLU = True)
            else:
                model = GCN(in_channels,hidden_channels1,hidden_channels2,hidden_channels3,hidden_channels4,out_channels)
        else:
            raise ValueError(
                "Model type not yet defined."
            )
    
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
    elif criterion_type == 'multimargin': # cuda crashed (similar to focal loss)
        criterion = [torch.nn.MultiMarginLoss()]
    elif criterion_type == 'mse-mse':
        criterion1 = torch.nn.MSELoss()
        criterion2 = torch.nn.MSELoss()
        criterion = [criterion1,criterion2]
    else:
        raise ValueError(
            "Criterion type not yet defined."
        )
    
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
    elif optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, weight_decay=0.0001, alpha=0.99, eps=1e-8, momentum=0.9)
    elif optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.0001, lr_decay=0.0001)
    else:
        raise ValueError(
            "Optimizer type not yet defined."
        )
    
    if scheduler_type == None:
        scheduler = None
    else:
        if scheduler_type == 'step':
            scheduler = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)]
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
            cycle_epochs = 5
            if optimizer_type in ['sgd','rmsprop']:
                scheduler_cyclic = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=cycle_epochs, cycle_momentum=True)
            else:
                scheduler_cyclic = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=cycle_epochs, cycle_momentum=False)
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cycle_epochs * 2)
            scheduler = [scheduler_cyclic,scheduler_cosine]
        else:
            raise ValueError(
                "Scheduler type not yet defined."
            )
    
    return model,criterion,optimizer,scheduler

def predict(gpu_bool,model,criterion_type,samples_x,samples_edge_index=[],samples_weights=[]):
    
    if len(samples_edge_index) > 0:
        if len(samples_edge_index) == 1:
            flag = True
        else:
            flag = False
    
    y_pred = []
    if gpu_bool:
        model = model.to('cuda:1')
    model.eval()
    with torch.no_grad():
        if model.out_channels == 1:
            if model.name == 'mlp':
                for x in samples_x:
                    if gpu_bool:
                        x = x.to('cuda:1')
                    pred_all = []
                    for j in range(x.shape[1]):
                        out = model(x[:,j].reshape(len(x[:,j]),1))  # Perform a single forward pass.
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        elif criterion_type in ['mse','l2','l1']:
                            pred = out.squeeze()
                        else:
                            pred = torch.round(out.squeeze())
                        pred_all.append(pred.cpu())
                    y_pred.append(np.array(pred_all).T)
            elif len(samples_weights) == 0:
                if flag:
                    edge_index = samples_edge_index[-1]
                    if gpu_bool:
                        edge_index = edge_index.to('cuda:1')
                    for x in samples_x:
                        if gpu_bool:
                            x = x.to('cuda:1')
                        pred_all = []
                        for j in range(x.shape[1]):
                            out = model(x[:,j].reshape(len(x[:,j]),1),edge_index)  # Perform a single forward pass.
                            if criterion_type in ['bce','ce','multimargin']:
                                pred = out.argmax(dim=1) #  Use the class with highest probability.
                            elif criterion_type in ['mse','l2','l1']:
                                pred = out.squeeze()
                            else:
                                pred = torch.round(out.squeeze())
                            pred_all.append(pred.cpu())
                        y_pred.append(np.array(pred_all).T)
                else:
                    for x,edge_index in list(zip(samples_x,samples_edge_index)):
                        if gpu_bool:
                            x = x.to('cuda:1')
                            edge_index = edge_index.to('cuda:1')
                        pred_all = []
                        for j in range(x.shape[1]):
                            out = model(x[:,j].reshape(len(x[:,j]),1),edge_index)  # Perform a single forward pass.
                            if criterion_type in ['bce','ce','multimargin']:
                                pred = out.argmax(dim=1) #  Use the class with highest probability.
                            elif criterion_type in ['mse','l2','l1']:
                                pred = out.squeeze()
                            else:
                                pred = torch.round(out.squeeze())
                            pred_all.append(pred.cpu())
                        y_pred.append(np.array(pred_all).T)
            else:
                if flag:
                    edge_index = samples_edge_index[-1]
                    weights = samples_weights[-1]
                    if gpu_bool:
                        edge_index = edge_index.to('cuda:1')
                        weights = weights.to('cuda:1')
                    for x in samples_x:
                        if gpu_bool:
                            x = x.to('cuda:1')
                        pred_all = []
                        for j in range(x.shape[1]):
                            out = model(x[:,j].reshape(len(x[:,j]),1),edge_index,weights)  # Perform a single forward pass.
                            if criterion_type in ['bce','ce','multimargin']:
                                pred = out.argmax(dim=1) #  Use the class with highest probability.
                            elif criterion_type in ['mse','l2','l1']:
                                pred = out.squeeze()
                            else:
                                pred = torch.round(out.squeeze())
                            pred_all.append(pred.cpu())
                        y_pred.append(np.array(pred_all).T)
                else:
                    for x,edge_index,weights in list(zip(samples_x,samples_edge_index,samples_weights)):
                        if gpu_bool:
                            x = x.to('cuda:1')
                            edge_index = edge_index.to('cuda:1')
                            weights = weights.to('cuda:1')
                        pred_all = []
                        for j in range(x.shape[1]):
                            out = model(x[:,j].reshape(len(x[:,j]),1),edge_index,weights)  # Perform a single forward pass.
                            if criterion_type in ['bce','ce','multimargin']:
                                pred = out.argmax(dim=1) #  Use the class with highest probability.
                            elif criterion_type in ['mse','l2','l1']:
                                pred = out.squeeze()
                            else:
                                pred = torch.round(out.squeeze())
                            pred_all.append(pred.cpu())
                        y_pred.append(np.array(pred_all).T)
        else:
            if model.name == 'mlp':
                for x in samples_x:
                    if gpu_bool:
                        x = x.to('cuda:1')
                    out = model(x)  # Perform a single forward pass.
                    if criterion_type in ['bce','ce','multimargin']:
                        pred = out.argmax(dim=1) #  Use the class with highest probability.
                    elif criterion_type in ['mse','l2','l1']:
                        pred = out.squeeze()
                    else:
                        pred = torch.round(out.squeeze())
                    y_pred.append(pred.cpu())
            elif len(samples_weights) == 0:
                if flag:
                    edge_index = samples_edge_index[-1]
                    if gpu_bool:
                        edge_index = edge_index.to('cuda:1')
                    for x in samples_x:
                        if gpu_bool:
                            x = x.to('cuda:1')
                        out = model(x,edge_index)  # Perform a single forward pass.
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        elif criterion_type in ['mse','l2','l1']:
                            pred = out.squeeze()
                        else:
                            pred = torch.round(out.squeeze())
                        y_pred.append(pred.cpu())
                else:
                    for x,edge_index in list(zip(samples_x,samples_edge_index)):
                        if gpu_bool:
                            x = x.to('cuda:1')
                            edge_index = edge_index.to('cuda:1')
                        out = model(x,edge_index)  # Perform a single forward pass.
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        elif criterion_type in ['mse','l2','l1']:
                            pred = out.squeeze()
                        else:
                            pred = torch.round(out.squeeze())
                        y_pred.append(pred.cpu())
            else:
                if flag:
                    edge_index = samples_edge_index[-1]
                    weights = samples_weights[-1]
                    if gpu_bool:
                        edge_index = edge_index.to('cuda:1')
                        weights = weights.to('cuda:1')
                    for x in samples_x:
                        if gpu_bool:
                            x = x.to('cuda:1')
                        out = model(x,edge_index,weights)  # Perform a single forward pass.
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        elif criterion_type in ['mse','l2','l1']:
                            pred = out.squeeze()
                        else:
                            pred = torch.round(out.squeeze())
                        y_pred.append(pred.cpu())
                else:
                    for x,edge_index,weights in list(zip(samples_x,samples_edge_index,samples_weights)):
                        if gpu_bool:
                            x = x.to('cuda:1')
                            edge_index = edge_index.to('cuda:1')
                            weights = weights.to('cuda:1')
                        out = model(x,edge_index,weights)  # Perform a single forward pass.
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        elif criterion_type in ['mse','l2','l1']:
                            pred = out.squeeze()
                        else:
                            pred = torch.round(out.squeeze())
                        y_pred.append(pred.cpu())
    model = model.to('cpu')
    return y_pred

def predict_allBatches(model,criterion_type,samples):
    gpu_bool = torch.cuda.is_available()
    y_pred_train = predict(gpu_bool, model, criterion_type, samples[0][0], samples[0][2], samples[0][3])
    y_pred_val = predict(gpu_bool, model, criterion_type, samples[1][0], samples[1][2], samples[1][3])
    y_pred_test = predict(gpu_bool, model, criterion_type, samples[2][0], samples[2][2], samples[2][3])
    return y_pred_train,y_pred_val,y_pred_test

def test(gpu_bool,model,criterion,criterion_type,samples_x,samples_y,samples_edge_index = [],samples_weights = []):
    
    if len(samples_edge_index) > 0:
        if len(samples_edge_index) == 1:
            flag = True
        else:
            flag = False
    
    t_loss = 0
    total_samples = 0
    y_pred = []
    model.eval()
    with torch.no_grad():
        if model.out_channels == 1:
            if len(samples_edge_index) == 0:
                for x,y in list(zip(samples_x,samples_y)):
                    if gpu_bool:
                        x = x.to('cuda:1')
                        y = y.to('cuda:1')
                    pred_all = []
                    for j in range(x.shape[1]):
                        out = model(x[:,j].reshape(len(x[:,j]),1))  # Perform a single forward pass.
                        t_loss += criterion[0](out.squeeze(), y[:,j])
                        total_samples += 1
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        elif criterion_type in ['mse','l2','l1','mse-mse']:
                            pred = out.squeeze()
                        else:
                            pred = torch.round(out.squeeze())
                        pred_all.append(pred.cpu())
                    y_pred.append(np.array(pred_all).T)
            elif len(samples_weights) == 0:
                if flag:
                    edge_index = samples_edge_index[-1]
                    if gpu_bool:
                        edge_index = edge_index.to('cuda:1')
                    for x,y in list(zip(samples_x,samples_y)):
                        if gpu_bool:
                            x = x.to('cuda:1')
                            y = y.to('cuda:1')
                        pred_all = []
                        for j in range(x.shape[1]):
                            out = model(x[:,j].reshape(len(x[:,j]),1),edge_index)  # Perform a single forward pass.
                            t_loss += criterion[0](out.squeeze(), y[:,j])
                            total_samples += 1
                            if criterion_type in ['bce','ce','multimargin']:
                                pred = out.argmax(dim=1) #  Use the class with highest probability.
                            elif criterion_type in ['mse','l2','l1','mse-mse']:
                                pred = out.squeeze()
                            else:
                                pred = torch.round(out.squeeze())
                            pred_all.append(pred.cpu())
                        y_pred.append(np.array(pred_all).T)
                else:
                    for x,y,edge_index in list(zip(samples_x,samples_y,samples_edge_index)):
                        if gpu_bool:
                            x = x.to('cuda:1')
                            y = y.to('cuda:1')
                            edge_index = edge_index.to('cuda:1')
                        pred_all = []
                        for j in range(x.shape[1]):
                            out = model(x[:,j].reshape(len(x[:,j]),1),edge_index)  # Perform a single forward pass.
                            t_loss += criterion[0](out.squeeze(), y[:,j])
                            total_samples += 1
                            if criterion_type in ['bce','ce','multimargin']:
                                pred = out.argmax(dim=1) #  Use the class with highest probability.
                            elif criterion_type in ['mse','l2','l1','mse-mse']:
                                pred = out.squeeze()
                            else:
                                pred = torch.round(out.squeeze())
                            pred_all.append(pred.cpu())
                        y_pred.append(np.array(pred_all).T)
            else:
                if flag:
                    edge_index = samples_edge_index[-1]
                    weights = samples_weights[-1]
                    if gpu_bool:
                        edge_index = edge_index.to('cuda:1')
                        weights = weights.to('cuda:1')
                    for x,y in list(zip(samples_x,samples_y)):
                        if gpu_bool:
                            x = x.to('cuda:1')
                            y = y.to('cuda:1')
                        pred_all = []
                        for j in range(x.shape[1]):
                            out = model(x[:,j].reshape(len(x[:,j]),1),edge_index,weights)  # Perform a single forward pass.
                            t_loss += criterion[0](out.squeeze(), y[:,j])
                            total_samples += 1
                            if criterion_type in ['bce','ce','multimargin']:
                                pred = out.argmax(dim=1) #  Use the class with highest probability.
                            elif criterion_type in ['mse','l2','l1','mse-mse']:
                                pred = out.squeeze()
                            else:
                                pred = torch.round(out.squeeze())
                            pred_all.append(pred.cpu())
                        y_pred.append(np.array(pred_all).T)
                else:
                    for x,y,edge_index,weights in list(zip(samples_x,samples_y,samples_edge_index,samples_weights)):
                        if gpu_bool:
                            x = x.to('cuda:1')
                            y = y.to('cuda:1')
                            edge_index = edge_index.to('cuda:1')
                            weights = weights.to('cuda:1')
                        pred_all = []
                        for j in range(x.shape[1]):
                            out = model(x[:,j].reshape(len(x[:,j]),1),edge_index,weights)  # Perform a single forward pass.
                            t_loss += criterion[0](out.squeeze(), y[:,j])
                            total_samples += 1
                            if criterion_type in ['bce','ce','multimargin']:
                                pred = out.argmax(dim=1) #  Use the class with highest probability.
                            elif criterion_type in ['mse','l2','l1','mse-mse']:
                                pred = out.squeeze()
                            else:
                                pred = torch.round(out.squeeze())
                            pred_all.append(pred.cpu())
                        y_pred.append(np.array(pred_all).T)
        else:
            if len(samples_edge_index) == 0:
                for x,y in list(zip(samples_x,samples_y)):
                    if gpu_bool:
                        x = x.to('cuda:1')
                        y = y.to('cuda:1')
                    out = model(x)  # Perform a single forward pass.
                    if criterion_type in ['bce']:
                        t_loss += criterion[0](out, torch.stack((1-y, y)).T)
                    elif criterion_type == 'mse-mse':
                        t_loss += 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                    elif criterion_type in ['ce','multimargin']:
                        t_loss += criterion[0](out, y) ## classification
                    else:
                        t_loss += criterion[0](out.squeeze(), y)
                    total_samples += 1
                    if criterion_type in ['bce','ce','multimargin']:
                        pred = out.argmax(dim=1) #  Use the class with highest probability.
                    elif criterion_type in ['mse','l2','l1','mse-mse']:
                        pred = out.squeeze()
                    else:
                        pred = torch.round(out.squeeze())
                    y_pred.append(pred.cpu())
            elif len(samples_weights) == 0:
                if flag:
                    edge_index = samples_edge_index[-1]
                    if gpu_bool:
                        edge_index = edge_index.to('cuda:1')
                    for x,y in list(zip(samples_x,samples_y)):
                        if gpu_bool:
                            x = x.to('cuda:1')
                            y = y.to('cuda:1')
                        out = model(x,edge_index)  # Perform a single forward pass.
                        if criterion_type in ['bce']:
                            t_loss += criterion[0](out, torch.stack((1-y, y)).T)
                        elif criterion_type == 'mse-mse':
                            t_loss += 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                        elif criterion_type in ['ce','multimargin']:
                            t_loss += criterion[0](out, y) ## classification
                        else:
                            t_loss += criterion[0](out.squeeze(), y)
                        total_samples += 1
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        elif criterion_type in ['mse','l2','l1','mse-mse']:
                            pred = out.squeeze()
                        else:
                            pred = torch.round(out.squeeze())
                        y_pred.append(pred.cpu())
                else:
                    for x,y,edge_index in list(zip(samples_x,samples_y,samples_edge_index)):
                        if gpu_bool:
                            x = x.to('cuda:1')
                            y = y.to('cuda:1')
                            edge_index = edge_index.to('cuda:1')
                        out = model(x,edge_index)  # Perform a single forward pass.
                        if criterion_type in ['bce']:
                            t_loss += criterion[0](out, torch.stack((1-y, y)).T)
                        elif criterion_type == 'mse-mse':
                            t_loss += 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                        elif criterion_type in ['ce','multimargin']:
                            t_loss += criterion[0](out, y) ## classification
                        else:
                            t_loss += criterion[0](out.squeeze(), y)
                        total_samples += 1
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        elif criterion_type in ['mse','l2','l1','mse-mse']:
                            pred = out.squeeze()
                        else:
                            pred = torch.round(out.squeeze())
                        y_pred.append(pred.cpu())
            else:
                if flag:
                    edge_index = samples_edge_index[-1]
                    weights = samples_weights[-1]
                    if gpu_bool:
                        edge_index = edge_index.to('cuda:1')
                        weights = weights.to('cuda:1')
                    for x,y in list(zip(samples_x,samples_y)):
                        if gpu_bool:
                            x = x.to('cuda:1')
                            y = y.to('cuda:1')
                        out = model(x,edge_index,weights)  # Perform a single forward pass.
                        if criterion_type in ['bce']:
                            t_loss += criterion[0](out, torch.stack((1-y, y)).T)
                        elif criterion_type == 'mse-mse':
                            t_loss += 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                        elif criterion_type in ['ce','multimargin']:
                            t_loss += criterion[0](out, y) ## classification
                        else:
                            t_loss += criterion[0](out.squeeze(), y)
                        total_samples += 1
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        elif criterion_type in ['mse','l2','l1','mse-mse']:
                            pred = out.squeeze()
                        else:
                            pred = torch.round(out.squeeze())
                        y_pred.append(pred.cpu())
                else:
                    for x,y,edge_index,weights in list(zip(samples_x,samples_y,samples_edge_index,samples_weights)):
                        if gpu_bool:
                            x = x.to('cuda:1')
                            y = y.to('cuda:1')
                            edge_index = edge_index.to('cuda:1')
                            weights = weights.to('cuda:1')
                        out = model(x,edge_index,weights)  # Perform a single forward pass.
                        if criterion_type in ['bce']:
                            t_loss += criterion[0](out, torch.stack((1-y, y)).T)
                        elif criterion_type == 'mse-mse':
                            t_loss += 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                        elif criterion_type in ['ce','multimargin']:
                            t_loss += criterion[0](out, y) ## classification
                        else:
                            t_loss += criterion[0](out.squeeze(), y)
                        total_samples += 1
                        if criterion_type in ['bce','ce','multimargin']:
                            pred = out.argmax(dim=1) #  Use the class with highest probability.
                        elif criterion_type in ['mse','l2','l1','mse-mse']:
                            pred = out.squeeze()
                        else:
                            pred = torch.round(out.squeeze())
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

def train(gpu_bool,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val,edge_index_train=[],edge_index_val=[],weights_train=[],weights_val=[]):
    
    if len(edge_index_train) > 0:
        if len(edge_index_train) == 1:
            flag = True
        else:
            flag = False
    
    model.train()
    if model.out_channels == 1:
        if len(edge_index_train) == 0:
            for x,y in list(zip(x_train,y_train)):
                optimizer.zero_grad()  # Clear gradients.
                if gpu_bool:
                    x = x.to('cuda:1')
                    y = y.to('cuda:1')
                for j in range(x.shape[1]):
                    out = model(x[:,j].reshape(len(x[:,j]),1))  # Perform a single forward pass.
                    t_loss = criterion[0](out.squeeze(), y[:,j])
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
        elif len(weights_train) == 0:
            if flag:
                edge_index = edge_index_train[-1]
                if gpu_bool:
                    edge_index = edge_index.to('cuda:1')
                for x,y in list(zip(x_train,y_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    if gpu_bool:
                        x = x.to('cuda:1')
                        y = y.to('cuda:1')
                    for j in range(x.shape[1]):
                        out = model(x[:,j].reshape(len(x[:,j]),1),edge_index)  # Perform a single forward pass
                        t_loss = criterion[0](out.squeeze(), y[:,j])
                        t_loss.backward()  # Derive gradients
                        optimizer.step()  # Update parameters based on gradients.
            else:
                for x,y,edge_index in list(zip(x_train,y_train,edge_index_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    if gpu_bool:
                        x = x.to('cuda:1')
                        y = y.to('cuda:1')
                        edge_index = edge_index.to('cuda:1')
                    for j in range(x.shape[1]):
                        out = model(x[:,j].reshape(len(x[:,j]),1),edge_index)  # Perform a single forward pass
                        t_loss = criterion[0](out.squeeze(), y[:,j])
                        t_loss.backward()  # Derive gradients
                        optimizer.step()  # Update parameters based on gradients.
        else:
            if flag:
                edge_index = edge_index_train[-1]
                weights = weights_train[-1]
                if gpu_bool:
                    edge_index = edge_index.to('cuda:1')
                    weights = weights.to('cuda:1')
                for x,y in list(zip(x_train,y_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    if gpu_bool:
                        x = x.to('cuda:1')
                        y = y.to('cuda:1')
                    for j in range(x.shape[1]):
                        out = model(x[:,j].reshape(len(x[:,j]),1),edge_index,weights)  # Perform a single forward pass.
                        t_loss = criterion[0](out.squeeze(), y[:,j])
                        t_loss.backward()  # Derive gradients
                        optimizer.step()  # Update parameters based on gradients.
            else:
                for x,y,edge_index,weights in list(zip(x_train,y_train,edge_index_train,weights_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    if gpu_bool:
                        x = x.to('cuda:1')
                        y = y.to('cuda:1')
                        edge_index = edge_index.to('cuda:1')
                        weights = weights.to('cuda:1')
                    for j in range(x.shape[1]):
                        out = model(x[:,j].reshape(len(x[:,j]),1),edge_index,weights)  # Perform a single forward pass.
                        t_loss = criterion[0](out.squeeze(), y[:,j])
                        t_loss.backward()  # Derive gradients
                        optimizer.step()  # Update parameters based on gradients.
    else:
        if len(edge_index_train) == 0:
            for x,y in list(zip(x_train,y_train)):
                optimizer.zero_grad()  # Clear gradients.
                if gpu_bool:
                    x = x.to('cuda:1')
                    y = y.to('cuda:1')
                out = model(x)  # Perform a single forward pass.
                if criterion_type in ['bce']:
                    t_loss = criterion[0](out, torch.stack((1-y, y)).T)
                elif criterion_type == 'mse-mse':
                    t_loss = 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                elif criterion_type in ['ce','multimargin']:
                    t_loss = criterion[0](out, y) ## classification
                else:
                    t_loss = criterion[0](out.squeeze(), y)
                t_loss.backward()  # Derive gradients
                optimizer.step()  # Update parameters based on gradients.
        elif len(weights_train) == 0:
            if flag:
                edge_index = edge_index_train[-1]
                if gpu_bool:
                    edge_index = edge_index.to('cuda:1')
                for x,y in list(zip(x_train,y_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    if gpu_bool:
                        x = x.to('cuda:1')
                        y = y.to('cuda:1')
                    out = model(x,edge_index)  # Perform a single forward pass
                    if criterion_type in ['bce']:
                        t_loss = criterion[0](out, torch.stack((1-y, y)).T)
                    elif criterion_type == 'mse-mse':
                        t_loss = 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                    elif criterion_type in ['ce','multimargin']:
                        t_loss = criterion[0](out, y) ## classification
                    else:
                        t_loss = criterion[0](out.squeeze(), y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
            else:
                for x,y,edge_index in list(zip(x_train,y_train,edge_index_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    if gpu_bool:
                        x = x.to('cuda:1')
                        y = y.to('cuda:1')
                        edge_index = edge_index.to('cuda:1')
                    out = model(x,edge_index)  # Perform a single forward pass
                    if criterion_type in ['bce']:
                        t_loss = criterion[0](out, torch.stack((1-y, y)).T)
                    elif criterion_type == 'mse-mse':
                        t_loss = 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                    elif criterion_type in ['ce','multimargin']:
                        t_loss = criterion[0](out, y) ## classification
                    else:
                        t_loss = criterion[0](out.squeeze(), y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
        else:
            if flag:
                edge_index = edge_index_train[-1]
                weights = weights_train[-1]
                if gpu_bool:
                    edge_index = edge_index.to('cuda:1')
                    weights = weights.to('cuda:1')
                for x,y in list(zip(x_train,y_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    if gpu_bool:
                        x = x.to('cuda:1')
                        y = y.to('cuda:1')
                    out = model(x,edge_index,weights)  # Perform a single forward pass.
                    if criterion_type in ['bce']:
                        t_loss = criterion[0](out, torch.stack((1-y, y)).T)
                    elif criterion_type == 'mse-mse':
                        t_loss = 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                    elif criterion_type in ['ce','multimargin']:
                        t_loss = criterion[0](out, y) ## classification
                    else:
                        t_loss = criterion[0](out.squeeze(), y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.
            else:
                for x,y,edge_index,weights in list(zip(x_train,y_train,edge_index_train,weights_train)):
                    optimizer.zero_grad()  # Clear gradients.
                    if gpu_bool:
                        x = x.to('cuda:1')
                        y = y.to('cuda:1')
                        edge_index = edge_index.to('cuda:1')
                        weights = weights.to('cuda:1')
                    out = model(x,edge_index,weights)  # Perform a single forward pass.
                    if criterion_type in ['bce']:
                        t_loss = criterion[0](out, torch.stack((1-y, y)).T)
                    elif criterion_type == 'mse-mse':
                        t_loss = 100*criterion[0](out[:, ::2], y[:, ::2]) + criterion[1](out[:, 1::2], y[:, 1::2])
                    elif criterion_type in ['ce','multimargin']:
                        t_loss = criterion[0](out, y) ## classification
                    else:
                        t_loss = criterion[0](out.squeeze(), y)
                    t_loss.backward()  # Derive gradients
                    optimizer.step()  # Update parameters based on gradients.

    train_loss, train_accuracy, train_sensitivity, train_specificity = test(gpu_bool,model,criterion,criterion_type,x_train,y_train,edge_index_train,weights_train)
    v_loss, v_accuracy, v_sensitivity, v_specificity = test(gpu_bool,model,criterion,criterion_type,x_val,y_val,edge_index_val,weights_val)

    if scheduler_type in ['step','exponential','cyclic','cosine']:
        scheduler[0].step()
    elif scheduler_type == 'reduce_on_plateau': 
        scheduler[0].step(v_loss)
    elif scheduler_type == 'cyclic-cosine':
        scheduler[0].step()
        scheduler[1].step()
    return train_loss,train_accuracy,train_sensitivity,train_specificity,v_loss,v_accuracy,v_sensitivity,v_specificity

def run(model_dir,samples,model_type,criterion_type,optimizer_type,scheduler_type,num_epochs=100,early_stopping_patience=None,save_model = False):

    title = 'out_channels = floor(sqrt(n))'
    gpu_bool = torch.cuda.is_available()

    x_train = samples[0][2][0]
    x_val = samples[1][2][0]
    x_test = samples[2][2][0]
    y_train = samples[0][2][1]
    y_val = samples[1][2][1]
    y_test = samples[2][2][1]
    edge_index_train = samples[0][2][2]
    edge_index_val = samples[1][2][2]
    edge_index_test = samples[2][2][2]
    weights_train = samples[0][2][3]
    weights_val = samples[1][2][3]
    weights_test = samples[2][2][3]

    in_channels = x_train[0].shape[1]
    out_channels = y_train[0].shape[1]
    model,criterion,optimizer,scheduler = build(in_channels, out_channels, model_type,criterion_type,optimizer_type,scheduler_type)
    file = model_dir+'/model_out'+str(out_channels)+'.pth'
    if os.path.exists(file):
        model.load_state_dict(torch.load(file))
    print(model)

    if gpu_bool:
        model = model.to('cuda:1')

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

    for epoch in range(1, num_epochs+1):
        if model_type == 'mlp':
            t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
        else:
            t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
        train_loss.append(t_loss)
        train_acc.append(t_acc)
        train_sen.append(t_sen)
        train_spec.append(t_spec)
        val_loss.append(v_loss)
        val_acc.append(v_acc)
        val_sen.append(v_sen)
        val_spec.append(v_spec)
        if epoch % 10 == 0:
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
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                no_improvement_count += 1
            if no_improvement_count >= early_stopping_patience:
                model.load_state_dict(torch.load('best_model.pth'))
                if gpu_bool:
                    model = model.to('cuda:1')
                break

    if model_type == 'mlp':
        test_loss, test_acc, test_sen, test_spec = test(gpu_bool,model,criterion,criterion_type,x_test, y_test)
    else:
        test_loss, test_acc, test_sen, test_spec = test(gpu_bool,model,criterion,criterion_type,x_test, y_test, edge_index_test, weights_test)
    
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
    plt.plot(x, train_loss, color = '#1f77b4', label = 'Training Loss', alpha = 0.75)
    plt.plot(x, val_loss, color = '#ff7f0e', label = 'Validation Loss', alpha = 0.75)
    if criterion_type in ['ce','bce','bcelogits','multimargin']:
        plt.plot(x, train_acc, color = '#2ca02c', label = 'Training Accuracy', alpha = 0.75)
        plt.plot(x, val_acc, color = '#d62728', label = 'Validation Accuracy', alpha = 0.75)
        plt.plot(x, train_sen, color = '#9467bd', label = 'Training Sensitivity', alpha = 0.75)
        plt.plot(x, val_sen, color = '#8c564b', label = 'Validation Sensitivity', alpha = 0.75)
        plt.plot(x, train_spec, color = '#e377c2', label = 'Training Specificity', alpha = 0.75)
        plt.plot(x, val_spec, color = '#7f7f7f', label = 'Validation Specificity', alpha = 0.75)
    plt.xlabel('Epoch')
    plt.ylabel('')
    plt.title('Training Results: '+title)
    plt.legend()
    filenames = os.listdir(dir)
    if '0.png' not in filenames:
        plt.savefig(dir+'/0.png')
    else:
        filenames = sorted([int(f[:-4]) for f in filenames])
        plt.savefig(dir+'/'+str(filenames[-1]+1)+'.png')   
    plt.show()

    model = model.to('cpu')
    if save_model:
        torch.save(model.state_dict(), model_dir+'/model_out'+str(model.out_channels)+'.pth')

    return model

def run_out1(model_dir,samples,model_type,criterion_type,optimizer_type,scheduler_type,num_epochs=100,early_stopping_patience=None,save_model = False):

    title = 'out_channels = 1'
    gpu_bool = torch.cuda.is_available()

    x_train = samples[0][2][0]
    x_val = samples[1][2][0]
    x_test = samples[2][2][0]
    y_train = samples[0][2][1]
    y_val = samples[1][2][1]
    y_test = samples[2][2][1]
    edge_index_train = samples[0][2][2]
    edge_index_val = samples[1][2][2]
    edge_index_test = samples[2][2][2]
    weights_train = samples[0][2][3]
    weights_val = samples[1][2][3]
    weights_test = samples[2][2][3]

    in_channels = 1
    out_channels = 1
    model,criterion,optimizer,scheduler = build(in_channels, out_channels, model_type,criterion_type,optimizer_type,scheduler_type)
    file = model_dir+'/model_out'+str(out_channels)+'.pth'
    if os.path.exists(file):
        model.load_state_dict(torch.load(file))
    print(model)

    if gpu_bool:
        model = model.to('cuda:1')

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

    for epoch in range(1, num_epochs+1):
        if model_type == 'mlp':
            t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,model,criterion,optimizer,scheduler,criterion_type,scheduler_type,x_train,x_val,y_train,y_val)
        else:
            t_loss,t_acc,t_sen,t_spec, v_loss, v_acc, v_sen, v_spec = train(gpu_bool,model,criterion,optimizer,scheduler,criterion_type,optimizer_type,x_train,x_val,y_train,y_val,edge_index_train,edge_index_val,weights_train,weights_val)
        train_loss.append(t_loss)
        train_acc.append(t_acc)
        train_sen.append(t_sen)
        train_spec.append(t_spec)
        val_loss.append(v_loss)
        val_acc.append(v_acc)
        val_sen.append(v_sen)
        val_spec.append(v_spec)
        if epoch % 10 == 0:
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
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                no_improvement_count += 1
            if no_improvement_count >= early_stopping_patience:
                model.load_state_dict(torch.load('best_model.pth'))
                if gpu_bool:
                    model = model.to('cuda:1')
                break

    if model_type == 'mlp':
        test_loss, test_acc, test_sen, test_spec = test(gpu_bool,model,criterion,criterion_type,x_test, y_test)
    else:
        test_loss, test_acc, test_sen, test_spec = test(gpu_bool,model,criterion,criterion_type,x_test, y_test, edge_index_test, weights_test)
    
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
    plt.plot(x, train_loss, color = '#1f77b4', label = 'Training Loss', alpha = 0.75)
    plt.plot(x, val_loss, color = '#ff7f0e', label = 'Validation Loss', alpha = 0.75)
    if criterion_type in ['ce','bce','bcelogits','multimargin']:
        plt.plot(x, train_acc, color = '#2ca02c', label = 'Training Accuracy', alpha = 0.75)
        plt.plot(x, val_acc, color = '#d62728', label = 'Validation Accuracy', alpha = 0.75)
        plt.plot(x, train_sen, color = '#9467bd', label = 'Training Sensitivity', alpha = 0.75)
        plt.plot(x, val_sen, color = '#8c564b', label = 'Validation Sensitivity', alpha = 0.75)
        plt.plot(x, train_spec, color = '#e377c2', label = 'Training Specificity', alpha = 0.75)
        plt.plot(x, val_spec, color = '#7f7f7f', label = 'Validation Specificity', alpha = 0.75)
    plt.xlabel('Epoch')
    plt.ylabel('')
    plt.title('Training Results: '+title)
    plt.legend()
    filenames = os.listdir(dir)
    if '0.png' not in filenames:
        plt.savefig(dir+'/0.png')
    else:
        filenames = sorted([int(f[:-4]) for f in filenames])
        plt.savefig(dir+'/'+str(filenames[-1]+1)+'.png')   
    plt.show()

    model = model.to('cpu')
    if save_model:
        torch.save(model.state_dict(), model_dir+'/model_out'+str(model.out_channels)+'.pth')
    
    return model