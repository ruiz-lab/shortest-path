import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, hidden_channels3, hidden_channels4, hidden_channels5, out_channels, sigmoid = False, reLU = False):
        super(MLP, self).__init__()
        self.name = 'mlp'
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
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, hidden_channels3, hidden_channels4, out_channels, sigmoid = False, reLU = False):
        super(GCN, self).__init__()
        self.name = 'gcn'
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
            if criterion_type in ['ce','bce','multimargin']:
                model = MLP(in_channels,hidden_channels1,hidden_channels2,hidden_channels3,hidden_channels4,out_channels,sigmoid = True)
            elif criterion_type in ['mse','l2','l1']: 
                model = MLP(in_channels,hidden_channels1,hidden_channels2,hidden_channels3,hidden_channels4,out_channels,reLU = True)
            else:
                model = MLP(in_channels,hidden_channels1,hidden_channels2,hidden_channels3,hidden_channels4,out_channels)
        elif model == 'gcn':
            if criterion_type in ['ce','bce','multimargin']:
                model = GCN(in_channels,hidden_channels1,hidden_channels2,hidden_channels3,hidden_channels4,out_channels,sigmoid = True)
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