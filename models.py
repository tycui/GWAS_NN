import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):     
        super(LinearLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        scale = 1. * np.sqrt(6. / (input_dim + output_dim))
        # approximated posterior
        self.w = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-scale, scale))
        self.bias = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-scale, scale))

    def forward(self, x):
        output = torch.mm(x, self.w) + self.bias          
        return output
    
class SparseLinearLayer(nn.Module):
    def __init__(self, gene_size, device):     
        super(SparseLinearLayer, self).__init__()
        self.device = device
        self.input_dim = sum(gene_size)
        self.output_dim = len(gene_size)
        self.mask = self._mask(gene_size).detach().to(self.device)
        
        scale = 1. * np.sqrt(6. / (self.input_dim + self.output_dim))
        # approximated posterior
        self.w = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-scale, scale).to(self.device) * self.mask) 
        self.bias = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-scale, scale).to(self.device))

    def forward(self, x):
        output = torch.mm(x, self.w * self.mask) + self.bias          
        return output
    
    def _mask(self, gene_size):
        index_gene = []
        index_gene.append(0)
        for i in range(len(gene_size)):
            index_gene.append(gene_size[i] + index_gene[i])
        sparse_mask = torch.zeros(sum(gene_size), len(gene_size))
        for i in range(len(gene_size)):
            sparse_mask[index_gene[i]:index_gene[i+1], i]=1
        return sparse_mask


class Encoder(nn.Module):
    def __init__(self, gene_size, device):
        super(Encoder, self).__init__()
        self.layer = SparseLinearLayer(gene_size, device)
        
    def forward(self, x):
        x = self.layer(x)
        return x, self.reg_layers()
    
    def reg_layers(self):
        reg = torch.norm(self.layer.w, 1)
        return reg 
    
class Predictor(nn.Module):
    def __init__(self, gene_size):
        super(Predictor, self).__init__()
        self.input_dim = len(gene_size)
        
        self.Layer1 = LinearLayer(self.input_dim, 100)
        self.Layer2 = LinearLayer(100, 1)
        self.activation_fn = nn.Softplus(beta = 10)
        
    def forward(self, x):
        x1 = self.activation_fn(self.Layer1(x))
        x2 = self.Layer2(x1)

        return x2, self.reg_layers()
    
    def reg_layers(self):
        reg = torch.norm(self.Layer1.w, 1) + torch.norm(self.Layer2.w, 1)
        return reg 
    


class Main_effect(nn.Module):
    def __init__(self, gene_size):
        super(Main_effect, self).__init__()
        self.input_dim = len(gene_size)
        self.Layer1 = LinearLayer(self.input_dim, 1)
        
    def forward(self, x):
        x = self.Layer1(x)
        return x, self.reg_layers()
    
    def reg_layers(self):
        reg = torch.norm(self.Layer1.w, 1)
        return reg 
    
class SparseNN(nn.Module):
    def __init__(self, encoder, predictor):
        super(SparseNN, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        
    def forward(self, x):
        x1, kl1 = self.encoder(x)
        x2, kl2 = self.predictor(x1)

        return x2, kl1, kl2    


class NNtraining(object):
    def __init__(self, 
                 model, 
                 learning_rate=0.001, 
                 batch_size=10000, 
                 num_epoch=200, 
                 early_stop_patience = 20,
                 reg_weight_encoder = 0.0,
                 reg_weight_predictor = 0.0,
                 use_cuda=False,
                 use_early_stopping = False):
        
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.best_val = 1e5
        self.early_stop_patience = early_stop_patience
        self.epochs_since_update = 0  # used for early stopping
        self.reg_weight_encoder = reg_weight_encoder
        self.reg_weight_predictor = reg_weight_predictor
        self.use_early_stopping = use_early_stopping
        
        self.use_cuda = use_cuda
        if use_cuda:
            self.model.cuda()
        
    def training(self, x, y, xval, yval):
  
        parameters = set(self.model.parameters())
        optimizer = optim.Adam(parameters, lr=self.learning_rate, eps=1e-3)
        criterion = nn.MSELoss()
        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()
        train_dl = DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True)       
        for epoch in range(self.num_epoch):
            for x_batch, y_batch in train_dl:
                optimizer.zero_grad()
                self.model.train()
                # calculate the training loss
                output, reg_encoder, reg_predictor = self.model(x_batch)
                loss = criterion(y_batch, output) + self.reg_weight_encoder * reg_encoder + self.reg_weight_predictor * reg_predictor
                # backpropogate the gradient
                loss.backward()
                # optimize with SGD
                optimizer.step()
            
            train_mse, train_pve = self.build_evaluation(x, y)
            val_mse, val_pve = self.build_evaluation(xval, yval)
            print('>>> Epoch {:5d}/{:5d} | train_mse={:.5f} | val_mse={:.5f} | train_pve={:.5f} | val_pve={:.5f}'.format(epoch,
                                                                                                                         self.num_epoch, 
                                                                                                                         train_mse, 
                                                                                                                         val_mse, 
                                                                                                                         train_pve, 
                                                                                                                         val_pve))
            if self.use_early_stopping:
                early_stop = self._early_stop(val_mse)
                if early_stop:
                    break
    
                
    def build_evaluation(self, x_test, y_test):
        criterion = nn.MSELoss()
        if self.use_cuda:
            x_test = x_test.cuda()
            y_test = y_test.cuda()
        self.model.eval()
        y_pred, _, _ = self.model(x_test)
        mse_eval = criterion(y_test, y_pred).detach()
        
        pve = (1. - torch.var(y_pred.view(-1) - y_test.view(-1)) / torch.var(y_test.view(-1))).detach() 
        return mse_eval, pve
    
    def _early_stop(self, val_loss):
        updated = False # flag
        current = val_loss
        best = self.best_val
        improvement = (best - current) / best
#         improvement  = best - current
        
        if improvement > 0.00:
            self.best_val = current
            updated = True
        
        if updated:
            self.epochs_since_update = 0
        else:
            self.epochs_since_update += 1
            
        return self.epochs_since_update > self.early_stop_patience
    