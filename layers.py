import numpy as np
import torch
import torch.nn as nn

class Meanfield_layer(nn.Module):
    def __init__(self, input_dim, output_dim, sigma = 0.6):     
        super(Meanfield_layer, self).__init__()      
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = sigma
        self.num_features = input_dim
        
        scale = 1. * np.sqrt(6. / (input_dim + output_dim))
        # approximated posterior
        self.mu_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-scale, scale))
        self.rho_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-4, -2))
        self.mu_bias = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-scale, scale))
        self.rho_bias = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-4, -2))
        
    def forward(self, x, training):
        eps = 1e-7
        sigma_beta = torch.log(1 + torch.exp(self.rho_beta))
        sigma_bias = torch.log(1 + torch.exp(self.rho_bias))
        if training:
            # forward passing with stochastic
            mean_output = torch.mm(x, self.mu_beta) + self.mu_bias
            sigma_output = torch.sqrt(torch.mm(x ** 2, sigma_beta ** 2) + sigma_bias ** 2)
            
            epsilon =  torch.randn(x.shape[0], self.output_dim)
            output = mean_output + sigma_output * epsilon

            # calculate KL
            KL_beta = torch.sum((sigma_beta ** 2 + self.mu_beta ** 2) / (2 * self.sigma ** 2) - torch.log(sigma_beta + eps) + np.log(self.sigma) - 0.5)
            KL_bias = torch.sum((sigma_bias ** 2 + self.mu_bias ** 2) / (2 * self.sigma ** 2) - torch.log(sigma_bias + eps) + np.log(self.sigma) - 0.5)
            
            KL = KL_beta + KL_bias 
            
            return output, KL   
        
        else:

            output = torch.mm(x, self.mu_beta) + self.mu_bias
            
            # calculate KL
            
            KL_beta = torch.sum((sigma_beta ** 2 + self.mu_beta ** 2) / (2 * self.sigma ** 2) - torch.log(sigma_beta + eps) + np.log(self.sigma) - 0.5)
            KL_bias = torch.sum((sigma_bias ** 2 + self.mu_bias ** 2) / (2 * self.sigma ** 2) - torch.log(sigma_bias + eps) + np.log(self.sigma) - 0.5)

            KL = KL_beta + KL_bias 
            
            return output, KL

class SpikeslabARD_layer(nn.Module):
    def __init__(self, input_dim, output_dim, slab_prob = 0.1, sigma = 0.6):     
        super(SpikeslabARD_layer, self).__init__()      
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_features = input_dim
        self.sigma = sigma   
        
        self.p_prior = torch.tensor(slab_prob)
        
        scale = 1. * np.sqrt(6. / (input_dim + output_dim))
        # approximated posterior
        self.mu_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-scale, scale))
        self.rho_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-4, -2))
        
        self.mu_bias = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-scale, scale))
        self.rho_bias = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-4, -2))
        
        init_min = 0.5; init_max = 0.5
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        
        self.p_logit = nn.Parameter(torch.empty(input_dim).uniform_(init_min, init_max))
        
    def forward(self, x, training):
        p = torch.sigmoid(self.p_logit)
        eps = 1e-7
        sigma_beta = torch.log(1 + torch.exp(self.rho_beta))
        sigma_bias = torch.log(1 + torch.exp(self.rho_bias))
        if training:
            # forward passing with stochastic
            x = self._hard_concrete_relaxation(p, x)

            mean_output = torch.mm(x, self.mu_beta) + self.mu_bias
            sigma_output = torch.sqrt(torch.mm(x ** 2, sigma_beta ** 2) + sigma_bias ** 2)
            
            epsilon =  torch.randn(x.shape[0], self.output_dim)
            output = mean_output + sigma_output * epsilon

            # calculate KL
            KL_beta = torch.sum((sigma_beta ** 2 + self.mu_beta ** 2) / (2 * self.sigma ** 2) - torch.log(sigma_beta + eps) + np.log(self.sigma) - 0.5)
            KL_bias = torch.sum((sigma_bias ** 2 + self.mu_bias ** 2) / (2 * self.sigma ** 2) - torch.log(sigma_bias + eps) + np.log(self.sigma) - 0.5)
            
            KL_entropy = torch.sum(p * torch.log(p+eps) + (1. - p) * torch.log(1. - p+eps))
            KL_prior = -torch.sum(p * torch.log(self.p_prior+eps) + (1. - p) * torch.log(1. - self.p_prior+eps)) 
            
            KL_spike = KL_entropy + KL_prior
            KL = KL_beta + KL_bias + KL_spike 
            
            return output, KL   
        
        else:
            output = torch.mm(x * p, self.mu_beta) + self.mu_bias
            
            # calculate KL
            
            KL_beta = torch.sum((sigma_beta ** 2 + self.mu_beta ** 2) / (2 * self.sigma ** 2) - torch.log(sigma_beta + eps) + np.log(self.sigma) - 0.5)
            KL_bias = torch.sum((sigma_bias ** 2 + self.mu_bias ** 2) / (2 * self.sigma ** 2) - torch.log(sigma_bias + eps) + np.log(self.sigma) - 0.5)

            KL_entropy = torch.sum(p * torch.log(p+eps) + (1. - p) * torch.log(1. - p+eps))
            KL_prior = -torch.sum(p * torch.log(self.p_prior+eps) + (1. - p) * torch.log(1. - self.p_prior+eps))
            KL_spike = KL_entropy + KL_prior
            KL = KL_beta + KL_bias + KL_spike 
            
            return output, KL
    
    def _hard_concrete_relaxation(self, p, x):
        eps = 1e-7
        temp = 0.1
        limit_left = -0.1; limit_right = 1.1
        unif_noise = torch.rand(x.shape[1])
        s = (torch.log(p + eps)  - torch.log(1 - p + eps) + torch.log(unif_noise + eps)  - torch.log(1 - unif_noise + eps))
        s = torch.sigmoid(s / temp); 
#         keep_prob = s
        
        s_bar = s * (limit_right - limit_left) + limit_left

        keep_prob = torch.min(torch.ones_like(s_bar), torch.max(torch.zeros_like(s_bar), s_bar))

        x  = x * keep_prob
        self.num_features = torch.sum(keep_prob)
        return x

class Encoder(nn.Module):
    def __init__(self, gene_size, sigma = 0.5):
        super(Encoder, self).__init__()
        self.dim_gene = gene_size
        self.sigma = sigma
        self.layers = nn.ModuleList([Meanfield_layer(gene_size[i], 1, sigma) for i in range(len(gene_size))])

    def forward(self, data_list, training):
        output = torch.zeros(data_list[0].shape[0], len(data_list))
        KL = torch.zeros([])
        
        for i, j in enumerate(self.layers):
            # forward passing for each gene
            output_i, kl_i = j(data_list[i], training)
            
            index_i = torch.zeros(data_list[0].shape[0], len(data_list))
            index_i[:, i] = 1
            output = output + index_i * output_i

            KL = KL + kl_i
            
        return output, KL

class Predictor_wide(nn.Module):
    def __init__(self, num_gene, p = 0.1, sigma = 0.5, num_hidden_nodes = 100):
        super(Predictor_wide, self).__init__()
        self.num_gene = num_gene
        self.p = p
        self.sigma = sigma
        
        self.Layer1 = Meanfield_layer(num_gene, num_hidden_nodes, sigma)
        self.Layer2 = SpikeslabARD_layer(num_hidden_nodes, 1, p, sigma)
        self.f = nn.Softplus(beta = 10)
        
    def forward(self, x, training):
        x1, kl1 = self.Layer1(x, training)
        x1 = self.f(x1)
        x2, kl2 = self.Layer2(x1, training)

        return x2, kl1+kl2

class Main_effect(nn.Module):
    def __init__(self, num_gene, sigma = 0.5):
        super(Main_effect, self).__init__()
        self.num_gene = num_gene
        self.sigma = sigma
        self.Layer1 = Meanfield_layer(num_gene, 1, sigma)
        
    def forward(self, x, training):
        x, kl = self.Layer1(x, training)
        return x, kl   
    
class SparseBNN(nn.Module):
    def __init__(self, encoder, predictor):
        super(SparseBNN, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        
    def forward(self, x):
        x1, kl1 = self.encoder(x, self.training)
        x2, kl2 = self.predictor(x1, self.training)

        return x2, kl1+kl2  



