import torch
import torch.nn as nn
import torch.distributions as Dis

# Neural Network Layers from Prior Distribution
class Gaussian_layer_prior(nn.Module):
    def __init__(self, input_dim, output_dim, scale):
        super(Gaussian_layer_prior, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = scale

    def forward(self, x):
        beta = torch.randn(self.input_dim, self.output_dim) * self.scale
        return torch.mm(x, beta)


class SS_layer_prior(nn.Module):
    def __init__(self, input_dim, output_dim, scale, prob):
        super(SS_layer_prior, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = scale
        self.prior_p = Dis.bernoulli.Bernoulli(probs=prob)

    def forward(self, x):
        beta = torch.randn(self.input_dim, self.output_dim) * self.scale
        indentity = self.prior_p.sample(sample_shape=torch.tensor([self.input_dim]))
        return torch.mm(x * indentity, beta)

# Neural Network priors
class Main_Prior(nn.Module):
    def __init__(self, num_feature, scale, num_output = 1):
        super(Main_Prior, self).__init__()
        self.num_feature = num_feature
        self.scale = scale
        self.Layer1 = Gaussian_layer_prior(num_feature, num_output, scale)

    def forward(self, x):
        x = self.Layer1(x)

        return x

class SparseBNN_Prior(nn.Module):
    def __init__(self, encoder_prior, predictor_prior):
        super(SparseBNN_Prior, self).__init__()
        self.encoder = encoder_prior
        self.predictor = predictor_prior
        
    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.predictor(x1)
        return x2 

class Predictor_wide_Prior(nn.Module):
    def __init__(self, num_feature, scale, prob = 0.1, num_hidden_nodes = 100):
        super(Predictor_wide_Prior, self).__init__()
        self.num_feature = num_feature
        self.scale = scale
        self.Layer1 = Gaussian_layer_prior(num_feature, num_hidden_nodes, scale)
        self.Layer2 = SS_layer_prior(num_hidden_nodes, 1, scale, prob)
        self.f = nn.Softplus(beta = 10)

    def forward(self, x):
        x1 = self.f(self.Layer1(x))
        x2 = self.Layer2(x1)
        return x2
    
class Encoder_Prior(nn.Module):
    def __init__(self, gene_size, sigma):
        super(Encoder_Prior, self).__init__()
        self.dim_gene = gene_size
        self.sigma = sigma
        self.layers = nn.ModuleList([Gaussian_layer_prior(gene_size[i], 1, sigma) for i in range(len(gene_size))])        
    
    def forward(self, data_list):
        output = torch.zeros(data_list[0].shape[0], len(data_list))
        
        for i, j in enumerate(self.layers):
            # forward passing for each gene
            output_i = j(data_list[i])
            
            index_i = torch.zeros(data_list[0].shape[0], len(data_list))
            index_i[:, i] = 1
            output = output + index_i * output_i
            
        return output
