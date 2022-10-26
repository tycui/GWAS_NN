import os
import csv
import numpy as np
import pandas as pd
import torch
from functions import load_data_permutation, preprocessing_permutation, GlobalSIS
from models import Encoder, Predictor, Main_effect, SparseNN, NNtraining

cwd = os.getcwd()
############ Hyper-parameters
num_run = 2
reg_weight_encoder_main = 0.001; reg_weight_predictor_main = 0.001;
reg_weight_encoder = 0.001; reg_weight_predictor = 0.001;
learning_rate = 0.001; num_epoch = 200000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Found device: {device}")
use_cuda=(True if torch.cuda.is_available() else False)
#############################
print('Encoder reg is: '+str(reg_weight_encoder)+', predictor reg is '+ str(reg_weight_predictor))
print('num_epoch is '+str(num_epoch))

## Load the gene structure
with open(cwd+'/data/snps_size.txt', newline='') as f:
    reader = csv.reader(f)
    tmp = list(reader)
gene_size = [int(float(i[0])) for i in tmp]
num_gene = len(gene_size)


## Load the data
x_path = cwd + "/data/genotype.csv"
y_path = cwd + "/data/phenotype.csv"
x, x_test, y, y_test, X_processed, Y_processed = load_data_permutation(x_path, y_path); batch_size = int(x.shape[0] / 100)


## main effect NN training
encoder = Encoder(gene_size, device = device); predictor = Main_effect(gene_size); menn = SparseNN(encoder, predictor);
MENN = NNtraining(menn, 
                  learning_rate=learning_rate, 
                  batch_size=batch_size, 
                  num_epoch=int(num_epoch), 
                  reg_weight_encoder=reg_weight_encoder_main, 
                  reg_weight_predictor=reg_weight_predictor_main,
                  use_cuda = use_cuda,
                  use_early_stopping = True)
MENN.training(x, y, x_test, y_test)


def one_permutation(index):
    """"
    save the interaction score on one permutation dataset to NN_idx
    """
    name = 'NN_' + str(index)
    ## Generate permutation datasets with main effects NN model
    torch.manual_seed(index)
    predicted_main, _, _ = MENN.model(X_processed.to(device)); predicted_main = predicted_main.detach()
    residual = Y_processed.reshape(-1,1).to(device) - predicted_main
    residual_perm = residual[torch.randperm(residual.shape[0]), :]
    Y_null = residual_perm + predicted_main
    x_null, x_test_null, y_null , y_test_null, X_processed_null, Y_processed_null = preprocessing_permutation(X_processed.to(device).numpy(), Y_null.to(device).numpy())

    ## Gene interaction NN training
    encoder = Encoder(gene_size, device = device); predictor = Predictor(gene_size); ginn_null = SparseNN(encoder, predictor);
    GINN_null = NNtraining(ginn_null, 
                          learning_rate=learning_rate, 
                          batch_size=batch_size, 
                          num_epoch=int(num_epoch), 
                          reg_weight_encoder=reg_weight_encoder, 
                          reg_weight_predictor=reg_weight_predictor,
                          use_cuda = use_cuda,
                          use_early_stopping = True)
    GINN_null.training(x_null, y_null, x_test_null, y_test_null)

    # Detect interactions from the trained gene interaction NN 
    ## Interaction detection
    num_samples = 200; 
    prediction_test, _, _ = GINN_null.model(x_test_null.to(device))
    prediction_test.detach_()
    value_high, idx_high = torch.topk(prediction_test.reshape(-1), int(num_samples/2))
    value_low, idx_low = torch.topk(prediction_test.reshape(-1), int(num_samples/2), largest=False)
    idx = torch.concat([idx_low, idx_high])
    gene_test, _ = GINN_null.model.encoder(x_test_null[idx].to(device)); 
    gene_test.detach_()
    baseline = torch.mean(gene_test, dim = 0).view(1,-1).to(device)
    GlobalSIS_NN, topGlobalSIS_NN, Shapely_NN = GlobalSIS(GINN_null.model.predictor, gene_test, baseline)
    np.savetxt(cwd+'/PermutationDistribution/'+name+'_1.csv', Shapely_NN, delimiter=",")

for i in range(num_run):
    one_permutation(i+1)