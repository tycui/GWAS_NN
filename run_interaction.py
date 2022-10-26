import os
import csv
import numpy as np
import pandas as pd
import torch
from functions import load_data, GlobalSIS
from models import Encoder, Predictor, SparseNN, NNtraining


cwd = os.getcwd()
idx = 1
############ Hyper-parameters
reg_weight_encoder = 0.001; reg_weight_predictor = 0.001;
learning_rate = 0.001; num_epoch = 200000
name = 'NN_' + str(idx)
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
x, x_test, y, y_test = load_data(x_path, y_path); batch_size = int(x.shape[0] / 100)


## Gene interaction NN training
encoder = Encoder(gene_size, device = device); predictor = Predictor(gene_size); ginn = SparseNN(encoder, predictor)
GINN = NNtraining(ginn, 
                learning_rate=learning_rate, 
                batch_size=batch_size, 
                num_epoch=int(num_epoch), 
                reg_weight_encoder=reg_weight_encoder, 
                reg_weight_predictor=reg_weight_predictor,
                use_cuda = use_cuda,
                use_early_stopping = True)
GINN.training(x, y, x_test, y_test)


## Interaction detection
num_samples = 200; 
prediction_test, _, _ = GINN.model(x_test.to(device))
prediction_test.detach_()
value_high, idx_high = torch.topk(prediction_test.reshape(-1), int(num_samples/2))
value_low, idx_low = torch.topk(prediction_test.reshape(-1), int(num_samples/2), largest=False)
idx = torch.concat([idx_low, idx_high])
gene_test, _ = GINN.model.encoder(x_test[idx].to(device)); 
gene_test.detach_()
baseline = torch.mean(gene_test, dim = 0).view(1,-1).to(device)
GlobalSIS_NN, topGlobalSIS_NN, Shapely_NN = GlobalSIS(GINN.model.predictor, gene_test, baseline)
np.savetxt(cwd+'/InteractionScore/'+name+'.csv', Shapely_NN, delimiter=",")