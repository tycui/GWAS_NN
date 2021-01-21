from training import *
from functions import *
import os
import csv

cwd = os.getcwd()

idx = 1
############ Hyper-parameters
pve_int = 1.0; sparsity = 0.1; num_hidden_nodes = 200
learning_rate = 2*1e-3; num_epoch = 200
name = 'NN_' + str(idx)
#############################

print('Prior pve is: '+str(pve_int)+', sparsity level is '+ str(sparsity))
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
x_train, x_test, phenotype_y, phenotype_y_test = load_data(x_path, y_path); batch_size = int(x_train.shape[0] / 100)

## Informative prior for main effects NN
torch.manual_seed(0); perm = torch.randperm(x_train.size(0)); idx = perm[:int(x_train.size(0)/100)]; x_prior = splite_data(x_train[idx], gene_size)
slope_main, intercept_main = informative_pve_main(x_prior, gene_size)
print('BLR slope is '+str(slope_main))
print('BLR intercept is '+str(intercept_main))

## Main effects NN training
sigma_main = tau_estimate(pve_int, slope_main, intercept_main)
encoder = Encoder(gene_size, sigma_main); main_effect = Main_effect(num_gene, sigma_main); BRRR = SparseBNN(encoder, main_effect)
train_errors_brrr, test_errors_brrr = training(BRRR, x_train, phenotype_y, x_test, phenotype_y_test, learning_rate, batch_size, num_epoch)

## Generate permutation datasets with main effects NN model
phenotype_y_perm = permutation_data_generation(BRRR, x_train, phenotype_y, gene_size)
phenotype_y_test_perm = permutation_data_generation(BRRR, x_test, phenotype_y_test, gene_size)

## Informative prior for gene interaction NN
slope_int, intercept_int = informative_pve(x_prior, sparsity, gene_size, num_hidden_nodes)
print('BNN slope is '+str(slope_int))
print('BNN intercept is '+str(intercept_int))

## Gene interaction NN training
sigma_int = tau_estimate(pve_int, slope_int, intercept_int)
encoder = Encoder(gene_size, sigma_int); predictor = Predictor_wide(num_gene, sparsity, sigma_int, num_hidden_nodes)
SSBNN = SparseBNN(encoder, predictor)
train_errors, test_errors = training(SSBNN, x_train, phenotype_y_perm, x_test, phenotype_y_test_perm, learning_rate, batch_size, num_epoch)
print('BNN model :')
splited_x_test = splite_data(x_test, gene_size)
PTVE_test(SSBNN, splited_x_test, phenotype_y_test_perm)

# Detect interactions from the trained gene interaction NN
num_samples = 50 ## number of samples of explanation
torch.manual_seed(0); perm = torch.randperm(x_test.size(0)); idx = perm[:num_samples]
gene_test, _ = SSBNN.encoder(splite_data(x_test[idx], gene_size), training = False); gene_test.detach_()
baseline = torch.mean(gene_test, dim = 0).view(1,-1)
GlobalSIS_BNN, topGlobalSIS_BNN, Shapely_BNN = GlobalSIS(SSBNN.predictor, gene_test, baseline)
np.savetxt(cwd+'/PermutationDistribution/'+name+'.csv', Shapely_BNN, delimiter=",")

