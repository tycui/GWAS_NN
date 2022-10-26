import torch
import numpy as np
import pandas as pd

def load_data(x_path, y_path):
    Y = np.genfromtxt(y_path, delimiter=",")
    X = np.genfromtxt(x_path, delimiter=",")

    ## scale each SNPs to have unit variance
    for i in range(X.shape[1]):
        X[:,i] = X[:,i] - np.mean(X[:,i])

        Y = (Y - np.mean(Y)) / np.std(Y)
    # seperate training and testing data
    np.random.seed(129)
    msk = np.random.rand(len(X)) < 0.7
    x_train = X[msk,:]; x_test = X[~msk,:]
    y_train = Y[msk]; y_test = Y[~msk]

    x_train = torch.tensor(x_train, dtype = torch.float)
    x_test = torch.tensor(x_test, dtype = torch.float)
    y_train = torch.tensor(y_train, dtype = torch.float)
    y_test = torch.tensor(y_test, dtype = torch.float)
    ## consider the first phenotype
    phenotype_y = y_train.view(-1,1)
    phenotype_y_test = y_test.view(-1,1)
    return x_train, x_test, phenotype_y, phenotype_y_test


def load_data_permutation(x_path, y_path):
    Y = np.genfromtxt(y_path, delimiter=",")
    X = np.genfromtxt(x_path, delimiter=",")

    ## scale each SNPs to have unit variance
    for i in range(X.shape[1]):
        X[:,i] = X[:,i] - np.mean(X[:,i])

        Y = (Y - np.mean(Y)) / np.std(Y)
    # seperate training and testing data
    np.random.seed(129)
    msk = np.random.rand(len(X)) < 0.7
    x_train = X[msk,:]; x_test = X[~msk,:]
    y_train = Y[msk]; y_test = Y[~msk]

    x_train = torch.tensor(x_train, dtype = torch.float)
    x_test = torch.tensor(x_test, dtype = torch.float)
    y_train = torch.tensor(y_train, dtype = torch.float)
    y_test = torch.tensor(y_test, dtype = torch.float)
    ## consider the first phenotype
    phenotype_y = y_train.view(-1,1)
    phenotype_y_test = y_test.view(-1,1)
    return x_train, x_test, phenotype_y, phenotype_y_test, torch.tensor(X, dtype = torch.float), torch.tensor(Y, dtype = torch.float)


def preprocessing_permutation(X, Y):
    X_prep = X; Y_prep = Y
    ## scale each SNPs to have unit variance
    for i in range(X.shape[1]):
        X_prep[:,i] = X[:,i] - np.mean(X[:,i])   

        Y = (Y - np.mean(Y)) / np.std(Y)

    # seperate training and testing data
    np.random.seed(129)
    msk = np.random.rand(len(X_prep)) < 0.7
    x_train = X_prep[msk,:]; x_test = X_prep[~msk,:]
    y_train = Y_prep[msk,:]; y_test = Y_prep[~msk,:]

    x_train = torch.tensor(x_train, dtype = torch.float); x_test = torch.tensor(x_test, dtype = torch.float)
    y_train = torch.tensor(y_train, dtype = torch.float); y_test = torch.tensor(y_test, dtype = torch.float)
    return x_train, x_test, y_train, y_test, torch.tensor(X_prep, dtype = torch.float), torch.tensor(Y_prep, dtype = torch.float)



## Interaction detection scores
def matric2dic(hessian, K):
    IS = {}
    for i in range(len(hessian[0])):
        for j in range(i+1, len(hessian[0])):
            tmp = 0
            interation = 'Interaction: '
            interation = interation + str(i + 1) + ' ' + str(j + 1) + ' '
            IS[interation] = hessian[i][j]
    Sorted_IS = [(k, IS[k]) for k in sorted(IS, key=IS.get, reverse=True)]
    return IS, Sorted_IS

def inputGradient(predictor, x):
    output, _ = predictor(x)
    first = torch.autograd.grad(output, x)
    return first[0].view(-1)

def inputHessian(predictor, x, device):
    Hessian = []
    output, _ = predictor(x)
    first = torch.autograd.grad(output, x, create_graph=True)
    num_gene = x.shape[1]
    for i in range(num_gene):
        gradient = torch.zeros(num_gene, dtype = torch.float).to(device)
        gradient[i] = 1.0
        second = torch.autograd.grad(first, x, grad_outputs=gradient.view(1,-1), retain_graph=True)
        Hessian.append(second[0][0].tolist())
    return Hessian

def IntegratedHessian(predictor, xi, baseline, device):
    num_gene = xi.shape[1]; m = 5; k = 5
    diff = xi - baseline
    Diff2 = torch.ger(diff.view(-1), diff.view(-1))
    PathHessian = torch.zeros([num_gene, num_gene]).to(device)
    PathGradient = torch.zeros([num_gene]).to(device)
    # discrete path integral
    for p in range(m):
        for l in range(k):
            x_eva = (baseline + (l+1) / k * (p+1) / m * diff).requires_grad_(True)
            PathHessian = PathHessian + (l+1) / k * (p+1) / m * torch.tensor(inputHessian(predictor, x_eva, device)).to(device) / (k * m)
            PathGradient = PathGradient + inputGradient(predictor, x_eva) / (k * m)
    ItgHessian = PathHessian * Diff2 + torch.diag(PathGradient * diff.view(-1))
    return ItgHessian

def GlobalIH(predictor, X, baseline, device):
    num_individual, num_gene = X.shape
    Hessian = torch.zeros([num_gene, num_gene]).to(device)
    
    for i in range(num_individual):
        x = X[i].clone().view(1,-1)
        Hessian = Hessian + torch.abs(IntegratedHessian(predictor, x, baseline, device))
    Hessian = Hessian / num_individual
    GlobalIH, topGlobalIH = matric2dic(Hessian, 10)
    return GlobalIH, topGlobalIH, Hessian

def copy_values(xi, baseline, index_set):
    tij = baseline.clone()
    for i in index_set:
        tij[i] = xi[i]
    return tij

def delta_main(predictor, xi, baseline, main_index):
    Ti = copy_values(xi, baseline, main_index).view(1,-1)
    T = copy_values(xi, baseline, []).view(1,-1)
    output_Ti, _ = predictor(Ti); output_T, _ = predictor(T); 
    return output_Ti.item() - output_T.item()

def deltaF(predictor, xi, baseline, interaction, T):
    Tij = copy_values(xi, baseline, T + interaction).view(1,-1)
    Ti = copy_values(xi, baseline, T + [interaction[0]]).view(1,-1)
    Tj = copy_values(xi, baseline, T + [interaction[1]]).view(1,-1)
    T = copy_values(xi, baseline, T).view(1,-1)
    output_Tij, _ = predictor(Tij); output_Ti, _ = predictor(Ti); output_Tj, _ = predictor(Tj); output_T, _ = predictor(T); 
    return output_Tij.item() - output_Ti.item() - output_Tj.item() + output_T.item()

def ShapleyValue(predictor, xi, baseline):
    num_gene = xi.shape[0]
    shapleyvalue = np.zeros([num_gene])
    for i in range(num_gene):
        shapleyvalue[i] = delta_main(predictor, xi, baseline, [i])
    return shapleyvalue

def ShapleyIS(predictor, xi, baseline, num_permutation):
    num_gene = xi.shape[0]
    SHAPLEYIS = np.zeros([num_gene, num_gene])
    for m in range(num_permutation):
        perm = list(np.random.permutation(num_gene)); T = []
        shapleyis = np.zeros([num_gene, num_gene])
        for i in range(len(perm)):
            if i >= 1:
                T.append(perm[i-1])
            for j in range(i+1, len(perm)):
                shapleyis[perm[i]][perm[j]] = deltaF(predictor, xi, baseline, [perm[i],perm[j]], T)
        SHAPLEYIS = SHAPLEYIS + shapleyis

    SHAPLEYIS = (SHAPLEYIS + SHAPLEYIS.T) / num_permutation
    SHAPLEYIS = SHAPLEYIS +  np.diag(ShapleyValue(predictor, xi, baseline)) 
    return SHAPLEYIS

def GlobalSIS(predictor, X, baseline, num_permutation = 10):
    num_individual, num_gene = X.shape
    Shapely = np.zeros([num_gene, num_gene])    
    for i in range(num_individual):
        x = X[i].clone()
        Shapely = Shapely + abs(ShapleyIS(predictor, x, baseline.view(-1), num_permutation))
    Shapely = Shapely / num_individual
    GlobalSIS, topGlobalSIS = matric2dic(Shapely, 10)
    return GlobalSIS, topGlobalSIS, Shapely 
