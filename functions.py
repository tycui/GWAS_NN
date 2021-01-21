from layers import *
from prior_NNs import *
from scipy.stats import linregress

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


def permutation_data_generation(BLR, X, Y, gene_size):
    splited_X = splite_data(X, gene_size)
    BLR.eval();
    Y_pred = BLR(splited_X)
    Y_pred = Y_pred[0].detach_().numpy().reshape(-1)
    Residual = Y.numpy()[:, 0] - Y_pred
    y_pred = Y_pred.reshape(-1, 1)
    residual = Residual.reshape(-1, 1)
    np.random.seed()
    index = np.random.permutation(residual.shape[0])
    print(index[0])
    residual = residual[index, :]
    Y_perm = residual + y_pred

    for i in range(Y.shape[1]):
        Y_perm[:, i] = (Y_perm[:, i] - np.mean(Y_perm[:, i])) / np.std(Y_perm[:, i])

    return torch.tensor(Y_perm, dtype=torch.float)

# Informative priors
def compute_PVE(x, NN):
    iteration = 500; PVEs = []; var_y = 1.
    for i in range(iteration):
        PVEs.append(torch.var(NN(x)) / var_y)
    return torch.tensor(PVEs)

def informative_pve(x_prior, sparsity, gene_size, num_hidden_nodes = 100):
    Scales = np.linspace(0.1,3.,20)
    muPVE = []
    print('number of hidden nodes is '+str(num_hidden_nodes))
    for i in range(len(Scales)):
        if ((i + 1) % 2) == 0:
            print('Current percentage: %d.' % ((i + 1) *5))
        encoder_prior = Encoder_Prior(gene_size, Scales[i]);
        predictor_prior = Predictor_wide_Prior(len(gene_size), Scales[i], sparsity, num_hidden_nodes)
        SSBNN_prior = SparseBNN_Prior(encoder_prior, predictor_prior)
        muPVE.append(torch.mean(compute_PVE(x_prior, SSBNN_prior)).item())
    slope_int, intercept_int, r_value, p_value, std_err = linregress(np.log(Scales), np.log(muPVE))
    print(r_value**2)
    return slope_int, intercept_int

def informative_pve_main(x_prior, gene_size):
    Scales = np.linspace(0.1,3.,20)
    muPVE = []
    for i in range(len(Scales)):
        if ((i + 1) % 2) == 0:
            print('Current percentage: %d.' % ((i + 1) *5))
        encoder_prior = Encoder_Prior(gene_size, Scales[i]);
        predictor_prior = Main_Prior(len(gene_size), Scales[i])
        SSBNN_prior = SparseBNN_Prior(encoder_prior, predictor_prior)
        muPVE.append(torch.mean(compute_PVE(x_prior, SSBNN_prior)).item())
    slope_int, intercept_int, r_value, p_value, std_err = linregress(np.log(Scales), np.log(muPVE))
    print(r_value**2)
    return slope_int, intercept_int

def tau_estimate(target_pve, slope, intercept):
    return np.exp((np.log(target_pve) - intercept) / slope)
def mu_pve(tau, slope, intercept):
    return np.exp(slope * np.log(tau) + intercept)

# splite combined gene_data into gene level
def splite_data(combined_gene, gene_size):
    # index of each gene
    index_gene = []
    index_gene.append(0)
    for i in range(len(gene_size)):
        index_gene.append(gene_size[i] + index_gene[i])
    
    splited_gene = []
    for i in range(len(index_gene)-1):
        splited_gene.append(combined_gene[:, index_gene[i] : index_gene[i+1]])
    return splited_gene

def PTVE_test(model, x, y):
    model.eval()
    y_pred, _ = model(x)
    print('Explained variance of 65 genes is: %.5f.' % (1. - torch.sum((y - y_pred).var(dim = 0)) / torch.sum(y.var(dim = 0))))
    return (1. - (y - y_pred).var(dim = 0) / y.var(dim = 0))


## Interaction detection scores
def matric2dic(hessian):
    IS = {}
    for i in range(len(hessian[0])):
        for j in range(i+1, len(hessian[0])):
            tmp = 0
            interation = 'Interaction: '
            interation = interation + str(i + 1) + ' ' + str(j + 1) + ' '
            IS[interation] = hessian[i][j]
    Sorted_IS = [(k, IS[k]) for k in sorted(IS, key=IS.get, reverse=True)]
    return IS, Sorted_IS

def copy_values(xi, baseline, index_set):
    tij = baseline.clone()
    for i in index_set:
        tij[i] = xi[i]
    return tij

def delta_main(predictor, xi, baseline, main_index):
    Ti = copy_values(xi, baseline, main_index).view(1,-1)
    T = copy_values(xi, baseline, []).view(1,-1)
    output_Ti, _ = predictor(Ti, False); output_T, _ = predictor(T, False); 
    return output_Ti.item() - output_T.item()

def deltaF(predictor, xi, baseline, interaction, T):
    Tij = copy_values(xi, baseline, T + interaction).view(1,-1)
    Ti = copy_values(xi, baseline, T + [interaction[0]]).view(1,-1)
    Tj = copy_values(xi, baseline, T + [interaction[1]]).view(1,-1)
    T = copy_values(xi, baseline, T).view(1,-1)
    output_Tij, _ = predictor(Tij, False); output_Ti, _ = predictor(Ti, False); output_Tj, _ = predictor(Tj, False); output_T, _ = predictor(T, False); 
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
    GlobalSIS, topGlobalSIS = matric2dic(Shapely)
    return GlobalSIS, topGlobalSIS, Shapely 