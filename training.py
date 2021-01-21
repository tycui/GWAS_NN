from sklearn.model_selection import KFold
from functions import *
import torch.optim as optim
import copy

def training_CV(mlp, x, y, gene_size, n_splits = 5, learning_rate=0.001, batch_size=50, num_epoch=1000):
    kf = KFold(n_splits)
    kf.get_n_splits(x)
    validation_PVE = []
    i = 1
    for train_index, val_index in kf.split(x):
        mlp_i = copy.deepcopy(mlp)
        print('Fold ' + str(i))
        xtrain, xval = x[train_index], x[val_index]
        ytrain, yval = y[train_index], y[val_index]
        
        training(mlp_i, xtrain, ytrain, xval, yval, learning_rate, batch_size, num_epoch)
        
        splited_x_val = splite_data(xval, gene_size)
        val_pve = PTVE_test(mlp_i, splited_x_val, yval)
        
        validation_PVE.append(val_pve.item())
        i = i + 1
    pve_mean = np.mean(validation_PVE)
    print('Mean of PVE is: ' + str(pve_mean))
    return pve_mean

def training(model, x, y, x_test, y_test,learning_rate=0.001, batch_size=50, num_epoch=1000):
    gene_size = model.encoder.dim_gene
    
    parameters = set(model.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate, eps=1e-3)
    criterion = nn.MSELoss()

    train_errors = []
    test_errors = []

    num_data, _ = x.shape
    _, num_output = y.shape
    data = torch.cat((x, y), 1)

    for epoch in range(num_epoch):
        # permuate the data
        data_perm = data[torch.randperm(len(data))]
        x = data_perm[:, :-num_output]
        y = data_perm[:, -num_output:]
        for index in range(int(num_data / batch_size)):
            # data comes in

            inputs = x[index * batch_size: (index + 1) * batch_size]
            labels = y[index * batch_size: (index + 1) * batch_size]
      
            ## split the combined gene
            splited_inputs = splite_data(inputs, gene_size)
            # initialize the gradient of optimizer
            optimizer.zero_grad()
    
            model.train()
            output, kl = model(splited_inputs)

            # calculate the training loss
            loss = criterion(labels, output) +  kl / num_data

            # backpropogate the gradient
            loss.backward()

            # optimize with SGD
            optimizer.step()
            
        # validation loss
        model.eval()
        
        # splite the training data
        splited_x = splite_data(x, gene_size)
        output_x_train, kl = model(splited_x)

        # splite the testing data
        splited_x_test = splite_data(x_test, gene_size)
        output_x_test, kl = model(splited_x_test)
        
        train_errors.append(criterion(output_x_train, y).detach())
        test_errors.append(criterion(output_x_test, y_test).detach())

        if (epoch % 100) == 0:
            print('EPOACH %d: TRAIN LOSS: %.4f; KL REG: %.4f; TEST LOSS IS: %.5f.' % (epoch + 1, train_errors[epoch], kl,  test_errors[epoch]))
        
    return train_errors, test_errors
