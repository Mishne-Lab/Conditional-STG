import os,glob
import math
import numpy as np
import pandas as pd
import scipy.io
import pickle
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import mean_squared_error as performance_metric
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils

import matplotlib.pyplot as plt
import argparse


device = 0
gpu = torch.device('cuda:{}'.format(device))
print(gpu)


input_dim,param_dim,output_dim = 0,0,0

def relu(x):
    return np.maximum(0, x)


def get_raw_data(n_samples,data="train",cv_cur=0,dim_x=30):
    # Set random seed for reproducibility
    if data=="train":
        np.random.seed(0+10*cv_cur)
    elif data=="val":
        np.random.seed(1+10*cv_cur)
    elif data=="test":
        np.random.seed(2+10*cv_cur)

    # Dimension of X
    dim_x = dim_x

    # Generate X
    X = np.random.randn(n_samples, dim_x) 

    # Initialize Y
    Y = np.zeros((n_samples, 1))

    # Generate Z
    Z = np.random.randint(0, 4, (n_samples, 1))

    # Compute Y according to Z and the rules
    for i in range(n_samples):
        if Z[i, 0] == 0:
            Y[i, 0] = relu(0.5*X[i, 0] + X[i, 1])
        elif Z[i, 0] == 1:
            Y[i, 0] = relu(X[i, 0] - 0.5*X[i, 1])
        elif Z[i, 0] == 2:
            Y[i, 0] = relu(0.5*X[i, 2] + X[i, 3])
        elif Z[i, 0] == 3:
            Y[i, 0] = relu(X[i, 2] - 0.5*X[i, 3])

    # Now, X, Y, and Z are NumPy arrays containing the data.
    # X: Input features
    # Y: Outputs
    # Z: Determines the relationship between X and Y
    return X,np.eye(4)[Z.flatten()],Y


def get_data(cv_cur,dim_x=30):
    
    X_train,T_train,y_train = get_raw_data(5000,data="train",cv_cur=cv_cur,dim_x=dim_x)
    X_val,T_val,y_val = get_raw_data(5000,data="val",cv_cur=cv_cur,dim_x=dim_x)
    X_test,T_test,y_test = get_raw_data(5000,data="test",cv_cur=cv_cur,dim_x=dim_x)
    
    input_dim,param_dim,output_dim = X_train.shape[-1],T_train.shape[-1],1
    
    trainset = data_utils.TensorDataset(torch.tensor(X_train), torch.tensor(y_train[:,None]),torch.tensor(T_train))
    valset = data_utils.TensorDataset(torch.tensor(X_val), torch.tensor(y_val[:,None]),torch.tensor(T_val))
    testset = data_utils.TensorDataset(torch.tensor(X_test), torch.tensor(y_test[:,None]),torch.tensor(T_test))
    
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(valset, batch_size=10000, shuffle=True,drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=True,drop_last=False)
    return train_dataloader,val_dataloader,test_dataloader,input_dim,param_dim,output_dim


train_dataloader,val_dataloader,test_dataloader,x_dim,z_dim,y_dim = get_data(0,dim_x=30)



class FeatureSelector(nn.Module):
    def __init__(self, hyper_input_dim,hyper_output_dim,hyper_hidden_dim, sigma=0.5,non_param_stg=False,train_sigma=False, is_cen=False):
        super(FeatureSelector, self).__init__()
        # hyper_input_dim: input dimension for the hyper network i.e dimensionality of contextual input
        # hyper_output_dim: output dimension for the hyper network i.e dimensionality of explanatory features
        # hyper_hidden_dim: dimensionals of hidden layers in the hyper network
        self.non_param_stg = non_param_stg
        self.hyper_output_dim = hyper_output_dim
        self.train_sigma = train_sigma
        self.is_cen = is_cen

        # Define hyper network 
        if self.non_param_stg:
            self.mu = torch.nn.Parameter(0.01*torch.randn(self.hyper_output_dim, )+0.5, requires_grad=True)
        else:
            self.hyper_dense_layers = nn.ModuleList()
            if len(hyper_hidden_dim):
                self.hyper_dense_layers.append(nn.Linear(hyper_input_dim, hyper_hidden_dim[0]))
                self.hyper_dense_layers.append(nn.ReLU())
                for i in range(len(hyper_hidden_dim)-1):
                    self.hyper_dense_layers.append(nn.Linear(hyper_hidden_dim[i], hyper_hidden_dim[i+1]))
                    self.hyper_dense_layers.append(nn.ReLU())
                self.hyper_dense_layers.append(nn.Linear(hyper_hidden_dim[-1], hyper_output_dim))
                self.hyper_last_weight_layer = nn.Linear(hyper_hidden_dim[-1], hyper_output_dim)
            else:
                self.hyper_dense_layers.append(nn.Linear(hyper_input_dim, hyper_output_dim))
                self.hyper_last_weight_layer = nn.Linear(hyper_input_dim, hyper_output_dim)
            self.hyper_dense_layers.append(nn.Sigmoid())
        
        self.noise = torch.randn(hyper_output_dim,) 
        self.sigma = nn.Parameter(torch.tensor([sigma]), requires_grad=train_sigma)

    def forward(self, prev_x, B, axis=2):
        # compute the feature importance given B
        stochastic_gate, weights = self.get_feature_importance(B)
        
        # mask the input with feature importance
        if self.non_param_stg:
            new_x = prev_x * stochastic_gate[None,:]
        else:
            new_x = prev_x * stochastic_gate[:,:]
            new_x = new_x * weights
        return new_x
    
    def get_feature_importance(self,B=None):
        # compute feature importance given contextual input (B)
        if not self.non_param_stg:
            self.mu = B
            self.weights = B
            for layer_idx,dense in enumerate(self.hyper_dense_layers):
                self.mu = dense(self.mu)
                if layer_idx<=len(self.hyper_dense_layers)-3:
                    self.weights = dense(self.weights)
                elif layer_idx==len(self.hyper_dense_layers)-2:
                    self.weights = self.hyper_last_weight_layer(self.weights)
                    
        if self.train_sigma:
            self.sigma = nn.ReLU(self.sigma)+0.01
            
        if self.is_cen:
            stochastic_gate = self.mu
            self.weights = None
        else:
            z = self.mu + (self.sigma)*self.noise.normal_()*self.training 
            stochastic_gate = self.hard_sigmoid(z)

        return stochastic_gate,self.weights
        
    
    def hard_sigmoid(self, x):
        return torch.clamp(x, 0.0, 1.0)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / self.sigma*math.sqrt(2))) 

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self


class LinearModel(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim,hyper_hidden_dim):
        super(LinearModel, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        
        self.gates = FeatureSelector(z_dim, x_dim, hyper_hidden_dim)
        self.reg = self.gates.regularizer 
        
    def forward(self, x, z):
        x = self.gates(x, z)
        out = torch.sum(x , dim=1)
        out = nn.ReLU()(out)
        return out

    



metric="r2_score"

if metric=="r2_score":
    from sklearn.metrics import r2_score as performance_metric
    comp_func = np.max
    fill_val = -np.inf
elif metric=="mean_squared_error":
    from sklearn.metrics import mean_squared_error as performance_metric
    comp_func = np.min
    fill_val = np.inf
elif metric=="RMSE":
    from sklearn.metrics import mean_squared_error as mean_squared_error
    def performance_metric(y_true,y_pred):
        return np.sqrt(mean_squared_error(y_true,y_pred))
    comp_func = np.min
    fill_val = np.inf

ML_model_name = "cSTG_param_prediction_model"

epochs = 2000
dropout = 0


hyper_hidden_dim = []
hidden_dims = []

for dim_x in [25]: 
    for learning_rate in [1e-2,5e-3,1e-3,5e-4,1e-4]: 
        for stg_regularizer in [10,5,4,3,2,1,0.75,5e-1,1e-1,5e-2,1e-2]:
            for rand_init_seed in range(1):
                for val_cur in range(5):

                    stg = True

                    print("rand_init_seed:{}, val_cur:{}".format(rand_init_seed,val_cur))

                    train_dataloader,val_dataloader,test_dataloader,input_dim,param_dim,output_dim = get_data(val_cur,dim_x=dim_x)
                    x_dim, z_dim, y_dim = input_dim,param_dim,output_dim


                    torch.manual_seed(rand_init_seed)
                    np.random.seed(rand_init_seed)

                    add_foldername = ""


                    root_fname = "./Trained_Models/XOR2" 
                    if not os.path.exists(root_fname):
                        os.mkdir(root_fname)

                    add_name = ""
                    add_name += "_"+"_".join(np.array([input_dim]+hidden_dims+[output_dim]).astype(str))

                    add_name += "_hyper_"+"_".join(np.array([param_dim]+hyper_hidden_dim+[input_dim]).astype(str))

                    model_path = "{}/{}_lr_{}_stg_lr_{}{}_init_{}_val_seed_{}_stg_{}.model".format(root_fname,ML_model_name,str(learning_rate).replace(".", "_"),str(stg_regularizer).replace(".","_"),add_name,val_cur,rand_init_seed,stg)

                    loss_path = model_path.replace("model","mat")

                    if os.path.exists(model_path):
                        continue

                 
                    model = LinearModel(x_dim, z_dim, y_dim, hyper_hidden_dim)
                    model = model.to(gpu).float()

                    torch.save(model.state_dict(), model_path)
                    print('Model saved!')

                    criterion = nn.MSELoss() 
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

                    train_metric_array = [fill_val]
                    val_metric_array = [fill_val]
                    test_metric_array = [fill_val]
                    train_loss_array = [0]
                    val_loss_array = [0]
                    test_loss_array = [0]

                    epoch_itr = tqdm(range(epochs), leave=True)
                    for epoch in epoch_itr:


                        train_loss = 0
                        val_loss = 0
                        train_count = 0
                        val_count = 0
                        train_metric_batch = []
                        for batch, (input, target, B) in enumerate(train_dataloader):

                            model.train()

                            input = input.to(gpu).float()
                            target = target.to(gpu).float().flatten()
                            B = B.to(gpu).float()
                            output = model(input,B)
                            output = torch.squeeze(output)
                            train_metric_batch.append(performance_metric(target.detach().cpu().numpy(),output.float().detach().cpu().numpy()))
                            loss = criterion(output, torch.squeeze(target)) #.long()
                            if stg:
                                temp = model.gates.mu
                                grads = temp.squeeze().cpu().detach().numpy()
                                stg_loss = torch.mean(torch.abs(model.reg((model.gates.mu)))) 
                                loss += stg_regularizer*stg_loss

                            optimizer.zero_grad()   
                            with torch.autograd.set_detect_anomaly(True):
                                loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                            optimizer.step()

                            loss_value = loss.item()
                            train_loss += loss.item()
                            train_count += len(input)


                        train_count = batch+1
                        train_metric = np.mean(train_metric_batch)
                        train_metric_array.append(train_metric)
                        train_loss_array.append(train_loss/train_count)

                        model.eval()

                        val_metric_batch = []
                        val_loss = 0
                        for batch, (input, target, B) in enumerate(val_dataloader):
                            input = input.to(gpu).float()
                            target = target.to(gpu).float().flatten()
                            B = B.to(gpu).float()
                            output = model(input,B)
                            output = torch.squeeze(output)

                            val_metric_batch.append(performance_metric(target.detach().cpu().numpy(),output.float().detach().cpu().numpy()))
                            loss = criterion(output, torch.squeeze(target))
                            if stg:
                                stg_loss = torch.mean(model.reg((model.gates.mu))) 
                                loss += stg_regularizer*stg_loss
                            val_loss += loss.item()
                            val_count += len(input)
                        val_count = batch+1
                        val_metric = np.mean(val_metric_batch)
                        val_metric_array.append(val_metric)
                        val_loss_array.append(val_loss/val_count)

                        test_metric_batch = []
                        test_loss = 0
                        test_count = 0
                        for batch, (input, target, B) in enumerate(test_dataloader):
                            input = input.to(gpu).float()
                            target = target.to(gpu).float().flatten()
                            B = B.to(gpu).float()
                            output = model(input,B)
                            output = torch.squeeze(output)

                            test_metric_batch.append(performance_metric(target.detach().cpu().numpy(),output.float().detach().cpu().numpy()))
                            loss = criterion(output, torch.squeeze(target))
                            if stg:
                                stg_loss = torch.mean(model.reg((model.gates.mu))) 
                                loss += stg_regularizer*stg_loss
                            test_loss += loss.item()
                            test_count += len(input)
                        test_count = batch+1
                        test_metric = np.mean(test_metric_batch)
                        test_metric_array.append(test_metric)
                        test_loss_array.append(test_loss/test_count)
                        if epoch%50==0:
                            print("train metric:{}, val metric:{}, test metric:{}".format(train_metric,val_metric,test_metric))

                        if (val_metric == comp_func(val_metric_array)):
                            torch.save(model.state_dict(), model_path)
                            if epoch%100==0:
                                print('Model saved! Validation metric improved from {:3f} to {:3f}'.format(comp_func(val_metric_array[:-1]), comp_func(val_metric_array)))
                        scipy.io.savemat(loss_path, {'train_loss_array': train_loss_array,'val_loss_array': val_loss_array,'test_loss_array': test_loss_array,'train_metric': train_metric_array,'val_metric': val_metric_array,'test_metric': test_metric_array})

