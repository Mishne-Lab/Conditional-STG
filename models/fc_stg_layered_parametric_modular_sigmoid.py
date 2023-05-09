# -*- coding: utf-8 -*-
"""
Created on Mon May 17 18:11:59 2021

@author: srist
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import captum.module

__all__ = ['FC_STG_Layered_Param_modular_model_sigmoid','fc_stg_layered_param_modular_model_sigmoid']

class FeatureSelector(nn.Module):
    def __init__(self, hyper_input_dim,hyper_output_dim,hyper_hidden_dim, sigma=0.5,non_param_stg=False,train_sigma=False):
        super(FeatureSelector, self).__init__()
        # hyper_input_dim: input dimension for the hyper network i.e dimensionality of contextual input
        # hyper_output_dim: output dimension for the hyper network i.e dimensionality of explanatory features
        # hyper_hidden_dim: dimensionals of hidden layers in the hyper network
        self.non_param_stg = non_param_stg
        self.hyper_output_dim = hyper_output_dim
        self.train_sigma = train_sigma
#         print(self.non_param_stg)
        # Define hyper network 
        if self.non_param_stg:
            self.mu = torch.nn.Parameter(0.01*torch.randn(self.hyper_output_dim, ), requires_grad=True)
        else:
            self.hyper_dense_layers = nn.ModuleList()
            if len(hyper_hidden_dim):
                self.hyper_dense_layers.append(nn.Linear(hyper_input_dim, hyper_hidden_dim[0]))
                self.hyper_dense_layers.append(nn.ReLU())
                for i in range(len(hyper_hidden_dim)-1):
                    self.hyper_dense_layers.append(nn.Linear(hyper_hidden_dim[i], hyper_hidden_dim[i+1]))
                    self.hyper_dense_layers.append(nn.ReLU())
                self.hyper_dense_layers.append(nn.Linear(hyper_hidden_dim[-1], hyper_output_dim))
            else:
                self.hyper_dense_layers.append(nn.Linear(hyper_input_dim, hyper_output_dim))

            self.hyper_dense_layers.append(nn.Sigmoid())
        
        self.noise = torch.randn(hyper_output_dim,) 
        self.sigma = nn.Parameter(torch.tensor([sigma]), requires_grad=train_sigma)

    def forward(self, prev_x, B, axis=2):
        # compute the feature importance given B
        stochastic_gate = self.get_feature_importance(B)
        # mask the input with feature importance
        if self.non_param_stg:
            new_x = prev_x * stochastic_gate[None,:]
        else:
            new_x = prev_x * stochastic_gate[:,:]
        return new_x
    
    def get_feature_importance(self,B):
        # compute feature importance given contextual input (B)
        if not self.non_param_stg:
            
            self.mu = B
            for dense in self.hyper_dense_layers:
                self.mu = dense(self.mu)
        if self.train_sigma:
            self.sigma = nn.ReLU(self.sigma)+0.01
        z = self.mu + (self.sigma)*self.noise.normal_()*self.training 
        stochastic_gate = self.hard_sigmoid(z)
        return stochastic_gate
        
    
    def hard_sigmoid(self, x):
        return torch.clamp(x, 0.0, 1.0)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / self.sigma*math.sqrt(2))) 

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self

class FC_STG_Layered_Param_modular_model_sigmoid(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, param_dim,hyper_hidden_dim, dropout, sigma, stg, classification, include_B_in_input=False,non_param_stg=False,train_sigma=False):
        super().__init__()
        # input_dim: input dimension for the network i.e dimensionality of explanatory features
        # hidden_dim: dimensionals of hidden layers in the network
        # output_dim: output dimension for the network
        # param_dim: input dimension for the hyper network i.e dimensionality of contextual input
        # hyper_hidden_dim: dimensionals of hidden layers in the hyper network
        # dropout: right now dropout is not being used in the network
        # sigma: sigma of gaussian distribution for STG gates
        # STG: True/False - enable/disable feature selection using parametric STG
        # classification: True/False - True: binary classification (adds sigmoid in the last layer), False: regression
        #                 ToDo: Didn't test this function after generalizing this to multiclass classification by adding softmax.
        
        
        self.stg = stg
        self.hidden_dim = hidden_dim
        self.sigma = sigma
        self.include_B_in_input = include_B_in_input
        self.non_param_stg = non_param_stg
        self.train_sigma = train_sigma
        # Define parametric-STG
        if self.stg:
            self.gates = FeatureSelector(param_dim, input_dim, hyper_hidden_dim, self.sigma, self.non_param_stg, self.train_sigma)
            self.reg = self.gates.regularizer 
#             self.sigma = self.gates.sigma
        # Define network architecture
        self.dense_layers = nn.ModuleList()
        if len(hidden_dim):
            self.dense_layers.append(nn.Linear(input_dim, hidden_dim[0]))
            self.dense_layers.append(nn.ReLU())
            for i in range(len(hidden_dim)-1):
                self.dense_layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                self.dense_layers.append(nn.ReLU())
            self.dense_layers.append(nn.Linear(hidden_dim[-1], output_dim))
        else:
            self.dense_layers.append(nn.Linear(input_dim, output_dim))
            
        if classification:
            if output_dim==1:
                self.dense_layers.append(nn.Sigmoid())
            else:
                self.dense_layers.append(nn.Softmax())
        
        
    def forward(self, x, B):
        if self.stg:
            x = self.gates(x, B)
        elif self.include_B_in_input:
            x = torch.hstack((B,x))

        for dense in self.dense_layers:
            x = dense(x)

        return x
    
def fc_stg_layered_param_modular_model_sigmoid(input_dim,hidden_dim=[10],output_dim=1, param_dim=1, hyper_hidden_dim=[500], dropout=0, sigma=0.5, stg=True,classification=True,include_B_in_input=False,non_param_stg=False,train_sigma=False):
    
    model = FC_STG_Layered_Param_modular_model_sigmoid(input_dim, hidden_dim, output_dim, param_dim, hyper_hidden_dim, dropout, sigma, stg,classification,include_B_in_input=include_B_in_input,non_param_stg=non_param_stg,train_sigma=train_sigma)
    return model

