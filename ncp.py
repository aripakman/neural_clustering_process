#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F





from utils import relabel

class MNIST_encoder(nn.Module):
    
    def __init__(self, params):
        
        super(MNIST_encoder, self).__init__()
        
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 256)
        self.fc2 = nn.Linear(256, params['h_dim'])

    def forward(self, x):
        
        x = x.unsqueeze(1)   # add channel index
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x




class Mixture_Gaussian_encoder(nn.Module):
    
    def __init__(self, params):
        
        super(Mixture_Gaussian_encoder, self).__init__()
        
        H = params['H_dim']
        self.h_dim = params['h_dim']        
        self.x_dim = params['x_dim']
        
        self.h = torch.nn.Sequential(
                torch.nn.Linear(self.x_dim, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),    
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, self.h_dim),
                )

    def forward(self, x):
        
        return self.h(x)







class NeuralClustering(nn.Module):
    
    
    def __init__(self, params):
        
        super(NeuralClustering, self).__init__()
        
        self.params = params
        self.previous_n = 0
        self.previous_K=1
        
        self.g_dim = params['g_dim']
        self.h_dim = params['h_dim']
        H = params['H_dim']        
        
        self.device = params['device']


        if self.params['model'] == 'Gauss2D':
            self.h = Mixture_Gaussian_encoder(params)         
        elif self.params['model'] == 'MNIST':
            self.h = MNIST_encoder(params)         
        else:
            raise NameError('Unknown model '+ self.params['model'])
    
        
        self.g = torch.nn.Sequential(
                torch.nn.Linear(self.h_dim, H),
                torch.nn.PReLU(),    
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),    
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, self.g_dim),
                )
        
        self.f = torch.nn.Sequential(
                torch.nn.Linear(self.g_dim +2*self.h_dim, H),
                torch.nn.PReLU(),    
                torch.nn.Linear(H, H),                
                torch.nn.PReLU(),    
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, 1, bias=False),
                )
        

        
    def forward(self,data, cs, n):
             
        # n =1,2,3..N
        # elements with index below or equal to n-1 are already assigned
        # element with index n is to be assigned. 
        # the elements from the n+1-th are not assigned

        assert(n == self.previous_n+1)
        self.previous_n = self.previous_n + 1 

        K = len(set(cs[:n]))  #num of already _assigned_clusters
        # K is the number of distinct classes in [0:n]          

        if n==1:
            
            self.batch_size = data.shape[0]
            self.N = data.shape[1]
            assert (cs==relabel(cs)).all()

            
            
            if self.params['model'] == 'Gauss2D':
                # The data comes as a numpy vector
                data = torch.tensor(data).float().to(self.device)                    
                data = data.view([self.batch_size*self.N, self.params['x_dim']])

            elif self.params['model'] == 'MNIST':
                # The data comes as a torch tensor, we just move it to the device 
                data = data.to(self.device)    
                data = data.view([self.batch_size*self.N, 28,28])
                                
            
            self.hs = self.h(data).view([self.batch_size,self.N, self.h_dim])            
            self.Q = self.hs[:,2:,].sum(dim=1)     #[batch_size,h_dim]
            
            self.Hs = torch.zeros([self.batch_size, 1, self.h_dim]).to(self.device)
            self.Hs[:,0,:] = self.hs[:,0,:]
            
            
        else:            
            if K == self.previous_K:            
                self.Hs[:, cs[n-1], :] += self.hs[:,n-1,:]
            else:
                self.Hs = torch.cat((self.Hs,self.hs[:,n-1,:].unsqueeze(1)), dim=1)


            if n==self.N-1:
                self.Q = torch.zeros([self.batch_size,self.h_dim]).to(self.device)    #[batch_size,h_dim]
                self.previous_n = 0
                
            else:
                self.Q -= self.hs[:,n,]
                
            
        self.previous_K = K
        
        assert self.Hs.shape[1] == K
        
        logprobs = torch.zeros([self.batch_size, K+1]).to(self.device)
            
        # loop over the K existing clusters for datapoint n to join
        for k in range(K):
            Hs2 = self.Hs.clone()
            Hs2[:,k,:] += self.hs[:,n,:]
            
            
            Hs2 = Hs2.view([self.batch_size*K, self.h_dim])                
            gs  = self.g(Hs2).view([self.batch_size, K, self.g_dim])
            Gk = gs.sum(dim=1)   #[batch_size,g_dim]

            uu = torch.cat((Gk,self.Q,self.hs[:,n,:]), dim=1)  #prepare argument for the call to f()
            logprobs[:,k] = torch.squeeze(self.f(uu))    
            
        
        # consider datapoint n creating a new cluster
        Hs2 = torch.cat((self.Hs,self.hs[:,n,:].unsqueeze(1)), dim=1)    
        Hs2 = Hs2.view([self.batch_size*(K+1), self.h_dim])                
    
        gs  = self.g(Hs2).view([self.batch_size, K+1, self.g_dim])
    
        Gk = gs.sum(dim=1)
    
        uu = torch.cat((Gk,self.Q,self.hs[:,n,:]), dim=1)   #prepare argument for the call to f()
        logprobs[:,K] = torch.squeeze(self.f(uu))    


        # Normalize
        m,_ = torch.max(logprobs,1, keepdim=True)        #[batch_size,1]
        logprobs = logprobs - m - torch.log( torch.exp(logprobs-m).sum(dim=1, keepdim=True))

        return logprobs



