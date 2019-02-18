#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.distributions import Categorical

class NCP_Sampler():
    
    
    def __init__(self, dpmm, data):
        

        self.h_dim = dpmm.params['h_dim']
        self.g_dim = dpmm.params['g_dim']
        self.device = dpmm.params['device']
        
        assert data.shape[0] == 1              
        self.N = data.shape[1]
        
        
        if dpmm.params['model'] == 'Gauss2D':
            data = torch.tensor(data).float().to(self.device)
            assert data.shape[2] == dpmm.params['x_dim']
            data = data.view([self.N, dpmm.params['x_dim']])
            
        elif dpmm.params['model'] == 'MNIST':
            data = data.clone().detach().to(self.device)
            data = data.view([self.N, 28,28])
        
        self.hs = dpmm.h(data)            
        self.qs = self.hs

        
        self.f = dpmm.f
        self.g = dpmm.g
        
        
    def sample(self, S):

        #input S: number of samples
             
        assert type(S)==int
        cs = torch.zeros([S,self.N], dtype=torch.int64)
        previous_maxK = 1 
        nll = torch.zeros(S)
        
        with torch.no_grad():
            
            for n in range(1,self.N):
                
                Ks, _ = cs.max(dim=1)
                Ks += 1
                maxK  = Ks.max().item()
                minK  = Ks.min().item()
                
                inds = {}
                for K in range(minK,maxK+1):
                    inds[K] = Ks==K
                    
                    
                if n==1:                
        
                    self.Q = self.qs[2:,:].sum(dim=0).unsqueeze(0)     #[1, q_dim]            
                    self.Hs = torch.zeros([S, 2, self.h_dim]).to(self.device)
                    self.Hs[:,0,:] = self.hs[0,:]
                    
                    
                else:            
                    if maxK > previous_maxK:            
                        new_h = torch.zeros([S, 1, self.h_dim]).to(self.device)
                        self.Hs = torch.cat((self.Hs, new_h), dim=1) 
        
        
                    
                    self.Hs[np.arange(S), cs[:,n-1], :] += self.hs[n-1,: ]
        
        
                    if n==self.N-1:
                        self.Q = torch.zeros([1,self.h_dim]).to(self.device)    #[1, h_dim]
                        
                    else:
                        self.Q[0,:] -= self.qs[n,:]
                        
                    
                previous_maxK = maxK
                
                assert self.Hs.shape[1] == maxK +1
                
                logprobs = torch.zeros([S, maxK+1]).to(self.device)
                rQ = self.Q.repeat(S,1)
                rhn = self.hs[n,:].unsqueeze(0).repeat(S,1)
        
                    
                for k in range(maxK+1):
                    Hs2 = self.Hs.clone()
                    Hs2[:,k,:] += self.hs[n,:]                
                    
                    Hs2 = Hs2.view([S*(maxK+1), self.h_dim])                
                    gs  = self.g(Hs2).view([S, (maxK+1), self.g_dim])                
                    
                    for K in range(minK,maxK+1):
                        if k < K:
                            gs[inds[K], K:, :] = 0   
                        elif k == K and K < maxK:
                            gs[inds[K], (K+1):, :] = 0   
        
                            
                    Gk = gs.sum(dim=1)
                    
                    uu = torch.cat((Gk,rQ,rhn), dim=1)
                    logprobs[:,k] = torch.squeeze(self.f(uu))    
                    

                for K in range(minK,maxK):
                    logprobs[inds[K], K+1:] = float('-Inf')
                    
                    
        
                # Normalize
                m,_ = torch.max(logprobs,1, keepdim=True)       
                logprobs = logprobs - m - torch.log( torch.exp(logprobs-m).sum(dim=1, keepdim=True))

                probs = torch.exp(logprobs)                            
                m = Categorical(probs)
                ss = m.sample()
                cs[:,n] = ss
                nll -= logprobs[np.arange(S), ss].to('cpu')



        cs = cs.numpy()
        nll = nll.numpy()
        
        sorted_nll =np.sort(list(set(nll)))    #sort the samples in order of increasing NLL
        Z = len(sorted_nll)                    #number of distinct samples among the S samples
        probs = np.exp(-sorted_nll)
        css = np.zeros([Z,self.N], dtype=np.int32) 
        
        for i in range(Z):
            snll= sorted_nll[i]
            r = np.nonzero(nll==snll)[0][0]
            css[i,:]= cs[r,:]
        
        return css, probs

