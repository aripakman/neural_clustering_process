#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
from torchvision import datasets, transforms


from utils import relabel



def get_generator(params):
    
    if params['model'] == 'MNIST':
        return MNIST_generator(params)        
    elif params['model'] == 'Gauss2D':         
        return Gauss2D_generator(params)   
    else:
        raise NameError('Unknown model '+ params['model'] )



class MNIST_generator():
    
    def __init__(self,params, train=True):
        
        self.Nmin = params['Nmin']
        self.Nmax = params['Nmax']
        
        self.params=params        
        self.dataset = datasets.MNIST('../data', train=train, download=True, \
                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
        
        
        all_labels = np.zeros(len(self.dataset), dtype= np.int32)
        for i in range(len(self.dataset)):
            all_labels[i] = self.dataset[i][1].item()
            
        
        self.label_data = {}
        for i in range(10):
            print('Processing label: ', i)
            label_inds = np.nonzero(all_labels == i)[0]            
            S = label_inds.shape[0]
            self.label_data[i] =torch.zeros([S,28,28])
            for s in range(S):
                self.label_data[i][s,:,:] = self.dataset[label_inds[s]][0][0,:,:]

        
    def generate(self,N=None, batch_size=1):
        
        K = 11
        while K>10:
            clusters, N, K = generate_CRP(self.params, N=N)
        
        data = torch.zeros([batch_size,N,28,28])
        
        cumsum = np.cumsum(clusters)
        
        for i in range(batch_size):
            labels = np.random.choice(10,size=K, replace = False )  #this is a sample from the 'base measure' for each cluster
            for k in range(K):
                l = labels[k]
                nk = clusters[k+1]
                inds = np.random.choice(self.label_data[l].shape[0],size=nk, replace = False )                
                data[i, cumsum[k]:cumsum[k+1], :,: ] = self.label_data[l][inds,:,:]

        cs = np.empty(N, dtype=np.int32)        
        for k in range(K):
            cs[cumsum[k]:cumsum[k+1]]= k
        
        
            
        arr = np.arange(N)
        np.random.shuffle(arr)
        cs = cs[arr]        
        data = data[:,arr,:,:]
        cs = relabel(cs)
        
        
        return data, cs, clusters, K    
        





class Gauss2D_generator():
    
    def __init__(self,params):
        self.params = params
        

    def generate(self,N=None, batch_size=1):        
        
        lamb = self.params['lambda']
        sigma = self.params['sigma']
        x_dim = self.params['x_dim']    
        
        clusters, N, num_clusters = generate_CRP(self.params, N=N)
            
        
        cumsum = np.cumsum(clusters)
        data = np.empty([batch_size, N, x_dim])
        cs =  np.empty(N, dtype=np.int32)
        
        for i in range(num_clusters):
            mu= np.random.normal(0,lamb, size = [x_dim*batch_size,1])
            samples= np.random.normal(mu,sigma, size=[x_dim*batch_size,clusters[i+1]] )
            
            samples = np.swapaxes(samples.reshape([batch_size, x_dim,clusters[i+1]]),1,2)        
            data[:,cumsum[i]:cumsum[i+1],:]  = samples
            cs[cumsum[i]:cumsum[i+1]]= i+1
            
        #%shuffle the assignment order
        arr = np.arange(N)
        np.random.shuffle(arr)
        cs = cs[arr]
        
        data = data[:,arr,:]
        
        # relabel cluster numbers so that they appear in order 
        cs = relabel(cs)
        
        #normalize data 
        #means = np.expand_dims(data.mean(axis=1),1 )    
        medians = np.expand_dims(np.median(data,axis=1),1 )    
        
        data = data-medians
        #data = 2*data/(maxs-mins)-1        #data point are now in [-1,1]
    
        return data, cs, clusters, num_clusters
        
            





def generate_CRP(params,N, no_ones=False):
    
    alpha = params['alpha']   #dispersion parameter of the Chinese Restaurant Process
    keep = True
    
    
    while keep:
        if N is None or N==0:
            N = np.random.randint(params['Nmin'],params['Nmax'])
            
                
        clusters = np.zeros(N+2)
        clusters[0] = 0
        clusters[1] = 1      # we start filling the array here in order to use cumsum below 
        clusters[2] = alpha
        index_new = 2
        for n in range(N-1):     #we loop over N-1 particles because the first particle was assigned already to cluster[1]
            p = clusters/clusters.sum()
            z = np.argmax(np.random.multinomial(1,p))
            if z < index_new:
                clusters[z] +=1
            else:
                clusters[index_new] =1
                index_new +=1
                clusters[index_new] = alpha
        
        clusters[index_new] = 0 
        clusters = clusters.astype(np.int32)
        
        if no_ones:
            clusters= clusters[clusters!=1]
        N = int(np.sum(clusters))
        keep = N==0                       
        
        
    K = np.sum(clusters>0)

    return clusters, N, K
