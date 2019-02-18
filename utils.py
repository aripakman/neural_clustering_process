#!/usr/bin/env python3
# -*- coding: utf-8 -*-



def get_parameters(model):

    params = {}
        
    if model == 'MNIST':
        params['model'] = 'MNIST'
        params['alpha'] = .7
        params['Nmin'] = 5
        params['Nmax'] = 100
        params['h_dim'] = 256
        params['g_dim'] = 512
        params['H_dim'] = 128
        
    elif model == 'Gauss2D':         
        params = {}
        params['model'] = 'Gauss2D'
        params['alpha'] = .7
        params['sigma'] = 1        # std for the Gaussian noise around the cluster mean 
        params['lambda'] = 10      # std for the Gaussian prior that generates de centers of the clusters
        params['Nmin'] = 5
        params['Nmax'] = 100
        params['x_dim'] = 2
        params['h_dim'] = 256
        params['g_dim'] = 512
        params['H_dim'] = 128
        
    else:
        raise NameError('Unknown model '+ model)
        
    return params



def relabel(cs):
    cs = cs.copy()
    d={}
    k=0
    for i in range(len(cs)):
        j = cs[i]
        if j not in d:
            d[j] = k
            k+=1
        cs[i] = d[j]        

    return cs

        






    



