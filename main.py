#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import time
import os
import torch

from ncp import NeuralClustering
from data_generators import get_generator
from plot_functions import plot_avgs, plot_samples_2d, plot_samples_MNIST
from utils import relabel, get_parameters



def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
        
    model = args.model
    params = get_parameters(model)
    params['device'] = torch.device("cuda:0" if args.cuda else "cpu")

    
    dpmm = NeuralClustering(params).to(params['device'])
    data_generator = get_generator(params)
    

    #define containers to collect statistics
    losses= []       # NLLs    
    accs =[]         # Accuracy of the classification prediction
    perm_vars = []   # permutation variance

    
    it=0      # iteration counter
    learning_rate = 1e-4
    weight_decay = 0.01
    optimizer = torch.optim.Adam( dpmm.parameters() , lr=learning_rate, weight_decay = weight_decay)
    
    perms = 6  # Number of permutations for each mini-batch. 
               # In each permutation, the order of the datapoints is shuffled.         
               
    batch_size = args.batch_size
    max_it = args.iterations
    
    dpmm = dpmm.train()

    if params['model'] == 'Gauss2D':
        if not os.path.isdir('saved_models/Gauss2D'):
            os.makedirs('saved_models/Gauss2D')
        if not os.path.isdir('figures/Gauss2D'):
            os.makedirs('figures/Gauss2D')


    elif params['model'] == 'MNIST':
        if not os.path.isdir('saved_models/MNIST'):
            os.makedirs('saved_models/MNIST')
        if not os.path.isdir('figures/MNIST'):
            os.makedirs('figures/MNIST')
    
    
    
    
    end_name = params['model']    
    learning_rates = {1200:5e-5, 2200:1e-5}
    
    
    
      
    t_start = time.time()
    itt = it
    while True:
            
    
            it += 1
    
            if it == max_it:
                break
            
            
            if it % args.plot_interval == 0:
                
                torch.cuda.empty_cache()                
                plot_avgs(losses, accs, perm_vars, 50, save_name='./figures/train_avgs_' + end_name + '.pdf')            
    
                if params['model'] == 'Gauss2D':
                    fig_name = './figures/Gauss2D/samples_2D_' + str(it) + '.pdf'
                    print('\nCreating plot at ' + fig_name + '\n')
                    plot_samples_2d(dpmm, data_generator, N=100, seed=it, save_name=fig_name)    
                    
                    
                elif params['model'] == 'MNIST':
                    fig_name = './figures/MNIST/samples_MNIST_' + str(it) + '.pdf'
                    print('\nCreating plot at ' + fig_name + '\n')
                    plot_samples_MNIST(dpmm, data_generator, N=20, seed=it, save_name= fig_name)
    
                
            if it % 100 == 0:
                if 'fname' in vars():
                    os.remove(fname)
                dpmm.params['it'] = it
                fname = 'saved_models/'+ end_name + '/'+ end_name +'_' + str(it) + '.pt'            
                torch.save(dpmm,fname)
    
                
            if it in learning_rates:            
                optimizer = torch.optim.Adam( dpmm.parameters() , lr=learning_rates[it], weight_decay = weight_decay)
    
    
            data, cs, clusters, K = data_generator.generate(None, batch_size)    
            N=data.shape[1]
            
            loss_values = np.zeros(perms)
            accuracies = np.zeros([N-1,perms])
            
            
            # The memory requirements change in each iteration according to the random values of N and K.
            # If both N and K are big, an out of memory RuntimeError exception might be raised.
            # When this happens, we capture the exception, reduce the batch_size to 3/4 of its value, and try again.
            
            while True:
                try:
    
                    loss = 0        
                    
                    for perm in range(perms):
                        arr = np.arange(N)
                        np.random.shuffle(arr)
                        cs = cs[arr]
                        data= data[:,arr,:]    
            
                        cs = relabel(cs)    # this makes cluster labels appear in cs[] in increasing order
                                
            
                        this_loss=0
                        dpmm.previous_n=0            
                        
                        for n in range(1,N):                
                        # points up to (n-1) are already assigned, the point n is to be assigned
                        
                            logprobs  = dpmm(data,cs,n)                
                            c = cs[n] 
                            accuracies[n-1, perm] = np.sum(np.argmax(logprobs.detach().to('cpu').numpy(),axis=1)==c)/logprobs.shape[0]            
                            
                            
                            this_loss -= logprobs[:,c].mean()
            
            
                        loss_values[perm] = this_loss.item()/N
                        loss += this_loss
                        
                        
                    
                    perm_vars.append(loss_values.var())
                    losses.append(loss.item()/N)
                    accs.append(accuracies.mean())
            
            
                    optimizer.zero_grad()    
                    loss.backward() 
                    optimizer.step()
            
    
                    print('{0:4d}  N:{1:2d}  K:{2}  Mean NLL:{3:.3f}   Mean Acc:{4:.3f}   Mean Variance: {5:.7f}  Mean Time/Iteration: {6:.1f}'\
                          .format(it, N, K , np.mean(losses[-50:]), np.mean(accs[-50:]), np.mean(perm_vars[-50:]), (time.time()-t_start)/(it - itt)    ))    
    
                    break
    
                except RuntimeError:
                    bsize = int(.75*data.shape[0])
                    if bsize > 2:
                        print('RuntimeError handled  ', 'N:', N, ' K:', K, 'Trying batch size:', bsize)
                        data = data[:bsize,:,:]
                    else:
                        break



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Neural Clustering Process')

    parser.add_argument('--model', type=str, default='Gauss2D', metavar='S',
                    choices = ['Gauss2D','MNIST'],
                    help='Generative Model: Gauss2D or MNIST (default: Gauss2D)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='batch size for training (default: 64)')
    parser.add_argument('--iterations', type=int, default=3500, metavar='N',
                    help='number of iterations to train (default: 3500)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--plot-interval', type=int, default=30, metavar='N',
                    help='how many iterations between training plots')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)

