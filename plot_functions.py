#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt



from ncp_sampler import NCP_Sampler



def plot_avgs(losses, accs, rot_vars, w, save_name=None):
    
    
    up = -1 #3500
    
    avg_loss = []
    for i in range(w, len(losses)):
        avg_loss.append(np.mean(losses[i-w:i]))
    
    avg_acc = []
    for i in range(w, len(accs)):
        avg_acc.append(np.mean(accs[i-w:i]))
    
    avg_var = []
    for i in range(w, len(rot_vars)):
        avg_var.append(np.mean(rot_vars[i-w:i]))
    
    
    plt.figure(22, figsize=(13,10))
    plt.clf()
    
    plt.subplot(312)
    plt.semilogy(avg_loss[:up])
    plt.ylabel('Mean NLL')
    plt.grid()
    
    plt.subplot(311)
    plt.plot(avg_acc[:up])
    plt.ylabel('Mean Accuracy')
    plt.grid()
    
    plt.subplot(313)
    plt.semilogy(avg_var[:up])
    plt.ylabel('Permutation Variance' )
    plt.xlabel('Iteration')
    plt.grid()

    if save_name:
        plt.savefig(save_name)







def plot_samples_2d(dpmm, data_generator, N= 50, seed = None, save_name=None):
    
    if seed:
        np.random.seed(seed=seed)


    data, cs, clusters, num_clusters = data_generator.generate(N, batch_size=1)
    
    
    plt.figure(1,figsize=(30,5))
    plt.clf()
    
    fig, ax = plt.subplots(ncols=6, nrows=1, num=1)
    ax = ax.reshape(6)
    
    #plt.clf()
    N = data.shape[1]
    s = 26  #size for scatter
    fontsize = 15
    
    
    #frame = plt.gca()        
    #frame.axes.get_xaxis().set_visible(False)
    #frame.axes.get_yaxis().set_visible(False)
    
        
    ax[0].scatter(data[0,:,0],data[0,:,1], color='gray',s=s)        
        
    K=len(set(cs))
    
    ax[0].set_title(str(N) + ' Points',fontsize=fontsize )

    for axis in ['top','bottom','left','right']:
      ax[0].spines[axis].set_linewidth(2)


    ncp_sampler = NCP_Sampler(dpmm,data)
    S = 5000 #number of samples 
    css, probs = ncp_sampler.sample(S)

    
    
    for i in range(5):
        ax[i+1].cla()
        cs= css[i,:]

        for j in range(N):        
            xs = data[0,j,0]
            ys = data[0,j,1]                
            ax[i+1].scatter(xs,ys, color='C'+str(cs[j]+1),s=s)
                
        K=len(set(cs))
        
        ax[i+1].set_title(str(K) + ' Clusters    Prob: '+ '{0:.2f}'.format(probs[i]), fontsize=fontsize)
        for axis in ['top','bottom','left','right']:
          ax[i+1].spines[axis].set_linewidth(0.8)
    
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
        
    return K, probs





def plot_samples_MNIST(dpmm, data_generator, N= 20, seed = None, save_name=None): 
    
    if seed:
        np.random.seed(seed=seed)

    N=20   #dataset size
    data, cs, clusters, K = data_generator.generate(N=N, batch_size=1)    
    
    ncp_sampler = NCP_Sampler(dpmm,data)
    S = 5000       #number of samples 
    css, probs = ncp_sampler.sample(S)
    
        
    rows = 5# number of samples to show
    
    
    W=10
    
    
    
    plt.figure(3,figsize=(15,8))
    plt.clf()
    
    step = 0.074
    
    for i in range(N):
        plt.subplot(W+1,26,i+1+1)
        plt.imshow(data[0,i,:,:], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    
    
    fontsize=25    
    plt.gcf().text(0.35, .9, str(N)+ ' Observations', fontsize=fontsize)
    plt.gcf().text(0.35, 0.76, str(rows) + ' Cluster Samples', fontsize=fontsize)
        
    
    for w in range(0,rows):
    
        it = css[w,:]
        K = len(set(it))
        
        dat = {}
        for k in range(K):
            dat[k]=data[0,np.where(it==k)[0],:,:]
            
            
        
        
        fontsize=15
        strtext = 'K = ' + str(K) + '  Pr: ' + '{0:.2f}'.format(probs[w]) 
        plt.gcf().text(0.03, 0.63-(w-1)*step, strtext, fontsize=fontsize)
        
        i= (w+2)*26
        for k in range(K):
            for j in range(len(dat[k])):
                plt.subplot(W+1,26,i+1+1)                    
                plt.imshow(dat[k][j,:,:], cmap='gray')
                plt.xticks([])
                plt.yticks([])
                i+=1
            i+=1
    
    if save_name:
        plt.savefig(save_name, bbox_inches="tight")




