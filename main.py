import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from distribution import dist
import mh
import pandas as pd
import sys

if __name__=='__main__':

    data_path = './fake_data.txt'
    true_params = './params.txt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--nsamp',type=int,default=5000)
    parser.add_argument('--prop',default='log')
    parser.add_argument('--params',default='params.txt')
    parser.add_argument('--data',default='fake_data.txt')
    parser.add_argument('--rundir',default=None)
    args = parser.parse_args()

    if not args.rundir:
        rundir='run_%s_%s'%(args.prop,args.nsamp)
    else:
        rundir=args.rundir

    if not os.path.exists(rundir):
        os.makedirs(rundir)

    fig_name = os.path.join(rundir,'conv.png')
    outfile = os.path.join(rundir,'samples.txt')

    if os.path.exists(outfile):
        os.remove(outfile)

    print args

    burn_in = int(args.nsamp * 0.2)
    data = np.loadtxt(args.data)
    params = np.loadtxt(args.params)

    times = data[:,0]
    outputs = data[:,1]
    print "TRUE: ",params

    init = np.ones(len(params))

    if args.prop=='log':
        mh.mh_lognormal(init,outputs,times,args.nsamp,fi=outfile)
    elif args.prop=='gau':
        mh.mh_gaussian(init,outputs,times,args.nsamp,fi=outfile)
    elif args.prop=='unif':
        mh.mh_uni(init,outputs,times,args.nsamp,wid=0.1,fi=outfile)
    
    
    samples = np.loadtxt(outfile)
    m = (len(params)-1)/2
    autocorr_times = mh.autocorr_times(samples[burn_in:])
    var_acf = mh.acf(samples[burn_in:][:,-1])
    autocorr_time = int(np.ceil(np.mean(autocorr_times)))
    # apply thinking, get posterior mean of samples
    print "autocorrelation time: ",autocorr_time
    final_params = np.mean(np.array([samples[x] for x in xrange(len(samples))\
    if x%autocorr_time==0]),0)

    print "true params: ",params
    print "est params: ", final_params
