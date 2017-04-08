import matplotlib.pyplot as plt
import mh
import numpy as np
import pandas as pd
import argparse
import generate_data as gen

def param_plot(est_params,true_params,fig_name='est_params.png'):
    """
    Plot estimated params against true params
    """
    m=len(est_params)/2
    axes = [1]*m+[2]*m+[3]
    plt.scatter(axes,est_params,c='blue')
    plt.scatter(axes,true_params,c='red')
    plt.xticks([1,2,3],['A','$\lambda$','$\sigma$'])
    plt.savefig(fig_name)
    plt.clf()

def posterior_plot(est_params,data,size=500,fig_name='posterior_samp.png'):
    """
    Plots samples from posterior compared to original data
    """
    m = len(est_params)/2
    times = np.random.uniform(0,40,size)
    post_samples = gen.generate_samples(times,est_params[0:m],est_params[m:2*m],est_params[-1])
    plt.scatter(times,post_samples,c='blue',label='posterior samples')
    plt.scatter(data[:,0],data[:,1],c='red',label='original data')
    plt.xlabel('Time')
    plt.ylabel("f(x)")
    plt.title("Original samples vs. Samples from Posterior")
    plt.legend()
    plt.savefig(fig_name)
    plt.clf()

def run_plots(samples,params,burn_in=200,fig_name='run.png'):
    m = len(params)/2
    f, axarr = plt.subplots(2,2)
    cmap = plt.cm.get_cmap("prism")
    for i in xrange(len(params)):
        lab = None
        if i < m:
            lab = "A_%s"%(i)
            axarr[0,0].plot(np.arange(len(samples)),samples[:,i],label=lab,color=cmap(i*10))
            axarr[0,0].scatter(len(samples)-1,params[i],label=lab,color='red')
            axarr[0,0].set_title("A Values vs. Steps")
        elif i < 2*m:
            lab = "lambda_%s"%(i-m)
            axarr[0,1].plot(np.arange(len(samples)),samples[:,i],label=lab,color=cmap(i*10))
            axarr[0,1].set_title("Lambda Values vs. Steps")
            axarr[0,1].scatter(len(samples)-1,params[i],label=lab,color='red')
        else:
            lab = "var"
            axarr[1,0].plot(np.arange(len(samples)),samples[:,i],label=lab)
            axarr[1,0].scatter(len(samples)-1,params[i],label=lab,color='red')
            axarr[1,0].set_title("Variance vs. Steps")

        pd.tools.plotting.autocorrelation_plot(samples[burn_in:][:,i],ax=axarr[1,1])
        axarr[1,1].set_title("Autocorrelation Plot")

    for i,ax in enumerate(f.axes):
        plt.sca(ax)
        plt.xticks(rotation=30)
        plt.xlim(0,len(samples)+int(0.05*len(samples)))
        if i<2:
            ax.xaxis.set_visible(False)

    plt.savefig(fig_name)
    plt.clf()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',default='fake_data.txt')
    parser.add_argument('--params',default='params.txt')
    parser.add_argument('--samps',default='./run_log_100000/samples.txt')
    args = parser.parse_args()
    samples = np.loadtxt(args.samps)
    data=np.loadtxt(args.data)
    params=np.loadtxt(args.params)
    #params[-1]=np.sqrt(params[-1])
    burn_in = int(0.2*len(samples))
    autocorr_time = int(np.ceil(np.mean(mh.autocorr_times(samples[burn_in:]))))
    print ("Autocorrelation time estimate: %s"%autocorr_time)
    final_samples = np.array([x for i,x in enumerate(samples[burn_in:]) if i%autocorr_time==0])
    print ("%s Samples of params"%len(final_samples))
    est_params = np.mean(np.array(final_samples),axis=0)
    print "EST PARAMS:",est_params
    run_plots(samples,params,burn_in=burn_in,\
            fig_name=args.samps.replace('samples.txt','run.png'))
    posterior_plot(est_params,data,fig_name=args.samps.replace('samples.txt','posterior_samp.png'))
    param_plot(est_params,params,fig_name=args.samps.replace('samples.txt','est_params.png'))
