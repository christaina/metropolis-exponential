import numpy as np
import scipy.stats
from distribution import dist

"""
Metropolis Hastings on sample data
"""
def mh_gaussian(init,ys,ts,iters,fi=None):
    """
    MH with gaussian proposals
    """
    print("Running Metropolis Algorithm with Gaussian proposals.")
    D = len(init)
    samples = np.zeros((iters,D))
    my_dist = dist(init)
    # initialize state and log-likelihood
    state = init.copy()
    Lp_state = my_dist.log_likelihood(ys,ts)
    accepts = 0.

    cov = (0.1)**2 * np.eye(D)*1./D
    for i in np.arange(0, iters):
        if fi is not None:
            write_samp(fi,state)
        # propose a new state
        prop = (np.random.multivariate_normal(state.ravel(), cov))
        move_p = np.log(scipy.stats.multivariate_normal(state,cov).pdf(prop))
        rev_p = np.log(scipy.stats.multivariate_normal(prop,cov).pdf(state))
        my_dist.set_params(prop)
        Lp_prop = my_dist.log_likelihood(ys,ts)
        rand = np.random.rand()

        if np.log(rand) < min(1,((Lp_prop+rev_p) - (Lp_state+move_p))):
            print ("acct bc %s < %s (iter %s)"%(np.log(rand),(Lp_prop-Lp_state),i))
            accepts += 1
            state = prop.copy()
            print state
            Lp_state = Lp_prop
        else:
            my_dist.set_params(state)
        samples[i] = state.copy()

    print 'Acceptance ratio', accepts/iters
    return samples

def mh_lognormal(init,ys,ts,iters,fi=None):
    """
    MH with lognormal proposals
    """
    print("Running Metropolis Algorithm with Log-Normal proposals.")
    D = len(init)
    my_dist = dist(init)
    # initialize state and log-likelihood
    state = init.copy()
    Lp_state = my_dist.log_likelihood(ys,ts)
    accepts = 0.

    cov = (0.1**2)*np.eye(D)*1./D
    for i in np.arange(0, iters):
        if fi is not None:
            write_samp(fi,state)

        # log(Rv) follow a normal distribution
        mu = np.log(state)
        # propose a new state
        prop = np.exp(np.random.multivariate_normal(mu, cov))
        move_p = np.log(scipy.stats.multivariate_normal(np.log(state),cov).\
                                                pdf(np.log(prop)))
        rev_p = np.log(scipy.stats.multivariate_normal(np.log(prop),cov).\
                                                pdf(np.log(state)))
        my_dist.set_params(prop)
        Lp_prop = my_dist.log_likelihood(ys,ts)
        #print Lp_prop+rev_p
        #print Lp_state+move_p
        rand = np.random.rand()
        prob = min(1,((Lp_prop+rev_p)-(Lp_state+move_p)))
        if np.log(rand) < prob:
            accepts += 1
            state = prop.copy()
            Lp_state = Lp_prop
            print ("acct bc %s < %s (iter %s)"%(np.log(rand),prob,i))
            print state
        else:
            my_dist.set_params(state)

    print 'Acceptance ratio', accepts/iters

def write_samp(fn,state):
    """
    Write a sample to file
    """
    with open(fn,'a') as f:
        f.write('%s\n'%' '.join(state.astype(str)))

def mh_uni(init,ys,ts,iters,wid=0.1,fi=None):
    """
    Component-wise MH with uniform proposals
    """
    print("Running Metropolis Algorithm with Uniform proposals.")
    D = len(init)
    samples = np.zeros((iters,D))
    my_dist = dist(init)
    # initialize state and log-likelihood
    state = init.copy()
    Lp_state = my_dist.log_likelihood(ys,ts)
    accepts = 0.

    for i in np.arange(0, iters):
        print '***Curr acct: %s'%(float(accepts)/((i+1)*4))
        for j,comp in enumerate(state):
            if fi is not None:
                write_samp(fi,state)
            # propose a new state
            prop = (np.random.uniform(comp-wid,comp+wid))
            move_p = np.log(scipy.stats.uniform(comp-wid,comp+wid).pdf(prop))
            rev_p = np.log(scipy.stats.uniform(prop-wid,prop+wid).pdf(comp))
            prop_state = state.copy()
            prop_state[j]=prop
            my_dist.set_params(prop_state)
            print "\t",prop_state
            Lp_prop = my_dist.log_likelihood(ys,ts)
            rand = np.random.rand()

            if np.log(rand) < min(1,((Lp_prop+rev_p) - (Lp_state+move_p))):
                print ("\tacct bc %s < %s (iter %s)"%(np.log(rand),\
                (Lp_prop-Lp_state),i))
                accepts += 1
                state = prop_state.copy()
                Lp_state = Lp_prop
            else:
                my_dist.set_params(state)
            samples[i] = state.copy()

    print 'Acceptance ratio', accepts/(iters*len(init))
    return samples

def acf(samples,last=None):
    n = len(samples)
    if not last:
        last = n/2
    data = np.asarray(samples)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        acf_l = (data[:n - h] - mean) * (data[h:] - mean)
        acf_lag = acf_l.sum() / float(n-h) / c0
        return round(acf_lag, 3)
    x = np.arange(1,last) # Avoiding lag 0 calculation
    acf_coeffs = map(r, x)
    return acf_coeffs

def autocorr_times(samples):
    """
    Computes autocorrelation time of each param of sampler
    """
    return [1+2*sum(np.abs(acf(samples[:,i]))) for i in xrange(samples.shape[1])]
