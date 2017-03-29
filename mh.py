import numpy as np
import scipy.stats
from distribution import dist

"""
Metropolis Hastings on sample data
"""
def mh_gaussian(init,ys,ts,iters):
    """
    MH with gaussian proposals
    """
    D = len(init)
    samples = np.zeros((iters,D))
    my_dist = dist(init)
    # initialize state and log-likelihood
    state = init.copy()
    Lp_state = my_dist.log_likelihood(ys,ts)
    accepts = 0.

    cov = (0.1)**2 * np.eye(D)*1./D
    for i in np.arange(0, iters):
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


def mh_adap(init,ys,ts,iters):
    """
    MH with gaussian proposals
    """
    D = len(init)
    samples = np.zeros((iters,D))
    my_dist = dist(init)
    # initialize state and log-likelihood
    state = init.copy()
    Lp_state = my_dist.log_likelihood(ys,ts)
    accepts = 0.
    beta = 0.05
    cov = np.eye(D)
    cov_b = (beta ** 2) * (0.1**2) * np.eye(D) * 1./D
    for i in np.arange(0, iters):
        cov_a = (1-beta)**2 * (2.38)**2 * cov * 1./D

        # propose a new state
        prop = np.random.multivariate_normal(state.ravel(), cov_a + cov_b)
        move_p = np.log(scipy.stats.multivariate_normal(state,cov_b+cov).pdf(prop))
        rev_p = np.log(scipy.stats.multivariate_normal(prop,cov_a+cov_b).pdf(state))
        my_dist.set_params(prop)
        Lp_prop = my_dist.log_likelihood(ys,ts)
        rand = np.random.rand()

        if np.log(rand) < min(1,((Lp_prop+rev_p) - (Lp_state+move_p))):
            print ("acct bc %s < %s (iter %s)"%(np.log(rand),(Lp_prop-Lp_state),i))
            accepts += 1
            state = prop.copy()
            Lp_state = Lp_prop
            cov = cov_a
        else:
            my_dist.set_params(state)
        samples[i] = state.copy()

    print 'Acceptance ratio', accepts/iters
    return samples

def mh_lognormal(init,ys,ts,iters):
    """
    MH with lognormal proposals
    """
    D = len(init)
    samples = np.zeros((iters,D))
    my_dist = dist(init)
    # initialize state and log-likelihood
    state = init.copy()
    Lp_state = my_dist.log_likelihood(ys,ts)
    accepts = 0.

    cov = (0.1**2)*np.eye(D)*1./D
    print cov
    for i in np.arange(0, iters):

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
        rand = np.random.rand()

        if np.log(rand) < min(1,((Lp_prop+rev_p) - (Lp_state+move_p))):
            print ("acct bc %s < %s (iter %s)"%(np.log(rand),(Lp_prop-Lp_state),i))
            accepts += 1
            state = prop.copy()
            Lp_state = Lp_prop
            print state
        else:
            my_dist.set_params(state)
        samples[i] = state.copy()

    print 'Acceptance ratio', accepts/iters
    return samples


def mh_uni(init,ys,ts,iters,wid=0.1):
    """
    Component-wise MH with uniform proposals
    """
    D = len(init)
    samples = np.zeros((iters,D))
    my_dist = dist(init)
    # initialize state and log-likelihood
    state = init.copy()
    Lp_state = my_dist.log_likelihood(ys,ts)
    accepts = 0.

    for i in np.arange(0, iters):
        for j,comp in enumerate(state):
            # propose a new state
            prop = (np.random.uniform(comp-wid,comp+wid))
            move_p = np.log(scipy.stats.uniform(comp-wid,comp+wid).pdf(prop))
            rev_p = np.log(scipy.stats.uniform(prop-wid,prop+wid).pdf(comp))
            prop_state = state.copy()
            prop_state[j]=prop
            my_dist.set_params(prop_state)
            print prop_state
            Lp_prop = my_dist.log_likelihood(ys,ts)
            rand = np.random.rand()

            if np.log(rand) < min(1,((Lp_prop+rev_p) - (Lp_state+move_p))):
                print ("acct bc %s < %s (iter %s)"%(np.log(rand),\
                (Lp_prop-Lp_state),i))
                accepts += 1
                state = prop_state.copy()
                Lp_state = Lp_prop
            else:
                my_dist.set_params(state)
            samples[i] = state.copy()

    print 'Acceptance ratio', accepts/(iters*len(init))
    return samples

def acf(samples):
    n = len(samples)
    data = np.asarray(samples)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        acf_l = (data[:n - h] - mean) * (data[h:] - mean)
        acf_lag = acf_l.sum() / float(n-h) / c0
        return round(acf_lag, 3)
    x = np.arange(1,n) # Avoiding lag 0 calculation
    acf_coeffs = map(r, x)
    return acf_coeffs

def autocorr_times(samples):
    """
    Computes autocorrelation time of each param of sampler
    """
    return [1+2*sum((acf(samples[:,i]))) for i in xrange(samples.shape[1])]
