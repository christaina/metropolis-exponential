import numpy as np
import scipy.stats

class dist(object):
    def __init__(self,params):
        self.A,self.lambdas = np.split(np.array(params[:-1]),2)
        self.var = params[-1]
        self.m = len(self.A)

    def set_params(self,params):
        self.A,self.lambdas = np.split(np.array(params[:-1]),2)
        self.var = params[-1]

    def prior_pdf(self,max_var=50):
        """
        Probability of parameters for prior
        distribution - indicator of 1 if A,var in [0,infty]
        and if lambas in [0,max_var]
        """
        if (np.all(self.lambdas <= max_var) & (np.all(self.A> 0)) \
        & (np.all(self.lambdas > 0)) & (self.var > 0)):
            return 1
        return 0

    def pdf(self,sample,time):
        """
        Evaluate probability of sample at a given time
        for given parameters
        """
        # sum exponential distributions
        exp_sum = np.sum([self.A[i]*np.exp(-1.*self.lambdas[i]*time)\
         for i in xrange(self.m)])
        #print (scipy.stats.norm(exp_sum,self.var).pdf(sample))
        return scipy.stats.norm(exp_sum,self.var).pdf(sample)

    def log_likelihood(self,samples,times):
        #print "prior, log-prior: %s,%s"%(self.prior_pdf(),np.log(self.prior_pdf()))
        prior_p = np.log(self.prior_pdf())
        sample_p = np.sum(np.log([self.pdf(s,t) for s,t in zip(samples,times)]))
        ll = prior_p + sample_p

        if np.isnan(ll):
            return -np.infty
        return ll
