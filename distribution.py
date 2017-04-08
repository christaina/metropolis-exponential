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

    def sum_exp(self,time):
        """
        Evaluate sum of exponentials given current params
        """
        sum = np.sum([self.A[i]*np.exp(-1.*self.lambdas[i]*time)\
         for i in xrange(self.m)])
        return sum

    def log_likelihood(self,samples,times):
        """
        Evaluate log likelihood of data
        """
        prior_p = np.log(self.prior_pdf())
        xform = [self.pdf(t) for t in times]
        lp = scipy.stats.norm(xform,np.sqrt(self.var)).pdf(samples)
        sample_p =np.sum(np.log(lp))
        ll = prior_p + sample_p

        if np.isnan(ll):
            return -np.infty
        return ll
