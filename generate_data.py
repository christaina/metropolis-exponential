import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
"""
Generate hierarchical fake data for sampler
"""

def dist_function(time,A,lambdas):
    """
    Evaluates sum of exponentials w/ params A and lambda at
    a given time
    """
    func = np.vectorize(lambda a,l: a * np.exp(-l*time))
    return sum(func(A,lambdas))

def generate_samples(times,A,lambdas,var):
    """
    Generate noisy samples from distribution given
    parameters
    input:
        times (list) - list of times for exp. dist
        params (list) - list of size 2m+1 where m is
                    the dimension of As and lambdas
    returns:
        samples (list) - a list of samples from sum
                        of exponential RVs
    """
    noise_func = np.vectorize(lambda t: dist_function(t,A,lambdas)\
    +np.sqrt(var)*np.random.normal())
    return noise_func(times)

if __name__=="__main__":

    m = 4
    max_time = 40
    data_size = 1000
    #times = np.random.randint(1,max_time,size=data_size)
    times = np.random.uniform(1,max_time,data_size)

    # generate parameters
    A = np.random.uniform(low=1, high=15,size=m)
    lambdas = np.random.uniform(low=0, high=1.5,size=m)
    var = np.random.uniform(low=0.1, high=3)
    
    params = np.concatenate((A,lambdas,np.array([var])))
    fake_data = np.expand_dims(generate_samples(times,A,lambdas,var),1)
    fake_data = np.concatenate((np.expand_dims(times,1),fake_data),1)
    true_values = [dist_function(t,A,lambdas) for t in np.arange(1,max_time)]
    plt.scatter(fake_data[:,0],fake_data[:,1],color='gray')
    plt.plot(np.arange(1,max_time),true_values,color='yellow')
    plt.xlabel("t")
    plt.ylabel("$\sum_{i=1}^m A_i e^{-\lambda_it}$")
    plt.title("Noisy Samples vs. True Value of $Y_i = \sum_{i=1}^m A_i e^{-\lambda_it}$")
    plt.savefig("figs/data_distr.png")
    plt.show()
    np.savetxt('./fake_data_4.txt',fake_data)
    np.savetxt('./params_4.txt',params)
