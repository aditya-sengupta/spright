'''
Script to test choices of M for SPRIGHT, when there's a prior on the distribution of W-H frequencies.
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import tqdm

import sys
sys.path.append("..")
import spright
from src import inputsignal
from src import query
from src.utils import bin_to_dec, dec_to_bin

n = 4
b = n // 2
num_to_get = n // b
lam = 6 * (n / 4) ** 2

num_runs = 1
eps = 1e-17

rv = stats.poisson(lam)
p = rv.pmf(np.arange(2 ** n))
p /= sum(p)

locs = np.random.choice(np.arange(2 ** n), p=p, size=2 ** b, replace=False)
strengths = np.random.randint(1, 6, size=2 ** b)
signal = inputsignal.Signal(n, locs, strengths)

transform = spright.SPRIGHT(**{"query_method" : "simple", "delays_method" : "random", "reconstruct_method" : "mle"}).transform

def num_to_Ms(num):
    return [x for x in dec_to_bin(num, num_to_get * n * b).reshape(num_to_get, n, b)]

def Ms_to_num(Ms):
    return bin_to_dec(np.array(Ms).flatten())

def init_func():
    return Ms_to_num(query.get_Ms_simple(n, b, num_to_get))

def proposal_func(num):
    return (num + 2 ** np.random.randint(num_to_get * n * b)) % (2 ** (num_to_get * n * b))

def get_score(c):
    for _ in range(num_runs):
        score = 0
        try:
            _, samples, loc = transform(signal, Ms=c, report=True)
        except KeyboardInterrupt:
            raise
        except BaseException:
            return eps
        if not loc == set(signal.loc):
            return eps
        else:
            score += - (samples - eps) / 2 ** n
    return score / num_runs

def scorer(current, candidate):
    return get_score(num_to_Ms(candidate)) / get_score(num_to_Ms(current))

def metropolis_hastings(proposal_func, init_func, score_func, num_iters, step=30):
    """
    Runs the metropolis-hastings algorithm for
    num_iters iterations, using proposal_func
    to generate samples and scorer to assign
    probability scores to samples.
      
    proposal_func -- function that proposes
        candidate state; takes in current state as
        argument and returns candidate state
    init_func -- function that proposes starting
        state; takes no arguments and returns a
        sample state
    score_func -- function that calculates f(y)/f(x)
        * g(y,x)/g(x,y); takes in two state samples
        (the current sample x then the candidate y).
    
    Returns a sequence of every step-th sample. You 
    should only sample on upon acceptance of a new
    proposal. Do not keep sampling the current state.
    
    Note the total number of samples will NOT be
    equal to num_iters. num_iters is the total number
    of proposals we generate.
    """
    samples = []
    sample = init_func()
    for i in tqdm.trange(num_iters):
        candidate = proposal_func(sample)
        acceptance_prob = min(1, score_func(sample, candidate))
        if np.random.uniform() < acceptance_prob:
            sample = candidate
            samples.append(sample)
    return samples[::step]

burn_in = 100
# samples = metropolis_hastings(proposal_func, init_func, scorer, 10000, step=1)[burn_in:]
# samples_as_ints = [bin_to_dec(np.array(x).flatten()) for x in samples]
scores = np.empty(2 ** (num_to_get * n * b))
for i in range(2 ** (num_to_get * n * b)):

best_int = max(set(samples_as_ints), key=samples_as_ints.count)
best_mats = [x for x in dec_to_bin(best_int, num_to_get * n * b).reshape(num_to_get, n, b)]
print("The best subsampling patterns for this signal are {}".format(best_mats))
print("The average ratio of untouched samples for this pattern is {}".format(get_score(best_mats)))