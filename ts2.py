

from math import exp
from scipy import stats
import math
import numpy as np
#from numba import jit


def lognorm_cdf(x, mu, sigma):
    shape = sigma
    loc = 0
    scale = exp(mu)
    return stats.lognorm.cdf(x, shape, loc, scale)

x      = 25
mu     = 2.0785
sigma  = 1.744
p      = lognorm_cdf(x, mu, sigma)



def cost(mu, sigma, p):
    steps = [i/10 for i in range(0,100)]
    cdfs = [lognorm.cdf(x,s=sigma,loc=0, scale=exp(mu)) for x in steps]
    fn = [(p-x)*cdf for x, cdf in zip(steps, cdfs)]
    return steps[fn.index(max(fn))], max(steps)

def gradient(x, p, mu, sigma):
    return stats.lognorm.cdf(x,s=sigma,loc=0, scale=exp(mu)) + (x-p)*stats.lognorm.pdf(x,s=sigma,loc=0, scale=exp(mu))

def cost(x, p, mu, sigma):
    return (p-x)*stats.lognorm.cdf(x, s=sigma, loc=0, scale=exp(mu))

p = 10
x = 1
lr = 0.5
for i in range(100):
    x = x - lr * gradient(x, p, mu, sigma)
    cost_v = cost(x, p, mu, sigma)
    print(f"{x} {cost_v}")



def find_optbid(bid, mu, sigma, itr=100, printcost=False):
    x = bid
    lr = bid/100
    for i in range(itr):
        x = x - lr * gradient(x, bid, mu, sigma)
        if printcost:
            cost_v = cost(x, bid, mu, sigma)
            print(f"{x} {cost_v}")
    return x

    
def grid_search(bid, mu, sigma, grid_size=100):
    cost_v = [cost(i/grid_size*bid, bid, mu, sigma) for i in range(grid_size)]
    max_i = cost_v.index(max(cost_v))
    return max_i/grid_size*bid
        
    
for i in range(101):
    x = i/100*bid
    cost_v = cost(x, bid, mu, sigma)
    print(f"{x} {cost_v}")      


#
# To activate this environment, use
#
#     $ conda activate gluon
#
# To deactivate an active environment, use
#
#     $ conda deactivate


K = 100

max_theta1 = 10
max_theta2 = 10

min_theta1 = 0.1
min_theta2 = -1


theta1 = 1
theta2 = 1

error1 = norm.rvs(loc=0, scale=0.005)
error2 = norm.rvs(loc=0, scale=0.005)
theta1 = math.exp(math.log(theta1) + error1)
theta2 = theta2 + error2


def weight_update(theta, win, q):
    sigma = theta[0]
    mu = theta[1]
    cdf_left = stats.lognorm.cdf(q, s=sigma, loc=0, scale=exp(mu))
    if win:
        return cdf_left
    else:
        return 1 - cdf_left


def normal_weight(w_vec):
    total_w = np.sum(w_vec)
    return [w/total_w for w in w_vec]



def lower_bound(nums, l, r, t):
  while l < r:
      m = l + (r - l) // 2
      if nums[m] >= t:
          r = m
      else:
          l = m + 1        
  return l

##check k_vec==0 particle??  
## k_vec stores all values of mu


def sample_theta(w_vec, k_vec):
    #w_vec is always normalized
    pre_sum = 0
    w_vec2 = w_vec.copy()
    for i in range(len(w_vec2)):
        pre_sum += w_vec2[i]
        w_vec2[i] = pre_sum  
    r = stats.uniform.rvs()
    index = lower_bound(w_vec2, 0, len(w_vec2), r)
    return index, k_vec[index]

    
def resample(w_vec, k_vec):
    new_w_vec = [0]*len(w_vec)
    for i in range(len(w_vec)):
        index, _ = sample_theta(w_vec, k_vec)
        new_w_vec[index] += 1
    return normal_weight(new_w_vec)

def resample2(w_vec, k_vec):
    new_w_vec = [0]*len(w_vec)
    for i in range(1000):
        index, _ = sample_theta(w_vec, k_vec)
        new_w_vec[index] += 1
    return normal_weight(new_w_vec)

def s_deg(w_vec):
    return 1/sum([w*w for w in w_vec])
    
            
#w_vec2  = resample(w_vec, theta_vec)
#simualtion
true_mu = 0.9
true_sigma = 0.9
self_mu = 0.5
self_sigma = 1
data_size = 1e4
take_rate = 0.20

def get_simulation_data(true_mu, true_sigma, self_mu, self_sigma, data_size, take_rate):
    max_other_bid = stats.lognorm.rvs(s=true_sigma,loc=0, scale=exp(true_mu),size = int(data_size))
    self_bid = stats.lognorm.rvs(s=self_sigma,loc=0, scale=exp(self_mu),size = int(data_size))
    true_win = np.array([1 if self_bid[i] > max_other_bid[i] else 0 for i in range(len(self_bid))])
    ideal_ps = np.array([self_bid[i] if self_bid[i] > max_other_bid[i] else 0 for i in range(len(self_bid))])
    ideal_margin = np.array([self_bid[i] - max_other_bid[i] if self_bid[i] > max_other_bid[i] else 0 for i in range(len(self_bid))])
    const_ps = np.array([self_bid[i] if self_bid[i]*(1 - take_rate) > max_other_bid[i] else 0 for i in range(len(self_bid))])
    const_win = np.array([1 if ps > 0 else 0 for ps in const_ps])
    const_margin = np.array([self_bid[i]*take_rate if const_ps[i] > 0 else 0 for i in range(len(self_bid))])
    ideal_ps_cum = np.cumsum(ideal_ps)
    ideal_margin_cum = np.cumsum(ideal_margin)
    ideal_win_cum = np.cumsum(true_win)
    const_ps_cum = np.cumsum(const_ps)
    const_margin_cum = np.cumsum(const_margin)
    const_win_cum = np.cumsum(const_win)
    return self_bid, max_other_bid, ideal_ps_cum, ideal_margin_cum, ideal_win_cum, const_ps_cum, const_margin_cum, const_win_cum

def get_simulation_data_cnst(true_mu, true_sigma, self_mu, self_sigma, data_size, take_rate):
    max_other_bid = stats.lognorm.rvs(s=true_sigma,loc=0, scale=exp(true_mu),size = int(data_size))
    self_bid = [2.5 for i in range(int(data_size))]
    true_win = np.array([1 if self_bid[i] > max_other_bid[i] else 0 for i in range(len(self_bid))])
    ideal_ps = np.array([self_bid[i] if self_bid[i] > max_other_bid[i] else 0 for i in range(len(self_bid))])
    ideal_margin = np.array([self_bid[i] - max_other_bid[i] if self_bid[i] > max_other_bid[i] else 0 for i in range(len(self_bid))])
    const_ps = np.array([self_bid[i] if self_bid[i]*(1 - take_rate) > max_other_bid[i] else 0 for i in range(len(self_bid))])
    const_win = np.array([1 if ps > 0 else 0 for ps in const_ps])
    const_margin = np.array([self_bid[i]*take_rate if const_ps[i] > 0 else 0 for i in range(len(self_bid))])
    ideal_ps_cum = np.cumsum(ideal_ps)
    ideal_margin_cum = np.cumsum(ideal_margin)
    ideal_win_cum = np.cumsum(true_win)
    const_ps_cum = np.cumsum(const_ps)
    const_margin_cum = np.cumsum(const_margin)
    const_win_cum = np.cumsum(const_win)
    return self_bid, max_other_bid, ideal_ps_cum, ideal_margin_cum, ideal_win_cum, const_ps_cum, const_margin_cum, const_win_cum
    
self_bid, max_other_bid, ideal_ps_cum, ideal_margin_cum, ideal_win_cum, const_ps_cum, const_margin_cum, const_win_cum = get_simulation_data(true_mu, true_sigma, self_mu, self_sigma, data_size, take_rate)
self_bid, max_other_bid, ideal_ps_cum, ideal_margin_cum, ideal_win_cum, const_ps_cum, const_margin_cum, const_win_cum = get_simulation_data_cnst(true_mu, true_sigma, self_mu, self_sigma, data_size, take_rate)

def take_rate_ps(self_bid, max_other_bid, take_rate):
    const_ps = np.array([self_bid[i] if self_bid[i]*(1 - take_rate) > max_other_bid[i] else 0 for i in range(len(self_bid))])
    const_ps_cum = np.cumsum(const_ps)
    const_margin = np.array([self_bid[i]*take_rate if const_ps[i] > 0 else 0 for i in range(len(self_bid))])
    const_margin_cum = np.cumsum(const_margin)
    return const_ps_cum[-1], const_margin_cum[-1]

take_rate_ps(self_bid, max_other_bid, 0.50)


def cost2(x, p, mu, sigma, r):
    return (p-x+r*x)*stats.lognorm.cdf(x, s=sigma, loc=0, scale=exp(mu))
def grid_search2(bid, mu, sigma, r, grid_size=100):
    cost_v = [cost2(i/grid_size*bid, bid, mu, sigma, r) for i in range(grid_size)]
    max_i = cost_v.index(max(cost_v))
    return max_i/grid_size*bid

opt_bid_o = grid_search2(self_bid[0], true_mu, true_sigma, 0)


def orcale_bid_cnst(self_bid, max_other_bid, true_mu, true_sigma, r):
    opt_bid_o = grid_search2(self_bid[0], true_mu, true_sigma, r)
    const_ps = np.array([self_bid[i] if opt_bid_o > max_other_bid[i] else 0 for i in range(len(self_bid))])
    const_margin = np.array([self_bid[i] - opt_bid_o if const_ps[i] > 0 else 0 for i in range(len(self_bid))])
    const_ps_cum = np.cumsum(const_ps)
    const_margin_cum = np.cumsum(const_margin)
    return const_ps_cum[-1], const_margin_cum[-1]


def orcale_bid(self_bid, max_other_bid, true_mu, true_sigma, r):
    const_ps = []
    const_margin = []
    const_win = []
    for i in range(len(self_bid)):
        opt_bid_o = grid_search2(self_bid[i], true_mu, true_sigma, r)
        const_ps.append(self_bid[i] if opt_bid_o > max_other_bid[i] else 0)
        const_margin.append(self_bid[i] - opt_bid_o if const_ps[i] > 0 else 0)
        const_win.append(1 if opt_bid_o > max_other_bid[i] else 0)
    const_ps_cum = np.cumsum(const_ps)
    const_margin_cum = np.cumsum(const_margin)
    const_win_cum = np.cumsum(const_win)
    return const_win_cum, const_ps_cum, const_margin_cum

 
def simulate_auction2(self_bid, max_other_bid, r):
    sigma_kvec = [i/10 for i in range(1,21,2)]
    mu_kvec = [i/10 for i in range(1,21,2)]
    theta_vec = [(sigma, mu) for sigma in sigma_kvec for mu in mu_kvec]
    w_vec = normal_weight([1]*len(theta_vec))
    total_margin, total_ps, win_total = 0, 0, 0
    win_total_list, total_ps_list, total_margin_list = [], [], []
    all_sampled_theta = []
    for i in range(len(self_bid)):
        if s_deg(w_vec) < 100/3:
            w_wec = resample(w_vec, theta_vec)
            print("resample")
        idx_theta, theta = sample_theta(w_vec, theta_vec)
        all_sampled_theta.append(theta)
        sigma, mu = theta[0], theta[1]
        opt_bid = grid_search2(self_bid[i], mu, sigma, r)
        win = opt_bid > max_other_bid[i]
        w_vec[idx_theta] *= weight_update(theta, win, opt_bid)
        w_vec = normal_weight(w_vec)
        if win:
            total_margin = total_margin + self_bid[i] - opt_bid
            total_ps = total_ps + self_bid[i]
            win_total = win_total + 1
            win_total_list.append(win_total)
            total_ps_list.append(total_ps)
            total_margin_list.append(total_margin)
        else:
            win_total_list.append(win_total)
            total_ps_list.append(total_ps)
            total_margin_list.append(total_margin)
    return all_sampled_theta, win_total_list, total_ps_list, total_margin_list, w_vec, theta_vec

all_sampled_theta, win_total_list, total_ps_list, total_margin_list, w_vec, theta_vec = simulate_auction2(self_bid, max_other_bid, r)


#ideal_ps_sum = 0
#ideal_margin_sum = 0
#win_ps_sum = 0
#for i in range(len(ideal_ps)):
#    ideal_ps_sum += ideal_ps[i]
#    ideal_ps_cum.append(ideal_ps_sum)
#    win_ps_sum += true_win[i]
#    true_win_cum.append(win_ps_sum)
#    ideal_margin_sum += ideal_margin[i]
#    ideal_margin_cum.append(ideal_margin_sum)
#ideal_ps_total = sum(ideal_ps)
#true_win_total = sum(true_win)
#sample from prior uniform(0,2) for mu and uniform(0,2) for sigma
#

##two simulations
##one know the prior distbution
## withou knoing distrubtion


    


def test_simulate_auction(self_bid, max_other_bid):
    sigma_kvec = [i/10 for i in range(1,21,2)]
    mu_kvec = [i/10 for i in range(1,21,2)]
    theta_vec = [(sigma, mu) for sigma in sigma_kvec for mu in mu_kvec]
    w_vec = normal_weight([1]*len(theta_vec))
    idx_theta, theta = sample_theta(w_vec, theta_vec)
    all_sampled_theta = []
    for i in range(len(self_bid)):
        idx_theta, theta = sample_theta(w_vec, theta_vec)
        all_sampled_theta.append(theta)
    return all_sampled_theta
    


theta_vec[w_vec.index(max(w_vec))]
#xy_mat = [[0]*10]*10 this initialization has reference issue.

def freq_map(all_sampled_theta):
    sigma_kvec = [i/10 for i in range(1,21,2)]
    mu_kvec = [i/10 for i in range(1,21,2)]
    xy_mat = [[0 for i in range(10) ] for j in range(10) ]
    for theta in all_sampled_theta:
        x_axis = sigma_kvec.index(theta[0])
        y_axis = mu_kvec.index(theta[1])
        xy_mat[x_axis][y_axis] += 1
    return xy_mat

xy_mat = freq_map(all_sampled_theta)

all_sampled_theta2 = test_simulate_auction(self_bid, max_other_bid)
xy_mat2 = freq_map(all_sampled_theta2)

import matplotlib.pyplot as plt

##plot freq heat map
#plt.imshow(xy_mat, cmap='hot', interpolation='nearest')
#plt.legend()
#plt.show()


heatmap=plt.pcolor(xy_mat)
plt.colorbar(heatmap)
plt.show()






