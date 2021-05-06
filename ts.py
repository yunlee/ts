from math import exp
from scipy import stats
import math
import numpy as np
from numba import jit


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
    return (p-x)*stats.lognorm.cdf(x,s=sigma,loc=0, scale=exp(mu))

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



self_bid, max_other_bid, ideal_ps_cum, ideal_margin_cum, ideal_win_cum, const_ps_cum, const_margin_cum, const_win_cum = get_simulation_data(true_mu, true_sigma, self_mu, self_sigma, data_size, take_rate)



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

def s_deg(w_vec):
    return 1/sum([w*w for w in w_vec])
    
    
def simulate_auction(self_bid, max_other_bid):
    sigma_kvec = [i/10 for i in range(1,21,2)]
    mu_kvec = [i/10 for i in range(1,21,2)]
    theta_vec = [(sigma, mu) for sigma in sigma_kvec for mu in mu_kvec]
    w_vec = normal_weight([1]*len(theta_vec))
    total_margin = 0
    total_ps = 0
    win_total = 0
    win_total_list = []
    total_ps_list = []
    total_margin_list = []
    total_margin_o = 0
    total_ps_o = 0
    win_total_o = 0
    win_total_list_o = []
    total_ps_list_o = []
    total_margin_list_o = []
    for i in range(len(self_bid)):
        if s_deg(w_vec) < 100/3:
            w_wec = resample(w_vec, theta_vec)
            print("resample")
        idx_theta, theta = sample_theta(w_vec, theta_vec)
        sigma, mu = theta[0], theta[1]
        opt_bid = grid_search(self_bid[i], mu, sigma)
        true_mu, true_sigma = 1, 1
        opt_bid_oracle = grid_search(self_bid[i], true_mu, true_sigma)
        win = opt_bid > max_other_bid[i]
        win_oracle = opt_bid_oracle > max_other_bid[i]
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
        if win_oracle:
            total_margin_o = total_margin_o + self_bid[i] - opt_bid_oracle
            total_ps_o = total_ps_o + self_bid[i]
            win_total_o = win_total_o + 1
            win_total_list_o.append(win_total_o)
            total_ps_list_o.append(total_ps_o)
            total_margin_list_o.append(total_margin_o)
        else:
            win_total_list_o.append(win_total_o)
            total_ps_list_o.append(total_ps_o)
            total_margin_list_o.append(total_margin_o)
    return win_total_list, total_ps_list, total_margin_list, win_total_list_o, total_ps_list_o, total_margin_list_o, w_vec, theta_vec

win_total_list, total_ps_list, total_margin_list, win_total_list_o, total_ps_list_o, total_margin_list_o, w_vec, theta_vec = simulate_auction(self_bid, max_other_bid)

theta_vec[w_vec.index(max(w_vec))]

#w_vec_2d

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.array([theta[0] for theta in theta_vec]).reshape((10, 10))
Y = np.array([theta[1] for theta in theta_vec]).reshape((10, 10))

Z = np.array(w_vec).reshape((10, 10))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0, max(w_vec)*1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
