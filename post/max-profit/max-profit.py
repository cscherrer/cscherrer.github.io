

import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import scipy

k = 20

# from sklearn.linear_model import BayesianRidge
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline
import pandas as pd


def savefig(filename):
    plt.savefig(f"plots/{filename}.png")
    plt.savefig(f"plots/{filename}.svg")
    plt.savefig(f"../max-profit-2/plots/{filename}.png")
    plt.savefig(f"../max-profit-2/plots/{filename}.svg")

logp0mean = np.log(np.linspace(30, 50, 5)).mean()

def fit0(p0, q0):
    with pm.Model() as m:
        loga = pm.Cauchy('loga', 0, 5)
        c = pm.Cauchy('c', 0, 5, testval=-5)
        μ0 = pm.Deterministic('μ0', np.exp(loga+c*np.log(p0)))
        qval = pm.Poisson('q', μ0, observed=q0)
        t = pm.sample()
    return t


def fit(p0, q0):
    with pm.Model() as m:
        α = pm.Cauchy('α', 0, 5)
        β = pm.Cauchy('β', 0, 5)
        logμ0 = α + β * (np.log(p0) - logp0mean)
        μ0 = pm.Deterministic('μ0', np.exp(logμ0))
        qval = pm.Poisson('q0', μ0, observed=q0)
        t = pm.sample()
    return t


def predict(t, p):
    p0 = np.linspace(30, 50, 5)
    μ = np.exp(t['α'].reshape(-1, 1) +
               np.outer(t['β'], np.log(p) - logp0mean))
    π = (p - k) * μ
    return μ.T, π.T


from sklearn.utils import resample


def boot(t, p, n):
    μ, π = predict(t, p)
    pboot = np.zeros(n)
    for j in range(n):
        newπ = resample(π.T).T
        pboot[j] = p[np.argmax(np.mean(newπ, 1))]
    return pboot

def fitrep(p0, μ0, n):
    qobs = np.random.poisson(np.outer(np.ones(n), μ0))
    with pm.Model() as m:
        α = pm.Cauchy('α', 0, 5, shape=n)
        β = pm.Cauchy('β', 0, 5, shape=n)
        μ0rep = np.exp(
            α + β * (np.log(p0) - logp0mean).reshape(-1, 1)).T
        qval = pm.Poisson('q0', μ0rep, observed=qobs)
        t = pm.sample()
    return t


def withrep(rep, p, f):
    result = []
    for j in range(rep.β.shape[1]):
        t = {'α': rep.α[:, j], 'β': rep.β[:, j]}
        result.append(f(t, p))
    return np.array(result)

def max_of_means(t, p):
    _, π = predict(t, p)
    πmean = np.mean(π, 1)
    jmax = np.argmax(πmean)
    return (p[jmax], πmean[jmax])


def median_of_maxes(t, p):
    β = np.median(t['β'][t['β'] < -1])
    pmax = 20*β/(1+β)
    _, πmax = predict(t, pmax)
    return (pmax, np.mean(πmax))

from matplotlib.patches import Ellipse

# From http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
def normalEllipse(mean, cov, level=0.95,color='C0',alpha=1,lw=1):
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigorder = np.argsort(-eigvals)
    
    theta = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))
    w, h = 2 * np.sqrt(eigvals * scipy.stats.chi2(2).ppf(level))
    return Ellipse(xy=mean,
        width=w, height=h,
        angle=theta, color=color,alpha=alpha,lw=lw)
