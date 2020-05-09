import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, entropy
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats
import math
import matplotlib.pyplot as plt

def getFMOT(data):
    u, var = [], []
    data = data.reshape(512,4)
    for i in range(data.shape[-1]):
        u.append(np.mean(data[:,i]))
        var.append(np.std(data[:,i])**2)
    return np.sum(u),np.sum(var)

def getFMOS(data):
    u, var = [], []
    data = data.reshape(512,4)
    for i in range(data.shape[0]):
        u.append(np.mean(data[i]))
        var.append(np.std(data[i])**2)
    return np.sum(u),np.sum(var)

def _get_kl_(embeddings, reference, method = 'naive', resolution=1000):
    if method == 'naive':
        mu = np.mean(reference)
        variance = np.var(reference)
    if method == 'FMOS':
        mu, variance = getFMOS(reference[None,:])
    if method == 'FMOT':
        mu, variance = getFMOT(reference[None,:])
    sigma = math.sqrt(variance)
    x = np.linspace(mu-3*sigma, mu+3*sigma, resolution)
    reference_dist = stats.norm.pdf(x, mu, sigma)
    kl_div = []
    for query in embeddings:
        if method == 'naive':
            mu_ = np.mean(query)
            variance_ = np.var(query)
        if method == 'FMOS':
            mu_, variance_ = getFMOS(query[None,:])
        if method == 'FMOT':
            mu_, variance_ = getFMOT(query[None,:])
        sigma_ = math.sqrt(variance_)
        query_dist = stats.norm.pdf(x, mu_, sigma_)
        div = entropy(query_dist,reference_dist)
        if div>1.0 or math.isnan(div):
            _x = np.linspace(min(mu-3*sigma, mu_-3*sigma_), max(mu+3*sigma, mu_+3*sigma_), resolution)
            _reference_dist = stats.norm.pdf(_x, mu, sigma)
            _query_dist = stats.norm.pdf(_x, mu_, sigma_)
            div = entropy(_query_dist,_reference_dist)
        kl_div.append(1/math.exp(div))
    return kl_div

def _get_js_(embeddings, reference, method = 'naive', resolution=1000):
    if method == 'naive':
        mu = np.mean(reference)
        variance = np.var(reference)
    if method == 'FMOS':
        mu, variance = getFMOS(reference[None,:])
    if method == 'FMOT':
        mu, variance = getFMOT(reference[None,:])
    sigma = math.sqrt(variance)
    x = np.linspace(mu-3*sigma, mu+3*sigma, resolution)
    reference_dist = stats.norm.pdf(x, mu, sigma)
    js_div = []
    for query in embeddings:
        if method == 'naive':
            mu_ = np.mean(query)
            variance_ = np.var(query)
        if method == 'FMOS':
            mu_, variance_ = getFMOS(query[None,:])
        if method == 'FMOT':
            mu_, variance_ = getFMOT(query[None,:])
        sigma_ = math.sqrt(variance_)
        query_dist = stats.norm.pdf(x, mu_, sigma_)
        div = jensenshannon(query_dist,reference_dist)
        if div>1.0 or math.isnan(div):
            _x = np.linspace(min(mu-3*sigma, mu_-3*sigma_), max(mu+3*sigma, mu_+3*sigma_), resolution)
            _reference_dist = stats.norm.pdf(_x, mu, sigma)
            _query_dist = stats.norm.pdf(_x, mu_, sigma_)
            div = entropy(_query_dist,_reference_dist)
        js_div.append(1/math.exp(div))
    return js_div

def _get_cs_(embeddings, reference):
    return cosine_similarity(embeddings,reference)

def _get_wasserstein_(embeddings, reference):
    emd = []
    for query in embeddings:
        emd.append(1-wasserstein_distance(query.flatten(),reference.flatten()))
    return emd

def _get_similiarity_(embeddings):
    reduced_embeddings = torch.stack(embeddings).flatten(1).cpu().detach().numpy()
    reference = reduced_embeddings[-1][None,:]
    cosine = _get_cs_(reduced_embeddings,reference)
    
    kl_div = _get_kl_(reduced_embeddings, reduced_embeddings[-1])
    js_div = _get_js_(reduced_embeddings, reduced_embeddings[-1])
    emd = _get_wasserstein_(reduced_embeddings, reduced_embeddings[-1])
    
    return cosine, kl_div, js_div, emd

class nop(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x): return x
