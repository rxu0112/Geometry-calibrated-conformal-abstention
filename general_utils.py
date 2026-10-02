import os
import torch
import numpy as np
import pickle

def cos_np(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def precompute_mahalanobis(X, eps=1e-6):
    """
    X: (n, m)
    returns:
        mu: (m,)
        Sigma_inv: (m, m)
    """
    mu = X.mean(axis=0)
    Sigma = np.cov(X, rowvar=False)
    Sigma += eps * np.eye(Sigma.shape[0])

    Sigma_inv = np.linalg.inv(Sigma)

    return mu, Sigma_inv

def covariance_matrix(X):
    """
    X: (n, m)
    returns:
        Sigma: (m, m)
    """
    return np.cov(X, rowvar=False)  

def mahalanobis_distance(x, mu, Sigma_inv):
    diff = x - mu
    return np.sqrt(diff.T @ Sigma_inv @ diff)

def append_pickle(obj, path):
    with open(path, "ab") as f:
        pickle.dump(obj, f)

def load_pickle_file(path): 
    objs = [] 
    with open(path, "rb") as f: 
        while True: 
            try: 
                objs.append(pickle.load(f)) 
            except EOFError: 
                break
    return objs