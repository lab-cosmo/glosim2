import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import numba as nb
from math import exp,sqrt

def avgKernel(envKernelDict,zeta):

    keys = envKernelDict.keys()
    rows = np.array([key[0] for key in keys])
    cols = np.array([key[1] for key in keys])
    N = rows.max() + 1
    M = cols.max() + 1
    Similarity = np.zeros((N,M),dtype=np.float64)
    for key,envKernel in envKernelDict.iteritems():
        Similarity[key[0],key[1]] = np.power(envKernel,zeta).mean()
    diag = np.diag(Similarity)
    return Similarity + Similarity.T - np.diag(diag)


def rematchKernel(envKernelDict, gamma=2., eps=1e-6, nthreads=8):

    nb_rematch = compile_rematch()

    keys = envKernelDict.keys()
    envKernels = envKernelDict.values()
    rows = np.array([key[0] for key in keys])
    cols = np.array([key[1] for key in keys])
    N = rows.max() + 1
    M = cols.max() + 1
    globalSimilarity = np.zeros((N, M), dtype=np.float64)

    if nthreads == 1:
        for key, envKernel in envKernelDict.iteritems():
            globalSimilarity[key[0], key[1]] = nb_rematch(envKernel, gamma, eps=eps)
    else:
        def nb_rematch_wrapper(kargs):
            return nb_rematch(**kargs)
        kargs = [{'envKernel': envKernel, 'gamma': gamma, 'eps': eps} for envKernel in envKernels]
        pool = ThreadPool(nthreads)
        results = pool.map(nb_rematch_wrapper, kargs)
        pool.close()
        pool.join()
        for key, result in zip(keys, results):
            globalSimilarity[key[0], key[1]] = result

    diag = np.diag(globalSimilarity)
    return globalSimilarity + globalSimilarity.T - np.diag(diag)


def rematch(envKernel, gamma, eps):
    n, m = envKernel.shape
    mf = float(m)
    nf = float(n)
    K = np.zeros((n, m))
    lamb = 1. / gamma
    for it in range(n):
        for jt in range(m):
            K[it, jt] = exp(- (1 - envKernel[it, jt]) * lamb)

    u = np.ones((n,)) / nf
    v = np.ones((m,)) / mf

    Kp = nf * K

    iii = 0
    err = 1

    while (err > eps):
        uprev = u
        vprev = v
        for jt in range(m):
            kk = 0.
            for it in range(n):
                kk += K[it, jt] * u[it]
            v[jt] = 1. / kk / mf

        for it in range(n):
            kk = 0.
            for jt in range(m):
                kk += Kp[it, jt] * v[jt]
            u[it] = 1. / kk

        if iii % 5:
            erru = 0.
            errv = 0.
            for it in range(n):
                erru += (u[it] - uprev[it]) * (u[it] - uprev[it]) / (u[it] * u[it])
            for jt in range(m):
                errv += (v[jt] - vprev[jt]) * (v[jt] - vprev[jt]) / (v[jt] * v[jt])
            err = erru + errv
        iii += 1

    RematchSimilarity = 0.
    for it in range(n):
        rrow = 0.
        for jt in range(m):
            rrow += K[it, jt] * envKernel[it, jt] * v[jt]
        RematchSimilarity += u[it] * rrow

    return RematchSimilarity

def np_rematch(envKernel, gamma, eps=1e-6):
    n, m = envKernel.shape
    K = np.exp(- (1 - envKernel) / gamma)
    u = np.ones((n,)) / n
    v = np.ones((m,)) / m

    a = np.ones((n,)) / float(n)
    b = np.ones((m,)) / float(m)

    Kp = (1 / a).reshape(-1, 1) * K

    iii = 0
    err = 1
    while (err > eps):
        uprev = u
        vprev = v
        v = np.divide(b, np.dot(K.T, u))
        u = 1. / np.dot(Kp, v)
        if iii % 5:
            err = np.sum((u - uprev) ** 2) / np.sum((u) ** 2) + np.sum((v - vprev) ** 2) / np.sum((v) ** 2)
        iii += 1

    rval = 0
    for it in range(n):
        rrow = np.sum(K[it, :] * envKernel[it, :] * v[:])
        rval += u[it] * rrow
    return rval

def compile_rematch():
    signatureRem = nb.double(nb.double[:, :], nb.double, nb.double)
    nb_rematch = nb.jit(signatureRem, nopython=True, nogil=True)(rematch)
    return nb_rematch


def normalizeKernel(kernel):
    n,m = kernel.shape
    diag = np.diag(kernel)
    for it in range(n):
        kernel[it,:] /= np.sqrt(diag[it] * diag)
    return kernel