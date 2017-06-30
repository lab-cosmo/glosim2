import numpy as np



def compute_Avg(kdict,zeta):
    rows = np.array([key[0] for key in kdict.keys()])
    cols = np.array([key[1] for key in kdict.keys()])
    N = rows.max() + 1
    M = cols.max() + 1
    Similarity = np.zeros((N,M),dtype=np.float64)
    for key,k in kdict.iteritems():
        Similarity[key[0],key[1]] = np.power(k,zeta).mean()

    diag = np.diag(Similarity)
    return Similarity + Similarity.T - np.diag(diag)


def normalizeKernel(kernel):
    n,m = kernel.shape
    for it in range(n):
        for jt in range(m):
            kernel[it,jt] = kernel[it,jt] / np.sqrt(kernel[it,it]* kernel[jt,jt])
    return kernel




