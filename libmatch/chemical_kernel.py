import numpy as np


def randKernel(spA ,spB ,seed=10):
    np.random.seed(( spA +spB ) *seed)
    return np.random.random()

def deltaKernel(spA ,spB):
    if spA == spB:
        return 1.
    else:
        return 0.

def Atoms2ChemicalKernelmat(atoms,chemicalKernel=deltaKernel):
    # unique sp in frames 1 and 2
    uk1 = []
    for frame in atoms:
        uk1.extend(frame.get_atomic_numbers())
    uk1 = list(set(uk1))
    Nsp1 = max(uk1)+1
    # 0 row and col are here but dont matter
    chemicalKernelmat = np.zeros((Nsp1,Nsp1))
    for it in uk1:
        for jt in uk1:
            chemicalKernelmat[it,jt] = chemicalKernel(it,jt)
    return chemicalKernelmat



def get_chemicalKernelmatFrames(frames1 ,frames2=None ,chemicalKernel=deltaKernel):
    # unique sp in frames 1 and 2
    uk1 = []
    for frame in frames1:
        uk1.extend(frame.get_atomic_numbers())
    uk1 = list(set(uk1))

    if frames2 is None:
        frames2 = frames1
        uk2 = uk1
    else:
        uk2 = []
        for frame in frames2:
            uk2.extend(frame.get_atomic_numbers())
        uk2 = list(set(uk2))

    Nsp1 = max(uk1 ) +1
    Nsp2 = max(uk2 ) +1
    # 0 row and col are here but dont matter
    chemicalKernelmat = np.zeros((Nsp1 ,Nsp2))
    for it in uk1:
        for jt in uk2:
            chemicalKernelmat[it ,jt] = chemicalKernel(it ,jt)
    return chemicalKernelmat

