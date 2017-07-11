import numpy as np
from multithreading import make_multithread_envKernel,make_singlethread_envKernel

try:
    import numba as nb
    nonumba = False
except:
    nonumba = True



def compile_envKernel_with_thread(nthreads=1):
    '''
    Compile with numba the nb_frameprod_upper function.

    :param nthreads: int. Number of threads each of which computes a block of the environmental matrix
    :return: Compiled inner_func_nbupper function with threads
    '''
    nd2d = nb.double[:,:]; nd2int = nb.uint32[:,:]; nd3d = nb.double[:,:,:]
    signatureEnv = nb.void(nd2d, nd2int,  nd3d,nd2int, nd3d, nd2d)

    inner_func_nbupper = nb.jit(signatureEnv, nopython=True,nogil=True,cache=True)(nb_frameprod_upper)

    if nthreads == 1:
        print('1 threaded calc')
        func_nbupper = make_singlethread_envKernel(inner_func_nbupper)
    elif nthreads in [2,4,6,9]:
        print('{:.0f} threaded calc'.format(nthreads))
        func_nbupper = make_multithread_envKernel(inner_func_nbupper, nthreads)
    else:
        print('Unsuported nthreads number\n 1 threaded calc')
        func_nbupper = make_singlethread_envKernel(inner_func_nbupper)
    return func_nbupper


def nb_frameprod_upper(result, keys1, vals1, keys2, vals2, chemicalKernelmat):
    '''
    Computes the environmental matrix between two AlchemyFrame.
    :param result: np.array. output
    :param keys1: np.array 2D. list of keys->(species1,species2), i.e. chemical channels, of AlchemyFrame1.
    :param vals1: np.array 3D. [environment center, chemical channel, soap vector].
    :param keys2: np.array 2D. list of keys->(species1,species2), i.e. chemical channels, of AlchemyFrame2.
    :param vals2: np.array 3D. [environment center, chemical channel, soap vector].
    :param chemicalKernelmat: np.array 2D.
    :return: np.array 2D. Environmental matrix.
    '''
    Nenv1, nA, nL = vals1.shape
    Nenv2, nB, nL = vals2.shape
    for it in range(Nenv1):
        for jt in range(Nenv2):
            EnvironmentalSimilarity = 0.
            for nt in range(nA):
                spA = keys1[nt, :]

                for mt in range(nB):
                    spB = keys2[mt, :]

                    theta1 = chemicalKernelmat[spA[0], spB[0]] * chemicalKernelmat[spA[1], spB[1]]
                    theta2 = chemicalKernelmat[spA[1], spB[0]] * chemicalKernelmat[spA[0], spB[1]]

                    if theta1 == 0. and theta2 == 0.:
                        continue

                    pp = 0.
                    for kt in range(nL):
                        pp += vals1[it, nt, kt] * vals2[jt, mt, kt]

                    # the symmetry of the chemicalKernel and chemical soap vector is a bit messy
                    if spA[0] != spA[1] and spB[0] != spB[1]:
                        EnvironmentalSimilarity += theta1 * pp * 2 + theta2 * pp * 2
                    elif (spA[0] == spA[1] and spB[0] != spB[1]) or (spA[0] != spA[1] and spB[0] == spB[1]):
                        EnvironmentalSimilarity += theta1 * pp + theta2 * pp
                    elif spA[0] == spA[1] and spB[0] == spB[1]:
                        EnvironmentalSimilarity += theta1 * pp

            result[it, jt] = EnvironmentalSimilarity


def np_frameprod1(keys1, vals1, keys2, vals2, chemicalKernelmat):
    Nenv1, Na, Nsoap = vals1.shape
    Nenv2, Nb, Nsoap = vals2.shape

    k = np.zeros((Nenv1, Nenv2))
    for it in range(Nenv1):
        for jt in range(Nenv2):
            similarity = 0.
            for nt, spA in enumerate(keys1):
                for mt, spB in enumerate(keys2):
                    theta = chemicalKernelmat[spA[0], spB[0]] * chemicalKernelmat[spA[1], spB[1]]
                    if theta != 0.:
                        similarity += theta * np.vdot(vals1[it, nt, :], vals2[jt, mt, :])

            k[it, jt] = similarity

    return k


def np_frameprod_upper(keys1, vals1, keys2, vals2, chemicalKernelmat):
    nenv1, Na, Nsoap = vals1.shape
    nenv2, Nb, Nsoap = vals2.shape
    k = np.zeros((nenv1, nenv2))
    for it in range(nenv1):
        for jt in range(nenv2):
            similarity = 0.
            for nt, spA in enumerate(keys1):
                for mt, spB in enumerate(keys2):

                    theta1 = chemicalKernelmat[spA[0], spB[0]] * chemicalKernelmat[spA[1], spB[1]]
                    theta2 = chemicalKernelmat[spA[1], spB[0]] * chemicalKernelmat[spA[0], spB[1]]
                    if theta1 == 0. and theta2 == 0.:
                        continue

                    pp = np.dot(vals1[it, nt, :], vals2[jt, mt, :])

                    # the symmetry of the chemicalKernel and chemical soap vector is a bit messy
                    if spA[0] != spA[1] and spB[0] != spB[1]:
                        similarity += (theta1  + theta2) * pp * 2
                    elif (spA[0] == spA[1] and spB[0] != spB[1]) or (spA[0] != spA[1] and spB[0] == spB[1]):
                        similarity += (theta1 + theta2) * pp
                    elif spA[0] == spA[1] and spB[0] == spB[1]:
                        similarity += theta1 * pp

            k[it, jt] = similarity

    return k

def np_frameprod3(keys1, vals1, keys2, vals2, chemicalKernelmat):
    Nenv1, Na, Nsoap = vals1.shape
    Nenv2, Nb, Nsoap = vals2.shape

    theta = np.zeros((Na, Nb))

    for nt, spA in enumerate(keys1):
        for mt, spB in enumerate(keys2):
            theta1 = chemicalKernelmat[spA[0], spB[0]] * chemicalKernelmat[spA[1], spB[1]]
            theta2 = chemicalKernelmat[spA[1], spB[0]] * chemicalKernelmat[spA[0], spB[1]]
            if theta1 == 0. and theta2 == 0.:
                continue
            # the symmetry of the chemicalKernel and chemical soap vector is a bit messy
            if spA[0] != spA[1] and spB[0] != spB[1]:
                theta[nt, mt] = theta1 * 2 + theta2 * 2
            elif (spA[0] == spA[1] and spB[0] != spB[1]) or (spA[0] != spA[1] and spB[0] == spB[1]):
                theta[nt, mt] = theta1  + theta2
            elif spA[0] == spA[1] and spB[0] == spB[1]:
                theta[nt, mt] = theta1

    k = np.einsum('kl,iko,jlo->ij', theta, vals1, vals2, optimize=True)

    return k


def framesprod(frames1, frames2=None, chemicalKernelmat=None, frameprodFunc=None):
    envkernels = {}
    if frames2 is None:
        frames2 = frames1
        for it, frame1 in enumerate(frames1):
            keys1, vals1 = frame1.get_arrays()
            for jt, frame2 in enumerate(frames2):
                if it > jt:
                    continue
                keys2, vals2 = frame2.get_arrays()
                kargs = {'keys1': keys1, 'keys2': keys2, 'vals1': vals1, 'vals2': vals2,
                         'chemicalKernelmat': chemicalKernelmat}
                envkernels[(it, jt)] = frameprodFunc(**kargs)

        return envkernels

    else:
        for it, frame1 in enumerate(frames1):
            keys1, vals1 = frame1.get_arrays()
            for jt, frame2 in enumerate(frames2):
                keys2, vals2 = frame2.get_arrays()
                kargs = {'keys1': keys1, 'keys2': keys2, 'vals1': vals1, 'vals2': vals2,
                         'chemicalKernelmat': chemicalKernelmat}
                envkernels[(it, jt)] = frameprodFunc(**kargs)

        return envkernels

