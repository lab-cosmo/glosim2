import numpy as np
import numba as nb
from ctypes import pythonapi, c_void_p
from multithreading import make_multithread_envKernel,make_singlethread


def compile_with_threads(nbfunc,nthreads=1):

    nd2d = nb.double[:,:]; nd2int = nb.uint32[:,:]; nd3d = nb.double[:,:,:]
    signatureEnv = nb.void(nd2d, nd2int,  nd3d,nd2int, nd3d, nd2d)

    inner_func_nbupper = nb.jit(signatureEnv, nopython=True,nogil=True)(nbfunc)

    if nthreads == 1:
        print('1 threaded calc')
        func_nbupper = make_singlethread(inner_func_nbupper)
    elif nthreads in [2,4,6,9]:
        print('{:.0f} threaded calc'.format(nthreads))
        func_nbupper = make_multithread_envKernel(inner_func_nbupper, nthreads)
    else:
        print('Unsuported nthreads number\n 1 threaded calc')
        func_nbupper = make_singlethread(inner_func_nbupper)
    return func_nbupper


def nb_frameprod_upper(result, keys1, vals1, keys2, vals2, chemicalKernelmat):
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


def np_frameprod3(keys1, vals1, keys2, vals2, chemicalKernelmat):
    Nenv1, Na, Nsoap = vals1.shape
    Nenv2, Nb, Nsoap = vals2.shape

    k = np.zeros((Nenv1, Nenv2))
    theta = np.zeros((Na, Nb))

    for nt, spA in enumerate(keys1):
        for mt, spB in enumerate(keys2):
            theta[nt, mt] = chemicalKernelmat[spA[0], spB[0]] * chemicalKernelmat[spA[1], spB[1]]

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
