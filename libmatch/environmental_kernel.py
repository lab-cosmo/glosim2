import numpy as np

from libmatch.chemical_kernel import Atoms2ChemicalKernelmat
from libmatch.soap import get_Soaps
from libmatch.utils import chunk_list, chunks1d_2_chuncks2d
from soap import get_Soaps
import multiprocessing as mp
import threading

try:
    import numba as nb
    nonumba = False

    signatureEnv = 'void(double[:, :], uint32[:, :], double[:, :, :], ' \
                   'uint32[:, :], double[:, :, :], double[:, :])'

    @nb.jit(signatureEnv, nopython=True, nogil=True, cache=True)
    def nb_frameprod_upper(result, keys1, vals1, keys2, vals2, chemicalKernelmat):
        '''
        Computes the environmental matrix between two AlchemyFrame. Only the upper
        chemical channels are actually computed. To be compiled with numba.
        :param result: np.array. output
        :param keys1: np.array 2D. list of keys->(species1,species2), i.e. chemical channels, of AlchemyFrame1.
        :param vals1: np.array 3D. [environment center, chemical channel, soap vector].
        :param keys2: np.array 2D. list of keys->(species1,species2), i.e. chemical channels, of AlchemyFrame2.
        :param vals2: np.array 3D. [environment center, chemical channel, soap vector].
        :param chemicalKernelmat: np.array 2D.
        :return: None. result is changed by 'reference'.
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

except:
    nonumba = True

def framesprod(frames1, frames2=None, chemicalKernelmat=None, frameprodFunc=None):
    '''
    Computes the environmental matrices between two list of AlchemyFrame.

    :param frames1: list of AlchemyFrame.
    :param frames2: list of AlchemyFrame.
    :param chemicalKernelmat:
    :param frameprodFunc: function to use to compute a environmental kernel matrix
    :return: dictionary of environmental kernel matrices -> (i,j):environmentalMatrix(frames1[i],frames2[j])
    '''
    envkernels = {}
    if frames2 is None:
        # when with itself only the upper global matrix is computed
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

def nb_frameprod_upper_multithread(**kargs):
    Nenv1, nA, nL = kargs['vals1'].shape
    Nenv2, nB, nL = kargs['vals2'].shape
    result = np.zeros((Nenv1, Nenv2), dtype=np.float64)

    keys1, keys2, vals1, vals2, chemicalKernelmat = [kargs['keys1'], kargs['keys2'], kargs['vals1'], kargs['vals2'], \
                                                     kargs['chemicalKernelmat']]

    numthreadsTot2nthreads = {2: (2, 1), 4: (2, 2), 6: (3, 2), 9: (3, 3)}
    numthreads1, numthreads2 = numthreadsTot2nthreads[4]

    chunks1, slices1 = chunk_list(vals1, numthreads1)
    chunks2, slices2 = chunk_list(vals2, numthreads2)

    chunks = []
    for it in range(numthreads1):
        for jt in range(numthreads2):
            chunks3 = result[slices1[it][0]:slices1[it][-1] + 1, slices2[jt][0]:slices2[jt][-1] + 1]
            a = {'result': chunks3, 'chemicalKernelmat': chemicalKernelmat, 'keys1': keys1, 'keys2': keys2}
            a.update(**{'vals1': chunks1[it]})
            a.update(**{'vals2': chunks2[jt]})

            chunks.append(a)

    threads = [threading.Thread(target=nb_frameprod_upper, kwargs=chunk) for chunk in chunks[:-1]]

    for thread in threads:
        thread.start()

    # the main thread handles the last chunk
    nb_frameprod_upper(**chunks[-1])

    for thread in threads:
        thread.join()
    return result

def nb_frameprod_upper_singlethread(**kargs):
    Nenv1, nA, nL = kargs['vals1'].shape
    Nenv2, nB, nL = kargs['vals2'].shape

    result = np.empty((Nenv1, Nenv2), dtype=np.float64)
    nb_frameprod_upper(result, **kargs)
    return result

def np_frameprod_upper(keys1, vals1, keys2, vals2, chemicalKernelmat):
    '''
    Computes the environmental matrix between two AlchemyFrame. Simplest implementation, very slow.
    Only the upperchemical channels are actually computed.

    :param keys1: np.array 2D. list of keys->(species1,species2), i.e. chemical channels, of AlchemyFrame1.
    :param vals1: np.array 3D. [environment center, chemical channel, soap vector].
    :param keys2: np.array 2D. list of keys->(species1,species2), i.e. chemical channels, of AlchemyFrame2.
    :param vals2: np.array 3D. [environment center, chemical channel, soap vector].
    :param chemicalKernelmat: np.array 2D.
    :return: np.array 2D. Environmental matrix.
    '''
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

def np_frameprod3_upper(keys1, vals1, keys2, vals2, chemicalKernelmat):
    '''
    Computes the environmental matrix between two AlchemyFrame. einsum implementaion, ~slow.
    Only the upperchemical channels are actually computed.

    :param keys1: np.array 2D. list of keys->(species1,species2), i.e. chemical channels, of AlchemyFrame1.
    :param vals1: np.array 3D. [environment center, chemical channel, soap vector].
    :param keys2: np.array 2D. list of keys->(species1,species2), i.e. chemical channels, of AlchemyFrame2.
    :param vals2: np.array 3D. [environment center, chemical channel, soap vector].
    :param chemicalKernelmat: np.array 2D.
    :return: np.array 2D. Environmental matrix.
    '''
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


def choose_envKernel_func(nthreads=4):
    '''
    Compile with numba the nb_frameprod_upper function.

    :param nthreads: int. Number of threads each of which computes a block of the environmental matrix
    :return: Compiled inner_func_nbupper function with threads
    '''
    if nonumba:
        print 'Using numpy version of envKernel function'
        get_envKernel = np_frameprod3_upper
    else:
        print 'Using compiled and threaded version of envKernel function'

        if nthreads == 1:
            print('1 threaded calc')
            get_envKernel = nb_frameprod_upper_singlethread
        elif nthreads in [2,4,6,9]:
            print('{:.0f} threaded calc'.format(nthreads))
            get_envKernel = nb_frameprod_upper_multithread
        else:
            print('Unsuported nthreads number\n 1 threaded calc')
            get_envKernel = nb_frameprod_upper_singlethread

    return get_envKernel


def framesprod_wrapper(kargs):
    keys = kargs.keys()
    get_envKernel = kargs.pop('frameprodFunc')

    if 'atoms1' in keys:
        atoms1 = kargs.pop('atoms1')
        atoms2 = kargs.pop('atoms2')
        chemicalKernelmat = kargs.pop('chemicalKernelmat')

        frames1 = get_Soaps(atoms1, **kargs)
        if atoms2 is not None:
            frames2 = get_Soaps(atoms2, **kargs)
        else:
            frames2 = None

        kargs = {'frames1': frames1, 'frames2': frames2, 'chemicalKernelmat': chemicalKernelmat}

    return framesprod(frameprodFunc=get_envKernel,**kargs)

class mp_framesprod(object):
    def __init__(self, chunks, nprocess, nthreads):

        self.nprocess = nprocess
        self.nthreads = nthreads
        self.func = framesprod_wrapper
        # get the function to compute an environmental kernel
        self.get_envKernel = choose_envKernel_func(nthreads)
        # add the frameprodFunc to the input chunks
        for chunk in chunks:
            chunk.update(**{'frameprodFunc': self.get_envKernel})
        self.chunks = chunks

    def run(self):
        pool = mp.Pool(self.nprocess)
        results = pool.map(self.func, self.chunks)

        pool.close()
        pool.join()

        return results


def join_envKernel(results, slices):
    rr = list(set([it for sl in slices for it in sl]))
    joined_results = {(it, jt): None for it in rr for jt in rr if jt >= it}

    iii = 0
    for nt, sl1 in enumerate(slices):
        for mt, sl2 in enumerate(slices):
            if nt > mt:
                continue

            if np.all(sl1 == sl2):

                for it, s1 in enumerate(sl1):
                    for jt, s2 in enumerate(sl2):
                        if s1 > s2:
                            continue
                        try:
                            joined_results[(s1, s2)] = results[iii][(it, jt)]
                        except:
                            print s1, s2, it, jt
            else:

                for it, s1 in enumerate(sl1):
                    for jt, s2 in enumerate(sl2):
                        try:
                            joined_results[(s1, s2)] = results[iii][(it, jt)]
                        except:
                            print s1, s2, it, jt

            iii += 1
    return joined_results


def get_environmentalKernels_mt_mp_chunks(atoms, nocenters=None, chem_channels=True, centerweight=1.0,
                             gaussian_width=0.5, cutoff=3.5,cutoff_transition_width=0.5,
                             nmax=8, lmax=6, chemicalKernelmat=None, chemicalKernel=None,
                             nthreads=4, nprocess=2, nchunks=2):
    if nocenters is None:
        nocenters = []

    # Builds the kernel matrix from the species present in the frames and a specified chemical
    # kernel function
    if chemicalKernelmat is not None:
        pass
    elif (chemicalKernelmat is None) and (chemicalKernel is not None):
        chemicalKernelmat = Atoms2ChemicalKernelmat(atoms, chemicalKernel=chemicalKernel)
    else:
        raise ValueError('wrong chemicalKernelmat and/or chemicalKernel input')

    # cut atomsList in chunks
    chunks1d, slices = chunk_list(atoms, nchunks=nchunks)

    soap_params = {'centerweight': centerweight, 'gaussian_width': gaussian_width,
                   'cutoff': cutoff, 'cutoff_transition_width': cutoff_transition_width,
                   'nmax': nmax, 'lmax': lmax, 'chemicalKernelmat': chemicalKernelmat,
                   'chem_channels': chem_channels, 'nocenters': nocenters,
                   }

    # create inputs for each block of the global kernel matrix
    chunks = chunks1d_2_chuncks2d(chunks1d, **soap_params)


    # get a list of environemental kernels
    pool = mp_framesprod(chunks, nprocess, nthreads)
    results = pool.run()
    # reorder the list of environemental kernels into a dictionary which keys are the (i,j) of the global kernel matrix
    environmentalKernels = join_envKernel(results, slices)

    return environmentalKernels


def get_environmentalKernels_singleprocess(atoms, nocenters=None, chem_channels=True, centerweight=1.0,
                                           gaussian_width=0.5, cutoff=3.5, cutoff_transition_width=0.5,
                                           nmax=8, lmax=6, chemicalKernelmat=None, chemicalKernel=None,
                                           nthreads=4, nprocess=0, nchunks=0):
    if nocenters is None:
        nocenters = []

    # Chooses the function to use to compute the kernel between two frames
    get_envKernel = choose_envKernel_func(nthreads)

    # Builds the kernel matrix from the species present in the frames and a specified chemical
    # kernel function
    if chemicalKernelmat is None and chemicalKernel is not None:
        chemicalKernelmat = Atoms2ChemicalKernelmat(atoms, chemicalKernel=chemicalKernel)
    else:
        raise ValueError('wrong chemicalKernelmat and/or chemicalKernel input')

    # get the soap for every local environement
    frames = get_Soaps(atoms, nocenters=nocenters, chem_channels=chem_channels,
                       centerweight=centerweight, gaussian_width=gaussian_width, cutoff=cutoff,
                       cutoff_transition_width=cutoff_transition_width, nmax=nmax, lmax=lmax)

    # get the environmental kernels as a dictionary
    environmentalKernels = framesprod(frames, frameprodFunc=get_envKernel, chemicalKernelmat=chemicalKernelmat)

    return environmentalKernels