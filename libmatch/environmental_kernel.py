import numpy as np

from libmatch.chemical_kernel import Atoms2ChemicalKernelmat
from libmatch.soap import get_Soaps
from libmatch.utils import chunk_list, chunks1d_2_chuncks2d,is_notebook,dummy_queue
from soap import get_Soaps
import multiprocessing as mp
import signal, psutil, os
import threading

import quippy as qp

if is_notebook():
    from tqdm import tqdm_notebook as tqdm_cs
else:
    from tqdm import tqdm as tqdm_cs

try:
    import numba as nb
    nonumba = False

    signatureEnv = 'void(double[:, :], uint32[:, :], double[:, :, :], ' \
                   'uint32[:, :], double[:, :, :], double[:, :])'

    @nb.njit(signatureEnv,parallel=True)
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
        for it in nb.prange(Nenv1):
            for jt in nb.prange(Nenv2):
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


    @nb.njit(signatureEnv,parallel=True)
    def nb_frameprod_upper_delta(result, keys1, vals1, keys2, vals2, chemicalKernelmat):

        Nenv1, nA, nL = vals1.shape
        Nenv2, nB, nL = vals2.shape

        mm = np.zeros((nA,), np.int32)
        union = np.zeros((nA, 2), np.int32)
        for it in nb.prange(nA):
            isUnion = False
            for jt in range(nB):
                if keys1[it][0] == keys2[jt][0] and keys1[it][1] == keys2[jt][1]:
                    mm[it] = jt
                    isUnion = True
                    continue

            if isUnion is True:
                union[it][0] = keys1[it][0]
                union[it][1] = keys1[it][1]

        for it in nb.prange(Nenv1):
            for jt in nb.prange(Nenv2):
                EnvironmentalSimilarity = 0.

                for nt in range(nA):

                    if union[nt, 0] == 0 and union[nt, 1] == 0:
                        continue

                    pp = 0.
                    for kt in range(nL):
                        pp += vals1[it, nt, kt] * vals2[jt, mm[nt], kt]

                    if union[nt, 0] == union[nt, 1]:
                        EnvironmentalSimilarity += pp
                    else:
                        EnvironmentalSimilarity += pp * 2


                result[it, jt] = EnvironmentalSimilarity

except:
    nonumba = True


# def framesprod(frames1, frames2=None, chemicalKernelmat=None, frameprodFunc=None):
#     '''
#     Computes the environmental matrices between two list of AlchemyFrame.
#
#     :param frames1: list of AlchemyFrame.
#     :param frames2: list of AlchemyFrame.
#     :param chemicalKernelmat:
#     :param frameprodFunc: function to use to compute a environmental kernel matrix
#     :return: dictionary of environmental kernel matrices -> (i,j):environmentalMatrix(frames1[i],frames2[j])
#     '''
#     envkernels = {}
#     if frames2 is None:
#         # when with itself only the upper global matrix is computed
#         frames2 = frames1
#         for it, frame1 in enumerate(frames1):
#             keys1, vals1 = frame1.get_arrays()
#             for jt, frame2 in enumerate(frames2):
#                 if it > jt:
#                     continue
#                 keys2, vals2 = frame2.get_arrays()
#                 kargs = {'keys1': keys1, 'keys2': keys2, 'vals1': vals1, 'vals2': vals2,
#                          'chemicalKernelmat': chemicalKernelmat}
#                 envkernels[(it, jt)] = frameprodFunc(**kargs)
#
#
#     else:
#         for it, frame1 in enumerate(frames1):
#             keys1, vals1 = frame1.get_arrays()
#             for jt, frame2 in enumerate(frames2):
#                 keys2, vals2 = frame2.get_arrays()
#                 kargs = {'keys1': keys1, 'keys2': keys2, 'vals1': vals1, 'vals2': vals2,
#                          'chemicalKernelmat': chemicalKernelmat}
#                 envkernels[(it, jt)] = frameprodFunc(**kargs)
#
#     return envkernels



def framesprod(frames1, frames2=None, chemicalKernelmat=None, frameprodFunc=None, queue=None,dispbar=False):
    '''
    Computes the environmental matrices between two list of AlchemyFrame.

    :param frames1: list of AlchemyFrame.
    :param frames2: list of AlchemyFrame.
    :param chemicalKernelmat:
    :param frameprodFunc: function to use to compute a environmental kernel matrix
    :return: dictionary of environmental kernel matrices -> (i,j):environmentalMatrix(frames1[i],frames2[j])
    '''


    if queue is None:
        if frames2 is None:
            Niter = len(frames1)*(len(frames1)+1)/2
        else:
            Niter = len(frames1)*len(frames2)
        queue = dummy_queue(Niter,'Env kernels',dispbar=dispbar)


    envkernels = {}

    if frames2 is None:
        # when with itself only the upper global matrix is computed
        frames2 = frames1
        for it, frame1 in enumerate(frames1):
            keys1, vals1 = frame1.get_arrays()
            # ii = 0
            for jt, frame2 in enumerate(frames2):
                if it > jt:
                    continue
                keys2, vals2 = frame2.get_arrays()
                kargs = {'keys1': keys1, 'keys2': keys2, 'vals1': vals1, 'vals2': vals2,
                         'chemicalKernelmat': chemicalKernelmat}
                envkernels[(it, jt)] = frameprodFunc(**kargs)
                # ii += 1

                queue.put(1)
    else:

        for it, frame1 in enumerate(frames1):
            keys1, vals1 = frame1.get_arrays()
            # ii = 0
            for jt, frame2 in enumerate(frames2):
                keys2, vals2 = frame2.get_arrays()
                kargs = {'keys1': keys1, 'keys2': keys2, 'vals1': vals1, 'vals2': vals2,
                         'chemicalKernelmat': chemicalKernelmat}
                envkernels[(it, jt)] = frameprodFunc(**kargs)
                # ii += 1
                queue.put(1)
    return envkernels

    # if frames2 is None:
    #     # when with itself only the upper global matrix is computed
    #     frames2 = frames1
    #     n = len(frames1)
    #     with tqdm_cs(total=int(n*(n-1)/2.),desc='Process {}'.format(proc_id),leave=True,position=pos,ascii=ascii,disable=disable_pbar) as pbar:
    #         for it, frame1 in enumerate(frames1):
    #             keys1, vals1 = frame1.get_arrays()
    #             ii = 0
    #             for jt, frame2 in enumerate(frames2):
    #                 if it > jt:
    #                     continue
    #                 keys2, vals2 = frame2.get_arrays()
    #                 kargs = {'keys1': keys1, 'keys2': keys2, 'vals1': vals1, 'vals2': vals2,
    #                          'chemicalKernelmat': chemicalKernelmat}
    #                 envkernels[(it, jt)] = frameprodFunc(**kargs)
    #                 ii += 1
    #             pbar.update(ii)
    #             queue.put(ii)
    # else:
    #     n1 = len(frames1)
    #     n2 = len(frames2)
    #     with tqdm_cs(total=int(n1*n2), desc='Process {}'.format(proc_id),leave=True,position=pos,ascii=True,disable=disable_pbar) as pbar:
    #         for it, frame1 in enumerate(frames1):
    #             keys1, vals1 = frame1.get_arrays()
    #             ii = 0
    #             for jt, frame2 in enumerate(frames2):
    #                 keys2, vals2 = frame2.get_arrays()
    #                 kargs = {'keys1': keys1, 'keys2': keys2, 'vals1': vals1, 'vals2': vals2,
    #                          'chemicalKernelmat': chemicalKernelmat}
    #                 envkernels[(it, jt)] = frameprodFunc(**kargs)
    #                 ii += 1
    #             pbar.update(ii)
    #             queue.put(ii)
    # return envkernels

def nb_frameprod_upper_multithread(**kargs):
    Nenv1, nA, nL = kargs['vals1'].shape
    Nenv2, nB, nL = kargs['vals2'].shape
    result = np.zeros((Nenv1, Nenv2), dtype=np.float64)

    keys1, keys2, vals1, vals2, chemicalKernelmat = [kargs['keys1'], kargs['keys2'], kargs['vals1'], kargs['vals2'], \
                                                     kargs['chemicalKernelmat']]

    numthreadsTot2nthreads = {2: (2, 1), 4: (2, 2), 6: (3, 2), 9: (3, 3),
                              12: (4, 3), 16: (4, 4), 25: (5, 5), 36: (6, 6),
                              48: (7, 7), 64: (8,8), 81: (9,9), 100: (10,10)}
    numthreads1, numthreads2 = numthreadsTot2nthreads[4]

    chunks1, slices1 = chunk_list(vals1, numthreads1)
    chunks2, slices2 = chunk_list(vals2, numthreads2)

    chunks = []
    for it in range(numthreads1):
        for jt in range(numthreads2):
            chunks3 = result[slices1[it][0]:slices1[it][-1] + 1, slices2[jt][0]:slices2[jt][-1] + 1]
            a = {'result': chunks3, 'chemicalKernelmat': chemicalKernelmat.copy(), 'keys1': keys1.copy(), 'keys2': keys2.copy()}
            a.update(**{'vals1': chunks1[it]})
            a.update(**{'vals2': chunks2[jt]})

            chunks.append(a)

    threads = [threading.Thread(target=nb_frameprod_upper, kwargs=chunk) for chunk in chunks]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return result

def nb_frameprod_upper_singlethread(**kargs):
    Nenv1, nA, nL = kargs['vals1'].shape
    Nenv2, nB, nL = kargs['vals2'].shape

    result = np.empty((Nenv1, Nenv2), dtype=np.float64)
    nb_frameprod_upper(result, **kargs)
    return result

def nb_frameprod_upper_delta_singlethread(**kargs):
    Nenv1, nA, nL = kargs['vals1'].shape
    Nenv2, nB, nL = kargs['vals2'].shape

    result = np.empty((Nenv1, Nenv2), dtype=np.float64)
    nb_frameprod_upper_delta(result, **kargs)
    return result

def nb_frameprod_upper_delta_multithread(**kargs):
    Nenv1, nA, nL = kargs['vals1'].shape
    Nenv2, nB, nL = kargs['vals2'].shape
    result = np.zeros((Nenv1, Nenv2), dtype=np.float64)

    keys1, keys2, vals1, vals2, chemicalKernelmat = [kargs['keys1'], kargs['keys2'], kargs['vals1'], kargs['vals2'], \
                                                     kargs['chemicalKernelmat']]

    numthreadsTot2nthreads = {2: (2, 1), 4: (2, 2), 6: (3, 2), 9: (3, 3),
                              12: (4, 3), 16: (4, 4), 25: (5, 5), 36: (6, 6),
                              48: (7, 7), 64: (8,8), 81: (9,9), 100: (10,10)}
    numthreads1, numthreads2 = numthreadsTot2nthreads[4]

    chunks1, slices1 = chunk_list(vals1, numthreads1)
    chunks2, slices2 = chunk_list(vals2, numthreads2)

    chunks = []
    for it in range(numthreads1):
        for jt in range(numthreads2):
            chunks3 = result[slices1[it][0]:slices1[it][-1] + 1, slices2[jt][0]:slices2[jt][-1] + 1]
            a = {'result': chunks3, 'chemicalKernelmat': chemicalKernelmat.copy(), 'keys1': keys1.copy(), 'keys2': keys2.copy()}
            a.update(**{'vals1': chunks1[it]})
            a.update(**{'vals2': chunks2[jt]})

            chunks.append(a)

    threads = [threading.Thread(target=nb_frameprod_upper_delta, kwargs=chunk) for chunk in chunks]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
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


def choose_envKernel_func(nthreads=4, isDeltaKernel=False,verbose=False):
    '''
    Compile with numba the nb_frameprod_upper function.

    :param isDeltaKernel: 
    :param nthreads: int. Number of threads each of which computes a block of the environmental matrix
    :return: Compiled inner_func_nbupper function with threads
    '''
    if nonumba:
        print 'Using numpy version of envKernel function'
        get_envKernel = np_frameprod3_upper
    else:
        if verbose:
            print 'Using compiled and threaded version of envKernel function'

        if nthreads == 1:
            if verbose:
                print('1 threaded calc')
            if isDeltaKernel:
                if verbose:
                    print 'with implicit delta kernel function'
                get_envKernel = nb_frameprod_upper_delta_singlethread
            else:
                if verbose:
                    print 'with explicit delta kernel function'
                get_envKernel = nb_frameprod_upper_singlethread
        elif nthreads in [2,4,6,9,12,16,25,36,48,64,81,100]:
            if verbose:
                print('{:.0f} threaded calc'.format(nthreads))
            if isDeltaKernel:
                if verbose:
                    print 'with implicit delta kernel function'
                get_envKernel = nb_frameprod_upper_delta_multithread
            else:
                get_envKernel = nb_frameprod_upper_multithread
        else:
            print('Unsuported nthreads number\n 1 threaded calc')
            get_envKernel = nb_frameprod_upper_singlethread

    return get_envKernel


def framesprod_wrapper(kargs):
    keys = kargs.keys()
    get_envKernel = kargs.pop('frameprodFunc')
    queue = kargs.pop('queue')
    # to disable the progressbar
    dispbar = kargs.pop('dispbar')

    if 'fpointers1' in keys:
        fpointers1 = kargs.pop('fpointers1')
        fpointers2 = kargs.pop('fpointers2')
        atoms1 = [qp.Atoms(fpointer=fpointer1) for fpointer1 in fpointers1]

        chemicalKernelmat = kargs.pop('chemicalKernelmat')

        frames1 = get_Soaps(atoms1,dispbar=dispbar, **kargs)
        if fpointers2 is not None:
            atoms2 = [qp.Atoms(fpointer=fpointer2) for fpointer2 in fpointers2]
            frames2 = get_Soaps(atoms2,dispbar=dispbar, **kargs)
        else:
            frames2 = None

        kargs = {'frames1': frames1, 'frames2': frames2,
                 'chemicalKernelmat': chemicalKernelmat}

    elif 'atoms1' in keys:
        atoms1 = kargs.pop('atoms1')
        atoms2 = kargs.pop('atoms2')
        chemicalKernelmat = kargs.pop('chemicalKernelmat')

        frames1 = get_Soaps(atoms1,dispbar=dispbar, **kargs)
        if atoms2 is not None:
            frames2 = get_Soaps(atoms2,dispbar=dispbar, **kargs)
        else:
            frames2 = None

        kargs = {'frames1': frames1, 'frames2': frames2, 'chemicalKernelmat': chemicalKernelmat}

    return framesprod(queue=queue,frameprodFunc=get_envKernel,**kargs)

# class mp_framesprod(object):
#     def __init__(self, chunks, nprocess, nthreads):
#
#         self.nprocess = nprocess
#         self.nthreads = nthreads
#         self.func = framesprod_wrapper
#         # get the function to compute an environmental kernel
#         self.get_envKernel = choose_envKernel_func(nthreads)
#         # add the frameprodFunc to the input chunks
#         for chunk in chunks:
#             chunk.update(**{'frameprodFunc': self.get_envKernel})
#         self.chunks = chunks
#
#     def run(self):
#         pool = mp.Pool(self.nprocess)
#         results = pool.map(self.func, self.chunks)
#
#         pool.close()
#         pool.join()
#
#         return results

class mp_framesprod(object):
    def __init__(self, chunks, nprocess, nthreads, Niter,isDeltaKernel,dispbar=False):
        super(mp_framesprod, self).__init__()
        self.func = framesprod_wrapper
        self.parent_id = os.getpid()
        self.get_envKernel = choose_envKernel_func(nthreads,isDeltaKernel)

        self.nprocess = nprocess
        self.nthreads = nthreads
        self.dispbar = dispbar
        manager = mp.Manager()
        self.queue = manager.Queue()

        for chunk in chunks:
            chunk.update(**{"queue": self.queue,'frameprodFunc': self.get_envKernel,
                            'dispbar':self.dispbar})
        self.chunks = chunks

        self.thread = threading.Thread(target=self.listener, args=(self.queue, Niter,dispbar))
        self.thread.start()
        self.pool = mp.Pool(nprocess, initializer=self.worker_init,maxtasksperchild=1)

    def run(self):
        res = self.pool.map(self.func, self.chunks)
        self.pool.close()
        self.pool.join()
        self.queue.put(None)
        self.thread.join()
        return res

    @staticmethod
    def listener(queue, Niter,dispbar):
        print 'listener ',dispbar
        tbar = tqdm_cs(total=int(Niter),desc='Env kernels',disable=dispbar)
        for ii in iter(queue.get, None):
            tbar.update(ii)
        tbar.close()

    # clean kill of the pool in interactive sessions
    def worker_init(self):
        def sig_int(signal_num, frame):
            print('signal: %s' % signal_num)
            parent = psutil.Process(self.parent_id)
            for child in parent.children():
                if child.pid != os.getpid():
                    print("killing child: %s" % child.pid)
                    child.kill()
            print("killing parent: %s" % self.parent_id)
            parent.kill()
            print("suicide: %s" % os.getpid())
            psutil.Process(os.getpid()).kill()

        signal.signal(signal.SIGINT, sig_int)


def join_envKernel(results, slices,slices_1=None):
    if slices_1 is None:
        slices_1 = slices
        diag = True
        rr = list(set([it for sl in slices for it in sl]))
        joined_results = {(it, jt): None for it in rr for jt in rr if jt >= it}
    else:
        diag = False
        rr1 = list(set([it for sl in slices for it in sl]))
        rr2 = list(set([it for sl in slices_1 for it in sl]))
        joined_results = {(it, jt): None for it in rr1 for jt in rr2}

    iii = 0
    for nt, sl1 in enumerate(slices):
        for mt, sl2 in enumerate(slices_1):
            if diag is True:
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
            else:
                for it, s1 in enumerate(sl1):
                        for jt, s2 in enumerate(sl2):
                            joined_results[(s1, s2)] = results[iii][(it, jt)]

            iii += 1
    return joined_results


def get_environmentalKernels_mt_mp_chunks(atoms, nocenters=None, chem_channels=True, centerweight=1.0,
                             gaussian_width=0.5, cutoff=3.5,cutoff_transition_width=0.5,
                             nmax=8, lmax=6, chemicalKernelmat=None, chemicalKernel=None,
                             chemicalProjection=None,
                             nthreads=4, nprocess=2, nchunks=2,islow_memory=False,isDeltaKernel=True,
                             dispbar=False,is_fast_average=False):
    if nocenters is None:
        nocenters = []

    # Builds the kernel matrix from the species present in the frames and a specified chemical
    # kernel function

    if chemicalKernelmat is not None:
        pass
    elif chemicalProjection is not None:
        pass
    elif (chemicalKernelmat is None) and (chemicalKernel is not None):
        chemicalKernelmat = Atoms2ChemicalKernelmat(atoms, chemicalKernel=chemicalKernel)
    else:
        raise ValueError('wrong chemicalKernelmat and/or chemicalKernel input')

    Natoms = len(atoms)
    NenvKernels = Natoms * (Natoms + 1) / 2.

    # fpointers = [frame._fpointer.copy() for frame in atoms]
    # chunks1d, slices = chunk_list(fpointers, nchunks=nchunks)
    # cut atomsList in chunks
    if islow_memory:
        frames = get_Soaps(atoms, nocenters=nocenters, chem_channels=chem_channels, centerweight=centerweight,
                           gaussian_width=gaussian_width, cutoff=cutoff,is_fast_average=is_fast_average,
                           chemicalProjection=chemicalProjection,
                           cutoff_transition_width=cutoff_transition_width, nmax=nmax, lmax=lmax, nprocess=nprocess)
        chunks1d, slices = chunk_list(frames, nchunks=nchunks)

    else:
        chunks1d, slices = chunk_list(atoms, nchunks=nchunks)

    soap_params = {'centerweight': centerweight, 'gaussian_width': gaussian_width,
                   'cutoff': cutoff, 'cutoff_transition_width': cutoff_transition_width,
                   'nmax': nmax, 'lmax': lmax, 'chemicalKernelmat': chemicalKernelmat,
                   'chemicalProjection':chemicalProjection,
                   'chem_channels': chem_channels, 'nocenters': nocenters, 'is_fast_average':is_fast_average,
                   }

    # create inputs for each block of the global kernel matrix
    chunks = chunks1d_2_chuncks2d(chunks1d, **soap_params)

    # new_atoms1 = {}
    # new_atoms2 = {}
    # for it,chunk in enumerate(chunks):
    #     atoms1 = chunk.pop('atoms1')
    #     atoms2 = chunk.pop('atoms2')
    #     # new_atoms1[it] = [qp.Atoms().copy_from(frame) for frame in atoms1]
    #     new_atoms1[it] = [frame.copy() for frame in atoms1]
    #     fpointers1 = [frame._fpointer.copy() for frame in new_atoms1[it]]
    #     if atoms2 is not None:
    #         # new_atoms2[it] = [qp.Atoms().copy_from(frame) for frame in atoms2]
    #         new_atoms2[it] = [frame.copy() for frame in atoms2]
    #         fpointers2 = [frame._fpointer.copy() for frame in new_atoms2[it]]
    #     else:
    #         fpointers2 = None
    #
    #     chunk.update(**{'fpointers1':fpointers1,'fpointers2':fpointers2})

    # get a list of environemental kernels
    pool = mp_framesprod(chunks, nprocess, nthreads, NenvKernels,
                         isDeltaKernel=isDeltaKernel,dispbar=dispbar)
    results = pool.run()
    # reorder the list of environemental kernels into a dictionary which keys are the (i,j) of the global kernel matrix
    environmentalKernels = join_envKernel(results, slices)

    return environmentalKernels


def get_environmentalKernels_singleprocess(atoms, nocenters=None, chem_channels=True, centerweight=1.0,
                                           gaussian_width=0.5, cutoff=3.5, cutoff_transition_width=0.5,
                                           nmax=8, lmax=6, chemicalKernelmat=None, chemicalKernel=None,
                                           chemicalProjection=None,
                                           nthreads=4, nprocess=0, nchunks=0,isDeltaKernel=True,
                                           dispbar=False,is_fast_average=False):
    if nocenters is None:
        nocenters = []

    # Chooses the function to use to compute the kernel between two frames
    get_envKernel = choose_envKernel_func(nthreads,isDeltaKernel)

    # Builds the kernel matrix from the species present in the frames and a specified chemical
    # kernel function
    if chemicalKernelmat is not None:
        pass
    elif chemicalProjection is not None:
        pass
    elif chemicalKernelmat is None and chemicalKernel is not None:
        chemicalKernelmat = Atoms2ChemicalKernelmat(atoms, chemicalKernel=chemicalKernel)
    else:
        raise ValueError('wrong chemicalKernelmat and/or chemicalKernel input')

    # get the soap for every local environement
    frames = get_Soaps(atoms, nocenters=nocenters, chem_channels=chem_channels, centerweight=centerweight,
                       gaussian_width=gaussian_width, cutoff=cutoff, cutoff_transition_width=cutoff_transition_width,
                       nmax=nmax, lmax=lmax, nprocess=nprocess,chemicalProjection=chemicalProjection,
                       dispbar=dispbar,is_fast_average=is_fast_average)

    # get the environmental kernels as a dictionary
    environmentalKernels = framesprod(frames, frameprodFunc=get_envKernel, chemicalKernelmat=chemicalKernelmat,
                                      dispbar=dispbar)


    return environmentalKernels