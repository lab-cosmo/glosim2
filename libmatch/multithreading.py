
import quippy as qp
import threading
import numpy as np
from copy import deepcopy
import multiprocessing as mp
import os


def chunk_list(lll, nchunks):
    N = len(lll)
    if nchunks == 1:
        slices = [range(N)]
        chunks = [lll]
    else:
        chunklen = N // nchunks
        chunkrest = N % nchunks
        slices = [range(i * chunklen, (i + 1) * chunklen) for i in range(nchunks)]
        for it in range(chunkrest):
            slices[-1].append(slices[-1][-1] + 1)
        chunks = [lll[slices[i][0]:slices[i][-1] + 1] for i in range(nchunks)]

    return chunks, slices


def chunks1d_2_chuncks2d(chunk_1d, **kargs):
    if isinstance(chunk_1d[0], qp.io.AtomsList):
        key = ['atoms1', 'atoms2']
    else:
        key = ['frames1', 'frames2']
    chunks = []
    iii = 0
    for nt, ch1 in enumerate(chunk_1d):
        for mt, ch2 in enumerate(chunk_1d):
            if nt > mt:
                continue
            if nt == mt:

                aa = {key[0]: deepcopy(ch1), key[1]: None}
                bb = deepcopy(kargs)
                aa.update(**bb)
                chunks.append(aa)
            else:

                aa = {key[0]: deepcopy(ch1), key[1]: deepcopy(ch2)}
                bb = deepcopy(kargs)
                aa.update(**bb)
                chunks.append(aa)
            iii += 1
    return chunks


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


def make_singlethread(inner_func):
    def func(**kargs):
        Nenv1, nA, nL = kargs['vals1'].shape
        Nenv2, nB, nL = kargs['vals2'].shape

        result = np.empty((Nenv1, Nenv2), dtype=np.float64)
        inner_func(result, **kargs)
        return result

    return func


def make_multithread_envKernel(inner_func, numthreadsTot):
    def func_mt(**kargs):
        Nenv1, nA, nL = kargs['vals1'].shape
        Nenv2, nB, nL = kargs['vals2'].shape
        result = np.zeros((Nenv1, Nenv2), dtype=np.float64)

        keys = kargs.keys()

        keys1, keys2, vals1, vals2, chemicalKernelmat = [kargs['keys1'], kargs['keys2'], kargs['vals1'], kargs['vals2'], \
                                                         kargs['chemicalKernelmat']]

        numthreadsTot2nthreads = {2: (2, 1), 4: (2, 2), 6: (3, 2), 9: (3, 3)}
        numthreads1, numthreads2 = numthreadsTot2nthreads[numthreadsTot]

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

        threads = [threading.Thread(target=inner_func, kwargs=chunk) for chunk in chunks[:-1]]

        for thread in threads:
            thread.start()

        # the main thread handles the last chunk
        inner_func(**chunks[-1])

        for thread in threads:
            thread.join()
        return result

    return func_mt
