import threading
import numpy as np

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


        keys1, keys2, vals1, vals2, chemicalKernelmat = [kargs['keys1'], kargs['keys2'], kargs['vals1'], kargs['vals2'], \
                                                         kargs['chemicalKernelmat']]

        numthreads = numthreadsTot // 2

        chunklen1 = (Nenv1 + 1) // numthreads
        kargs1 = {'vals1': vals1}
        chunks1 = [{key: arg[i * chunklen1:(i + 1) * chunklen1] for key, arg in kargs1.iteritems()} for i in
                   range(numthreads)]

        chunklen2 = (Nenv2 + 1) // numthreads
        kargs2 = {'vals2': vals2}
        chunks2 = [{key: arg[i * chunklen2:(i + 1) * chunklen2] for key, arg in kargs2.iteritems()} for i in
                   range(numthreads)]

        chunks = []
        result = np.zeros((Nenv1, Nenv2), dtype=np.float64)

        for it in range(numthreads):
            for jt in range(numthreads):
                # chunks3 = result[it * chunklen1:(it + 1) * chunklen1,jt * chunklen2:(jt + 1) * chunklen2]
                chunks3 = result[it * chunklen1:(it + 1) * chunklen1, jt * chunklen2:(jt + 1) * chunklen2]
                a = {'result': chunks3, 'chemicalKernelmat': chemicalKernelmat.copy(),
                     'keys1': keys1.copy(), 'keys2': keys2.copy()}

                a.update(**chunks1[it])
                a.update(**chunks2[jt])

                chunks.append(a)

        # You should make sure inner_func is compiled at this point, because
        # the compilation must happen on the main thread. This is the case
        # in this example because we use jit().
        threads = [threading.Thread(target=inner_func, kwargs=chunk) for chunk in chunks[:-1]]

        for thread in threads:
            thread.start()

        # the main thread handles the last chunk
        inner_func(**chunks[-1])

        for thread in threads:
            thread.join()
        return result

    return func_mt