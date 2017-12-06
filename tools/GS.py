import numpy as np
from krr import score,KRR
from CV import CrossValidation
from sklearn.model_selection import ParameterGrid
import sys,os,signal,psutil
import argparse
from Pool.mpi_pool import MPIPool
from multiprocessing import Pool
# to import from libmatch
sys.path.insert(0,os.path.dirname(os.path.realpath(__file__))+'/..')
from libmatch.utils import  is_notebook

if is_notebook():
    from tqdm import tqdm_notebook as tqdm_cs
else:
    from tqdm import tqdm as tqdm_cs


def GridSearch(kernel, prop, params=None,Nfold=4, seed=10,nprocess=1,
               disable_pbar=False):
    if params is None:
        params = dict(sigma=[0.001,0.01],csi=[1,2,4])

    gs_params = ParameterGrid(params)

    chunks = [dict(kernel=kernel, prop=prop, params=params,
                   Nfold=Nfold, seed=seed, verbose=False)
                    for params in gs_params]

    pool = mp_GS(chunks,nprocess,disable_pbar=disable_pbar)
    bestParams,scores,gs_params = pool.run()

    return bestParams,scores,gs_params

def CrossValidation_wrapper(kwargs):
    params = kwargs['params']
    scoreVal, _= CrossValidation(**kwargs)
    return (params,scoreVal)

class mp_GS(object):
    def __init__(self, chunks, nprocess,disable_pbar=False):
        super(mp_GS, self).__init__()
        self.func_wrap = CrossValidation_wrapper
        self.dispbar = disable_pbar
        self.parent_id = os.getpid()
        self.nprocess = nprocess
        self.chunks = chunks

    def run(self):
        Nit = len(self.chunks)
        pbar = tqdm_cs(total=Nit,desc='GridSearch',disable=self.dispbar)
        scores = []
        gs_params = []
        if self.nprocess > 1:
            pool = Pool(self.nprocess, initializer=self.worker_init,
                                maxtasksperchild=1)

            for param, res in pool.imap_unordered(self.func_wrap, self.chunks):
                scores.append(res)
                gs_params.append(param)
                pbar.update()

            pool.close()
            pool.join()

        elif self.nprocess == 1:
            for chunk in self.chunks:
                param, res = self.func_wrap(chunk)
                scores.append(res)
                gs_params.append(param)

                pbar.update()
        else:
            print 'Nproces: ',self.nprocess
            raise NotImplementedError('need at least 1 process')

        pbar.close()

        scores = np.asarray(scores)

        ii = np.argmin(scores[:, 0])
        bestParams = gs_params[ii]
        bestParams.pop('memory_eff')
        return bestParams,scores,gs_params
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Computes the Global average/rematch kernel.""")

    parser.add_argument("filename", nargs=1, help="Name of the LibAtom formatted xyz input file")
