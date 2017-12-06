from libmatch.soap import get_soap
from libmatch.utils import get_spkit,get_spkitMax,is_notebook,ase2qp
from libmatch.global_kernel import compile_rematch,normalizeKernel
import multiprocessing as mp
import os,signal,psutil,sys
import numpy as np
import quippy as qp

if is_notebook():
    from tqdm import tqdm_notebook as tqdm_cs
else:
    from tqdm import tqdm as tqdm_cs


def compute_global_kernel(envk, strides, kernel_type='average', zeta=2, gamma=1.0, eps=1e-06,
                          normalize_global_kernel=False):
    Nframe = len(strides) - 1
    gbk = np.empty((Nframe, Nframe))

    if kernel_type == 'rematch':
        rematch = compile_rematch()

    for iframe1, (st1, nd1) in enumerate(zip(strides[:-1], strides[1:])):
        for iframe2, (st2, nd2) in enumerate(zip(strides[:-1], strides[1:])):
            if iframe1 <= iframe2:
                kkk = envk[st1:nd1, st2:nd2]
                if kernel_type == 'average':
                    gbk[iframe1, iframe2] = gbk[iframe2, iframe1] = np.mean(np.power(kkk, zeta))
                if kernel_type == 'rematch':
                    gbk[iframe1, iframe2] = gbk[iframe2, iframe1] = rematch(kkk, gamma, eps)

    if normalize_global_kernel:
        gbk = normalizeKernel(gbk)

    return gbk



def compute_env_kernel(feature_matrix):
    return np.dot(feature_matrix,feature_matrix.T)

def get_raw_soap_dim(frame ,soap_params):
    raw_soap = get_soap(frame ,spkit=get_spkit(frame) ,**soap_params)
    Ncenter ,Ncoef = raw_soap.shape
    return Ncoef

def get_feature_dim(frames ,spkitMax ,nocenters=None ,is_fast_average=False):
    ids_list = []
    flat_mapping = []
    z_mapping = {z :[] for z in spkitMax}
    strides = [0]

    Nfeature ,ii = 0 ,0
    for it ,frame in enumerate(frames):
        Nat = 0
        if is_fast_average:
            z_mapping = {'AVG' :[]}
            Nat = 1
            flat_mapping.append([it ,it ,0 ,0])
            z_mapping['AVG'].append([it ,it ,0])
        else:
            for jt ,z in enumerate(frame.get_atomic_numbers()):
                if nocenters is None:
                    flat_mapping.append([ii ,it ,jt ,z])
                    z_mapping[z].append([ii ,it ,jt])
                    Nat += 1
                    ii += 1
                else:
                    if z not in nocenters:
                        flat_mapping.append([ii ,it ,jt ,z])
                        z_mapping[z].append([ii ,it ,jt])
                        Nat += 1
                        ii += 1

        strides.append(Nat)
        ids_list.append(np.arange(Nat ) +Nfeature)
        Nfeature += Nat

    flat_mapping = np.asarray(flat_mapping ,np.int32)
    z_mapping = {z :np.asarray(val ,np.int32) for z ,val in z_mapping.iteritems()}
    strides = np.cumsum(strides)

    return Nfeature ,ids_list ,flat_mapping ,z_mapping ,strides

def get_raw_soaps(frames ,ids_list ,feature_matrix=None ,feature_shape=None ,spkitMax=None, nocenters=None,
                  centerweight=1.0, gaussian_width=0.5, cutoff=3.5, cutoff_transition_width=0.5,
                  nmax=8, lmax=6, is_fast_average=False ,nprocess=4 ,disable_pbar=False):

    if feature_matrix is None and feature_shape is not None:
        feature_matrix = np.empty(feature_shape)
    elif feature_matrix is not None:
        pass
    else:
        raise ValueError('must give one of feature_matrix or feature_shape at least')

    if spkitMax is None:
        spkitMax = get_spkitMax(frames)

    soap_params = dict( spkitMax=spkitMax, nocenters=nocenters,
                        centerweight=centerweight, gaussian_width=gaussian_width, cutoff=cutoff,
                        cutoff_transition_width=cutoff_transition_width,
                        nmax=nmax, lmax=lmax, is_fast_average=is_fast_average)


    compute = mp_soap(frames ,ids_list ,soap_params ,nprocess=nprocess ,disable_pbar=disable_pbar)

    compute.run(feature_matrix)

    return feature_matrix


def get_raw_soap_wrapper(kwargs):
    fpointer = kwargs.pop('fpointer')
    # need to copy the frame to avoid seg fault in quip if rerun the procedure on the same name space
    frame = qp.Atoms(fpointer=fpointer).copy()
    kwargs.update({'atoms' :frame ,'spkit' :get_spkit(frame)})
    idx = kwargs.pop('idx')
    return (idx ,get_soap(**kwargs))

class mp_soap(object):
    def __init__(self ,frames ,ids_list ,soap_params ,nprocess=4 ,disable_pbar=False):
        super(mp_soap, self).__init__()
        self.func_wrap = get_raw_soap_wrapper
        self.disable_pbar = disable_pbar
        self.parent_id = os.getpid()
        self.nprocess = nprocess
        self.ids_list = ids_list

        self.chunks = []
        for it ,frame in enumerate(frames):
            aa = {"idx": it ,'fpointer' :frame._fpointer}
            aa.update(soap_params)
            self.chunks.append(aa)


    def run(self ,out):
        Nit = len(self.chunks)
        pbar = tqdm_cs(total=Nit ,desc='SOAP vectors' ,disable=self.disable_pbar)

        results = out
        ids_list = self.ids_list

        if self.nprocess > 1:
            pool = mp.Pool(self.nprocess, initializer=self.worker_init,
                           maxtasksperchild=10)

            for idx, res in pool.imap_unordered(self.func_wrap, self.chunks):
                results[ids_list[idx] ,:] = res
                pbar.update()

            pool.close()
            pool.join()

        elif self.nprocess == 1:
            for chunk in self.chunks:
                idx ,res = self.func_wrap(chunk)
                results[ids_list[idx] ,:] = res

                pbar.update()
        else:
            print 'Nproces: ' ,self.nprocess
            raise NotImplementedError('need at least 1 process')

        pbar.close()

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