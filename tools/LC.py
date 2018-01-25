import numpy as np
import numpy.random as npr
from krr import validation
from sklearn.model_selection import KFold
import sys,os,signal,psutil
import argparse
from multiprocessing import Pool
import seaborn as sns
import matplotlib.pyplot as plt
# to import from libmatch
sys.path.insert(0,os.path.dirname(os.path.realpath(__file__))+'/..')
from libmatch.utils import  is_notebook

if is_notebook():
    from tqdm import tqdm_notebook as tqdm_cs
else:
    from tqdm import tqdm as tqdm_cs

def plot_learning_curve(nTrains,scoreTest,err_scoreTest,Nfold,fout=None):
    with sns.plotting_context("notebook", font_scale=1.5):
        with sns.axes_style("ticks"):
            #plt.loglog(nTrains,scoreTest,'-ob',basex=10,basey=10);
            plt.errorbar(nTrains, scoreTest, yerr=err_scoreTest,fmt='o', capthick=1.5,capsize=3);
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Number of Training Samples')
            plt.ylabel('Test MAE [kJ/mol]')
            plt.title('Learning Curve testing on {:.0f}% of the dataset'.format(100*(1./Nfold)))
            if fout:
                plt.savefig(fout,dpi=300,bbox_inches='tight')

def LearningCurve(kernel,prop,params,nprocess=1,Nfold=4,nSet=None,train_fractions=np.linspace(0.1,0.9,8),seed=10,verbose=False):
    if nSet is None:
        nSet = kernel.shape[0]

    seeds = {f: npr.randint(0, 500, size=(int(it),)) for it, f in zip(np.linspace(50,10,len(train_fractions)), train_fractions)}

    cv = KFold(n_splits=Nfold, shuffle=True, random_state=seed)
    chunks = []
    if nprocess > 1:
        params.update({'memory_eff':True})
    for fraction in train_fractions:

        Ntrain = int(nSet * fraction)
        chunks.append(dict(kernel=kernel, prop=prop, params=params, cv=cv, Ntrain=Ntrain, seeds=seeds[fraction]))

    pool = mp_LC(chunks, nprocess, disable_pbar=False)

    Ntrains,scores_test, scores_test_err, scores_train, scores_train_err = pool.run()

    return Ntrains,scores_test, scores_test_err, scores_train, scores_train_err


def point(kernel, prop, params, cv, Ntrain, seeds):
    if isinstance(kernel,str):
        kernel = np.load(kernel)
    score_outer_test = []
    score_outer_train = []
    for it, (train, test) in enumerate(cv.split(kernel)):
        score_inner = []
        score_inner_train = []

        for jt, seed in enumerate(seeds):
            npr.seed(seed)
            r_train = train.copy()
            npr.shuffle(r_train)
            train_ids = r_train[:Ntrain]
            test_ids = test

            ypred_val, ytrue_val, sc_val, ypred_train, ytrue_train, sc_train, model = \
                validation(kernel, prop, train_ids, test_ids, params, verbose=False)

            score_inner.append(sc_val)
            score_inner_train.append(sc_train)

        score_outer_test.append(np.mean(score_inner, axis=0))
        score_outer_train.append(np.mean(score_inner_train, axis=0))

    return score_outer_test, score_outer_train


def point_wrapper(kwargs):
    Ntrain = kwargs['Ntrain']
    score_outer_test, score_outer_train = point(**kwargs)
    return (Ntrain,score_outer_test, score_outer_train)


class mp_LC(object):
    def __init__(self, chunks, nprocess, disable_pbar=False):
        super(mp_LC, self).__init__()
        self.func_wrap = point_wrapper
        self.dispbar = disable_pbar
        self.parent_id = os.getpid()
        self.nprocess = nprocess
        self.chunks = chunks

    def run(self):
        Nit = len(self.chunks)
        pbar = tqdm_cs(total=Nit, desc='LearningCurve', disable=self.dispbar)
        scores_test = []
        scores_train = []
        scores_test_err = []
        scores_train_err = []

        Ntrains = []
        if self.nprocess > 1:
            pool = Pool(self.nprocess, initializer=self.worker_init,
                        maxtasksperchild=1)
            #self.func_wrap(self.chunks[0])
            #print '########'
            for Ntrain, res_test,res_train in pool.imap_unordered(self.func_wrap, self.chunks):
                scores_test.append(np.mean(res_test,axis=0))
                scores_train.append(np.mean(res_train,axis=0))
                scores_test_err.append(np.std(res_test, axis=0))
                scores_train_err.append(np.std(res_train, axis=0))
                Ntrains.append(Ntrain)
                pbar.update()

            pool.close()
            pool.join()

        elif self.nprocess == 1:
            for chunk in self.chunks:
                Ntrain, res_test, res_train = self.func_wrap(chunk)
                scores_test.append(np.mean(res_test, axis=0))
                scores_train.append(np.mean(res_train, axis=0))
                scores_test_err.append(np.std(res_test, axis=0))
                scores_train_err.append(np.std(res_train, axis=0))
                Ntrains.append(Ntrain)

                pbar.update()
        else:
            print 'Nproces: ', self.nprocess
            raise NotImplementedError('need at least 1 process')

        pbar.close()

        scores_test = np.asarray(scores_test)
        scores_train = np.asarray(scores_train)
        scores_test_err = np.asarray(scores_test_err)
        scores_train_err = np.asarray(scores_train_err)

        return Ntrains,scores_test,scores_test_err,scores_train,scores_train_err

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
