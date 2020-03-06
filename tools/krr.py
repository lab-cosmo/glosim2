import sys,os
import numpy as np
from scipy.stats.mstats import spearmanr
from sklearn.metrics import r2_score
from scipy.linalg import cho_solve,cho_factor
import json

# to import from libmatch
sys.path.insert(0,os.path.dirname(os.path.realpath(__file__))+'/..')

def dump_json(fn,data):
    with open(fn,'w') as f:
        json.dump(data,f,sort_keys=True,indent=2)

def load_json(fn):
    with open(fn,'r') as f:
        data = json.load(f)
    return data

def dump_data(fn,metadata,data,is_sparse=False,compressed=False):
    data_fn = os.path.join(os.path.dirname(fn),metadata['fn'])
    if is_sparse is False:
        np.save(data_fn,data)
    else:
        save_npz(data_fn,data,compressed=compressed)
    dump_json(fn,metadata)

def load_data(fn,mmap_mode='r',is_sparse=False):
    metadata = load_json(fn)
    data_fn = os.path.join(os.path.dirname(fn),metadata['fn'])
    if is_sparse is False:
        data = np.load(data_fn,mmap_mode=mmap_mode)
    else:
        data = load_npz(data_fn)
    return metadata,data


def validation(kernel, prop, train_ids, validation_ids, params, verbose=False):
    y = prop.reshape((-1, 1))

    # train model
    model = KRR(**params)
    # kernel is copied here
    model.train(kernel[np.ix_(train_ids, train_ids)], y[train_ids])

    # kernel is copied here
    ypred_train = model.predict(kernel[np.ix_(train_ids, train_ids)])
    ytrue_train = y[train_ids].reshape((-1,))
    sc_train = score(ypred_train, ytrue_train)

    ypred_val = model.predict(kernel[np.ix_(train_ids, validation_ids)])
    ytrue_val = y[validation_ids].reshape((-1,))
    sc_val = score(ypred_val, ytrue_val)

    if verbose:
        print('TRAIN MAE={:.3e} RMSE={:.3e} SUP={:.3e} R2={:.3e} CORR={:.3e}'.format(*sc_train))
        print('VALIDATION MAE={:.3e} RMSE={:.3e} SUP={:.3e} R2={:.3e} CORR={:.3e}'.format(*sc_val))

    return ypred_val,ytrue_val,sc_val,ypred_train,ytrue_train,sc_train,model


def prediction(kernel_train,kernel_test, prop_train,prop_test, params, verbose=False):
    prop_train = prop_train.reshape((-1, 1))
    prop_test = prop_test.reshape((-1, 1))

    model = KRR(**params)
    model.train(kernel_train, prop_train)

    ypred_train = model.predict(kernel_train)
    ytrue_train = prop_train.reshape((-1,))
    sc_train = score(ypred_train, ytrue_train)

    ypred_test = model.predict(kernel_test)
    ytrue_test = prop_test.reshape((-1,))
    sc_test = score(ypred_test, ytrue_test)

    if verbose:
        print('Train MAE={:.3e} RMSE={:.3e} SUP={:.3e} R2={:.3e} CORR={:.3e}'.format(*sc_train))
        print('TEST MAE={:.3e} RMSE={:.3e} SUP={:.3e} R2={:.3e} CORR={:.3e}'.format(*sc_test))

    return ypred_test, ytrue_test, sc_test, ypred_train, ytrue_train, sc_train,model



def score(ypred,y):
    def mae(ypred,y):
        return np.mean(np.abs(ypred-y))
    def rmse(ypred,y):
        return np.sqrt(np.mean((ypred-y)**2))
    def sup(ypred,y):
        return np.amax(np.abs((ypred-y)))
    def spearman(ypred,y):
        corr,_ = spearmanr(ypred,y)
        return corr
    return mae(ypred,y),rmse(ypred,y),sup(ypred,y),r2_score(ypred,y),spearman(ypred,y)


def dummy(a):
    return a

class KRR(object):
    def __init__(self,sigma=None,csi=None,sampleWeights=None,memory_eff=False):

        self.sigma = sigma
        self.csi = csi

        # Weights of the krr model
        self.alpha = None
        self.sampleWeights = sampleWeights
        self.memory_eff = memory_eff

    def train(self,kernel,labels):
        '''Train the krr model with trainKernel and trainLabel. If sampleWeights are set then they are used as a multiplicative factor.'''
        nTrain, _ = kernel.shape

        # uses the sample weights from default or leave one out procedure
        if self.sampleWeights is None:
            sampleWeights = np.ones((nTrain,))
        else:
            sampleWeights = np.array(self.sampleWeights)

        # learn a function of the label
        trainLabel = labels

        diag = kernel.diagonal().copy()
        self.lower = False
        reg = np.multiply(
            np.divide(np.multiply(self.sigma ** 2, np.mean(diag)), np.var(trainLabel)),
            sampleWeights)
        self.reg = reg
        if self.memory_eff:
            # kernel is modified here

            np.fill_diagonal(np.power(kernel, self.csi, out=kernel),
                                np.add(np.power(diag,self.csi,out=diag), reg,out=diag))

            kernel, lower = cho_factor(kernel, lower=False, overwrite_a=True, check_finite=False)
            L = kernel
        else:
            # kernel is not modified here
            reg = np.diag(reg)
            L, lower = cho_factor(np.power(kernel, self.csi) + reg, lower=False, overwrite_a=False,check_finite=False)

        # set the weights of the krr model
        self.alpha = cho_solve((L, lower), trainLabel,overwrite_b=False).reshape((1,-1))

    def predict(self,kernel):
        '''kernel.shape is expected as (nTrain,nPred)'''
        if self.memory_eff:
            # kernel is modified in place here
            return np.dot(self.alpha, np.power(kernel,self.csi,out=kernel)).reshape((-1))
        else:
            # kernel is not modified here
            return np.dot(self.alpha, np.power(kernel,self.csi) ).reshape((-1))

    def get_params(self):
        state = dict(
            sigma=self.sigma,
            csi=self.csi,
            memory_eff=self.memory_eff,
        )
        if self.sampleWeights is None:
            state['sampleWeights'] = None
        else:
            state['sampleWeights'] = self.sampleWeights.tolist()
        return state
    def set_params(self,params):
        self.sigma = params['sigma']
        self.csi = params['csi']
        if params['sampleWeights'] is None:
            self.sampleWeights = None
        else:
            self.sampleWeights = np.array(params['sampleWeights'])
        self.memory_eff = params['memory_eff']

    def pack(self):
        params = self.get_params()
        data = dict(alpha=self.alpha.tolist())
        state = dict(data=data,
                     params=params)
        return state
    def unpack(self,state):
        self.set_params(state['params'])
        self.alpha = np.array(state['data']['alpha'])
        return self


####################  ##########################
def func(N):
    np.random.seed(10)
    X = np.random.rand(N,1000)
    kernel = np.dot(X,X.T)
    trainLabel = np.random.rand(N)
    sampleWeights =  np.ones(N)
    sigma = 3
    csi = 1
    diag = kernel.diagonal().copy()
    reg = np.divide(np.multiply(sigma ** 2 , np.mean(diag) ) ,np.multiply( np.var(trainLabel) , sampleWeights ) )
    np.fill_diagonal(np.power(kernel,csi,out=kernel), np.add(diag, reg) )
    lower = False
    kernel,lower = cho_factor(kernel, lower=lower, overwrite_a=True,check_finite=False)
    alpha = cho_solve((kernel,lower),trainLabel)
    return alpha

def func_ref(N):
    np.random.seed(10)
    X = np.random.rand(N,1000)
    kernel = np.dot(X,X.T)
    trainLabel = np.random.rand(N)
    sampleWeights =  np.ones(N)
    sigma = 3
    csi = 1
    reg = np.diag(sigma ** 2 * np.mean(np.diag(kernel)) / np.var(trainLabel) / sampleWeights)
    aa = np.add(np.power(kernel,csi),reg)
    alpha = np.linalg.solve(aa ,trainLabel)

    return alpha


