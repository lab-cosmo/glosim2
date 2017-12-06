import numpy as np
from krr import score,KRR
import sys,os
from sklearn.model_selection import KFold
import argparse
# to import from libmatch
sys.path.insert(0,os.path.dirname(os.path.realpath(__file__))+'/..')

def CrossValidation(kernel,prop,params,Nfold=4,seed=10,verbose=False):
    y = prop.reshape((-1, 1))
    cv = KFold(n_splits=Nfold, shuffle=True, random_state=seed)

    inner_score = []
    params.update(dict(memory_eff=True))

    for train, test in cv.split(kernel):
        model = KRR(**params)
        # copy the sub matrix
        model.train(kernel[np.ix_(train, train)], y[train])

        ypred = model.predict(kernel[np.ix_(train, test)])

        ytrue = y[test].reshape((-1,))
        MAE, RMSE, SUP, R2, CORR = score(ypred, ytrue)
        inner_score.append([MAE, RMSE, SUP, R2, CORR])

    scoreTest = np.mean(inner_score, axis=0)
    err_scoreTest = np.std(inner_score, axis=0)

    if verbose:
        vv = []
        from uncertainties import ufloat
        for m, s in zip(scoreTest, err_scoreTest):
            vv.append(ufloat(m, s))
        print '{:.1uL} & {:.1uL} & {:.1uL} & {:.1uL} & {:.1uL} '.format(vv[0], vv[1], vv[2], vv[3], vv[4]).replace(
            '\pm', '$\pm$')

    return scoreTest, err_scoreTest


