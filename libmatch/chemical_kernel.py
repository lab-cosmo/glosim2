import numpy as np

from numba import jitclass,typeof ,vectorize ,prange,njit ,jit  # import the decorator
from numba import int32, float64 , void   # import the types
from collections import MutableMapping

def randKernel(spA ,spB ,seed=10):
    np.random.seed(( spA +spB ) *seed)
    return np.random.random()

def deltaKernel(spA ,spB):
    if spA == spB:
        return 1.
    else:
        return 0.

def Atoms2ChemicalKernelmat(atoms,atoms2=None,chemicalKernel=deltaKernel):
    # unique sp in frames 1 and 2
    uk1 = []
    for frame in atoms:
        uk1.extend(frame.get_atomic_numbers())
    if atoms2 is not None:
        for frame in atoms2:
            uk1.extend(frame.get_atomic_numbers())
    uk1 = list(set(uk1))
    Nsp1 = max(uk1)+1
    # 0 row and col are here but dont matter
    chemicalKernelmat = np.zeros((Nsp1,Nsp1))
    for it in uk1:
        for jt in uk1:
            chemicalKernelmat[it,jt] = chemicalKernel(it,jt)
    return chemicalKernelmat


def get_chemicalKernelmatFrames(frames1 ,frames2=None ,chemicalKernel=deltaKernel):
    # unique sp in frames 1 and 2
    uk1 = []
    for frame in frames1:
        uk1.extend(frame.get_atomic_numbers())
    uk1 = list(set(uk1))

    if frames2 is None:
        frames2 = frames1
        uk2 = uk1
    else:
        uk2 = []
        for frame in frames2:
            uk2.extend(frame.get_atomic_numbers())
        uk2 = list(set(uk2))

    Nsp1 = max(uk1 ) +1
    Nsp2 = max(uk2 ) +1
    # 0 row and col are here but dont matter
    chemicalKernelmat = np.zeros((Nsp1 ,Nsp2))
    for it in uk1:
        for jt in uk2:
            chemicalKernelmat[it ,jt] = chemicalKernel(it ,jt)
    return chemicalKernelmat

############## MEMORY LEAK WITH PARALLEL=TRUE
@jit(float64[:, :](int32[:, :], float64[:, :, :], float64[:, :]), parallel=False, nopython=True, nogil=True, cache=True)
def nb_partial_kernels2kernel(keys, partial_mats, chemicalKernelmat):
    K, N, M = partial_mats.shape
    kernel = np.zeros((N, M), dtype=np.float64)

    for it in range(K):
        spA, spB = (keys[it, 0], keys[it, 1]), (keys[it, 2], keys[it, 3])

        theta1 = chemicalKernelmat[spA[0], spB[0]] * chemicalKernelmat[spA[1], spB[1]]
        theta2 = chemicalKernelmat[spA[1], spB[0]] * chemicalKernelmat[spA[0], spB[1]]

        if theta1 == 0. and theta2 == 0.:
            continue
        # the symmetry of the chemicalKernel and chemical soap vector is a bit messy
        if spA[0] != spA[1] and spB[0] != spB[1]:
            kernel += theta1 * partial_mats[K, :, :] * 2 + theta2 * partial_mats[K, :, :] * 2
        elif (spA[0] == spA[1] and spB[0] != spB[1]) or (spA[0] != spA[1] and spB[0] == spB[1]):
            kernel += theta1 * partial_mats[K, :, :] + theta2 * partial_mats[K, :, :]
        elif spA[0] == spA[1] and spB[0] == spB[1]:
            kernel += theta1 * partial_mats[K, :, :]

    return kernel




class PartialKernels(MutableMapping):
    def __init__(self, fingerprintsA, fingerprintsB=None, chemicalKernelmat=None, nthreads=4):

        self.dtype = 'float64'
        self.nthreads = nthreads
        try:
            import mkl
            mkl.set_num_threads(self.nthreads)
        except:
            raise Warning('NUMPY DOES NOT SEEM TO BE LINKED TO MKL LIBRARY SO NTHREADS IS IGNORED')

        self.fingerprintsA = fingerprintsA
        self.fingerprints_infoA = self.get_info(fingerprintsA)
        pairsA = self.fingerprints_infoA['pairs']
        Nframe = len(fingerprintsA)
        if fingerprintsB is not None:
            self.fingerprintsB = fingerprintsB
            self.fingerprints_infoB = self.get_info(fingerprintsB)
            pairsB = self.fingerprints_infoB['pairs']
            Mframe = len(fingerprintsB)
        else:
            self.fingerprintsB = None
            pairsB = pairsA
            Mframe = Nframe

        # initialize data container
        self._storage = {pA + pB: np.zeros((Nframe, Mframe), dtype=self.dtype)
                         for pA in pairsA for pB in pairsB}

        self.set_partial_kernels()

        self.chemicalKernelmat = chemicalKernelmat
        self.set_kernel(chemicalKernelmat)

    def get_dense_values(self):
        values = np.asarray(self.values())
        return values

    def get_dense_keys(self):
        keys = np.asarray(self.keys())
        return keys

    def get_dense_arrays(self):
        return self.get_dense_keys(), self.get_dense_values()

    def get_info(self, fingerprints):
        ii = 0
        ll = []
        fings_info = {}
        for it, fing1 in enumerate(fingerprints):
            ll.extend(fing1['AVG'].keys())
            for pA in fing1['AVG'].keys():
                ii += 1

        fings_info['types'] = np.unique(ll)
        fings_info['lin_length'] = ii

        fings_info['pairs'] = [(t1, t2) for t1 in fings_info['types']
                               for t2 in fings_info['types'] if t1 <= t2]

        soapParams = fingerprints[0].get_soapParams()
        nmax = soapParams['nmax']
        lmax = soapParams['lmax']
        fings_info['soapLen'] = nmax ** 2 * (lmax + 1)

        fings_info['dtype'] = fingerprints[0]['AVG'].dtype

        return fings_info

    def set_kernel(self, chemicalKernelmat):
        if chemicalKernelmat is None:
            self.kernel = None
        else:
            _keys, _partial_mats = self.get_dense_arrays()
            self.chemicalKernelmat = chemicalKernelmat
            self.kernel = nb_partial_kernels2kernel(_keys, _partial_mats, chemicalKernelmat)

    def get_kernel(self):
        return self.kernel

    def get_linear_array(self, fingerprints, fings_info):

        dtype = fings_info['dtype']
        lin_length = fings_info['lin_length']
        soapLen = fings_info['soapLen']
        pairs = fings_info['pairs']

        lin_array = np.zeros((lin_length, soapLen), dtype=self.dtype)

        pair2ids = {pA: {'frame_ids': [], 'linear_ids': []} for pA in pairs}

        jj = 0
        for it, fing1 in enumerate(fingerprints):
            for pA, pp in fing1['AVG'].iteritems():
                lin_array[jj] = np.asarray(pp, dtype=self.dtype)
                pair2ids[pA]['frame_ids'].append(it)
                pair2ids[pA]['linear_ids'].append(jj)
                jj += 1

        return lin_array, pair2ids

    def set_partial_kernels_from_linear_prod(self, linear_prod, pair2idsA, pair2idsB):
        for pA, itemA in pair2idsA.iteritems():
            it_idsA, jj_idsA = itemA['frame_ids'], itemA['linear_ids']
            for pB, itemB in pair2idsB.iteritems():
                it_idsB, jj_idsB = itemB['frame_ids'], itemB['linear_ids']
                self._storage[pA + pB][np.ix_(it_idsA, it_idsB)] = linear_prod[np.ix_(jj_idsA, jj_idsB)]

    def set_partial_kernels(self):

        lin_arrayA, pair2idsA = self.get_linear_array(self.fingerprintsA, self.fingerprints_infoA)

        if self.fingerprintsB is None:
            lin_arrayB, pair2idsB = lin_arrayA, pair2idsA
        else:
            lin_arrayB, pair2idsB = self.get_linear_array(self.fingerprintsB, self.fingerprints_infoB)

        linear_prod = np.dot(lin_arrayA, lin_arrayB.T)

        self.set_partial_kernels_from_linear_prod(linear_prod, pair2idsA, pair2idsB)

    def __cmp__(self, dict):
        return cmp(self._storage, dict)

    def __contains__(self, item):
        return item in self._storage

    def __iter__(self):
        for key in self.keys():
            yield key

    def __unicode__(self):
        return unicode(repr(self._storage))

    def __del__(self):
        for values in self.__dict__.values():
            del values

    def __setitem__(self, key, item):
        # asarray does not copy if the types are matching
        self._storage[key] = np.asarray(item, dtype=self.dtype)

    def __getitem__(self, key):
        return self._storage[key]

    def get(self, key):
        return self._storage[key]

    def __repr__(self):
        return repr(self._storage)

    def __len__(self):
        return len(self.keys())

    def __delitem__(self, key):
        del self._storage[key]

    def has_key(self, key):
        return self._storage.has_key(key)

    def pop(self, key, d=None):
        return self._storage.pop(key, d)

    def update(self, *args, **kwargs):
        return self._storage.update(*args, **kwargs)

    def keys(self):
        return self._storage.keys()

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):
        return [(key, self[key]) for key in self.keys()]


class PartialKernels_slow(MutableMapping):
    def __init__(self, fingerprintsA, fingerprintsB=None, chemicalKernelmat=None, nthreads=4):

        self.dtype = 'float64'
        self.nthreads = nthreads
        try:
            import mkl
            mkl.set_num_threads(self.nthreads)
        except:
            raise Warning('NUMPY DOES NOT SEEM TO BE LINKED TO MKL LIBRARY SO NTHREADS IS IGNORED')

        self.fingerprintsA = fingerprintsA
        self.fingerprints_infoA = self.get_info(fingerprintsA)
        pairsA = self.fingerprints_infoA['pairs']
        Nframe = len(fingerprintsA)
        if fingerprintsB is not None:
            self.fingerprintsB = fingerprintsB
            self.fingerprints_infoB = self.get_info(fingerprintsB)
            pairsB = self.fingerprints_infoB['pairs']
            Mframe = len(fingerprintsB)
        else:
            pairsB = pairsA
            Mframe = Nframe

        self._storage = {pA + pB: np.zeros((Nframe, Mframe), dtype=self.dtype)
                         for pA in pairsA for pB in pairsB}

        self.set_partial_kernels(fingerprintsA, fingerprintsB)

        self.set_kernel(chemicalKernelmat)

    def __del__(self):
        for values in self.__dict__.values():
            del values

    def __setitem__(self, key, item):
        # asarray does not copy if the types are matching
        self._storage[key] = np.asarray(item, dtype=self.dtype)

    def __getitem__(self, key):
        return self._storage[key]

    def get(self, key):
        return self._storage[key]

    def __repr__(self):
        return repr(self._storage)

    def __len__(self):
        return len(self.keys())

    def __delitem__(self, key):
        del self._storage[key]

    def has_key(self, key):
        return self._storage.has_key(key)

    def pop(self, key, d=None):
        return self._storage.pop(key, d)

    def update(self, *args, **kwargs):
        return self._storage.update(*args, **kwargs)

    def keys(self):
        return self._storage.keys()

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def get_dense_values(self):
        values = np.asarray(self.values())
        return values

    def get_dense_keys(self):
        keys = np.asarray(self.keys())
        return keys

    def get_dense_arrays(self):
        return self.get_dense_keys(), self.get_dense_values()

    def __cmp__(self, dict):
        return cmp(self._storage, dict)

    def __contains__(self, item):
        return item in self._storage

    def __iter__(self):
        for key in self.keys():
            yield key

    def __unicode__(self):
        return unicode(repr(self._storage))

    def get_info(self, fingerprints):
        ii = 0
        ll = []
        fings_info = {}
        for it, fing1 in enumerate(fingerprints):
            ll.extend(fing1['AVG'].keys())
            for pA in fing1['AVG'].keys():
                ii += 1

        fings_info['types'] = np.unique(ll)
        fings_info['lin_length'] = ii

        fings_info['pairs'] = [(t1, t2) for t1 in fings_info['types']
                               for t2 in fings_info['types'] if t1 <= t2]

        soapParams = fingerprints[0].get_soapParams()
        nmax = soapParams['nmax']
        lmax = soapParams['lmax']
        fings_info['soapLen'] = nmax ** 2 * (lmax + 1)

        fings_info['dtype'] = fingerprints[0]['AVG'].dtype

        return fings_info

    def get_kernel(self):
        return self.kernel

    def set_kernel(self, chemicalKernelmat):
        if chemicalKernelmat is None:
            self.kernel = None
        else:
            kk = self.keys()
            N, M = self[kk[0]].shape
            kernel = np.zeros((N, M), dtype=self.dtype)

            for key, mat in self.iteritems():
                spA, spB = (key[0], key[1]), (key[2], key[3])

                theta1 = chemicalKernelmat[spA[0], spB[0]] * chemicalKernelmat[spA[1], spB[1]]
                theta2 = chemicalKernelmat[spA[1], spB[0]] * chemicalKernelmat[spA[0], spB[1]]

                if theta1 == 0. and theta2 == 0.:
                    continue
                # the symmetry of the chemicalKernel and chemical soap vector is a bit messy
                if spA[0] != spA[1] and spB[0] != spB[1]:
                    kernel += theta1 * mat * 2 + theta2 * mat * 2
                elif (spA[0] == spA[1] and spB[0] != spB[1]) or (spA[0] != spA[1] and spB[0] == spB[1]):
                    kernel += theta1 * mat + theta2 * mat
                elif spA[0] == spA[1] and spB[0] == spB[1]:
                    kernel += theta1 * mat
            self.kernel = kernel

    def set_partial_kernels(self, fingerprintsA, fingerprintsB=None):
        fings_infoA = self.get_info(fingerprintsA)

        if fingerprintsB is None:
            fingerprintsB = fingerprintsA
            fings_infoB = fings_infoA
        else:
            fings_infoB = self.get_info(fingerprintsB)

        Nframe, Mframe = len(fingerprintsA), len(fingerprintsB)

        pairsA = fings_infoA['pairs']
        pairsB = fings_infoB['pairs']

        partial_kernels = {pA + pB: np.zeros((Nframe, Mframe), dtype=np.float64) for pA in pairsA for pB in pairsB}

        for it, fing1 in enumerate(fingerprintsA):
            for jt, fing2 in enumerate(fingerprintsB):
                for sk1, pp1 in fing1['AVG'].iteritems():
                    for sk2, pp2 in fing2['AVG'].iteritems():
                        partial_kernels[sk1 + sk2][it, jt] = np.dot(pp1, pp2)
        return partial_kernels


def test_implementation(fingerprintsA, fingerprintsB=None):
    partial_kernels = PartialKernels(fingerprintsA, fingerprintsB)
    partial_kernels_ref = PartialKernels_slow(fingerprintsA, fingerprintsB)
    is_equal = []
    not_equal = []
    for key in partial_kernels_ref:
        eee = np.allclose(partial_kernels_ref[key], partial_kernels[key])
        is_equal.append((key, eee))
        if not eee:
            not_equal.append((key, eee))
    if len(not_equal) == 0:
        print 'partial matrices are identical'
    else:
        print 'partial matrices are not identical in:'
        print not_equal