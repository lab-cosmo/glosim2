from collections import MutableMapping,OrderedDict
import numpy as np
from utils import atomicnb_to_symbol,get_localEnv


garbageKey = ['positions','numbers','species','map_shift','n_neighb']


class AlchemySoap(MutableMapping):
    '''
    Container class for the soap vectors in their alchemy format.

    '''
    def __init__(self ,qpatoms ,soapParams ,centerIdx ,nocenters=None):
        # keys is a list of all the possible key that are needed in the dictionary
        if nocenters is None:
            nocenters = []

        z = list(set(qpatoms.get_atomic_numbers()))
        a = []
        for z1 in z:
            if z1 in nocenters: continue
            for z2 in z:
                if z2 in nocenters: continue
                a.append((z1 ,z2))

        self._allKeys = a

        self._soapParams = soapParams
        self.dtype = np.float64

        nmax = self._soapParams['nmax']
        lmax = self._soapParams['lmax']
        Nsoap = nmax ** 2 * (lmax + 1)

        self._empty = np.zeros((Nsoap,) ,dtype=self.dtype)

        self._frameIdx = centerIdx
        self._position = qpatoms.positions[centerIdx ,:]
        self._atomic_number = qpatoms.get_atomic_numbers()[centerIdx]
        self._cell = qpatoms.get_cell()
        self._chemical_symbol = qpatoms.get_chemical_symbols()[centerIdx]
        self._info = {}
        for key,item in qpatoms.arrays.iteritems():
            if key in garbageKey:
                continue
            self._info[key] = item[centerIdx].copy()

        self._localEnvironementDict = get_localEnv(qpatoms, centerIdx,
                                                   self._soapParams['cutoff'], onlyDict=True)

        upperKeys = []

        self._storage = dict()
        for key in self._allKeys:
            upperKeys.append(tuple(sorted(key)))
            # store only the upper keys (key[0] <= key[1])
            self[key] = self._empty
        self._upperKeys = list(set(upperKeys))

    def __del__(self):
        for values in self.__dict__.values():
            del values

    def from_dict(self ,dictionary):
        filledKeys = []
        for key ,item in dictionary.iteritems():
            if key[0] > key[1]:
                # only upper part is considered (key[0] <= key[1])
                continue
            elif np.allclose(item ,self._empty):
                # keep the self.empty reference instead of reassigning a vector of zeros
                # so only one self.empty is kept in memory
                continue
            else:
                # use __setitem__ method
                filledKeys.append(tuple(sorted(key)))
                self[key] = item
        self._filledKeys = list(set(filledKeys))
        self._emptyKeys = list(set(self.get_upperKeys() ) -set(filledKeys))

    def get_upperKeys(self):
        return self._upperKeys
    def get_allKeys(self):
        return self._allKeys
    def get_centerInfo(self):
        from ase import Atoms as aseAtoms
        info = {'z' :self._atomic_number ,'position' :self._position ,
                'cell' :self._cell ,'idx' :self._frameIdx,
                'symbol':self._chemical_symbol,'env': aseAtoms(**self._localEnvironementDict)}
        info.update(**self._info)
        return info
    def get_localEnvironement(self):
        from ase import Atoms as aseAtoms
        return aseAtoms(**self._localEnvironementDict)
    def get_filledKeys(self):
        return self._filledKeys
    def get_emptyKeys(self):
        return self._emptyKeys
    def get_soapParams(self):
        return self._soapParams


    def __setitem__(self, key, item):
        # asarray does not copy if the types are matching
        self._storage[tuple(sorted(key))] = np.asarray(item ,dtype=self.dtype)

    def __getitem__(self, key):
        return self._storage[tuple(sorted(key))]
    def get(self ,key):
        return self[key]

    def __repr__(self):
        return repr(self._storage)

    def __len__(self):
        return len(self.keys())

    def __delitem__(self, key):
        skey = tuple(sorted(key))
        del self._storage[skey]

    def has_key(self, key):
        skey = tuple(sorted(key))
        return self._storage.has_key(skey)

    def pop(self, key, d=None):
        skey = tuple(sorted(key))
        return self._storage.pop(skey, d)

    def update(self, *args, **kwargs):
        return self._storage.update(*args, **kwargs)

    def keys(self):
        return  self._upperKeys

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):
        return [(key ,self[key]) for key in self.keys()]

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


class AtomicFrame(object):
    def __init__(self, qpatoms, nocenters, soapParams):
        # qpatoms is a libatom Atom object
        super(AtomicFrame, self).__init__()
        self._atomic_numbers = qpatoms.get_atomic_numbers()
        self._chemical_formula = qpatoms.get_chemical_formula(mode='hill')
        self._positions = qpatoms.get_positions()
        self._nocenters = nocenters
        self._cell = qpatoms.get_cell()
        self._pbc = qpatoms.get_pbc()
        self._soapParams = soapParams

        self._infoDict = {}
        for key,item in qpatoms.arrays.iteritems():
            if key in garbageKey:
                continue
            self._infoDict[key] = item.copy()

    def __del__(self):
        for values in self.__dict__.values():
            del values

    def get_info(self):
        return self._infoDict

    def get_atomic_numbers(self):
        return self._atomic_numbers

    def get_chemical_formula(self):
        return self._chemical_formula

    def get_positions(self):
        return self._positions

    def get_nocenters(self):
        return self._nocenters

    def get_cell(self):
        return self._cell

    def get_pbc(self):
        return self._pbc

    def get_atom(self):
        from ase import Atoms as aseAtoms
        frame = aseAtoms(numbers=self._atomic_numbers,
                        positions=self._positions,
                        cell=self._cell,
                        pbc=self._pbc)
        for key,item in self._infoDict.iteritems():
            frame.set_array(key,item)
        return frame

    def get_soapParams(self):
        return self._soapParams


class AlchemyFrame(AtomicFrame, MutableMapping):
    def __init__(self, atom, nocenters, soapParams):
        # atom is a libatom Atom object
        super(AlchemyFrame, self).__init__(atom, nocenters, soapParams)

        self.valdtype = np.float64
        self.keydtype = np.uint32

        self._storage = OrderedDict()

        uniquez = list(set(self.get_atomic_numbers()))

        self._count = {z: 0 for z in uniquez}
        self._int2symb = {}

        allKeys = []
        for z1 in uniquez:
            #if z1 in nocenters: continue
            for z2 in uniquez:
                #if z2 in nocenters: continue
                allKeys.append((z1, z2))

        self._allKeys = allKeys
        upperKeys = []
        for key in self._allKeys:
            upperKeys.append(tuple(sorted(key)))
        self._upperKeys = np.array(list(set(upperKeys)))
    def __del__(self):
        for values in self.__dict__.values():
            del values
    def __setitem__(self, key, item):
        # asarray does not copy if the types are matching
        assert isinstance(item, AlchemySoap)
        z = key
        nb = self._count[z]
        self._count[z] += 1

        self._int2symb[len(self._storage)] = atomicnb_to_symbol(z) + str(nb)

        self._storage[atomicnb_to_symbol(z) + str(nb)] = item

    def __getitem__(self, key):
        if isinstance(key, int):
            key = self._int2symb[key]
        return self._storage[key]

    def get(self, key):
        return self[key]

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

    def __iter__(self):
        for key in self.keys():
            yield key

    def get_upperKeys(self):
        return self._upperKeys

    def get_allKeys(self):
        return self._allKeys

    def get_arrays(self):
        nmax = self._soapParams['nmax']
        lmax = self._soapParams['lmax']
        Nsoap = nmax ** 2 * (lmax + 1)

        upperKeys = np.asarray(self.get_upperKeys(), dtype=self.keydtype)

        Nkey = len(upperKeys)

        envs = self.values()
        Nenv = len(envs)

        alchemyArray = np.zeros((Nenv, Nkey, Nsoap), dtype=self.valdtype)
        for it, env in enumerate(envs):
            alchemyArray[it, :, :] = env.get_dense_values()

        return upperKeys, alchemyArray

    def get_arrays_emp(self):
        nmax = self._soapParams['nmax']
        lmax = self._soapParams['lmax']
        Nsoap = nmax ** 2 * (lmax + 1)

        upperKeys = self.get_upperKeys()
        Nkey = len(upperKeys)

        upperKeysnp = np.asarray(upperKeys, dtype=self.keydtype)

        envs = self.values()
        Nenv = len(envs)

        emptyIds_outter = []
        alchemyArray = np.zeros((Nenv, Nkey, Nsoap), dtype=self.valdtype)
        emptyIds = np.zeros((Nenv, Nkey), dtype=np.bool)
        for it, env in enumerate(envs):
            alchemyArray[it, :, :] = env.get_dense_values()

            emptyKeys = env.get_emptyKeys()
            for nt, emptykey in enumerate(emptyKeys):
                ids = [jt for jt, upperkey in enumerate(upperKeys) if np.all(emptykey == upperkey)]
                emptyIds[it, ids] = True

        return upperKeysnp, alchemyArray, emptyIds

    def get_full_arrays(self):
        nmax = self._soapParams['nmax']
        lmax = self._soapParams['lmax']
        Nsoap = nmax ** 2 * (lmax + 1)

        allKeys = np.asarray(self.get_allKeys(), dtype=self.keydtype)
        Nkey = len(allKeys)

        envs = self.values()
        Nenv = len(envs)

        alchemyArray = np.zeros((Nenv, Nkey, Nsoap), dtype=self.valdtype)
        for it, env in enumerate(envs):
            for jt, key in enumerate(allKeys):
                alchemyArray[it, jt, :] = env[key]

        return allKeys, alchemyArray


# TODO impelement AverageFrame