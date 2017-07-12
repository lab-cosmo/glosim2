from copy import deepcopy

import numpy  as np
import quippy as qp


def s2hms(time):
    m = time // 60
    s = int(time % 60)
    h = int(m // 60)
    m = int(m % 60)
    return '{:02d}:{:02d}:{:02d} (h:m:s)'.format(h,m,s)

def atomicnb_to_symbol(atno):
    pdict={1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Ha', 106: 'Sg', 107: 'Ns', 108: 'Hs', 109: 'Mt', 110: 'Unn', 111: 'Unu'}
    return pdict[atno]

def get_spkitMax(atoms):
    '''
    Get the set of species their maximum number across atoms.

    :param atoms: list of quippy Atoms object
    :return: Dictionary with species as key and return its
                largest number of occurrence
    '''
    spkitMax = {}

    for at in atoms:
        atspecies = {}
        for z in at.z:
            if z in atspecies:
                atspecies[z] += 1
            else:
                atspecies[z] = 1

        for (z, nz) in atspecies.iteritems():
            if z in spkitMax:
                if nz > spkitMax[z]: spkitMax[z] = nz
            else:
                spkitMax[z] = nz

    return spkitMax

def get_spkit(atom):
    '''
    Get the set of species their number across atom.

    :param atom: One quippy Atoms object
    :return:
    '''
    spkit = {}
    for z in atom.z:
        if z in spkit:
            spkit[z] += 1
        else:
            spkit[z] = 1
    return spkit

def are_envKernel_same(knp,knb):
    a = True
    for key in knb:
        if not np.allclose(knp[key],knb[key]):
            a = False
            print('##### {}'.format(key))
    print('the two are same ? -> {}'.format(a))
    return a

def envIdx2centerIdxMap(atoms,spkit,nocenters=None):
    if nocenters is None:
        nocenters = []
    spInFrame = spkit.keys()
    # makes sure that the nocenters is propely adapted to the species present in the frame
    nocenterInFrame = []
    for nocenter in nocenters:
        if nocenter in spInFrame:
            nocenterInFrame.append(nocenter)
    dd = {}
    ii = 0
    for it,z in enumerate(atoms.get_atomic_numbers()):
        if z not in nocenterInFrame:
            dd[ii] = it
            ii += 1
    return dd

def qp2ase(qpatoms):
    from ase import Atoms as aseAtoms
    positions = qpatoms.get_positions()
    cell = qpatoms.get_cell()
    symbols = qpatoms.get_chemical_formula()
    pbc = qpatoms.get_pbc()
    return aseAtoms(symbols=symbols,cell=cell,positions=positions,pbc=pbc)
def ase2qp(aseatoms):
    from quippy import Atoms as qpAtoms
    positions = aseatoms.get_positions()
    cell = aseatoms.get_cell()
    numbers = aseatoms.get_atomic_numbers()
    pbc = aseatoms.get_pbc()
    return qpAtoms(numbers=numbers,cell=cell,positions=positions,pbc=pbc)


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

                aa = {key[0]: ch1, key[1]: None}
                bb = kargs
                aa.update(**bb)
                chunks.append(deepcopy(aa))
            else:

                aa = {key[0]: ch1, key[1]: ch2}
                bb = kargs
                aa.update(**bb)
                chunks.append(deepcopy(aa))
            iii += 1
    return chunks