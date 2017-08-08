import argparse
import time

import numpy  as np
import quippy as qp
import cPickle as pck

from libmatch.chemical_kernel import Atoms2ChemicalKernelmat, deltaKernel
from libmatch.environmental_kernel import get_environmentalKernels_mt_mp_chunks, \
    get_environmentalKernels_singleprocess
from libmatch.global_kernel import avgKernel, rematchKernel, normalizeKernel
from libmatch.utils import s2hms


def get_environmentalKernels_cluster(atoms, nocenters=None, chem_channels=True, centerweight=1.0,
                                     gaussian_width=0.5, cutoff=3.5, cutoff_transition_width=0.5,
                                     nmax=8, lmax=6, chemicalKernel=deltaKernel,
                                     nthreads=4, nprocess=2, nchunks=2, islow_memory=False):
    '''
    Compute the environmental kernels for every atoms (frame) pairs. Wrapper function around several setup.

    :param atoms: AtomsList object from LibAtoms library. List of atomic configurations.
    :param nocenters: 
    :param chem_channels:
    :param centerweight: Center atom weight
    :param gaussian_width: Atom Gaussian std
    :param cutoff: Cutoff radius for each atomic environment in the unit of cell and positions.
    :param cutoff_transition_width: Steepness of the smooth environmental cutoff radius. Smaller -> steeper
    :param nmax: Number of radial basis functions.
    :param lmax: Number of Spherical harmonics.
    :param chemicalKernel:
    :param frameprodFunc:
    :param nthreads:
    :param nprocess:
    :param nchunks:
    :return: Dictionary of environmental kernels which keys are the (i,j) of the global kernel matrix
    '''

    if nocenters is None:
        nocenters = []

    # Builds the kernel matrix from the species present in the frames and a specified chemical
    # kernel function
    chemicalKernelmat = Atoms2ChemicalKernelmat(atoms, chemicalKernel=chemicalKernel)

    soap_params = {
        'atoms': atoms, 'centerweight': centerweight, 'gaussian_width': gaussian_width,
        'cutoff': cutoff, 'cutoff_transition_width': cutoff_transition_width,
        'nmax': nmax, 'lmax': lmax, 'chemicalKernelmat': chemicalKernelmat,
        'chem_channels': True, 'nocenters': nocenters,
    }


    kargs = {'nthreads': nthreads}
    kargs.update(**soap_params)
    # get the environmental kernels as a dictionary
    environmentalKernels = get_environmentalKernels_singleprocess(**kargs)


    return environmentalKernels