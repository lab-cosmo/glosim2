
import argparse
import time

import numpy  as np
import quippy as qp

from libmatch.chemical_kernel import Atoms2ChemicalKernelmat, deltaKernel
from libmatch.environmental_kernel import get_environmentalKernels_mt_mp_chunks, \
    get_environmentalKernels_singleprocess
from libmatch.global_kernel import avgKernel, rematchKernel, normalizeKernel
from libmatch.utils import s2hms

try:
    import numba as nb
    nonumba = False
except:
    print 'Numba is not installed... this will be much slower.'
    nonumba = True


def get_globalKernel(environmentalKernels,kernel_type='average',zeta=2,gamma=1.,eps=1e-6,nthreads=8,
                     normalize_global_kernel=False):
    '''
    Reduce the environemental kernels dictionary into a global kernel.

    :param kernel_type:
    :param zeta:
    :param gamma:
    :param eps:
    :param nthreads:
    :param normalize_global_kernel:
    :return:
    '''

    if kernel_type == 'average':
        globalKernel = avgKernel(environmentalKernels, zeta)
    elif kernel_type == 'rematch':
        globalKernel = rematchKernel(environmentalKernels, gamma=gamma, eps=eps, nthreads=nthreads)
    else:
        raise ValueError('This kernel type: {}, does not exist.'.format(kernel_type))

    # Normalize the global kernel
    if normalize_global_kernel:
        globalKernel = normalizeKernel(globalKernel)

    return globalKernel


def get_environmentalKernels(atoms, nocenters=None, chem_channels=True, centerweight=1.0,
                             gaussian_width=0.5, cutoff=3.5,cutoff_transition_width=0.5,
                             nmax=8, lmax=6, chemicalKernel=deltaKernel,
                             nthreads=4, nprocess=2, nchunks = 2):
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
              'atoms':atoms,'centerweight': centerweight, 'gaussian_width': gaussian_width,
              'cutoff': cutoff, 'cutoff_transition_width': cutoff_transition_width,
              'nmax': nmax, 'lmax': lmax, 'chemicalKernelmat': chemicalKernelmat,
              'chem_channels': True ,'nocenters': nocenters,
                   }
    
    if nchunks == 1:
        kargs = {'nthreads':nthreads}
        kargs.update(**soap_params)
        # get the environmental kernels as a dictionary
        environmentalKernels = get_environmentalKernels_singleprocess(**kargs)
    else:
        kargs = {'nthreads':nthreads,'nprocess':nprocess, 'nchunks':nchunks}
        kargs.update(**soap_params)


        environmentalKernels = get_environmentalKernels_mt_mp_chunks(**kargs)
        
    return environmentalKernels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Computes the Global average/rematch kernel.""")

    parser.add_argument("filename", nargs=1, help="Name of the LibAtom formatted xyz input file")
    parser.add_argument("-n", type=int, default=8, help="Number of radial functions for the descriptor")
    parser.add_argument("-l", type=int, default=6, help="Maximum number of angular functions for the descriptor")
    parser.add_argument("-c", type=float, default=3.5, help="Radial cutoff")
    parser.add_argument("-cotw", type=float, default=0.5, help="Cutoff transition width")
    parser.add_argument("-g", type=float, default=0.5, help="Atom Gaussian sigma")
    parser.add_argument("-cw", type=float, default=1.0, help="Center atom weight")
    parser.add_argument("-k","--kernel", type=str, default="average",
                        help="Global kernel mode (e.g. --kernel average / rematch ")
    parser.add_argument("-gm","--gamma", type=float, default=1.0,
                        help="Regularization for entropy-smoothed best-match kernel")
    parser.add_argument("-z", "--zeta", type=int, default=2, help="Power for the environmental matrix")
    parser.add_argument("--prefix", type=str, default='', help="Prefix for output files (defaults to input file name)")
    parser.add_argument("--first", type=int, default='0', help="Index of first frame to be read in")
    parser.add_argument("--last", type=int, default='0', help="Index of last frame to be read in")
    parser.add_argument("--outformat", type=str, default='text', help="Choose how to dump the alchemySoaps, e.g. pickle (default) or text (same as from glosim --verbose)")
    parser.add_argument("-nt","--nthreads", type=int, default=4, help="Number of threads (1,2,4,6 or 9).")
    parser.add_argument("-np","--nprocess", type=int, default=4, help="Number of processes to run in parallel.")
    parser.add_argument("-nc","--nchunks", type=int, default=4, help="Number of chunks to divide the global kernel matrix in.")
    parser.add_argument("--nocenters", type=str, default="",help="Comma-separated list of atom Z to be ignored as environment centers (e.g. --nocenter 1,2,4)")
    parser.add_argument("-ngk","--normalize-global-kernel", action='store_true', help="Normalize global kernel")

    args = parser.parse_args()


###### Reads parameters input ######
    filename = args.filename[0]
    prefix = args.prefix
    centerweight = args.cw
    gaussian_width = args.g
    cutoff = args.c
    cutoff_transition_width = args.cotw
    nmax = args.n
    lmax = args.l

    global_kernel_type = args.kernel
    zeta = args.zeta
    gamma = args.gamma

    nthreads = args.nthreads
    nprocess = args.nprocess
    nchunks = args.nchunks
    normalize_global_kernel = args.normalize_global_kernel


    first = args.first if args.first>0 else None
    last = args.last if args.last>0 else None

    if args.outformat in ['text','pickle']:
        outformat = args.outformat
    else:
        raise Exception('outformat is not recognised')

    # reads the nocenters list and transforms it into a list
    if args.nocenters == "":
        nocenters = []
    else:
        nocenters = map(int, args.nocenters.split(','))
    nocenters = sorted(list(set(nocenters)))

    ###### Start the app ######
    print "{}".format(time.ctime())
    print "Start Computing the global {} kernel of {}".format(global_kernel_type,filename)

    # format of the output name
    if prefix == "": prefix = filename
    if prefix.endswith('.xyz'): prefix = prefix[:-4]
    prefix += "-n"+str(nmax)+"-l"+str(lmax)+"-c"+str(cutoff)+\
             "-g"+str(gaussian_width)+ "-cw"+str(centerweight)+ \
             "-cotw" +str(cutoff_transition_width)
    if global_kernel_type == 'average':
        prefix += '-average-zeta{:.0f}'.format(zeta)
    elif global_kernel_type == 'rematch':
        prefix += '-rematch-gamma{:.2f}'.format(gamma)
    else:
        raise ValueError
    if not normalize_global_kernel: prefix += "-nonorm"
    print  "using output prefix =", prefix


    st = time.time()

    # Reads the file and create a list of quippy frames object
    atoms = qp.AtomsList(filename, start=first, stop=last)
    n = len(atoms)


    soap_params = {
        'atoms': atoms, 'centerweight': centerweight, 'gaussian_width': gaussian_width,
        'cutoff': cutoff, 'cutoff_transition_width': cutoff_transition_width,
        'nmax': nmax, 'lmax': lmax, 'chemicalKernel': deltaKernel,
        'chem_channels': True, 'nocenters': nocenters,
    }

    print 'Reading {} input atomic structure from {}: done {}'.format(n,filename,s2hms(time.time() - st))

    if nchunks == 1:
        print 'Compute soap and environmental kernels with {} : {}'.format(nthreads,s2hms(time.time() - st))
    else:
        print 'Compute soap and environmental kernels with ' \
              'a pool of {} workers and {} threads over {} chunks: {}'.format(nprocess, nthreads,
                                                                              nchunks * (nchunks + 1) // 2,
                                                                              s2hms(time.time() - st))
    environmentalKernels = get_environmentalKernels(
                                nthreads=nthreads,nprocess=nprocess, nchunks=nchunks,
                                **soap_params)

    print 'Compute environmental kernels: done {}'.format(s2hms(time.time() - st))

    # Reduce the environemental kernels into global kernels
    globalKernel = get_globalKernel(kernel_type=global_kernel_type, zeta=zeta, gamma=gamma,
                                    eps=1e-6, nthreads=8,
                                    normalize_global_kernel=normalize_global_kernel)

    if global_kernel_type == 'average':
        print 'Compute global average kernel with zeta={} : done {}'.format(zeta, s2hms(time.time() - st))
    elif global_kernel_type == 'rematch':
        print 'Compute global rematch kernel with gamma={} : done {}'.format(gamma, s2hms(time.time() - st))


    # Normalize the global kernel
    if normalize_global_kernel:
        globalKernel = normalizeKernel(globalKernel)

    # Save the global kernel
    if outformat == 'text':
        np.savetxt(prefix + ".k",globalKernel)
    elif outformat == 'pickle':
        raise NotImplementedError
