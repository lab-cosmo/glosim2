import argparse
import time

import numpy  as np
import cPickle as pck

from libmatch.chemical_kernel import Atoms2ChemicalKernelmat, deltaKernel
from libmatch.environmental_kernel import get_environmentalKernels_mt_mp_chunks, \
    get_environmentalKernels_singleprocess
from libmatch.global_kernel import avgKernel, rematchKernel, normalizeKernel
from libmatch.utils import s2hms
from GlobalSimilarity import get_globalKernel

from libmatch.chemical_kernel import Atoms2ChemicalKernelmat
from libmatch.soap import get_Soaps
from libmatch.utils import chunk_list, chunks1d_2_chuncks2d,is_notebook
from libmatch.environmental_kernel import choose_envKernel_func,framesprod
import os
from tqdm import tqdm
import quippy as qp

try:
    import numba as nb

    nonumba = False
except:
    print 'Numba is not installed... this will be much slower.'
    nonumba = True





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Computes the Global average/rematch kernel.""")

    parser.add_argument("filename", nargs=1, help="Name of the LibAtom formatted xyz input file")
    parser.add_argument("-n", type=int, default=8, help="Number of radial functions for the descriptor")
    parser.add_argument("-l", type=int, default=6, help="Maximum number of angular functions for the descriptor")
    parser.add_argument("-c", type=float, default=3.5, help="Radial cutoff")
    parser.add_argument("-cotw", type=float, default=0.5, help="Cutoff transition width")
    parser.add_argument("-g", type=float, default=0.5, help="Atom Gaussian sigma")
    parser.add_argument("-cw", type=float, default=1.0, help="Center atom weight")
    parser.add_argument("-k", "--kernel", type=str, default="average",
                        help="Global kernel mode (e.g. --kernel average / rematch ")
    parser.add_argument("-gm", "--gamma", type=float, default=1.0,
                        help="Regularization for entropy-smoothed best-match kernel")
    parser.add_argument("-z", "--zeta", type=int, default=2, help="Power for the environmental matrix")
    parser.add_argument("--prefix", type=str, default='', help="Prefix for output files (defaults to input file name)")
    parser.add_argument("--xlim", type=str, default='', help="Index of first frame to be read in")
    parser.add_argument("--ylim", type=str, default='', help="Index of last frame to be read in")
    parser.add_argument("--outformat", type=str, default='text',
                        help="Choose how to dump the alchemySoaps, e.g. pickle (default) or text (same as from glosim --verbose)")
    parser.add_argument("-nt", "--nthreads", type=int, default=4, help="Number of threads (1,2,4,6,9,12,16,25,36,48,64,81,100).")
    parser.add_argument("-np", "--nprocess", type=int, default=4, help="Number of processes to run in parallel.")
    parser.add_argument("-nc", "--nchunks", type=int, default=4,
                        help="Number of chunks to divide the global kernel matrix in.")
    parser.add_argument("--nocenters", type=str, default="",
                        help="Comma-separated list of atom Z to be ignored as environment centers (e.g. --nocenter 1,2,4)")
    parser.add_argument("-ngk", "--normalize-global-kernel", action='store_true', help="Normalize global kernel")
    parser.add_argument("-sek", "--save-env-kernels", action='store_true', help="Save environmental kernels")
    parser.add_argument("-lm", "--low-memory", action='store_true',
                        help="Computes the soap vectors in each thread when nchunks > 1")

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
    save_env_kernels = args.save_env_kernels
    islow_memory = args.low_memory

    try:
        a = args.xlim.split(',')
        b = args.ylim.split(',')
        xlim = [int(a[0]), int(a[1])]
        ylim = [int(b[0]), int(b[1])]
    except:
        if not args.xlim or not args.ylim:
            raise Exception('must provide xlim or ylim')
        else:
            print 'xlim {}'.format(args.xlim)
            print 'ylim {}'.format(args.ylim)
            raise Exception('xlim or ylim format not recognised')


    if args.outformat in ['text', 'pickle']:
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
    print "Start Computing the environmental kernels with " \
          "nmax={} lmax={} c={} g={} cw={} cotw={} from filename {}\n xlim={} ylim={}" \
          "".format(nmax,lmax,cutoff,gaussian_width,centerweight,cutoff_transition_width,
                    filename,xlim,ylim)

    # format of the output name
    if prefix == "": prefix = filename
    if prefix.endswith('.xyz'): prefix = prefix[:-4]
    prefix += "-n" + str(nmax) + "-l" + str(lmax) + "-c" + str(cutoff) + \
              "-g" + str(gaussian_width) + "-cw" + str(centerweight) + \
              "-cotw" + str(cutoff_transition_width)

    fn_env_kernels = prefix + '-env_kernels.pck'

    print  "using output prefix =", prefix

    st = time.time()
    ################################################################################
    # Reads the file and create a list of quippy frames object
    atoms1 = qp.AtomsList(filename, start=xlim[0], stop=xlim[1])
    print 'Reading {} input atomic structure from {} with index {}: done {}'.format(len(atoms1), filename,xlim, s2hms(time.time() - st))

    if xlim != ylim:
        atoms2 = qp.AtomsList(filename, start=ylim[0], stop=ylim[1])
        print 'Reading {} input atomic structure from {} with index {}: done {}'.format(len(atoms2), filename,ylim, s2hms(time.time() - st))
    else:
        atoms2 = None
        print 'no atoms 2, Computing upper triangular sub matrix'
    soap_params = {
        'centerweight': centerweight, 'gaussian_width': gaussian_width,
        'cutoff': cutoff, 'cutoff_transition_width': cutoff_transition_width,
        'nmax': nmax, 'lmax': lmax, 'chem_channels': True, 'nocenters': nocenters,
    }

    # DELTA CHEMICAL KERNEL hard coded
    chemicalKernelmat = Atoms2ChemicalKernelmat(atoms1,atoms2=atoms2, chemicalKernel=deltaKernel)
    # Chooses the function to use to compute the kernel between two frames
    get_envKernel = choose_envKernel_func(nthreads,isDeltaKernel=True)

    ####################################################################################
    # get the soap for every local environement
    frames1 = get_Soaps(atoms1, nprocess=nprocess,**soap_params )

    print 'Compute xSoap {} with {} process from {}: done {}'.format(xlim,nprocess, filename, s2hms(time.time() - st))
    if atoms2 is None:
        frames2 = None
        print 'no atoms 2, Computing upper triangular sub matrix'
    else:
        frames2 = get_Soaps(atoms2, nprocess=nprocess, **soap_params )
        print 'Compute ySoap {} process from {}: done {}'.format(ylim,nprocess, filename, s2hms(time.time() - st))

    ########################################################################################
    # get the environmental kernels as a dictionary

    print 'Compute soap and environmental kernels with {} threads: start {}'.format(nthreads, s2hms(time.time() - st))
    environmentalKernels = framesprod(frames1, frames2=frames2, frameprodFunc=get_envKernel,
                                      chemicalKernelmat=chemicalKernelmat)

    shift_environmentalKernels = {}
    for key,val in environmentalKernels.iteritems():
        shift_environmentalKernels[( int(key[0]+xlim[0]), int(key[1]+ylim[0]) )] = val

    print 'Compute environmental kernels: done {}'.format(s2hms(time.time() - st))

    if save_env_kernels:
        with open(fn_env_kernels, 'wb') as f:
            pck.dump(shift_environmentalKernels, f, protocol=pck.HIGHEST_PROTOCOL)
    print 'Save environmental kernels in {}: done {}'.format(fn_env_kernels,s2hms(time.time() - st))

    print "Exit program \n {}".format(time.ctime())
