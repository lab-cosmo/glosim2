
import quippy as qp
import numpy  as np
import argparse
import time
from libmatch.soap import get_Soaps
from libmatch.chemical_kernel import Atoms2ChemicalKernelmat,deltaKernel,randKernel
from libmatch.environmental_kernel import framesprod,compile_envKernel_with_thread,np_frameprod3,np_frameprod_upper
from libmatch.global_kernel import avgKernel,rematchKernel,normalizeKernel
from libmatch.multithreading import chunk_list,chunks1d_2_chuncks2d,join_envKernel
from libmatch.utils import s2hms
import multiprocessing as mp

try:
    import numba as nb
    nonumba = False
except:
    print 'Numba is not installed... this will be much slower.'
    nonumba = True


def framesprod_wrapper(kargs):
    keys = kargs.keys()

    if 'atoms1' in keys:
        atoms1 = kargs.pop('atoms1')
        atoms2 = kargs.pop('atoms2')
        chemicalKernelmat = kargs.pop('chemicalKernelmat')

        frames1 = get_Soaps(atoms1, **kargs)
        if atoms2 is not None:
            frames2 = get_Soaps(atoms2, **kargs)
        else:
            frames2 = None

        kargs = {'frames1': frames1, 'frames2': frames2, 'chemicalKernelmat': chemicalKernelmat}

    return framesprod(frameprodFunc=get_envKernel, **kargs)

def mp_framesprod(chunks, nprocess):

    pool = mp.Pool(nprocess)
    results = pool.map(framesprod_wrapper, chunks)

    pool.close()
    pool.join()

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Computes the SOAP vectors of a list of atomic frame 
            and differenciate the chemical channels. Ready for alchemical kernel.""")

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

    params = {'centerweight': centerweight, 'gaussian_width': gaussian_width,
              'cutoff': cutoff, 'cutoff_transition_width': cutoff_transition_width,
              'nmax': nmax, 'lmax': lmax}

    globalkernel = args.kernel
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
    print "Start Computing the global {} kernel of {}".format(globalkernel,filename)

    # format of the output name
    if prefix == "": prefix = filename
    if prefix.endswith('.xyz'): prefix = prefix[:-4]
    prefix += "-n"+str(nmax)+"-l"+str(lmax)+"-c"+str(cutoff)+\
             "-g"+str(gaussian_width)+ "-cw"+str(centerweight)+ \
             "-cotw" +str(cutoff_transition_width)
    if globalkernel == 'average':
        prefix += '-average-zeta{:.0f}'.format(zeta)
    elif globalkernel == 'rematch':
        prefix += '-rematch-gamma{:.2f}'.format(gamma)
    else:
        raise ValueError
    if not normalize_global_kernel: prefix += "-nonorm"
    print  "using output prefix =", prefix

    print "Reading input file", filename

    st = time.time()

    # Reads the file and create a list of quippy frames object
    atoms = qp.AtomsList(filename, start=first, stop=last)
    n = len(atoms)

    # Builds the kernel matrix from the species present in the frames and a specified chemical
    # kernel function
    chemicalKernelmat = Atoms2ChemicalKernelmat(atoms, chemicalKernel=deltaKernel)

    # Chooses the function to use to compute the kernel between two frames
    if nonumba:
        print 'Using numpy version of envKernel function'

        get_envKernel = np_frameprod3
    else:
        print 'Using compiled and threaded of envKernel function'
        get_envKernel = compile_envKernel_with_thread(nthreads)

    # Chooses between the 1 core implementation and the multiprocess multithread implementation
    # of the environmental matrices
    if nchunks == 1:
        kargs = {'chemicalKernelmat': chemicalKernelmat, 'chem_channels': True, 'nocenters': nocenters}
        kargs.update(**params)
        # get the alchemical soap for every local environement
        frames = get_Soaps(atoms, **kargs)
        # get the environmental kernels as a dictionary
        environmentalKernels = framesprod(frames, frameprodFunc=get_envKernel,  chemicalKernelmat=chemicalKernelmat)

    else:
        # cut atomsList in chunks
        chunks1d, slices = chunk_list(atoms, nchunks=nchunks)

        pp = {'chemicalKernelmat': chemicalKernelmat, 'chem_channels': True, 'nocenters': nocenters}
        pp.update(**params)
        # create inputs for each block of the global kernel matrix
        chunks = chunks1d_2_chuncks2d(chunks1d, **pp)

        print 'Compute soap and environmental kernels with ' \
              'a pool of {} workers and {} threads over {} chunks: {}'.format(nprocess,nthreads,nchunks*(nchunks+1)//2, s2hms(time.time() - st) )

        # get a list of environemental kernels
        results = mp_framesprod(chunks,nprocess)
        # reorder the list of environemental kernels into a dictionary which keys are the (i,j) of the global kernel matrix
        environmentalKernels = join_envKernel(results, slices)

    print 'Compute environmental kernels: done {}'.format(s2hms(time.time() - st))

    # Reduce the environemental kernels into global kernels
    if globalkernel == 'average':
        globalKernel = avgKernel(environmentalKernels, zeta)
        print 'Compute global average kernel with zeta={} : done {}'.format(zeta, s2hms(time.time() - st))

    elif globalkernel == 'rematch':
        globalKernel = rematchKernel(environmentalKernels, gamma=gamma, eps=1e-6, nthreads=8)
        print 'Compute global rematch kernel with gamma={} : done {}'.format(gamma, s2hms(time.time() - st))
    else:
        raise ValueError

    # Normalize the global kernel
    if normalize_global_kernel:
        globalKernel = normalizeKernel(globalKernel)

    # Save the global kernel
    if outformat == 'text':
        np.savetxt(prefix + ".k",globalKernel)
    elif outformat == 'pickle':
        raise NotImplementedError
