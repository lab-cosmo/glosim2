
import quippy as qp
import numpy  as np
import argparse
import time
from libmatch.soap import get_Soaps
from libmatch.chemical_kernel import Atoms2ChemicalKernelmat,deltaKernel,randKernel
from libmatch.environmental_kernel import compile_with_threads,framesprod,nb_frameprod_upper
from libmatch.global_kernel import avgKernel,normalizeKernel
from libmatch.multithreading import chunk_list,chunks1d_2_chuncks2d,join_envKernel
from libmatch.utils import s2hms
import multiprocessing as mp

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

    return framesprod(frameprodFunc=nb_mtfunc, **kargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Computes the SOAP vectors of a list of atomic frame 
            and differenciate the chemical channels. Ready for alchemical kernel.""")

    parser.add_argument("filename", nargs=1, help="Name of the LibAtom formatted xyz input file")
    parser.add_argument("-n", type=int, default='8', help="Number of radial functions for the descriptor")
    parser.add_argument("-l", type=int, default='6', help="Maximum number of angular functions for the descriptor")
    parser.add_argument("-c", type=float, default='3.5', help="Radial cutoff")
    parser.add_argument("-cotw", type=float, default='0.5', help="Cutoff transition width")
    parser.add_argument("-g", type=float, default='0.5', help="Atom Gaussian sigma")
    parser.add_argument("-cw", type=float, default='1.0', help="Center atom weight")
    parser.add_argument("--prefix", type=str, default='', help="Prefix for output files (defaults to input file name)")
    parser.add_argument("--first", type=int, default='0', help="Index of first frame to be read in")
    parser.add_argument("--last", type=int, default='0', help="Index of last frame to be read in")
    parser.add_argument("--outformat", type=str, default='text', help="Choose how to dump the alchemySoaps, e.g. pickle (default) or text (same as from glosim --verbose)")
    parser.add_argument("-z","--zeta", type=int, default=2,help="Power for the environmental matrix")
    parser.add_argument("--nthreads", type=int, default=4, help="Number of threads (1,2,4,6 or 9).")
    parser.add_argument("--nprocess", type=int, default=4, help="Number of processes to run in parallel.")
    parser.add_argument("--nchunks", type=int, default=4, help="Number of chunks to divide the global kernel matrix in.")
    parser.add_argument("--nocenters", type=str, default="",help="Comma-separated list of atom Z to be ignored as environment centers (e.g. --nocenter 1,2,4)")
    parser.add_argument("--nonorm",type=bool, default=True, help="Does not normalize structural kernels")

    args = parser.parse_args()

    filename = args.filename[0]
    prefix = args.prefix
    centerweight = args.cw
    gaussian_width = args.g
    cutoff = args.c
    cutoff_transition_width = args.cotw
    nmax = args.n
    lmax = args.l
    zeta = args.zeta

    params = {'centerweight': centerweight, 'gaussian_width': gaussian_width,
              'cutoff': cutoff, 'cutoff_transition_width': cutoff_transition_width,
              'nmax': nmax, 'lmax': lmax}

    nthreads = args.nthreads
    nprocess = args.nprocess
    nchunks = args.nchunks
    nonorm = args.nonorm

    first = args.first if args.first>0 else None
    last = args.last if args.last>0 else None

    if args.outformat in ['text','pickle']:
        outformat = args.outformat
    else:
        raise Exception('outformat is not recognised')

    if args.nocenters == "":
        nocenters = []
    else:
        nocenters = map(int, args.nocenters.split(','))

    nocenters = sorted(list(set(nocenters)))

    print "Start Computing average kernel"

    if prefix=="": prefix=filename
    if prefix.endswith('.xyz'): prefix=prefix[:-4]
    prefix += "-n"+str(nmax)+"-l"+str(lmax)+"-c"+str(cutoff)+\
             "-g"+str(gaussian_width)+ "-cw"+str(centerweight)+ \
             "-cotw" +str(cutoff_transition_width)
    if nonorm: prefix += "-nonorm"
    print  "using output prefix =", prefix
    # Reads input file using quippy
    print "Reading input file", filename

    st = time.time()

    # Reads the file and create a list of quippy frames object
    atoms = qp.AtomsList(filename, start=first, stop=last)
    n = len(atoms)

    chemicalKernelmat = Atoms2ChemicalKernelmat(atoms, chemicalKernel=deltaKernel)

    # print 'Load {:.0f} frames: done ({:.3f} sec)'.format(n,(time.time() - st))
    #
    # frames = get_Soaps(atoms, chem_channels=True, nocenters=nocenters, **params)
    #
    # print 'Compute Soaps: done ({:.3f} sec)'.format((time.time() - st))

    nb_mtfunc = compile_with_threads(nb_frameprod_upper, nthreads=nthreads)

    chunks1d, slices = chunk_list(atoms, nchunks=nchunks)

    pp = {'chemicalKernelmat': chemicalKernelmat, 'chem_channels': True, 'nocenters': []}
    pp.update(**params)

    chunks = chunks1d_2_chuncks2d(chunks1d, **pp)

    print 'Init Pool of {} workers: {}'.format(nprocess, s2hms(time.time() - st) )

    pool = mp.Pool(nprocess)

    res = pool.map_async(framesprod_wrapper, chunks)

    results = res.get()

    pool.close()
    pool.join()

    environmentalKernels = join_envKernel(results, slices)

    print 'Compute environmental kernels: done {}'.format(s2hms(time.time() - st))

    globalKernel = avgKernel(environmentalKernels, zeta)

    print 'Compute global average kernel: done {}'.format(s2hms(time.time() - st))

    if outformat == 'text':
        np.savetxt(prefix + ".k",globalKernel)
    elif outformat == 'pickle':
        raise NotImplementedError
