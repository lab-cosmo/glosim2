
import quippy as qp
import numpy  as np
import argparse
import time
from libmatch.soap import get_Soaps
from libmatch.chemical_kernel import get_chemicalKernelmatFrames,deltaKernel,randKernel
from libmatch.environmental_kernel import compile_with_threads,framesprod,nb_frameprod_upper
from libmatch.global_kernel import avgKernel,normalizeKernel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Computes the SOAP vectors of a list of atomic frame 
            and differenciate the chemical channels. Ready for alchemical kernel.""")

    parser.add_argument("filename", nargs=1, help="Name of the LibAtom formatted xyz input file")
    parser.add_argument("-n", type=int, default='8', help="Number of radial functions for the descriptor")
    parser.add_argument("-l", type=int, default='6', help="Maximum number of angular functions for the descriptor")
    parser.add_argument("-c", type=float, default='5.0', help="Radial cutoff")
    parser.add_argument("-cotw", type=float, default='0.5', help="Cutoff transition width")
    parser.add_argument("-g", type=float, default='0.5', help="Atom Gaussian sigma")
    parser.add_argument("-cw", type=float, default='1.0', help="Center atom weight")
    parser.add_argument("--prefix", type=str, default='', help="Prefix for output files (defaults to input file name)")
    parser.add_argument("--first", type=int, default='0', help="Index of first frame to be read in")
    parser.add_argument("--last", type=int, default='0', help="Index of last frame to be read in")
    parser.add_argument("--outformat", type=str, default='text', help="Choose how to dump the alchemySoaps, e.g. pickle (default) or text (same as from glosim --verbose)")
    parser.add_argument("-zeta", type=int, default=2,help="Power for the environmental matrix")
    parser.add_argument("--nthreads", type=int, default=4, help="Number of threads.")
    parser.add_argument("--nocenters", type=str, default="",help="Comma-separated list of atom Z to be ignored as environment centers (e.g. --nocenter 1,2,4)")
    parser.add_argument("--nonorm",type=bool, default=False, help="Does not normalize structural kernels")

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

    print 'Load {:.0f} frames: done ({:.3f} sec)'.format(n,(time.time() - st))

    frames = get_Soaps(atoms, chem_channels=True, nocenters=nocenters, **params)

    print 'Compute Soaps: done ({:.3f} sec)'.format((time.time() - st))

    chemicalKernelmat = get_chemicalKernelmatFrames(frames, chemicalKernel=deltaKernel)

    nb_mtfunc = compile_with_threads(nb_frameprod_upper, nthreads=nthreads)

    environmentalKernels = framesprod(frames, frames2=None, chemicalKernelmat=chemicalKernelmat, frameprodFunc=nb_mtfunc)

    print 'Compute environmental kernels: done ({:.3f} sec)'.format((time.time() - st))

    globalKernel = avgKernel(environmentalKernels, zeta)

    print 'Compute global average kernel: done ({:.3f} sec)'.format((time.time() - st))

    if outformat == 'text':
        np.savetxt(prefix + ".k",globalKernel)
    elif outformat == 'pickle':
        raise NotImplementedError
