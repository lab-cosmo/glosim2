from subprocess import Popen
from time import time,ctime
from libmatch.utils import s2hms
import cPickle as pck
from GlobalSimilarity import get_globalKernel
import numpy as np
from Pool.mpi_pool import MPIPool
import sys,os,argparse
import quippy as qp

def func(command):
    p = Popen(command, shell=True)
    return p.wait()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Computes the Global average/rematch kernel. Needs MPI to run, 
    mpiexec -n 4 python """)

    parser.add_argument("filename", nargs=1, help="Name of the LibAtom formatted xyz input file")
    parser.add_argument("-pe", "--path-to-executable", type=str, default="/home/musil/git/glosim2/",
                        help="Path to the executable that runs, i.e. GlobalSimilarity_cluster.py")
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
    parser.add_argument("-nt","--nthreads", type=int, default=4, help="Number of threads (1,2,4,6,9,12,16,25,36,48,64,81,100).")
    parser.add_argument("-np","--nprocess", type=int, default=4, help="Number of processes to run in parallel.")
    parser.add_argument("-cl","--chunklen", type=str, default='', help="Lenght of chunks to divide the global kernel matrix in. (comma separated)")
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
    normalize_global_kernel = args.normalize_global_kernel
    save_env_kernels = True


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
    pool = MPIPool()

    if not pool.is_master():
        # Wait for instructions from the master process.
        pool.wait()
        sys.exit(0)

    print "Start dirty parallelisation for cluster: {}".format(ctime())
    print "Start Computing the global {} kernel of {}".format(global_kernel_type, filename)
    # run commands in parallel
    st = time()

    frames = qp.AtomsList(filename,start=first,stop=last)
    Nframe = len(frames)

    if args.chunklen:
        aa = args.chunklen.split(',')
        if len(aa) > 1:
            xchunklen, ychunklen = int(aa[0]), int(aa[1])
        elif len(aa) == 1:
            xchunklen, ychunklen = int(aa[0]), int(aa[0])
        else:
            raise Exception('chunk lenght format not recognised')
    else:
        print 'Default chunk lenght'
        xchunklen, ychunklen = Nframe // 5, Nframe // 5

    xNchunk = Nframe // xchunklen
    yNchunk = Nframe // ychunklen

    xslices = [(it*xchunklen,(it+1)*xchunklen) for it in range(xNchunk)]
    yslices = [(it*ychunklen,(it+1)*ychunklen) for it in range(yNchunk)]
    xslices.append(((xNchunk)*xchunklen,Nframe))
    yslices.append(((yNchunk)*ychunklen,Nframe))

    params = "-n" + str(nmax) + "-l" + str(lmax) + "-c" + str(cutoff) + \
             "-g" + str(gaussian_width) + "-cw" + str(centerweight) + \
             "-cotw" + str(cutoff_transition_width)

    if prefix:
        abspath = os.path.abspath(prefix)
    else:
        abspath = os.path.abspath(filename)
    path,name = os.path.split(abspath)

    if name.endswith('.xyz'):
        name = name[:-4]
    path += '/'
    outpath = path

    suffix = 0
    while os.path.exists(path+name+params+'_tmp{}'.format(suffix)):
        suffix += 1
    tmp_path = path+name+ params +'_tmp{}/'.format(suffix)
    print 'TMP output is in ' + tmp_path
    os.makedirs(tmp_path)



    fn_env_kernels = [tmp_path+name+'-{xf},{xl}-{yf},{yl}'.format(xf=xsl[0],xl=xsl[1],yf=ysl[0],yl=ysl[1])
                      +params + '-env_kernels.pck'
                      for xsl in xslices for ysl in yslices  if ysl[0] >= xsl[0]]

    path2GlobSim = 'GlobalSimilarity_cluster.py'
    commands = ['python {path2exec} {filename} ' \
                '-n {nmax} -l {lmax} -c {cutoff} -g {gaussian_width} -cw {centerweight} ' \
                '-cotw {cutoff_transition_width} -z {zeta} -gm {gamma} -k {kernel} ' \
                '-nt {nthreads} -np {nprocess} -nc 1 --xlim {xf},{xl} --ylim {yf},{yl}  ' \
                '--prefix {prefix}{name}-{xf},{xl}-{yf},{yl} -sek ' \
                '2>&1 | tee {prefix}log-{xf},{xl}-{yf},{yl} >/dev/null'
                .format(path2exec=path2GlobSim,filename=filename,nmax=nmax,lmax=lmax,cutoff=cutoff,
                        gaussian_width=gaussian_width,centerweight=centerweight,
                        cutoff_transition_width=cutoff_transition_width,
                        zeta=zeta,gamma=gamma,kernel=global_kernel_type,
                        nthreads=nthreads,nprocess=nprocess,
                        xf=xsl[0], xl=xsl[1], yf=ysl[0], yl=ysl[1],
                        prefix=tmp_path,name=name)
                for xsl in xslices for ysl in yslices if ysl[0] >= xsl[0]
                ]

    pool.map(func,commands)

    pool.close()

    env_kernels = {}
    for fn in fn_env_kernels:
        with open(fn,'rb') as f:
            aa = pck.load(f)
            env_kernels.update(**aa)

    gkt = ''
    norm = ''
    if global_kernel_type == 'average':
        gkt = '-average-zeta{:.0f}'.format(zeta)
    elif global_kernel_type == 'rematch':
        gkt = '-rematch-gamma{:.2f}'.format(gamma)
    else:
        raise ValueError
    if not normalize_global_kernel: norm = "-nonorm"

    globalKernel = get_globalKernel(env_kernels,kernel_type=global_kernel_type,zeta=zeta,nthreads=1)

    fn = outpath + name + params + '-env_kernels.pck'
    print 'Saving env kernels in ' + fn
    with open(fn, 'wb') as f:
        pck.dump(env_kernels,f,protocol=pck.HIGHEST_PROTOCOL)


    fn = outpath + name + params + gkt + norm + '.k'
    print 'Saving global kernel in ' + fn
    np.savetxt(fn,globalKernel)

    print 'Finished in: {}'.format(s2hms(time()-st))
    print 'Closing app: {}'.format(ctime())