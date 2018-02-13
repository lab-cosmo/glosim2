import quippy as qp
import numpy  as np
from utils import get_spkit,get_spkitMax,envIdx2centerIdxMap
from data_model import AlchemyFrame,AlchemySoap,ProjectedAlchemySoap
import multiprocessing as mp
from libmatch.utils import is_notebook
import os,signal,threading,psutil

if is_notebook():
    from tqdm import tqdm_notebook as tqdm_cs
else:
    from tqdm import tqdm as tqdm_cs

def get_alchemy_frame( spkit, spkitMax,atoms=None,fpointer=None, nocenters=None, centerweight=1., gaussian_width=0.5,cutoff=3.5,
                      chemicalProjection=None,
                      cutoff_transition_width=0.5, nmax=8, lmax=6,chem_channels=True,is_fast_average=False,queue=None):
    if nocenters is None:
        nocenters = []

    if atoms is None and fpointer is not None:
        atoms = qp.Atoms(fpointer=fpointer)
    elif atoms is not None and fpointer is None:
        atoms = atoms
    elif atoms is not None and fpointer is not None:
        atoms = atoms
    else:
        raise NotImplementedError('At least atoms or fpointer needs to be given')
    spkit = get_spkit(atoms)
    soapParams = {'spkit': spkit, 'spkitMax': spkitMax, 'nocenters': nocenters,
                  'centerweight': centerweight, 'gaussian_width': gaussian_width,
                  'cutoff': cutoff, 'cutoff_transition_width': cutoff_transition_width,
                  'nmax': nmax, 'lmax': lmax,'is_fast_average':is_fast_average}


    rawsoaps = get_soap(atoms, **soapParams)

    zList = atoms.get_atomic_numbers()

    mm = envIdx2centerIdxMap(atoms, spkit, nocenters)
    # chemical channel separation for each central atom species
    # and each atomic environment
    alchemyFrame = AlchemyFrame(atom=atoms, nocenters=nocenters, soapParams=soapParams,is_fast_average=is_fast_average)
    Nenv, Npowerspectrum = rawsoaps.shape
    if chem_channels:
        for it in xrange(Nenv):
            # soap[it] is (1,Npowerspectrum) so need to transpose it
            #  convert the soap vector of an environment from quippy descriptor to soap vectors
            # with chemical channels.
            alchemySoapdict = Soap2AlchemySoap(rawsoaps[it, :], spkitMax, nmax, lmax)

            alchemySoap = AlchemySoap(qpatoms=atoms, soapParams=soapParams, centerIdx=mm[it],
                                      is_fast_average=is_fast_average)

            alchemySoap.from_dict(alchemySoapdict)
            if chemicalProjection is not None:
                alchemySoap = ProjectedAlchemySoap(alchemySoap,chemicalProjection)

            centerZ = zList[mm[it]] if not is_fast_average else 'AVG'
            alchemyFrame[centerZ] = alchemySoap
    else:
        for it in xrange(Nenv):
            centerZ = zList[mm[it]] if not is_fast_average else 'AVG'
            alchemyFrame[centerZ] = rawsoaps[it, :]

    return alchemyFrame


def get_alchemy_frame_wrapper(kargs):
    idx = kargs.pop('idx')
    return (idx,get_alchemy_frame(**kargs))


class mp_soap(object):
    def __init__(self, chunks, nprocess,dispbar=False):
        super(mp_soap, self).__init__()
        self.func_wrap = get_alchemy_frame_wrapper
        self.dispbar = dispbar
        self.parent_id = os.getpid()
        self.nprocess = nprocess

        for it,chunk in enumerate(chunks):
            chunk.update(**{"idx": it})
        self.chunks = chunks

    def run(self):
        Nit = len(self.chunks)
        pbar = tqdm_cs(total=Nit,desc='SOAP vectors',disable=self.dispbar)
        results = {}
        if self.nprocess > 1:
            pool = mp.Pool(self.nprocess, initializer=self.worker_init,
                                maxtasksperchild=10)

            for idx, res in pool.imap_unordered(self.func_wrap, self.chunks):
                results[idx] = res
                pbar.update()

            pool.close()
            pool.join()

        elif self.nprocess == 1:
            for chunk in self.chunks:
                idx,res = self.func_wrap(chunk)
                results[idx] = res

                pbar.update()
        else:
            print 'Nproces: ',self.nprocess
            raise NotImplementedError('need at least 1 process')

        pbar.close()

        Frames = [results[it] for it in range(Nit)]

        return Frames
    # clean kill of the pool in interactive sessions
    def worker_init(self):
        def sig_int(signal_num, frame):
            print('signal: %s' % signal_num)
            parent = psutil.Process(self.parent_id)
            for child in parent.children():
                if child.pid != os.getpid():
                    print("killing child: %s" % child.pid)
                    child.kill()
            print("killing parent: %s" % self.parent_id)
            parent.kill()
            print("suicide: %s" % os.getpid())
            psutil.Process(os.getpid()).kill()

        signal.signal(signal.SIGINT, sig_int)




def get_Soaps(atoms, nocenters=None, chem_channels=False, centerweight=1.0, gaussian_width=0.5, cutoff=3.5,
              cutoff_transition_width=0.5, nmax=8, lmax=6, spkitMax=None, nprocess=1,chemicalProjection=None,
              dispbar=False,is_fast_average=False):
    '''
    Compute the SOAP vectors for each atomic environment in atoms and
    reorder them into chemical channels.

    :param is_fast_average: 
    :param nprocess: 
    :param atoms: list of quippy Atoms object
    :param centerweight: Center atom weight
    :param gaussian_width: Atom Gaussian std
    :param cutoff: Cutoff radius for each atomic environment in the unit of cell and positions.
    :param cutoff_transition_width: Steepness of the smooth environmental cutoff radius. Smaller -> steeper
    :param nmax: Number of radial basis functions.
    :param lmax: Number of Spherical harmonics.
    :return: Nested List/Dictionary: list->atoms,
                dict->(keys:atomic number,
                items:list of atomic environment), list->atomic environment,
                dict->(keys:chemical channel, (sp1,sp2) sp* is atomic number
                      inside the atomic environment),
                       items: SOAP vector, flat numpy array)
    '''
    if nocenters is None:
        nocenters = []


    # get the set of species their maximum number across atoms
    if spkitMax is None:
        spkitMax = get_spkitMax(atoms)

    soapParams = []
    not_considered = []
    for it,frame in enumerate(atoms):
        ## if the frame is empty because of the nocenters then don't add it to the computation
        ## the frame/fingerprint ordering will be changed
        spkit = get_spkit(frame)
        sps = spkit.keys()
        intersec = list(set(sps).difference(nocenters))
        if len(intersec) > 0:
            soapParam = \
                { 'spkit': spkit , 'spkitMax': spkitMax,
                 'nocenters': nocenters, 'is_fast_average': is_fast_average,
                 'chem_channels': chem_channels,'chemicalProjection':chemicalProjection,
                 'centerweight': centerweight, 'gaussian_width': gaussian_width,
                 'cutoff': cutoff, 'cutoff_transition_width': cutoff_transition_width,
                 'nmax': nmax, 'lmax': lmax}

            if nprocess > 1:
                soapParam.update(**{'fpointer': frame._fpointer.copy()})
            elif nprocess == 1:
                soapParam.update(**{'atoms': frame})
            soapParams.append(soapParam)
        else:
            not_considered.append(it)
            #print 'frame {} does not contain centers'.format(it)
    if len(not_considered)>0:
        print 'frames\n {} \ndo not contain centers'.format(not_considered)
    compute_soaps = mp_soap(soapParams,nprocess,dispbar=dispbar)

    Frames = compute_soaps.run()

    return Frames


def get_soap(atoms, spkit, spkitMax, nocenters=None, centerweight=1., gaussian_width=0.5,
             cutoff=3.5, cutoff_transition_width=0.5, nmax=8, lmax=6,is_fast_average=False):
    '''
    Get the soap vectors (power spectra) for each atomic environments in atom.

    :param atoms: A quippy Atoms object
    :param spkit: Dictionary with specie as key and number of corresponding atom as item.
                    Returned by get_spkit(atom).
    :param spkitMax: Dictionary with species as key and return its largest number of occurrence.
                        Returned by get_spkitMax(atoms) .
    :param nocenters: list of atomic numbers to exclude as center for the soap evaluation
    :param centerweight: Center atom weight
    :param gaussian_width: Atom Gaussian std
    :param cutoff: Cutoff radius for each atomic environment in the unit of cell and positions.
    :param cutoff_transition_width: Steepness of the smooth environmental cutoff radius. Smaller -> steeper
    :param nmax: Number of radial basis functions.
    :param lmax: Number of Spherical harmonics.
    :param is_fast_average: bool. Compute the average soap in quippy and only returns it.
    :return: Soap vectors of atom. 2D array shape=(Nb of centers,Length of the rawsoap). The ordering of the centers is identical as in atom.
    '''

    if nocenters is None:
        nocenters = []

    zsp = spkitMax.keys()
    zsp.sort()
    lspecies = 'n_species=' + str(len(zsp)) + ' species_Z={ '
    for z in zsp:
        lspecies = lspecies + str(z) + ' '
    lspecies = lspecies + '}'

    if is_fast_average:
        ##### AVERAGE FROM QUIPPY IS NOT THE SAME AS AVERAGING OVER THE CENTERS
        average = 'F'
        Ncenters = 1
    else:
        average = 'F'
        Ncenters = 0
        for z in atoms.get_atomic_numbers():
            if z in nocenters:
                continue
            Ncenters += 1

    atoms.set_cutoff(cutoff)
    atoms.calc_connect()


    spInFrame = spkit.keys()
    # makes sure that the nocenters is propely adapted to the species present in the frame
    nocenterInFrame = []
    for nocenter in nocenters:
        if nocenter in spInFrame:
            nocenterInFrame.append(nocenter)

    centers = ' n_Z=' + str(len(spInFrame ) -len(nocenterInFrame)) + ' Z={ '
    for z in spInFrame:
        if z in nocenterInFrame:
            continue
        centers += str(z) + ' '
    centers += '} '

    soapstr = "average="+average+" normalise=T soap central_reference_all_species=F " \
              " central_weight=" + str(centerweight )+ \
              " covariance_sigma0=0.0 atom_sigma=" + str(gaussian_width) + \
              " cutoff=" + str(cutoff) + \
              " cutoff_transition_width=" + str(cutoff_transition_width) + \
              " n_max=" + str(nmax) + " l_max=" + str(lmax) + ' ' \
              + lspecies + centers

    desc = qp.descriptors.Descriptor(soapstr)
    # computes the soap descriptors for the full frame (atom)
    soap = desc.calc(atoms ,grad=False)["descriptor"]

    if is_fast_average:
        soap = soap.mean(axis=0)


    return soap.reshape((Ncenters,-1))


def Soap2AlchemySoap(rawsoap, spkitMax, nmax, lmax):
    '''
    Convert the soap vector of an environment from quippy descriptor to soap vectors
     with chemical channels.

    :param rawsoap: numpy array dim:(N,) containing the soap vector of one environment
    :param spkitMax: Dictionary with specie as key and number across all frames considered.
                    Returned by get_spkit(atom).
    :param nmax: Number of radial basis functions.
    :param lmax: Number of Spherical harmonics.
    :return: Dictionary  (keys: species tuples (sp1,sp2),
                            items: soap vector, numpy array dim:(nmax ** 2 * (lmax + 1),) )
    '''
    # spkitMax keys are the center species in the all frames
    zspecies = sorted(spkitMax.keys())
    nspecies = len(spkitMax.keys())

    alchemyKeys = []
    for z1 in zspecies:
        for z2 in zspecies:
            alchemyKeys.append((z1, z2))

    LENalchsoap = nmax ** 2 * (lmax + 1)
    alchemySoapdict = {}

    ipair = {}
    # initialize the alchemical soap
    for s1 in xrange(nspecies):
        for s2 in xrange(
                nspecies):  # range(s1+1): we actually need to store also the reverse pairs if we want to go alchemical
            alchemySoapdict[(zspecies[s2], zspecies[s1])] = np.zeros(LENalchsoap, float)
            ipair[(zspecies[s2], zspecies[s1])] = 0

    isoap = 0
    isqrttwo = 1.0 / np.sqrt(2.0)

    # selpair and revpair are modified and in turn modify soaps because they are all pointing at the same memory block
    for s1 in xrange(nspecies):
        for n1 in xrange(nmax):
            for s2 in xrange(s1 + 1):
                selpair = alchemySoapdict[(zspecies[s2], zspecies[s1])]
                # we need to reconstruct the spectrum for the inverse species order, that also swaps n1 and n2.
                # This is again only needed to enable alchemical combination of e.g. alpha-beta and beta-alpha. Shit happens
                revpair = alchemySoapdict[(zspecies[s1], zspecies[s2])]
                isel = ipair[(zspecies[s2], zspecies[s1])]
                for n2 in xrange(nmax if s2 < s1 else n1 + 1):
                    for l in xrange(lmax + 1):
                        # print s1, s2, n1, n2, isel, l+(self.lmax+1)*(n2+self.nmax*n1), l+(self.lmax+1)*(n1+self.nmax*n2)
                        # selpair[isel] = rawsoap[isoap]
                        if (s1 != s2):
                            selpair[isel] = rawsoap[
                                                isoap] * isqrttwo  # undo the normalization since we will actually sum over all pairs in all directions!
                            revpair[l + (lmax + 1) * (n1 + nmax * n2)] = selpair[isel]
                        else:
                            # diagonal species (s1=s2) have only half of the elements.
                            # this is tricky. we need to duplicate diagonal blocks "repairing" these to be full.
                            # this is necessary to enable alchemical similarity matching, where we need to combine
                            # alpha-alpha and alpha-beta environment fingerprints
                            selpair[l + (lmax + 1) * (n2 + nmax * n1)] = rawsoap[isoap] * (1 if n1 == n2 else isqrttwo)
                            selpair[l + (lmax + 1) * (n1 + nmax * n2)] = rawsoap[isoap] * (1 if n1 == n2 else isqrttwo)
                        # selpair[l + (lmax + 1) * (n2 + nmax * n1)] = selpair[l + (lmax + 1) * (n1 + nmax * n2)]  \
                        #                                                                                                   =  rawsoap[isoap] * (1 if n1 == n2 else isqrttwo)
                        isoap += 1
                        isel += 1
                ipair[(zspecies[s2], zspecies[s1])] = isel

    return alchemySoapdict

#  https://libatoms.github.io/QUIP/descriptors.html
