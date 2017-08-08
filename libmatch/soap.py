import quippy as qp
import numpy  as np
from utils import get_spkit,get_spkitMax,envIdx2centerIdxMap
from data_model import AlchemyFrame,AlchemySoap
import multiprocessing as mp
from libmatch.utils import chunk_list


def get_alchemy_frame(fpointer, spkit, spkitMax, nocenters, centerweight=1., gaussian_width=0.5,
                      cutoff=3.5, cutoff_transition_width=0.5, nmax=8, lmax=6):
    atoms = qp.Atoms(fpointer=fpointer)
    atm = atoms
    spkit = get_spkit(atm)
    soapParams = {'spkit': spkit, 'spkitMax': spkitMax, 'nocenters': nocenters,
                  'centerweight': centerweight, 'gaussian_width': gaussian_width,
                  'cutoff': cutoff, 'cutoff_transition_width': cutoff_transition_width,
                  'nmax': nmax, 'lmax': lmax}

    rawsoaps = get_soap(atm, **soapParams)

    zList = atm.get_atomic_numbers()

    mm = envIdx2centerIdxMap(atm, spkit, nocenters)

    alchemyFrame = AlchemyFrame(atom=atm, nocenters=nocenters, soapParams=soapParams)
    Nenv, Npowerspectrum = rawsoaps.shape

    for it in xrange(Nenv):
        # soap[it] is (1,Npowerspectrum) so need to transpose it
        #  convert the soap vector of an environment from quippy descriptor to soap vectors
        # with chemical channels.
        alchemySoapdict = Soap2AlchemySoap(rawsoaps[it, :], spkitMax, nmax, lmax)

        alchemySoap = AlchemySoap(qpatoms=atm, soapParams=soapParams, centerIdx=mm[it])

        alchemySoap.from_dict(alchemySoapdict)

        centerZ = zList[mm[it]]
        alchemyFrame[centerZ] = alchemySoap
    return alchemyFrame


def get_alchemy_frame_wrapper(kargs):
    return get_alchemy_frame(**kargs)



def get_Soaps(atoms, nocenters=None, chem_channels=False, centerweight=1.0, gaussian_width=0.5, cutoff=3.5,
              cutoff_transition_width=0.5, nmax=8, lmax=6, spkitMax=None, nprocess=1):
    '''
    Compute the SOAP vectors for each atomic environment in atoms and
    reorder them into chemical channels.

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

    Frames = []
    # get the set of species their maximum number across atoms
    if spkitMax is None:
        spkitMax = get_spkitMax(atoms)

    if nprocess == 1:

        soapParams = {'centerweight': centerweight, 'gaussian_width': gaussian_width,
                      'cutoff': cutoff, 'cutoff_transition_width': cutoff_transition_width,
                      'nmax': nmax, 'lmax': lmax}

        for atom in atoms:

            # to avoid side effect due to pointers
            atm = atom.copy()
            # get the set of species their number across atom
            spkit = get_spkit(atm)
            # get the soap vectors (power spectra) for each atomic environments in atm
            rawsoaps = get_soap(atm, spkit, spkitMax, nocenters, **soapParams)

            zList = atm.get_atomic_numbers()

            mm = envIdx2centerIdxMap(atm, spkit, nocenters)
            # chemical channel separation for each central atom species
            # and each atomic environment
            if chem_channels:
                alchemyFrame = AlchemyFrame(atom=atm, nocenters=nocenters, soapParams=soapParams)
                Nenv, Npowerspectrum = rawsoaps.shape

                for it in xrange(Nenv):
                    # soap[it] is (1,Npowerspectrum) so need to transpose it
                    #  convert the soap vector of an environment from quippy descriptor to soap vectors
                    # with chemical channels.
                    alchemySoapdict = Soap2AlchemySoap(rawsoaps[it, :], spkitMax, nmax, lmax)

                    alchemySoap = AlchemySoap(qpatoms=atm, soapParams=soapParams, centerIdx=mm[it])

                    alchemySoap.from_dict(alchemySoapdict)

                    centerZ = zList[mm[it]]
                    alchemyFrame[centerZ] = alchemySoap

                # gather soaps over the atom
                Frames.append(alchemyFrame)
            # output rawSoaps
            else:
                raise NotImplementedError()
                Frames.append(rawsoaps)
    elif nprocess > 1:
        if chem_channels:
            soapParams = [
                {'fpointer': frame._fpointer.copy(), 'spkit': get_spkit(frame), 'spkitMax': spkitMax,
                 'nocenters': nocenters,
                 'centerweight': centerweight, 'gaussian_width': gaussian_width,
                 'cutoff': cutoff, 'cutoff_transition_width': cutoff_transition_width,
                 'nmax': nmax, 'lmax': lmax} for frame in atoms]

            pool = mp.Pool(nprocess, maxtasksperchild=10)

            Frames = pool.map(get_alchemy_frame_wrapper, soapParams)
            pool.close()
            pool.join()
        else:
            raise NotImplementedError()

    return Frames


def get_soap(atoms, spkit, spkitMax, nocenters=None, centerweight=1., gaussian_width=0.5,
             cutoff=3.5, cutoff_transition_width=0.5, nmax=8, lmax=6):
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

    soapstr = "soap central_reference_all_species=F central_weight=" + str(centerweight )+ \
              "  covariance_sigma0=0.0 atom_sigma=" + str(gaussian_width) + \
              " cutoff=" + str(cutoff) + \
              " cutoff_transition_width=" + str(cutoff_transition_width) + \
              " n_max=" + str(nmax) + " l_max=" + str(lmax) + ' ' \
              + lspecies + centers

    desc = qp.descriptors.Descriptor(soapstr)
    # computes the soap descriptors for the full frame (atom)
    soap = desc.calc(atoms ,grad=False)["descriptor"]

    return soap


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

# TODO impelement get_avgSoap see https://libatoms.github.io/QUIP/descriptors.html
