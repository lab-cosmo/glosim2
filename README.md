# Glosim2

This package is not further developed. We might be able to incorporate small bugfixes - particularly if you propose a fix and submit a pull request - but the original developers do not have time to properly maintain it. You may want to check https://github.com/cosmo-epfl/librascal where most of the development efforts are currently focused

## Installation

### Python Packages and Code

#### Python

```
pip install numpy ase numba tqdm psutil
```

Optional:
```
pip install seaborn mpi4py scikit-learn
```


#### Code

QUIP code (http://libatoms.github.io/QUIP/index.html)

+ Download the GAP code from http://www.libatoms.org/gap/gap_download.html

+ Follow the installation instructions of QUIP http://libatoms.github.io/QUIP/install.html. During the configuration enable GAP support and copy the GAP folder form the previous point in the QUIP/src/ folder before compilation.



## Organisation of the code

The main functionality of the package are exposed in GlobalSimilarity.py and GlobalSim_cluster.py is an attempt to make a robust parallelization using mpi4py library.

## Examples

An example on how to use the library in several scenarios can be found as a jupyter notebook in the examples folder.
