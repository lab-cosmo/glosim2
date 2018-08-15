# Glosim2


## Instalation

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
