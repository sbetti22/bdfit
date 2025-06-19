# bdfit
bdbit is a spectral fitting routine for low resolution brown dwarf NIR spectra using the species package 

author: Sarah Betti

## Installation 
To use the module, download or clone the repository above and install it with the following 
```
$ git clone https://github.com/sbetti22/bdfit.git
$ cd bdfit
$ python setup.py install

After installation, install ```[species](https://species.readthedocs.io/en/latest/index.html)``` and ``[PyMultiNest](https://johannesbuchner.github.io/PyMultiNest/install.html#building-the-libraries)``` following their specific installation instructions.
````

## Requirements
```bdfit``` primarily uses the ```species``` package.  Therefore, ```species``` and ```pymultinest``` are required to be installed.  
  - [species](https://species.readthedocs.io/en/latest/index.html)
  - [PyMultiNest](https://johannesbuchner.github.io/PyMultiNest/install.html#building-the-libraries)
  - numpy
  - scipy
  - pandas
  - matplotlib
  - corner
  - [astropy](https://www.astropy.org)


## To Run
The example notebook ```bdfit_tutorial.ipynb``` shows how you can use the package to run your data through the spectral fitting code.  The notebook shows you have to compare to empirical templates and do a full MCMC fitting routine.  
There is also ```bdfit_species_tutorial.ipynb``` which extracts the bdfit.py class attributes into separate jupyter notebook cells, lending a less "black box" feel to the fitting.   

## Credits
This package extensively uses the ```species``` package. Please cite [Stolker et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26A...635A.182S) if you use this package.  

Additionally, if you use CASPAR to determine extinctions or other literature parameters, please cite [Betti et al. (2023b)](https://ui.adsabs.harvard.edu/abs/2023AJ....166..262B/abstract).

Finally, if you use the example .fits file, please cite [Betti et al. 2023a](https://ui.adsabs.harvard.edu/abs/2023PhDT........13B/abstract)
