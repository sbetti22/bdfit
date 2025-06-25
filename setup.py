from setuptools import setup, find_packages

setup(name="bdfit", 
        version = "1.0",
        description='spectral fitting routine for low resolution brown dwarf NIR spectra using the species package',
        author = "Sarah Betti",
        author_email = "sbetti@stsci.edu",
        license='MIT',
        url = "https://github.com/sbetti22/bdfit",
        classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research ',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
        ],
        keywords='NIR spectral fitting',
        packages = ['bdfit'],
        install_requires=[
        'numpy', 'scipy', 'astropy', 'matplotlib', 
        ], 
        package_data={'bdfit': ['*']},
        zip_safe=False
        
        )