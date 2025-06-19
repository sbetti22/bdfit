from setuptools import setup, find_packages

# setup(name="bdfit", 
#         version = "1.0",
#         author = "Sarah Betti",
#         author_email = "sbetti@stsci.edu",
#         url = "https://github.com/sbetti22/bdfit",
#         packages = find_packages(),
#         )

setup(
    name='bdfit',
    version='1.0',
    description='spectral fitting routine for low resolution brown dwarf NIR spectra using the species package',
    url='https://github.com/sbetti22/bdfit',
    author='Sarah Betti',
    author_email='sbetti@stsci.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research ',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
    keywords='NIR spectral fitting',
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'astropy', 'matplotlib', 
    ],
    include_package_data=True,
    zip_safe=False
)