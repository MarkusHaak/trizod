import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

from trizod import META_DATA, __version__
VERSION = __version__

setuptools.setup(
    name=META_DATA['package_name'].lower(),
    version=VERSION,
    author=META_DATA['author'],
    author_email=META_DATA['author_email'],
    description=META_DATA['description'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=META_DATA['github_url'],
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'tqdm',
        'pandarallel',
        'pynmrstar',
        'scipy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU AFFERO GENERAL PUBLIC LICENSE Version 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research"
    ],
    entry_points={
    'console_scripts': [
        'trizod=trizod.trizod:main',
    ],
},
)