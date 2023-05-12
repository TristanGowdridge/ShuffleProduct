# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:56:04 2023

@author: trist
"""

from setuptools import setup


# The text of the README file
with open('README.md') as f:
    rm = f.read()

# This call to setup() does all the work
setup(
    name                            = "shuffleproduct",
    version                         = "0.0.1",
    description                     = "shuffleproduct of Generating Series",
    long_description                = rm,
    long_description_content_type   = "text/markdown",
    url                             = "https://github.com/TristanGowdridge",
    author                          = "Tristan Gowdridge",
    author_email                    = "tristan.gowdridge@gmail.com",
    license                         = "MIT",
    classifiers                     = [
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
    ],
    packages                        = ['shuffleproduct'],
    package_dir                     = {'':'ShuffleProduct'},
    include_package_data            = False,
    install_requires                = [
        "numpy",
        "sympy"
    ],
)