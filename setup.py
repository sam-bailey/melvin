#!/usr/bin/env python3

# Set this to True to enable building extensions using Cython.
# Set it to False to build extensions from the C file (that
# was previously created using Cython).
# Set it to 'auto' to build with Cython if available, otherwise
# from the C file.
import sys

from setuptools import find_packages, setup

requirements = ["numpy", "scipy", "matplotlib", "jax", "jaxlib", "tqdm", "pandas"]

requirements_dev = ["jupyter", "pre-commit"]

setup(
    name="melvin",
    version="0.1.0",
    description="Maximum likelihood estimation and statistics, built on top of Jax",
    author="Sam Bailey",
    author_email="samcbailey90@gmail.com",
    # url="http://github.com/sam-bailey/tyme",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    long_description=open("README.md").read(),
    install_requires=requirements,
    extras_require={"dev": requirements_dev},
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="statistics maximum likelihood estimation",
)
