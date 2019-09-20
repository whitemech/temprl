#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The setup script."""
import os

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

here = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(here, 'temprl', '__version__.py'), 'r') as f:
    exec(f.read(), about)

requirements = [
    "gym",
    "numpy",
    "flloat==1.0.0a0",
    "pythomata==1.0.0a0"
]

setup(
    name=about["__title__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description=about["__description__"],
    install_requires=requirements,
    license=about["__version__"],
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='temprl, reinforcement-learning, temporal-logic',
    packages=find_packages(include=['temprl*']),
    test_suite='tests',
    tests_require=["tox"],
    url=about["__url__"],
    version=about["__version__"],
    zip_safe=False,
)
