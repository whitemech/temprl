# TempRL 

[![](https://img.shields.io/pypi/v/temprl.svg)](https://pypi.python.org/pypi/temprl)
[![](https://img.shields.io/travis/sapienza-rl/temprl.svg)](https://travis-ci.org/sapienza-rl/temprl)

Framework for Reinforcement Learning with Temporal Goals defined by LTLf/LDLf formulas.


## Install

Install from `master` branch:

- with `pip`:


        pip3 install git+https://github.com/sapienza-rl/temprl.git


- or, clone the repository and install:


        git clone htts://github.com/sapienza-rl/temprl.git
        cd temprl
        pip install .


## Tests

To run the tests:

    tox

To run only the code style checks:

    tox -e flake8

## Docs

To build the docs:


    mkdocs build
    

To view documentation in a browser


    mkdocs serve


and then go to [http://localhost:8000](http://localhost:8000)


## License

Copyright 2018-2019 Marco Favorito

