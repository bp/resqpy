# resqpy: Python API for working with RESQML models

[![License](https://img.shields.io/pypi/l/resqpy)](https://github.com/bp/resqpy/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/resqpy/badge/?version=latest)](https://resqpy.readthedocs.io/en/latest/?badge=latest)
[![Python CI](https://github.com/bp/resqpy/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/bp/resqpy/actions/workflows/ci-tests.yml)
![Python version](https://img.shields.io/pypi/pyversions/resqpy)
[![PyPI](https://img.shields.io/pypi/v/resqpy)](https://badge.fury.io/py/resqpy)
![Status](https://img.shields.io/pypi/status/resqpy)
[![codecov](https://codecov.io/gh/bp/resqpy/branch/master/graph/badge.svg)](https://codecov.io/gh/bp/resqpy)

## Introduction

**resqpy** is a pure Python package which provides a programming interface (API) for
reading, writing, and modifying reservoir models in the RESQML format. It gives
you the ability to work with reservoir models programmatically, without having
to know the details of the RESQML standard.

The package is written and maintained by bp, and is made available under the MIT
license as a contribution to the open-source community.

**resqpy** was created by Andy Beer. For enquires about resqpy, please contact
Nathan Lane (Nathan.Lane@bp.com)

### Documentation

See the complete package documentation on
[readthedocs](https://resqpy.readthedocs.io/).

### About RESQML

RESQML™ is an industry initiative to provide open, non-proprietary data exchange
standards for reservoir characterization, earth and reservoir models. It is
governed by the [Energistics
consortium](https://www.energistics.org/portfolio/resqml-data-standards/).

Resqpy provides specialized classes for a subset of the RESQML high level object
classes, as described in the docs. Furthermore, not all variations of these
object types are supported; for example, radial IJK grids are not yet catered
for, although the RESQML standard does allow for such grids.

It is envisaged that the code base will be expanded to include other classes of
object and more fully cover the options permitted by the RESQML standard.

Modification functionality at the moment focuses on changes to grid geometry.

## Installation

Resqpy can be installed with pip:

```bash
pip install resqpy
```

Alternatively, to install your working copy of the code in "editable" mode:

```bash
pip install -e /path/to/repo/
```

## Contributing

Contributions of all forms are welcome and encouraged! Please feel free to open
issues on the GitHub issue tracker, or submit Pull Requests. Instructions with how to set up your own development environment can be found at [Development environment setup](https://github.com/bp/resqpy/blob/master/docs/CONTRIBUTING.rst#development-environment-setup). Please read the
[Contributing Guide](docs/CONTRIBUTING.rst) before submitting patches.
