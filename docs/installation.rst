Installation
============

resqpy is written for Python 3.

It is recommended to use a conda environment for each new project.

.. code-block:: bash

    $ conda create -n my_env python=3
    $ conda activate my_env

Install using pip:

.. code-block:: bash

    $ pip install resqpy

To install a development version on your local machine, use:

.. code-block:: bash

    $ pip install -e /path/to/working/copy

To run unit tests (requires pytest):

.. code-block:: bash

    $ python -m pytest tests/

To build the documentation locally (requires sphinx):

.. code-block:: bash

    $ sphinx-build docs docs/html
