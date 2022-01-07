Contributing to resqpy
======================

Resqpy is an open source project released under the MIT license. Contributions
of all forms are most welcome!

Resqpy was created by Andy Beer.

All contributors (alphabetical order):

* Andrew Ediriscoriya
* Andy Beer
* Casey Warshauer
* Connor Tann
* Duncan Hunter
* Emma Nesbit
* Jeremy Tillay
* Kadija Hassanali
* Nathan Lane
* Nirjhor Chakraborty

Ways of contributing
--------------------

* Submitting bug reports and feature requests (using the `GitHub issue tracker <https://github.com/bp/resqpy/issues>`_)
* Contributing code (in the form of `Pull requests <https://github.com/bp/resqpy/pulls>`_)
* Documentation or test improvements
* Publicity and support

Checklist for pull requests
---------------------------

1. Changes or additions should have appropriate unit tests (see below)
2. Follow the PEP8 style guide as far as possible (with caveats below).
3. All public functions and classes should have
   `Google-style docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_ 
4. Code should be formatted with ``yapf``
5. All GitHub checks should pass

Development environment setup
-----------------------------

1. Clone the repo

   Create a fork of the repository using the GitHub website. Note: this step can be
   skipped if you already have write access to the main repo. Then, clone your fork
   locally to your computer to your working area:

   .. code-block:: bash

      git clone <url from GitHub>
      cd resqpy

2. Set up a Python environment

   It is recommended that you set up an isolated Python environment, using conda or virtualenv. 

   .. code-block:: bash

      conda create -n resqpy python=3.7
      conda activate resqpy
        
   You should then make an “editable” installation of the package into your
   local environment. This will also install required dependencies, including
   extra packages required for running unit tests and building documentation.

   .. code-block:: bash

      # Get pinned versions of 3rd party libs for repeatable setup
      pip install -r requirements.txt

      # Intall resqpy lib in editable mode (adds local clone to pythonpath)
      pip install --editable .[tests,docs]

   Be sure to execute the above command from the top level of the repository.
   The full stop ``.`` instructs pip to find the Python package in the current
   working directory.
    
3. Make a Pull Request

   Create a new branch from master:

   .. code-block:: bash

      git checkout master
      git pull
      git checkout -b <your-branch-name>

   You can then commit and push your changes as usual. Open a Pull Request on
   GitHub to submit your code to be merged into master.

Code Style
----------

We use the yapf auto-formatter with the style configured in the repository. 
Most IDEs allow you to configure a formatter to run automatically when you save
a file. Alternatively, you can run the following command before commiting any
changes:

.. code-block:: bash

   # Reformat all python files in the repository
   yapf -ir .

Please try to write code according to the
`PEP8 Python style guide <https://www.python.org/dev/peps/pep-0008/>`_, which
defines conventions such as variable naming and capitalisation. A consistent
style makes it much easier for other developers to read and understand your
code.

Note the existing code base differs from PEP8 in using 3 spaces for indentation
rather than the usual 4.

See `Static analysis`_ for how to check your code for conformance to PEP8 style.

Tests
-----

Why write tests?
^^^^^^^^^^^^^^^^

Automated tests are used to check that code does what it is supposed to do. This
is absolutely key to maintaining quality: for example, automated tests enable
maintainers to check whether anything breaks when new versions of 3rd party
libraries are released.

As a rule of thumb: if you want your code to still work in 6 months' time,
ensure it has some unit tests!

Writing tests
^^^^^^^^^^^^^

pytest is a framework for running automated tests in Python. It is a high-level
framework, so very little code is required to write a test.

Tests are written in the form of functions with the prefix `test_`. Look in the
tests directory for examples of existing tests. A typical pattern is
“Arrange-Act-Assert”:

.. code:: python

    def test_a_thing():
        """ Test to check that MyClass behaves as expected """

        # Arrange
        my_obj = resqml.MyClass()

        # Act
        result = my_obj.do_calculation()

        # Assert
        expected = [1,2,3]
        assert result == expected

Running tests
^^^^^^^^^^^^^

The easiest way to run the tests is simply to open a Pull Request on GitHub.
This automatically triggers the unit tests, run in several different Python
environments. Note that if your MR references an outside fork of the repo, then
a maintainer may need to manually approve the CI suite to run.

Alternatively, you can run the tests against your local clone of the code base
from the command line:

.. code:: bash

    pytest

There are several command line options that can be appended, for example:

.. code:: bash

    pytest -k foobar  # selects just tests with "foobar" in the name
    pytest -rA        # prints summary of all executed tests at end

Static analysis
^^^^^^^^^^^^^^^

We use `flake8 <https://flake8.pycqa.org/en/latest/user/invocation.html>`_ to
scan for obvious code errors. This is automatically run part as part of the CI
tests, and can also be run locally with:

.. code:: bash

    flake8 .

The configuration of which
`error codes <https://gist.github.com/sharkykh/c76c80feadc8f33b129d846999210ba3>`_
are checked by default is configured in the repo in
`setup.cfg <https://github.com/bp/resqpy/blob/master/setup.cfg>`_.

By default in resqpy:

* ``F-`` Logical errors (i.e. bugs) are enabled
* ``E-`` Style checks (i.e. PEP8 compliance) are disabled

You can test for PEP8 compliance by running flake8 with further error codes:

.. code:: bash

    flake8 . –select=F,E2,E3,E4,E7

Documentation
-------------

The docs are built automatically when code is merged into master, and are hosted
at `readthedocs <https://resqpy.readthedocs.io/>`_.

There a few different versions of the documentation available, tied to different
versions of the code:

+------------------------------------------+------------------------------+--------+
| URL                                      | Version                      | Hidden |
+==========================================+==============================+========+
| https://resqpy.readthedocs.io/en/latest/ | The `master` branch, default | No     |
+------------------------------------------+------------------------------+--------+
| https://resqpy.readthedocs.io/en/stable/ | The most recent git tag      | No     |
+------------------------------------------+------------------------------+--------+
| https://resqpy.readthedocs.io/en/docs/   | The `docs` branch            | Yes    |
+------------------------------------------+------------------------------+--------+

These automatically re-build when the relevant branch is updated, or when a new
tag is pushed.

The `docs` version is intended for previewing changes to documentation. Just
create a new feature branch called `docs` and push changes there; you can then
use the link above to check it renders correctly. One can delete the `docs` git
branch as usual when closing a PR, and re-create it when needed.

You may find it helpful to run a linter to check that the syntax of your
ReStructured text is correct: the python package `restructuredtext-lint` is
pretty good for this purpose. Similarly, many IDEs or plugins have a "rewrap"
function that inserts line endings for uniform line lengths, which can make text
more readable and visually pleasing.

You can also build the docs locally, providing you have installed all required
dependencies as described above:

.. code:: bash

   sphinx-build docs docs/html

The autoclasstoc extension is used to group some of the most commonly-used methods
together at the top of the class summary tables. To make a method appear in this list,
add `:meta common:` to the bottom of the method docstring.

Making a release
----------------

To make a release at a given commit, simply make a git tag:

.. code:: bash

   # Make a tag
   git tag -a v0.0.1 -m "Incremental release with some bugfixes"

   # Push tag to github
   git push origin v0.0.1

The tag must have the prefix ``v`` and have the form ``MAJOR.MINOR.PATCH``.

Following [semantic versioning](https://semver.org/), increment the:

* ``MAJOR`` version when you make incompatible API changes,
* ``MINOR`` version when you add functionality in a backwards compatible manner, and
* ``PATCH`` version when you make backwards compatible bug fixes.

Interpreting version numbers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The version number is made available to users as an attribute of the module:

.. code:: python

   >>> import resqpy
   >>> print(resqpy.__version__)
   '1.6.1'

When working with a development version of the code that does not correspond to
a tagged release, the version number will look a little different, for example
``1.6.2.dev301+gddfbf6c``.

This can be interpreted as:

* ``1.6.2`` : is the *next* expected release. The previous release would be ``1.6.1``.
* ``dev301`` : 301 commits added since the previous release.
* ``+gddfbf6c`` : a ``+g`` prefix followed by current commit ID: ``ddfbf6c``.

How the version is retreived
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The git history defines the version, and consequently the version number cannot
be written in a file that is itself under source control.

The package `setuptools_scm <https://github.com/pypa/setuptools_scm>`_ is used to
extract the version number from the git history:

* In a prod setup, the version is hard-coded in a file ``resqpy/version.py``.
* In a development setup, the local git history is analysed.

Get in touch
------------

For bug reports and feature requests, please use the GitHub issue page.

For other queries about resqpy please feel free to get in touch at Nathan.Lane@bp.com

Code of Conduct
---------------

We abide by the Contributor-covenant standard:

https://www.contributor-covenant.org/version/1/4/code-of-conduct/code_of_conduct.md
