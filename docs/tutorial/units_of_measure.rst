The Units of Measure system
===========================

Resqpy implements the full RESQML units of measure system,
making it easy for you to track the of the units in your reservoir model rigorously.

Resqpy also has some helper functions for coercing invalid units, and for converting values between any compatible units.

Resqpy's unit systems module is :mod:`resqpy.weights_and_measures`, and is typically imported as:

.. code-block:: python

    import resqpy.weights_and_measures as wam

The Energistics Unit of Measure Standard
----------------------------------------

The RESQML standard has a rigorous unit system, which is shared with other Energistics standards PRODML and WITSML.
The standard defines the concepts of Quantities, Dimensions, and Units of Measure (uoms),
and also defines the set of allowed values for each.

All properties in a RESQML model are stored with a corresponding unit of measure.

Resqpy contains a database of this unit system, along with helper functions to coerce invalid units into RESQML-compliant units, and to convert between units.

Quantities
----------

A Quantity represents a set of units with the same dimension and same underlying measurement concept.

To get a set of all possible quantities, use :func:`~resqpy.weights_and_measures.valid_quantities`:

.. code-block:: python

    >>> wam.valid_quantities()
    {'area', 'volume', 'length', ...}

To see details about each quantity, such as the list of supported units of measure, use:

.. code-block:: python

    >>> wam.valid_quantities(return_attributes=True)
    {'area': {
        'dimension': 'L2',
        'baseForConversion': 'm2',
        'members': ['acre', 'b', 'cm2', 'ft2', 'ha', 'in2', 'km2', ...]
        },
     'volume': ...
    }

Units of Measure
----------------

A RESQML unit of measure (or "uom") is a unit symbol, such as "m" or "bbl".

Each uom has an associated dimension, and may be compatible with multiple quantities.

Resqpy can try to coerce an input string into a valid RESQML unit of measure. :func:`~resqpy.weights_and_measures.rq_uom` understands some common aliases:

.. code-block:: python

    >>> wam.rq_uom("metre")
    "m"
    >>> wam.rq_uom("scf")
    "ft3"
    >>> wam.rq_uom("p.u.")
    "%"

To see the valid set of units of measure, use :func:`~resqpy.weights_and_measures.valid_uoms`:

.. code-block:: python

    >>> wam.valid_uoms()
    {'%', '%[area]', '%[mass]', '%[molar]', '%[vol]', '(bbl/d)/(bbl/d)', ...}

You can filter to a given Quantity of interest:

.. code-block:: python


    >>> wam.valid_uoms(quantity="volume")
    {'1000 bbl',  '1000 ft3', '1000 gal[UK]', '1000 gal[US]', ...}
 
To see details of each unit of measure such as the name and dimension, pass :code:`return_attributes=True` to return a dictionary.
For example, for the "indian foot" unit of measure:

.. code-block:: python

    >>> wam.valid_uoms(return_attributes=True)["ft[Ind]"]
    {'name': 'indian foot',
    'dimension': 'L',
    'isSI': False,
    'category': 'atom',
    'baseUnit': 'm',
    'conversionRef': 'EPSG',
    'isExact': False,
    'A': 0,
    'B': 12,
    'C': 39.370142,
    'D': 0,
    'description': "Indian Foot = 0.99999566 British feet (A.R.Clarke 1865). 
        British yard (= 3 British feet) taken to be J.S.Clark's 1865 value of 0.9144025 metres."}
    

Converting between units
------------------------

Each unit has four associated conversion factors `A`, `B`, `C` and `D`, which define how one can convert to and from a base unit.

A value `x` can be converted into the base unit with the formula:

.. math::

   y = \frac{A + Bx}{C + Dx}

where `y` represents a value in the base unit.

Use :func:`~resqpy.weights_and_measures.convert` to convert values between any compatible units of measure:

.. code-block:: python

    >>> wam.convert(1, unit_from="ft", unit_to="m")
    0.3048
    >>> wam.convert(1, unit_from="ft", unit_to="ft[US]")
    0.999998

This will also work with numpy arrays, pandas dataframes or even distributed dask objects:

.. code-block:: python

    >>> import numpy as np
    >>> x = np.array([1,2,3])
    >>> wam.convert(x, unit_from="km", unit_to="m")
    np.array([1000, 2000, 3000])

You can also convert arrays in-place:

    >>> z = np.array([1,2,3])
    >>> wam.convert(z, unit_from="km", unit_to="m", inplace=True)
    >>> z
    np.array([1000, 2000, 3000])
