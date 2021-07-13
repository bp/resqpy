Coordinate Reference Systems
============================

In this resqpy tutorial, we will take a look at a RESQML coordinate reference system object.

Opening a model
---------------
We can open a model in the usual way, as shown in previous tutorials:

.. code-block:: python

    import resqpy.model as rq
    model = rq.Model('s_bend.epc')

About RESQML coordinate reference systems
-----------------------------------------
The RESQML standard requires all objects that involve a geometry (in 3D space) to have a related coordinate reference system (CRS). There are actually two classes of CRS:

* ``obj_LocalTime3dCrs`` which has time based z values, for seismic data
* ``obj_LocalDepth3dCrs`` which has length based z values, for everything else

(In these tutorials RESQML object classes will sometimes be shown for brevity without the leading ``obj_``. The resqpy code also usually accepts these class names with or without the ``obj_``)

Both these classes of CRS consist only of xml metadata – there is no associated array data, so no group in the hdf5 file. The metadata includes the units of measure (uom) for the x & y values, and an independent uom for the z values. It also indicates whether the z values are increasing upwards (away from the centre of the Earth), or downwards, and how the x & y axes relate to the compass directions.

The local coordinate reference system may also be placed within a parent CRS, with an xyz origin which locates the local point (0.0, 0.0, 0.0) within another frame of reference. A rotation in the projected (plan) view may also be specified. The parent CRS may optionally be identified as another RESQML CRS or by specifying an EPSG code. (For more information on EPSG codes visit https://epsg.org) The parent may also be left unspecified, in which case the implication is that all CRS objects within the RESQML dataset share the same parent frame of reference.

The rest of this tutorial will focus on a depth based CRS (LocalDepth3dCrs).

Identifying a CRS object
------------------------
Usually when reading a CRS, it has been referenced in some other object such as a grid, a surface or a well trajectory. The reference contains the universally unique identifier (uuid) of the CRS and the uuid can be thought of as a primary key for the object. A later tutorial will look more at object references.

Alternatively, if we are not following a reference, we can list the uuids of depth based CRS objects with the ``uuids()`` method of the Model class, which we encountered in an earlier tutorial:

.. code-block:: python

    crs_uuid_list = model.uuids(obj_type = 'LocalDepth3dCrs')

The S-bend example dataset only has one CRS object, so this list should only contain one uuid. If the calling code knows that will be the case, it can instead use the singular method:

.. code-block:: python

    crs_uuid = model.uuid(obj_type = 'LocalDepth3dCrs')

or, of course, we could pick a single item out of the list, for example with:

.. code-block:: python

    crs_uuid = crs_uuid_list[0]

Instantiating a resqpy Crs object
---------------------------------
Many of the RESQML object classes have corresponding resqpy Python classes available, and that includes the CRS classes. Note that there is not always a one-to-one correspondence between RESQML and resqpy classes though. (The next tutorial discusses this in more detail.) The resqpy Crs class caters for both the RESQML CRS classes: ``LocalTime3dCrs`` and ``LocalDepth3dCrs``

Having found the uuid, we can instantiate a resqpy Crs object:

.. code-block:: python

    import resqpy.crs as rqc
    crs = rqc.Crs(model, uuid = crs_uuid)

A similar approach is used to instantiate objects for all the resqpy classes, when reading an existing dataset.

Older releases of resqpy used the xml root instead of the uuid, when instantiating resqpy objects for existing RESQML
objects. This is now deprecated (from v0.3.0). 

Inspecting the resqpy Crs object
--------------------------------
The resqpy API allows calling code to make direct use of attributes within high level objects. Three commonly accessed attributes in a Crs object are:

.. code-block:: python

    crs.xy_units
    crs.z_units
    crs.z_inc_down

Note that these attribute names are not generally identical to the RESQML schema definition field names. In this case, for example, resqpy uses ``xy_units`` where the RESQML xsd uses ``ProjectedUom``

Using resqpy Crs methods
------------------------
Of course the resqpy classes provide methods for working with the objects. An example from the Crs class is a method which checks whether one Crs is equivalent to another. The following should always return ``True`` !:

.. code-block:: python

    crs.is_equivalent(other_crs = crs)

Another Crs method determines the handedness of the xyz axes:

.. code-block:: python

    crs.is_right_handed_xyz()

The S-bend dataset only has one CRS. If it had more, the following Crs methods could be used to convert xyz data from one to another:

.. code-block:: python

    crs.convert_to(another_crs, xyz)  # returns a new tuple for a single xyz point
    crs.convert_array_to(another_crs, xyz_array)  # converts in situ a numpy float array of shape (..., 3)

The two conversion methods above assume that the xyz data is starting in the space of this ``crs`` and being converted to ``another_crs``. There are an equivalent pair of methods for converting from the other crs (ie. the one passed as an argument), so the following two lines would have exactly the same effect as the two above:

.. code-block:: python

    another_crs.convert_from(crs, xyz)
    another_crs.convert_array_from(crs, xyz_array)

Along with some other simple resqpy classes, Crs includes a definition for __eq__() and __ne__(), so that the == and != operators can be used to test for equivalence between two coordinate reference system objects (behind the scenes this is calling the *is_equivalent()* method):

.. code-block:: python

    if crs == another_crs:
        print('no coordinate transformation needed')

The Crs class includes other methods but those mentioned above are the most commonly used ones.

RESQML Units of Measure (uom)
-----------------------------
The RESQML standard includes a comprehensive set of data for handling physical units, which is shared with the sister standards PRODML and WITSML. Some components of this data include:

* a comprehensive list of quantity classes, such as volume flow rate
* the physical dimensionality of each quantity class (in terms of Mass, Length, Time etc.), e.g. L3/T
* a reference unit of measure for each quantity class (called the base unit), e.g. m3/s
* a comprehensive list of units of measure
* unit prefixes, e.g. *nano*
* conversion factors for compatible units of measure to and from the base unit, and for the prefixes

There is also a list of standard *property kinds* of relevance to reservoir modelling, such as *porosity*.

The resqpy library does not yet make full use of the RESQML units data. So, for example, the Crs conversion methods currently only recognize the following length units: m, ft, ft[US]. However, a release coming soon will include support for the full RESQML uom system.
