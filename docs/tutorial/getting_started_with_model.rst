Getting started with the Model class
====================================

This tutorial covers opening an existing RESQML dataset and identifying the high level objects contained within it.

Prerequisites
-------------
You will need to have resqpy installed in your Python environment, along with its dependencies, before proceeding.

You will also need an example RESQML dataset (some are available within the resqpy repository). The RESQML dataset will consist of two files, one with extension .epc and the other .h5. This pair of files should have the same name prior to the extension and be located in the same directory (folder). You can use any dataset for this exercise – the detailed output from each step will vary depending on the data.

Note: Example file names shown here and in other resqpy tutorials are for a Unix environment. If you are working in a Windows environment, the file paths would be in the corresponding format.

Importing the **model** module
------------------------------
In this tutorial, we will be using the :class:`resqpy.model.Model` class which is contained in :mod:`resqpy.model`. This can be imported with:

.. code-block:: python

    import resqpy.model as rq

The rest of this tutorial assumes the import statement shown above. However, you can vary it according to your preferred style. Other examples are:

.. code-block:: python

    import resqpy.model
    from resqpy.model import Model

Opening an existing RESQML dataset
----------------------------------
The dataset is accessed via the epc file. It is opened with:

.. code-block:: python

    model = rq.Model('/path/to/my_file.epc')

Tip: the ``Model`` initialiser method has some optional arguments which are needed when creating a new dataset or copying an existing dataset before opening the duplicate.

As a convenient shorthand, models can be opened using the :class:`resqpy.model.ModelContext` context manager:

.. code-block:: python

    with rq.ModelContext("my_model.epc", mode="read/write") as model:
        print(model.uuids())

When a RESQML dataset is opened in this way, file handles are safely closed when the "with" clause exits and optionally changes can be written to disk.

A ``Model`` object is almost always the starting point for code using resqpy. The other resqpy object classes require a Model object which is treated as a 'parent'. The resqpy Model class is not equivalent to any of the RESQML classes, rather it should be thought of as equivalent to a whole epc file.

The ``Model`` class includes many methods. In this tutorial we will focus on some of the more essential ones when reading a model.

Keys to the RESQML high level objects
-------------------------------------
A RESQML dataset is a collection of high level objects, also called parts. There are four primary data items that code is likely to work with when handling these parts:

* A *uuid* (universally unique identifier), which is an object of class uuid.UUID. The uuid module is a standard Python module. A uuid is sometimes referred to as a guid (globally unique identifier). The resqpy code base sticks with the term uuid as preferred by Energistics and the underlying ISO standard which these identifiers adhere to. As the uuids are often presented as a hexadecimal string, the resqpy code generally allows uuids to be passed around either as UUID objects or as strings.
* A *part name*, which is a string representing an internal 'file name' within the epc package. A part name usually consists of a high level object class followed by a uuid (see next point) in hexadecimal form and a .xml extension. Where a resqpy argument is named part or part_name, it refers to such a part name.
* An *xml root node*. The metadata for each part is held within the epc in xml format. The lowest level of resqpy code reads this xml into an internal tree structure using the lxml standard Python module, which is compatible with elementTree. Where a resqpy argument name contains root or root_node, it is referring to the root node in the internal tree representation of the xml for the part. Such a root is an object of type lxml._Element and does not have a meaningful human readable form.
* A *citation title*, which is a human readable string held within the citation block of the xml for the part. This is what a human would consider to be the name of the high level object. However, there is no requirement for the citation titles to be unique within a RESQML dataset, so they should generally not be used as a primary key. Where a resqpy argument is named citation_title, or simply title, it is referring to this item of data.

Within a ``Model`` object, there is a one-to-one correspondence between a part name and a uuid, and between a part name and a root node. There are methods for moving from one of these to another and also for finding the (possibly non-unique) citation title.

The ``Model`` class contains four similar methods each of which returns a list of items, corresponding to the four points above. The methods have the names:

* :meth:`resqpy.model.Model.uuids`
* :meth:`resqpy.model.Model.parts`
* :meth:`resqpy.model.Model.roots`
* :meth:`resqpy.model.Model.titles`

If applied to a ``Model`` object without any arguments, a full list is returned, i.e. with one item per high level object.

Selectively listing high level objects
--------------------------------------
The four methods mentioned above have similar lists of optional arguments, some of which allow for filtering of the list:

* ``obj_type`` (string): only objects of this RESQML high level object class are included in the returned list. The leading 'obj_' may be omitted from the class name. Examples:
    .. code-block:: python

        model.parts(obj_type = 'obj_LocalDepth3dCrs')
        model.titles(obj_type = 'DeviationSurveyRepresentation')

* ``uuid`` (UUID object or string): the list will contain the one high level object which matches this uuid, eg.:
    .. code-block:: python

        model.roots(uuid = '27e11404-231b-11ea-8971-80e650222718')

* ``related_uuid`` (UUID object or string): the list will only contain those high level objects which have a relationship with the object identified by this uuid, e.g.:
    .. code-block:: python

        model.parts(related_uuid = '27e11404-231b-11ea-8971-80e650222718')

* ``extra`` (dictionary of key:value pairs): if a non-empty dictionary is provided, only those high level objects with extra metadata including all the key:value pairs in this dictionary will be in the returned list, eg.:
    .. code-block:: python

        model.roots(obj_type = 'WellboreTrajectoryRepresentation',
                    extra = {'development_phase': 2, 'planned_use': 'injection'})

* ``title`` (string): the list will only contain high level objects whose citation title matches this string, e.g.:
    .. code-block:: python

        model.uuids(title = 'WELL_A')

By default, the ``title`` argument results in a case insensitive string comparison with the objects' citation titles. However, other optional arguments may be used to modify this behaviour:

* ``title_case_sensitive`` (boolean, default ``False``): if set ``True``, the comparison will be case sensitive
* ``title_mode`` (string, default 'is'): one of 'is', 'starts', 'ends', 'contains', 'is not', 'does not start', 'does not end', 'does not contain'

If multiple filtering arguments are supplied, then only those high level objects meeting all the criteria will be included ('and' logic).

Rather than starting from the full list of high level objects present in the model, it is also possible to pass in a starting list to apply other filters to:

* ``parts_list`` (list of strings): if present, a list of 'input' part names to which any other filtering arguments are applied, eg:
    * roots(parts_list = ['obj_IjkGridRepresentation_27e10fc2-231b-11ea-8971-80e650222718.xml', 'obj_IjkGridRepresentation_319154f4-5f3e-11eb-9d8d-80e650222718.xml'], title = 'ROOT')

The return list will not be in any particular order unless a further argument is supplied:

* ``sort_by`` (string): if not None then one of 'newest', 'oldest', 'title', 'uuid', 'type'

Finding a single high level object
----------------------------------
Each of the above four methods has a corresponding method which can be used if it is expected that at most one high level object will meet the criteria:

* :meth:`resqpy.model.Model.uuid`
* :meth:`resqpy.model.Model.part`
* :meth:`resqpy.model.Model.root`
* :meth:`resqpy.model.Model.title`

For example:

* ``model.title(uuid = '27e11404-231b-11ea-8971-80e650222718')``

The filtering arguments for these singleton methods are the same as for the list methods. If no objects match the criteria then None is returned. There is a further argument which controls the behaviour when more than one object matches the criteria:

* multiple_handling (string, default 'exception'): one of 'exception', 'none', 'first', 'oldest', 'newest'

Other methods in the Model class
--------------------------------
Although the Model class contains many other methods, the eight listed above are the crucial ones when reading a RESQML dataset. Most of the other methods are involved with writing or modifying datasets, which are more complicated operations and will be covered by other tutorials.

There are three other methods worth mentioning in passing here, which are involved with accessing the hdf5 file:

* :meth:`resqpy.model.Model.h5_file_name`
* :meth:`resqpy.model.Model.h5_uuid`
* :meth:`resqpy.model.Model.h5_release`

The first of these, ``h5_file_name()``, returns the full path of the hdf5 file for the model. By default, any hdf5 filename(s) stored within the xml in the epc file are ignored and a path for a single hdf5 file is returned, based on the epc filename supplied when initialising the model. This protocol makes it much easier to move RESQML datasets around and rename them but it assumes a simple one-to-one pairing of epc and h5 files. Optional arguments allow for other ways of working.

The ``h5_uuid()`` method returns the uuid for the hdf5 'external part'. Although not a normal RESQML high level object, the hdf5 file(s) associated with the epc are treated as special parts and each gets its own uuid. Calling code does not usually need to be concerned with this if the simple file naming protocol is being used.

The last of the three methods, ``h5_release()``, ensures that the hdf5 file is closed, assuming that it has been accessed by other resqpy operations. This is more important when writing a dataset, to ensure the hdf5 file is released ready for other code to access.

The model.py module also contains a tiny convenience function for creating a new, empty RESQML dataset (overwriting any existing files with the same name):

* :func:`resqpy.model.new_model('new_file.epc')`

Summary
-------
In this tutorial, we have seen how to open an existing RESQML dataset and discover what high level objects it contains.
