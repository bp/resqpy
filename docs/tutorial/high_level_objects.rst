High Level Objects
==================

This tutorial discusses some concepts that are important when working with high level objects in resqpy.

RESQML and resqpy classes
-------------------------
The RESQML standard defines many classes of high level objects and specifies precisely how they are to be represented in persistent storage (files). However, application code making use of resqpy will not usually interact directly with the RESQML objects but rather with the closely related resqpy classes of object. Whilst there is a degree of correspondence between RESQML high level classes and resqpy classes, there are some differences which should be borne in mind:

* Class names are usually different
* Some resqpy classes cater for more than one RESQML class
* There are a few circumstances where a RESQML object can be represented by more than one resqpy class
* RESQML is purely concerned with what data is stored for a class, whilst a resqpy class also contains methods to provide different ways of viewing or processing the data
* Whereas a RESQML class is defined in a hierarchical way, and makes use of inheritance (xsd extension base) and abstract classes, the comparable resqpy class is flattened with data elements held as simple attributes
* Some resqpy classes use class inheritance to allow common functionality to be implemented in a base class – this is a different hierarchy to that used in the RESQML schema definition
* Not all RESQML classes are yet catered for (except in the lowest level generic layer of code)
* Some RESQML objects have optional attributes or multiple possible representations of an attribute – some of the options might not yet be implemented in resqpy

Apart from the last two of these points, the differences are due to the different aims of RESQML and resqpy: RESQML aims to provide a comprehensive and unambiguous standard for efficient storage of reservoir models, whereas resqpy aims to provide high level functionality to facilitate processing of the models.

The table at the end of this page shows which resqpy class implements each RESQML class.

Reading and writing objects
---------------------------
From the discussion above, it is evident that the same information can exist in two different representations: in a file in RESQML format, or in memory as resqpy objects. When reading a dataset, the transformation is from RESQML to resqpy. When writing, the transformation is from resqpy to RESQML. However, for efficiency of processing, things are more complicated than that and the representation of a conceptual object can exist in one of a number of states.

Firstly, the resqpy code differentiates between RESQML classes depending on how much array data they involve:

* Classes with no array data, for example measured depth datum (*obj_MdDatum* in RESQML, *MdDatum* in resqpy)
* Classes with modest amounts of array data, eg. wellbore trajectory (*obj_WellboreTrajectoryRepresentation* in RESQML, *Trajectory* in resqpy)
* Classes with large amounts of array data, eg. ijk cellullar grid (*obj_IjkGridRepresentation* in RESQML, *Grid* in resqpy)

The rest of this tutorial will refer to these volumes of array data as none, small or large respectively. Note that the behaviour of the resqpy code is based on the typical amounts of array data for a given class, not the actual size of the arrays for a specific object.

When **reading**, the representation of an object passes through these states:

1. Only in files: metadata in xml compressed into the epc file; any array data in the hdf5 file
2. Metadata loaded into equivalent data structure in memory; any array data still only in the hdf5 file
3. In memory resqpy object instantiated; metadata in object attributes; if small array(s), array data also in memory as attributes
4. For classes with large arrays, Individual arrays are cached as attributes on demand

Step 2 in this sequence occurs with the instantiation of a Model object for an existing epc. The metadata for each part is loaded into a Python lxml tree (which is compatible with elementTree). Application code does not usually interact directly with this representation, though the root node of the lxml tree for an object is sometimes used as an argument to resqpy function calls. Here is an example of code that moves all objects in the s_bend dataset into state 2:

.. code-block:: python

    import resqpy.model as rq
    model = rq.Model('s_bend.epc')

Step 3 occurs when the application code instantiates a resqpy object for one of the parts in the model. At this point, the lxml metadata is interrogated to set the values of the class-specific attributes. The naming and definition of these attributes is often very similar to the equivalent metadata fields in the RESQML class. If the class has a small volume of array data, then it is also loaded at this point into numpy array attributes. The resqpy class might also have derived attributes which are not stored in the RESQML object but are set for the convenience of application code. The following lines will create a resqpy Grid object in state 3 for one of the IjkGridRepresentation parts in the s_bend model:

.. code-block:: python

    import resqpy.grid as grr
    faulted_grid_uuid = model.uuid(obj_type = 'IjkGridRepresentation', title = 'FAULTED GRID')
    faulted_grid = grr.Grid(model, uuid = faulted_grid_uuid)

Step 4 only pertains to classes with large amounts of array data. To minimize memory and time usage, these arrays are not loaded until application code requests them using specific methods in the class. The names of these methods usually contain terms like *cached* and/or *array_ref*. There is often another method allowing for the uncaching of such arrays, which has the effect of deleting the associated attribute from the resqpy object. The following example loads a numpy boolean array from the hdf5 file (unless it has already been cached), indicating which cells in a resqpy Grid object have geometry defined; the array is stored as an attribute of the object (cached) and also returned by the method:

.. code-block:: python

    faulted_grid.cell_geometry_is_defined_ref()

The Grid class also has a method which ensures that all arrays are cached:

.. code-block:: python

    faulted_grid.cache_all_geometry_arrays()

Note that these steps are triggered by application code calling resqpy methods. Apart from step 4, the calling code needs to keep track of which state the information for a particular object is in – resqpy itself is not generally keeping a handle on high level objects as they are instantiated.

When **writing**, the representation of an object typically passes through these states:

1. Only in memory, as a resqpy object, with metadata and any array data held as attributes
2. Metadata and any array data held as attributes of resqpy object; any array data also written to the hdf5 file
3. The metadata is also stored in an lxml tree, in memory, in a form ready to be written to the epc file
4. When all parts have been through the steps above, the metadata for all parts is written to the epc file from the lxml trees

Step 1 in this sequence is achieved by calling the initialization method of the resqpy class with arguments set to indicate import from a different format. Or an empty resqpy object can be instantiated and all the attributes set by the calling code. Only when the object's attributes are fully populated can the representation proceed with the rest of the steps. The s_bend dataset, unrealistically, uses a single measured depth datum for 4 wells. Here is some example code for creating a new resqpy MdDatum object in state 1, located 5 metres to the east of the existing datum:

.. code-block:: python

    import resqpy.well as rqw
    existing_md_uuid = model.uuid(obj_type = 'obj_MdDatum')  # we happen to know there is only one MdDatum object
    existing_md_datum = rqw.MdDatum(model, uuid = existing_md_uuid)
    x, y, z = existing_md_datum.location
    x += 5.0
    new_md_datum = rqw.MdDatum(model,
                               crs_root = existing_md_datrum.crs_root,
                               location = (x, y, z))

Step 2 is achieved by the application code calling a method, usually named ``write_hdf5()``, for the resqpy object. As the obj_MdDatum class does not involve any array data, this step does not apply to our example.

Step 3 Each resqpy class has a method named ``create_xml()`` which generates the lxml tree representation of the metadata, in memory, and adds the part to the parent resqpy Model object, also creating relationship data structures. Here is the line for the newly created MdDatum object instantiated above:

.. code-block:: python

    new_md_datum.create_xml()

Step 4 is achieved by the application code calling the ``store_epc()`` method of the Model object when all objects have been prepared as far as step 3. So in the example above, when the application code has generated all the required objects, the call is simply:

.. code-block:: python

    model.store_epc()

At this point the data is stored persistently in the epc file (and hdf5 file) and the application can exit, or delete the model and other objects.

Temporary object states
-----------------------
The two situations discussed above – reading and writing – are the most common ways of working with resqpy objects. However, resqpy has been designed to support processing of models and for this a third situation can arise: the need for temporary objects. Such objects are not written to the epc file (nor their arrays to the hdf5 file) but exist only in memory as resqpy objects.

As an example of working with temporary objects, imagine an application that generates many undrilled well trajectories and then tests them against a reservoir model to select the best trajectory. The trajectories could all be saved, using the sequence for writing resqpy objects outlined above. However, perhaps there is only the need to keep the trajectory that has been selected as best. The other trajectories would be temporary.

The simplest way to work with a temporary object is simply to instantiate it. This is equivalent to step 1 of the writing sequence above. Such an object can be used for most processing purposes. Note, however, that it has not been added as a part to the nominal parent Model object, nor does any xml exist for it. Some of the resqpy method and function calls require these other steps to have been taken.

Another approach for working with temporary objects is to create a separate, temporary, Model object and to instantiate the temporary high level objects with the temporary model as the parent. The ``create_xml()`` methods of the high level objects can be called without calling the ``write_hdf5()`` methods. If the temporary model's ``store_epc()`` method is not called, nothing will be written to the persistent file system. This is equivalent to steps 1 and 3 of the writing sequence discussed above.

Managing resqpy objects
-----------------------
Although a resqpy high level object is associated with a Model object (and contains a reference to the Model as an argument), the Model does not maintain a list of resqpy objects which have been instantiated for it. The Model does contain the list of RESQML parts, each of which can be used to instantiate a resqpy object (at least for the classes catered for in resqpy).

The exception is the resqpy Grid class (RESQML obj_IjkGridRepresentation), for which the Model class includes methods for optionally managing a list of resqpy Grid objects. This exception is made because grids can be memory and time intensive to instantiate, and are fundamental to all processing when working with a cellular model.

In general, though, it is up to the application code to manage the lifecycle of the resqpy objects.

RESQML to resqpy class mapping
------------------------------
The table below shows which high level resqpy class is used to represent each RESQML class. The blank rows indicate that a high level resqpy class has not yet been implemented for the RESQML class. (The lowest level resqpy code is generic, so steps 1 & 2 of the reading sequence above will function for all RESQML classes, as will step 4 of the writing sequence.)

+--------------------------------------------------------+------------+-----------------------------------------------------------+
| RESQML class                                           | array data | primary resqpy class                                      |
+========================================================+============+===========================================================+
| obj_Activity                                           |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_ActivityTemplate                                   |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_BlockedWellboreRepresentation                      | small      | :class:`resqpy.well.BlockedWell`                          |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_BoundaryFeature                                    | none       | :class:`resqpy.organize.BoundaryFeature`                  |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_BoundaryFeatureInterpretation                      |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_CategoricalProperty                                | large      | :class:`resqpy.property.PropertyCollection`               |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_CategoricalPropertySeries                          |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_CommentProperty                                    |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_CommentPropertySeries                              |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_ContinuousProperty                                 | large      | :class:`resqpy.property.PropertyCollection`               |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_ContinuousPropertySeries                           |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_DeviationSurveyRepresentation                      | small      | :class:`resqpy.well.DeviationSurvey`                      |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_DiscreteProperty                                   | large      | :class:`resqpy.property.PropertyCollection`               |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_DiscretePropertySeries                             |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_DoubleTableLookup                                  |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_EarthModelInterpretation                           | none       | :class:`resqpy.organize.EarthModelInterpretation`         |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_EpcExternalPartReference                           |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_FaultInterpretation                                | none       | :class:`resqpy.organize.FaultInterpretation`              |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_FluidBoundaryFeature                               | none       | :class:`resqpy.organize.FluidBoundaryFeature`             |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_FrontierFeature                                    | none       | :class:`resqpy.organize.FrontierFeature`                  |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_GenericFeatureInterpretation                       |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_GeneticBoundaryFeature                             | none       | :class:`resqpy.organize.GeneticBoundaryFeature`           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_GeobodyBoundaryInterpretation                      | none       | :class:`resqpy.organize.eobodyBoundaryInterpretation`     |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_GeobodyFeature                                     | none       | :class:`resqpy.organize.GeobodyFeature`                   |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_GeobodyInterpretation                              | none       | :class:`resqpy.organize.GeobodyInterpretation`            |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_GeologicUnitFeature                                | none       | :class:`resqpy.organize.GeologicUnitFeature`              |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_GeologicUnitInterpretation                         | none       | :class:`resqpy.strata.GeologicUnitInterpretation`         |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_GlobalChronostratigraphicColumn                    |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_GpGridRepresentation                               |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_Grid2dRepresentation                               | large      | :class:`resqpy.surface.Mesh`                              |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_Grid2dSetRepresentation                            |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_GridConnectionSetRepresentation                    | large      | :class:`resqpy.fault.GridConnectionSet`                   |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_HorizonInterpretation                              | none       | :class:`resqpy.organize.HorizonInterpretation`            |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_IjkGridRepresentation                              | large      | :class:`resqpy.grid.Grid`                                 |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_LocalDepth3dCrs                                    | none       | :class:`resqpy.crs.Crs`                                   |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_LocalGridSet                                       |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_LocalTime3dCrs                                     | none       | :class:`resqpy.crs.Crs`                                   |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_MdDatum                                            | none       | :class:`resqpy.well.MdDatum`                              |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_NonSealedSurfaceFrameworkRepresentation            |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_OrganizationFeature                                | none       | :class:`resqpy.organize.OrganizationFeature`              |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_PlaneSetRepresentation                             |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_PointSetRepresentation                             | large      | :class:`resqpy.surface.PointSet`                          |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_PointsProperty                                     | large      | :class:`resqpy.property.PropertyCollection`               |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_PolylineRepresentation                             | small      | :class:`resqpy.lines.Polyline`                            |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_PolylineSetRepresentation                          | small      | :class:`resqpy.lines.PolylineSet`                         |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_PropertyKind                                       | none       | :class:`resqpy.property.PropertyKind`                     |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_PropertySet                                        | none       | :class:`resqpy.property.PropertyCollection`               |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_RedefinedGeometryRepresentation                    |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_RepresentationIdentitySet                          |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_RepresentationSetRepresentation                    |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_RockFluidOrganizationInterpretation                |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_RockFluidUnitFeature                               | none       | :class:`resqpy.organize.RockFluidUnitFeature`             |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_RockFluidUnitInterpretation                        |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_SealedSurfaceFrameworkRepresentation               |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_SealedVolumeFrameworkRepresentation                |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_SeismicLatticeFeature                              |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_SeismicLineFeature                                 |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_SeismicLineSetFeature                              |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_StratigraphicColumn                                | none       | :class:`resqpy.strata.StratigraphicColumn`                |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_StratigraphicColumnRankInterpretation              | none       | :class:`resqpy.strata.StratigraphicColumnRank`            |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_StratigraphicOccurrenceInterpretation              |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_StratigraphicUnitFeature                           | none       | :class:`resqpy.strata.StratigraphicUnitFeature`           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_StratigraphicUnitInterpretation                    | none       | :class:`resqpy.strata.StratigraphicUnitInterpretation`    |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_StreamlinesFeature                                 |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_StreamlinesRepresentation                          |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_StringTableLookup                                  | none       | :class:`resqpy.property.StringLookup`                     |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_StructuralOrganizationInterpretation               |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_SubRepresentation                                  |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_TectonicBoundaryFeature                            | none       | :class:`resqpy.organize.TectonicBoundaryFeature`          |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_TimeSeries                                         | none       | :class:`resqpy.time_series.TimeSeries`                    |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_TriangulatedSetRepresentation                      | large      | :class:`resqpy.surface.Surface`                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_TruncatedIjkGridRepresentation                     |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_TruncatedUnstructuredColumnLayerGridRepresentation |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_UnstructuredColumnLayerGridRepresentation          |            |                                                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_UnstructuredGridRepresentation                     | large      | :class:`resqpy.unstructured.UnstructuredGrid`             |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_WellboreFeature                                    | none       | :class:`resqpy.organize.WellboreFeature`                  |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_WellboreFrameRepresentation                        | small      | :class:`resqpy.well.WellboreFrame`                        |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_WellboreInterpretation                             | none       | :class:`resqpy.organize.WellboreInterpretation`           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_WellboreMarkerFrameRepresentation                  | small      | :class:`resqpy.well.WellboreMarkerFrame`                  |
+--------------------------------------------------------+------------+-----------------------------------------------------------+
| obj_WellboreTrajectoryRepresentation                   | small      | :class:`resqpy.well.Trajectory`                           |
+--------------------------------------------------------+------------+-----------------------------------------------------------+