A first look at Well Objects
============================

This tutorial introduces the classes relating to wells and goes into more detail for some of the basic ones. Other tutorials will cover the remaining well classes in depth.

The RESQML and resqpy classes for wells
---------------------------------------

The RESQML standard contains several classes of object that relate to wells. Each of these has an equivalent resqpy class, named (in parenthesis) in this list:

* MdDatum (MdDatum) - a simple class holding a datum location for measured depths
* DeviationSurveyRepresentation (DeviationSurvey) - inclination and azimuth at given measured depths
* WellboreTrajectoryRepresentation (Trajectory) - xyz coordinates at given measured depths
* WellboreFrameRepresentation (WellboreFrame) - list of measured depths supporting well log properties
* WellboreMarkerFrameRepresentation (WellboreMarkerFrame) - list of picks (well markers)
* BlockedWellboreRepresentation (BlockedWell) - list of cells visited or perforated by a well

The resqpy WellboreFrame and BlockedWell support related properties, which can be handled with the PropertyCollection and/or Property classes. However, for working with well logs, the resqpy property module includes the following classes for convenience:

* (WellLogCollection) - for managing logs, including a method for exporting in LAS format
* (WellLog) - for simpler access to a single log

RESQML also has organisational classes relating to wells:

* WellboreFeature (WellboreFeature) - a named entity representing a real, planned or conceptual well
* WellboreInterpretation (WellboreInterpretation) - one possible incarnation of a wellbore feature

There are various relationships between these classes. For example, a deviation survey or a trajectory must refer to a measured depth datum, and a blocked well must refer to a trajectory. Any of the representation objects can relate to a wellbore interpretation, which in turn must relate to a wellbore feature. The use of these optional organisational objects is encouraged and some software requires them to be present.

In resqpy, the default behaviour is to use the same well name as the citation title for any of the well objects that are in use for a given well. Note that if there are multiple competing interpretations, then it is best to assign a different title to each of the interpretations (and any related representations).

Most of the well related resqpy classes are contained in the `well.py` module. The feature and interpretation classes are in the `organize.py` module. Code snippets in this tutorial assume the following imports:

.. code-block:: python

    import resqpy.model as rq
    import resqpy.well as rqw
    import resqpy.organize as rqo

The measured depth datum class: MdDatum
---------------------------------------
A measured depth datum is a simple object which locates the datum for measured depths within a coordinate reference system. A direct reference to an MD datum is required in both a deviation survey and a trajectory. (And the other well representation objects refer to a trajectory, so an MD datum is always needed.)

When reading an existing dataset, a resqpy MdDatum object can be instantiated in the usual way by first identifying the required uuid. In this example we follow relationships from an interpretation object:

.. code-block:: python

    model = rq.Model('existing_model.epc')
    pq13b_sidetrack_interpretation_uuid = model.uuid(obj_type = 'WellboreInterpretation',
                                                     title = 'PQ13B_SIDETRACK')
    assert pq13b_sidetrack_interpretation_uuid is not None
    pq13b_sidetrack_survey_uuid = model.uuid(obj_type = 'DeviationSurveyRepresentation',
                                             related_uuid = pq13b_sidetrack_interpretation_uuid)
    assert pq13b_sidetrack_survey_uuid is not None
    pq13_md_datum_uuid = model.uuid(obj_type = 'MdDatum',
                                    related_uuid = pq13b_sidetrack_survey_uuid)
    assert pq13_md_datum_uuid is not None
    pq13_md_datum = rqw.MdDatum(model, uuid = pq13_md_datum_uuid)

The MdDatum class doesn't have any exciting methods. Code accessing such an object will usually simply refer to some of the attributes, such as:

* crs_uuid: the uuid of the coordinate reference system within which the datum is located
* location: a triple float being the xyz location of the datum
* md_reference: a human readable string from a prescribed set, such as 'mean sea level', or 'kelly bushing'

The list of valid MD reference strings is defined in the RESQML schema definition and is available in the resqpy well module as:

.. code-block:: python

    rqw.valid_md_reference_list

Creating a new measured depth datum object
------------------------------------------
Most of the tutorials so far have focussed on reading existing data. As the MdDatum is such a simple object, it is a good place to start looking at how we create new objects using resqpy. In this example, we will add a new MdDatum, located fifteen metres to the east and two metres north of the existing datum which we identified above:

.. code-block:: python

    # prepare whatever data we need to populate the new object
    pq13_location = np.array(pq13_md_datum.location)
    new_location = tuple(pq13_location + (15.0, 2.0, 0.0))

    # instantiate the resqpy object using data
    pq14_md_datum = rqw.MdDatum(model,
                                crs_uuid = pq13_md_datum.crs_uuid,
                                location = new_location,
                                md_reference = pq13_md_datum.md_reference,
                                title = 'PQ14')

    # the md datum class does not involve any arrays, so no need to write anything to hdf5

    # create an xml tree (in memory) and add it to the model's dictionary of parts
    pq14_md_datum.create_xml()

    # update the epc file on disc (more typically done after creating a bunch of new objects)
    model.store_epc()

Note that for a real well that has been drilled, the actual location of the datum should be available from the drilling information, so the example above is rather unrealistic.

Other resqpy objects can be created in a similar way. Note, however:

* most classes are much more complex than MdDatum, so much more data needs to be prepared
* resqpy includes import options for some classes, for reading the data from other formats
* many classes include array data, which require an extra step writing to the hdf5 file
* it is usual to call the model's `store_epc()` method once after a batch of objects have been added

The Trajectory class
--------------------
The WellboreTrajectoryRepresentation class (Trajectory in resqpy) plays a central role in the modelling of wells in a RESQML dataset. Apart from a deviation survey, the other well representation classes all require a reference to a trajectory. It is the class which holds information about the path taken by a wellbore in physical space.

To instantiate a resqpy Trajectory for an existing RESQML WellboreTrajectoryRepresentation use the familiar methods:

.. code-block:: python

    pq13b_traj_uuid = model.uuid(obj_type = 'WellboreTrajectoryRepresentation',
                                 title = 'PQ13B_SIDETRACK')
    pq13b_trajectory = rqw.Trajectory(model, uuid = pq13b_traj_uuid)

As the amount of array data is modest for a trajectory, it is all loaded into memory at the time of instantiation. The main data of interest are the list of xyz points defining the path of the wellbore (within a coordinate reference system). The xyz data is available as a numpy array of shape (N, 3) in the `control_points` attribute, e.g.:

.. code-block:: python

    td_z = pq13b_trajectory.control_points[-1, 2]

The measured depths corresponding to the xyz control points are also available in a numpy vector of shape (N,) e.g.:

.. code-block:: python

    td_md = q13b_trajectory.measured_depths[-1]

There are several other attributes, including:

* crs_uuid
* md_uom: the unit of measure (usually 'm' or 'ft') for the measured depths, which don't belong in any crs as such
* md_datum: an MdDatum object
* knot_count: an integer being the number of 'knots', or points in the arrays (i.e. the value of N above)
* line_kind_index: an integer in the range -1..5 indicating how the control points should be interpreted (see below)

It is common practice for application code to treat the trajectory as a piecewise linear spline between the control points. The `line_kind_index` indicates how the data can be interpreted more rigorously. It may have the following values:

* -1: null value, there is no line!
* 0: vertical: the trajectory follows a vertical path beneath the MdDatum location; control points need not be supplied
* 1: linear spline: a piecewise linear spline with sudden changes in direction at control points
* 2: natural cubic spline: a cubic spline with direction control at the two end points
* 3: cubic spline: a cubic spline with no sudden changes in direction
* 4: z linear cubic spline: another form of cubic spline
* 5: minimum curvature spline: the path which has least severe rate of change of direction

When converting from inclination and azimuth data, as acquired by a deviation survey, the minimum curvature interpretation is invariably applied, so the line kind index often has the value 5, even though applications often interpret the trajectory as if it had value 1. For many applications, the differences will be insignificant.

A resqpy Trajectory object has other attributes – some of the optional ones are:

* tangent_vectors: a numpy array of shape (N, 3) holding tangent vectors for the control points
* deviation_survey: a DeviationSurvey object from which the trajectory has been derived
* wellbore_interpretation: a related WellboreInterpretation object
* wellbore_feature: a WellboreFeature object indirectly related via an interpretation object

The Trajectory class offers some methods for setting up a new trajectory from other data sources. These can be triggered by use of appropriate arguments to the initialisation function. The methods are:

* compute_from_deviation_survey(): derives a trajectory from inclination and azimuth data on a minimum curvature basis
* load_from_dataframe(): takes MD, X, Y & Z values from columns of a pandas dataframe
* load_from_ascii_file(): similar to load_from_dataframe() but with the data in a tabular file
* load_from_cell_list(): sets the control points to the cell centres, for a list of cells in a grid
* load_from_wellspec(): similar to load_from_cell_list() but starting from a Nexus WELLSPEC file
* splined_trajectory(): from an existing trajectory, create a new one with more control points following a cubic spline

There is one commonly used method for finding the xyz location for a given measured depth:

* xyz_for_md(): returns the interpolated xyz point based on a simple piecewise linear spline interpretation

Creating a new trajectory object
--------------------------------

In this example we will add a new Trajectory given the following pandas dataframe (the numbers are made up and might not be realistic!):

.. code-block:: python

    df = pd.DataFrame(((2170.00, 450123.45, 5013427.21, 2100.00),
                       (2227.00, 450108.95, 5013432.77, 2150.00),
                       (2288.00, 450081.02, 5013434.25, 2200.00),
                       (2349.00, 450067.83, 5013433.91, 2250.00),
                       (2399.82, 450064.05, 5013433.44, 2300.00)),
              columns = ('MD',     'X',       'Y',       'Z'))

We need to establish an MdDatum object for the well. Here we will assume that the datum is vertically above the first control point in our dataframe. We will also assume that the coordinate reference system object already exists:

.. code-block:: python

    datum_xyz = df['X'][0], df['Y'][0], df['Z'][0] - df['MD'][0]
    md_datum = rqw.MdDatum(model,
                           crs_uuid = model.crs_uuid,  # handy if all your objects use the same crs
                           location = datum_xyz,
                           md_reference = 'ground level',
                           title = 'spud datum')
    md_datum.create_xml()

Now we have enough to instantiate the resqpy Trajectory:

.. code-block:: python

    trajectory = rqw.Trajectory(model,
                                md_datum = md_datum,
                                data_frame = df,
                                length_uom = 'm',  # this is the md_uom
                                well_name = 'Wildcat 1')

The trajectory now exists in memory as a resqpy object but it has not been added to the model in any persistent way. For temporary objects, this state is sometimes fine to work with. However we usually want to add the new object fully. Before doing that, we can optionally call the following method to create an interpretation object and a feature for the well:

.. code-block:: python

    trajectory.create_feature_and_interpretation()

Now we are ready to fully add the trajectory (and related objects) with:

.. code-block:: python

    trajectory.write_hdf5()
    trajectory.create_xml()

This is followed by writing to the epc with the following, which will include all the new objects:

.. code-block:: python

    model.store_epc()

If the model contained just a Crs object before the sequence shown above, then after execution the model.parts() method will return something like:

.. code-block:: python

    ['obj_LocalDepth3dCrs_672bbf86-e4be-11eb-a560-80e650222718.xml',
     'obj_MdDatum_6733ad4a-e4be-11eb-a560-80e650222718.xml',
     'obj_WellboreFeature_67368eb6-e4be-11eb-a560-80e650222718.xml',
     'obj_WellboreInterpretation_67369082-e4be-11eb-a560-80e650222718.xml',
     'obj_WellboreTrajectoryRepresentation_67352698-e4be-11eb-a560-80e650222718.xml']

The other well classes will be covered in later tutorials.
