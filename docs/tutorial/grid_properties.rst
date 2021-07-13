Grid Properties
===============

This tutorial is about working with RESQML grid property arrays. However, much of what is presented here is also applicable to property data attached to other classes of objects, for example well logs. Some of the notes refer to the Nexus simulator as resqpy includes import and export functions for working with Nexus. Similar considerations would apply to other simulators, though the import and export functions would need to be developed separately.

You should edit the file paths in the examples to point to your own resqml dataset.

Quick start for getting at property arrays in a RESQML dataset
--------------------------------------------------------------
There are different routes to tracking down a property array, in these examples we go via the host grid object.

The first step is always to open the dataset as a resqpy Model object:

.. code-block:: python

    epc_path = '/sd/sdaq_2.epc'  # an existing RESQML dataset
    import resqpy.model as rq
    model = rq.Model(epc_path)

If the dataset is for a single realisation, or has a shared grid used by all realisations, we can get a Grid object for the ROOT grid with:

.. code-block:: python

    grid = model.grid()

See the previous tutorial for information on finding the right grid in a multi-grid dataset.

A Grid object comes with a collection of properties as an attribute. It is a good idea to set a short variable name to this collection, as we are going to work with it intensively. Here we use pc but it has nothing to do with capillary pressure, so if you find that confusing, pick another name!

.. code-block:: python

    pc = grid.property_collection

The property collection is a resqpy object of class property.PropertyCollection which has many methods available. Each individual property array is referred to as a part (and also exists in the RESQML dataset as an individual object in its own right, though in these examples we always access the properties via the collection for a given grid).

The simplest PropertyCollection method simply returns the number of parts (arrays) in the collection:

.. code-block:: python

    pc.number_of_parts()

If we know the title of the property array we are interested in, and if the title is unique, we can get at it as a numpy array with, eg.:

.. code-block:: python

    pore_volume_array = pc.single_array_ref(citation_title = 'PVR')

Where a RESQML dataset has been constructed from Nexus data using the resqpy import functions, the citation title for grid properties will have been set to the Nexus keyword used in the vdb dataset or ascii files.

Note that arrays are not actually loaded into memory until they are requested with methods such as the one shown above.

Grid property arrays with one value per cell have the shape (nk, nj, ni) – **note the order of the indexing: [k, j, i]**. Also note that the indices for these numpy arrays begin at zero (the Python way), rather than 1 which is used by simulators such as Nexus (and is the default in Fortran). So, for an index of a cell in simulator format: (sim_i, sim_j, sim_k), the property value for that cell is found with:

.. code-block:: python

    pore_volume_array[sim_k - 1, sim_j - 1, sim_i - 1]

This arrangement means that the actual array of data is laid out on disc or in memory in exactly the same way in the two systems.

If instead of the actual array, we want to get the part name (which is often needed as an argument to other methods in the property collection class), we can use:

.. code-block:: python

    pore_volume_part = pc.singleton(citation_title = 'PVR')

Each property array is a high level object in its own right, and the part name is the same as that used by the Model class when managing the high level objects.

Using RESQML property kinds and facets
--------------------------------------
In the examples above, we are using the citation title to uniquely identify a property array. That can work if the source of the dataset is known in advance, so that the values and uniqueness of citation titles is ensured. However, to write code that will work with RESQML data that has come from other sources, it is better to use the *property kind* to find the array of interest. The resqpy Nexus vdb import code also sets the property kind, so the following should work regardless of the source of the RESQML data:

.. code-block:: python

    pore_volume_array = pc.single_array_ref(property_kind = 'pore volume')

There is a fixed list of standard property kinds, defined in the RESQML standard, though extra 'local property kinds' can be defined when needed. The standard property kinds that are most often used can be found as:

.. code-block:: python

    rqp.supported_property_kind_list

which evaluates to:

.. code-block:: python

    ['code', 'index', 'depth', 'rock volume', 'pore volume', 'volume',
     'thickness', 'length', 'cell length', 'net to gross ratio', 'porosity',
     'permeability thickness', 'permeability length', 'permeability rock',
     'rock permeability', 'fluid volume', 'transmissibility', 'pressure',
     'saturation', 'solution gas-oil ratio', 'vapor oil-gas ratio',
     'property multiplier', 'thermodynamic temperature',
     'continuous', 'discrete', 'categorical']

That list is a small subset of the standard resqml property kinds – the subset which resqpy has some 'understanding' of. For the full list, see the definition of ResqmlPropertyKind in the RESQML schema definition file property.xsd, or find the same list in json format in the resqpy repository file: resqml/olio/data/properties.json. Using property kinds that are not in the supported_property_kind_list should usually be okay.

The following method returns a list of the distinct property kinds found within the collection:

.. code-block:: python

    property_kinds_present = pc.property_kind_list()

Some of the property kinds may have an associated directional indication, which is stored as a property *facet*, with a facet type of 'direction'. So to get at PERMZ using the property kind, we would need:

.. code-block:: python

    vertical_perm_array = pc.single_array_ref(property_kind = 'permeability rock', facet_type = 'direction', facet = 'K')

or facet = 'I'  or facet = 'J'  for 'horizontal' permeability arrays.

Here are the facet types and facet values currently used by resqpy:

* facet_type = 'direction': facet = 'I', 'J', 'K', 'IJ', or 'IJK', used for 'permeability rock', 'transmissibility', 'property multiplier' for transmissibility
* facet_type = 'netgross': facet = 'net' or 'gross', sometimes used for property kinds 'rock volume' and 'thickness'
* facet_type = 'what': facet = 'oil', 'water' or 'gas', used for saturations

The exact use of facets is not really pinned down in the RESQML standard, so we might choose to work with the citation titles in some situations.

The RESQML standard allows for a property object to have any number of facets. However, the resqpy PropertyCollection class currently handles at most one facet per property.

Identifying basic static properties
-----------------------------------
The PropertyCollection class includes a convenience method for identifying 5 basic static properties: net to gross ratio, porosity, and 3 permeabilities (I, J & K). The following method returns a tuple of 5 part names:

.. code-block:: python

    ntg_part, porosity_part, perm_i_part, perm_j_part, perm_k_part = pc.basic_static_property_parts(share_perm_parts = True)

Given a part name for a property, the numpy array can be accessed with:

.. code-block:: python

    ntg_array = pc.cached_part_array_ref(ntg_part)

The share_perm_parts argument allows the same part to be returned for more than one of the three permeability keys. So, for example, if only one permeability rock array is found and it doesn't have any direction facet information, then it will be returned for all three permeability dictionary entries. The array caching mechanism means that the actual array data will not be duplicated, even if 3 array variables are set up.

There is a similar method which returns the UUIDs of the same 5 basic static properties:

.. code-block:: python

    ntg_uuid, porosity_uuid, perm_i_uuid, perm_j_uuid, perm_k_uuid = pc.basic_static_property_uuids(share_perm_parts = True)

Continuous, discrete and categorical properties
-----------------------------------------------
The RESQML standard distinguishes between three classes of property, depending on the type of an individual datum:

* **continuous**: for real (floating point) data
* **categorical**: for integer data where the set of possible values is limited and a value can be used as an index into a lookup table (e.g. facies)
* **discrete**: for other integer or boolean data

Both categorical and discrete make use of a numpy array of integers. In terms of the data structures, the difference is that a categorical property also has a reference to a string lookup table. The following example shows how to get at the lookup table. (Note that at present the resqpy code for converting from Nexus vdb to RESQML does not create any lookup tables, so the datasets only contain continuous and discrete properties, not categorical.)

.. code-block:: python

    facies_part = pc.singleton(citation_title = 'FACIES')
    lookup_table = pc.string_lookup_for_part(facies_part)

The lookup table is an object of resqpy class StringLookup (equivalent to RESQML class StringTableLookup). It maps integer values to strings. Given an integer, the string can be looked up with:

.. code-block:: python

    facies_name = lookup_table.get_string(2)

To go in the opposite direction, i.e. discover the integer value for a given string, use:

.. code-block:: python

    facies_int_for_mouthbar = lookup_table.get_index_for_string('MOUTHBAR')

If you are not sure what class a property is, the property collection has some methods to help:

.. code-block:: python

    pc.continuous_for_part(facies_part)  # returns True if the property is continuous, False for categorical or discrete
    pc.part_is_categorical(facies_part)  # returns True it the property is categorical, False otherwise

Note that the resqpy code tends to treat categorical as a special case of discrete, so some methods have a boolean argument to distinguish between continuous and discrete – in which case the argument should be set to the value for discrete data when handling a categorical property.

Units of measure
----------------
The RESQML standard includes a comprehensive handling of units of measure – uom. Any continuous property must have an associated uom which can be accessed, for example, with:

.. code-block:: python

    pv_part = pc.singleton(property_kind = 'pore volume')
    pv_uom = pc.uom_for_part(pv_part)  # for volumes, the uom will be 'm3' or 'ft3' for our datasets

The RESQML standard includes a full (very long) list of allowable units. Here are a few of the common ones we might be using:

* length: 'm', 'ft'
* area: 'm2', 'ft2'
* volume: 'm3', 'ft3', 'bbl'
* volume ratios: 'm3/m3', 'ft3/ft3', 'ft3/bbl', '1000 ft3/bbl' (the first two are used for net to gross ratio, porosity, saturation)
* volume rate: 'm3/d', 'bbl/d', 'ft3/d', '1000 ft3/d'
* permeability: 'mD'
* pressure: 'kPa', 'bar', 'psi'
* unitless: 'Euc' (but preferable to use ratio units where they exist, for dimensionless ratios such as the volume ratios above)

The RESQML units definition is shared with the other Energistic standards: PRODML & WITSML. It is very thorough and well thought out. Here we only touch on it in the most minimal way. The full list of units of measure is to be found in the RESQML common schema definition file QuantityClass.xsd, and is also available in json format in the resqpy repository file: resqml/olio/data/properties.json. The resqpy *weights_and_measures* module also has functions for retrieving such information.

Discrete and categorical properties do not have a unit of measure.

Resqpy includes support for the full Energistics uom system, including a general unit conversion capability: See *The Units of Measure system* tutorial.

Null values and masked arrays
-----------------------------
RESQML continuous properties use the special floating point value Not-a-Number, or NaN (np.NaN), as the null value. This is convenient as the numpy array operations can generally handle the null values without much extra coding effort. For discrete (including categorical) properties, a null value can be explicitly identified in the metadata. It is common to use -1 as the null value unless this is a valid value for the property.

To discover the null value for a discrete (or categorical) part, use something like:

.. code-block:: python

    irock_part = pc.singleton(title = 'IROCK')
    irock_null_value = pc.null_value_for_part(irock_part)

The null_value_for_part() method will return an integer if a null value has been defined (or None if a null value has not been defined in the metadata) for a discrete property, or np.NaN if the part is a continuous property.

The property collection methods which return an array of property data, such as single_array_ref(), return a simple numpy array by default. However, there is the option to return a numpy masked array instead. Masked arrays contain not only the data but also a boolean mask indicating which elements to exclude. When a masked array is requested, the resqpy code sets the mask to be the inactive cell mask. There is also an option to mask out elements containing the null value. Numpy operations working with a masked array as an operand will also return a masked array. Furthermore, numpy operations such as sum, mean etc. will ignore masked out values.

To get a masked version of a property array, use one of the following forms:

.. code-block:: python

    depth_masked_array = pc.single_array_ref(property_kind = 'depth', masked = True)  # exludes inactive cells
    mean_active_depth = np.mean(depth_masked_array)

    # following also excludes null value cells
    facies_masked_array = pc.single_array_ref(title = 'FACIES', masked = True, exclude_null = True)

The cached_part_array_ref() method also has the same optional arguments.

Universally unique identifiers
------------------------------
From the earlier discussion, it is clear that sometimes we might struggle to identify a particular property object. To help with this problem, RESQML makes use of Universally Unique Identifiers (also known as GUIDs, globally unique identifiers). They are used by RESQML as a key to uniquely identify high level objects. Every part in a RESQML dataset has a UUID assigned to it, including the individual property objects.

Behind the scenes, a UUID is a 128 bit integer, but it is usually presented in ascii in a specific hexadecimal form (see example below). All of this is the subject of an ISO standard, as these UUIDs are used all over place, not just in the oil industry.

As every part of a RESQML model has a UUID, and as the name suggests it is unique, this can be thought of as a primary key for the objects or parts in the dataset. Many of the resqpy methods work with UUIDs as a way of identifying a part. Here is an example of the single_array_ref() method we saw earlier, but now using the UUID for a particular property array:

.. code-block:: python

    ntg_array = pc.single_array_ref(uuid = 'fa52e6a2-dbbb-11ea-b158-248a07af10b2')

These UUIDs are not very human-friendly, so the examples don't tend to focus on them. However, for scripts running as part of automated jobs, their use is to be encouraged. The basic static property parts method we saw earlier is also available in a version that returns UUIDs instead of part names:

.. code-block:: python

    ntg_uuid, porosity_uuid, perm_i_uuid, perm_j_uuid, perm_k_uuid = pc.basic_static_property_uuids(share_perm_parts = True)

Working with recurrent properties
---------------------------------
The examples above will only uniquely identify a property array if it is a static property and the grid only has property data for a single realisation. To handle recurrent properties (i.e. properties that vary over time) or multiple realisations, more is needed...

Within the property collection, each instance of a recurrent property has a time index associated with it, along with a reference to a time series object which can be used to look up an actual date for a given time index value. If the property collection has come from the import of a single Nexus case, all the time indices will relate to the same time series. The model may additionally contain other time series objects. In particular, when importing from Nexus output, the resqpy code attempts to create 2 time series: one with all the Nexus timesteps and the other limited to the steps where recurrent properties were output which will usually be the one referred to by the property collection.

To find the UUID of the time series in use in the property collection, use:

.. code-block:: python

    ts_uuid_list = pc.time_series_uuid_list()
    assert len(ts_uuid_list) == 1
    ts_uuid = ts_uuid_list[0]

Given the UUID of the time series, we can instantiate a resqpy TimeSeries object:

.. code-block:: python

    import resqml.time_series as rqts
    time_series = rqts.TimeSeries(model, time_series_root = model.root(uuid = ts_uuid))

The TimeSeries class includes various methods, for example:

.. code-block:: python

    ti_count = time_series.number_of_timestamps()
    for time_index in range(ti_count):
    print(time_index, time_series.timestamp(time_index))

The time indices relevant to a time series are in the range zero to number_of_timestamps() - 1. The list of indices at use in a property collection can be found with:

.. code-block:: python

    time_indices_list = pc.time_index_list()

Note that not all the recurrent properties will necessarily exist for all the time indices. Furthermore, the time indices are not generally the same as Nexus timestep numbers, because they usually refer to the reduced time series rather than the full Nexus time series.

TheTimeSeries.timestamp() method, shown in the for loop above, returns an ascii string representation of a date, or date and time, also in a format that is specified by an ISO standard. If you want to find the time index for a given date, use one of the following:

.. code-block:: python

    time_index = time_series.index_for_timestamp('2006-10-01')  # exact match required; note format: YYYY-MM-DD
    # following includes time of day; format: YYYY-MM-DDTHH:MM:SSZ
    time_index = time_series.index_for_timestamp('2006-10-01T00:00:00Z')
    # an alternative method not requiring an exact match
    time_index = time_series.index_for_timestamp_not_later_than('2006-10-01T18:00:00Z')

Given a time index, we can use it as a criterion when identifying an individual array for a recurrent property. For example:

.. code-block:: python

    final_time_index = time_series.number_of_timestamps() - 1  # time indices count up starting at zero
    final_water_saturation_array = pc.single_array_ref(citation_title = 'SW', time_index = final_time_index)

The examples shown above will work for a RESQML dataset holding data from a single Nexus case, because we know that all the recurrent arrays will refer to the same time series. In the more general case, we might need to instantiate a separate time series object for each recurrent property: the UUID of the related time series is stored for each property array and can be found with:

.. code-block:: python

    initial_pressure_part = pc.singleton(property_kind = 'pressure', time_index = 0)  # time_index of zero will be earliest
    pressure_specific_ts_uuid = pc.time_series_uuid_for_part(initial_pressure_part)
    pressure_time_series = rqts.TimeSeries(model, time_series_root = model.root(uuid = pressure_specific_ts_uuid))

The resqpy time_series.py module also includes a TimeDuration class for working with time periods, ie. the interval between two timestamps.

Working with groups of properties
---------------------------------
The collection of arrays for a recurrent property, at different reporting timesteps, form a logical group of properties. The resqpy property module provides functions and methods to help with these groupings. The first approach we'll look at involves creating a new property collection object for the group. Bear in mind that the actual arrays of data are only loaded on demand, so having multiple property collections instantiated is not a problem.

Here's a general way to create a new property collection as a subset of an existing one:

.. code-block:: python

    import resqpy.property as rqp
    pressure_pc = rqp.selective_version_of_collection(pc, property_kind = 'pressure')

The selection criteria can involve any of the items we've seen before, such as citation_title or time_index (amongst others). Eg.:

.. code-block:: python

    inital_saturations_pc = rqp.selective_version_of_collection(pc, property_kind = 'saturation', time_index = 0)

There are some convenience functions in the property module for common groupings. Here is a function which will look for a particular simulator keyword as the citation title:

.. code-block:: python

    oil_sat_pc = rqp.property_collection_for_keyword(pc, 'SO')

If we have identified one part for a recurrent property, we can use it as an example to group other parts that only differ by time index:

.. code-block:: python

    pressure_pc = rqp.property_over_time_series_from_collection(pc, initial_pressure_part)

We can also merge a second property collection into a primary one, for example:

.. code-block:: python

    hydrocarbon_saturations_pc = rqp.property_collection_for_keyword(pc, 'SG')
    hydrocarbon_saturations_pc.inherit_parts_from_other_collection(oil_sat_pc)

Note that the example above is not calculating a hydrocarbon saturation, it is merely collecting the oil and gas saturation arrays into a single property collection.

There is another mechanism for working with groups of properties (which we won't look at in detail here), and that is via a RESQML PropertySet object. This also groups together a set of property arrays, with the grouping also being an object in the dataset. The vdb import functions support generating some PropertySet objects, if desired. For example, the import_vdb_ensemble() function has an optional boolean argument create_property_set_per_realization. And one way to instantiate a respqy PropertyCollection object is for a given RESQML PropertySet object.

Working with multiple realisations
----------------------------------
A RESQML property includes an optional realisation number. These are set by the resqpy functions to match the case number, when importing an ensemble of vdbs from a TDRM/Fortuna job. The resqpy PropertyCollection methods for selecting arrays accept a realization number as an optional argument. For example:

.. code-block:: python

    case_23_pore_volume_array = pc.single_array_ref(property_kind = 'pore volume', realization = 23)

The set of realisation numbers present in a PropertyCollection can be found with the following method. Note that this does not imply that all properties are present for all the realisations, though for an ensemble built from a set of successful Nexus runs, that will usually be the case.

.. code-block:: python

    realization_list = pc.realization_list()

Depending on how one wants to work with the properties, the methods already discussed can be used to build property collections covering different subsets of all the arrays:

* all properties, for all realisations, for all timesteps
* all properties, for all realisations, for a single timestep
* all properties, for one realisation, for all timesteps
* all properties, for one realisation, for a single timestep
* any of the above combinations for a single property

Of course, the timestep options only apply to recurrent properties.

Supporting representation and indexable elements
------------------------------------------------
Everything discussed so far about accessing RESQML properties applies not only to grid properties but also, for example, well logs and blocked well properties, amongst other things. The same classes and methods can be used when handling all these sorts of properties. (Though for convenience resqpy also has some derived classes such as WellLogCollection.) In RESQML, the object providing the discrete geometrical frame for the properties is referred to as the supporting representation, which for our purposes here is the grid.

The dimensionality of the underlying property arrays depends on the number of dimensions used to index an indexable element of the supporting representation. In the case of Nexus grid property arrays, the indexable elements are 'cells' and the K,J,I indexing is 3D. (All references to grids here refer to the IjkGridRepresentation RESQML class – other classes of grid are available in the standard!) But the same grid object could also have some properties where the indexable element is set to 'columns' and the array is 2D, indexed by J,I. Or how about an efficient representation of zonation with a categorical property where the indexable element is 'layers' – just a single zone number would be held for each layer of the grid, indicating which zone the layer is assigned to.

Another example could be transmissibility multipliers: simulators such as Nexus rather clumsily assign I-face multipliers to the cell either on the plus side of the face, or the minus side – and different simulators have adopted opposite protocols. In RESQML, 'faces' is also a valid indexable element for a grid, which makes more explicit where the data is applicable.

For Ijk Grid properties (excluding radial grids), the full list of possible indexable elements is:

* cells
* column edges
* columns
* coordinate lines
* edges
* edges per column
* faces
* faces per cell
* hinge node faces
* interval edges
* intervals
* I0
* I0 edges
* J0
* J0 edges
* layers
* nodes
* nodes per cell
* nodes per edge
* nodes per face
* pillars
* subnodes

High dimensional numpy arrays
-----------------------------
Returning to the cell based grid properties... Despite the mechanisms for grouping property arrays, the data is actually stored in the hdf5 file as individual 3D numpy arrays. The 3 dimensions cover the K, J & I axes of the grid.

There are three methods in the PropertyCollection class for presenting a group of arrays as a single 4D numpy array. For example:

.. code-block:: python

    pore_volume_pc = rqp.selective_version_of_collection(pc, property_kind = 'pore volume')
    pore_volume_4d_array = pore_volume_pc.realizations_array_ref()  # numpy array indexed by R, K, J, I

Of course such arrays could be very large, so they should be used with caution – for example reducing the data to zonal values before creating the 4D array. The advantage is that extremely efficient numpy operations can then be used. For example to compute the cell-by-cell mean pore volume across all realizations:

.. code-block:: python

    mean_across_ensemble_pv_3d_array = np.nanmean(pore_volume_4d_array, axis = 0)

The other high dimensional array methods currently offered by the PropertyCollection class are for handling facets and time indices. Here is a facet example:

.. code-block:: python

    permeability_pc = rqp.selective_version_of_collection(pc, property_kind = 'permeability rock')
    facet_list = permeability_pc.facet_list()  # could return ['K', 'I'], for example, if we have PERMZ and PERMX data
    permeability_4d_array = permeability_pc.facets_array_ref()
    # numpy array above indexed by F, K, J, I where F is also an index into facet_list

And for a 4D property array where the extra axis covers time indices:

.. code-block:: python

    pressure_pc = rqp.selective_version_of_collection(pc, property_kind = 'pressure')
    time_index_list = pressure_pc.time_index_list()
    pressure_4d_array = pressure_pc.time_series_array_ref()
    # numpy array above indexed by T, K, J, I where T is also an index into time_index_list

Beyond these 4D arrays, we could combine some of these higher dimensions to produce, for example, 5D arrays covering realisations and time indices, or 6D arrays covering realisations, time indices and facets, as well as the K, J, I of the cell indices of course!

Creating new grid property objects
----------------------------------
The discussion so far has focussed on accessing property arrays from a RESQML dataset – making them available to application code as numpy arrays. At some point though, we might want to store a new property array in the dataset. The resqml.derived_model module has a function for this. Note that all the functions in the derived model module work from and to datasets stored on disc. After calling such a function it is necessary to re-instantiate a Model object in order to pick up on the changes.

To add a property, first create the data as a numpy array. Here, for example, we compute pressure change:

.. code-block:: python

    initial_pressure_part = pc.singleton(property_kind = 'pressure', time_index = 0)
    initial_pressure_array = pc.cached_part_array_ref(initial_pressure_part)
    pressure_units = pc.uom_for_part(initial_pressure_part)

    final_pressure_array = pc.single_array_ref(property_kind = 'pressure', time_index = final_time_index)
    # see earlier notes for finding final_time_index

    pressure_change_array = final_pressure_array - initial_pressure_array  # example calculation

Then call the function to add the new array as shown below. The full argument list is shown here to facilitate the discussion which follows. In practice, for this example, all the arguments after uom could be omitted.

.. code-block:: python

    import resqpy.derived_model as rqdm

    rqdm.add_one_grid_property_array(epc_file = epc_path,
                                     a = pressure_change_array,
                                     property_kind = 'pressure',
                                     grid_uuid = grid.uuid,
                                     source_info = 'final pressure minus initial',
                                     title = 'PRESSURE CHANGE',
                                     discrete = False,
                                     uom = pressure_units,
                                     time_index = None,
                                     time_series_uuid = None,
                                     string_lookup_uuid = None,
                                     null_value = None,
                                     indexable_element = 'cells',
                                     facet_type = None, facet = None,
                                     realization = None,
                                     local_property_kind_uuid = None,
                                     count_per_element = 1,
                                     new_epc_file = None)

The paragraphs below look at the argument list for that function in some more detail.

To re-open the model after calling a function in the derived_model module, simply re-instatiate a Model object:

.. code-block:: python

    model = rq.Model(epc_path)

**epc_file**

The first argument is the RESQML epc file which contains the grid. By default the new property will be added to this RESQML dataset (both the epc and h5 files will be updated). Another argument, new_epc_file, can be used as well if a new dataset is required instead of an update (see below).

**a**

The second argument is the numpy array holding the new property. It should have the appropriate shape for the grid (taking into consideration the indexable_element and count_per_element arguments). Assuming the default value of 'cells' for the indexable element (and 1 for count_per_element), the required shape is (nk, nj, ni).

The dtype (element data type) of the array should also be appropriate. Numpy arrays tend to default to a dtype of float, which will be a 64 bit floating point representation. For discrete data, be sure to use an integer data type such as int (64 bit) or int32, or int8 or bool for boolean data.

**property_kind**

This argument must be set and should be one of the supported property kinds, unless a local property kind is needed for the array (see below).

**grid_uuid**

This should be set to the UUID of the grid to which the array pertains.

**source_info**

The source info is a human readable string that should be set in such a way to help people understand where the data has come from. It is not used for any automated processing purposes.

**title**

The title is used to populate the citation title in the metadata for the new property object. Application code later in the workflow might rely on this to find the correct array.

**discrete**

This is a boolean indicating whether the data is discrete (True) or continuous (False). Set to True for any integer or boolean array data, including categorical data.

**uom**

The units must be specified. See earlier section for a list of the most common units we work with.

**time_index & time_series_uuid**

If the new property is part of a recurrent series, these two arguments should be specified. Here they are left as None because we are computing a single pressure change array. If we were generating a series of arrays, indicating the pressure change per reporting timestep, then these arguments would be needed.

**string_lookup_uuid**

If the property is categorical, this argument must be set to the UUID of the string lookup table object. The lookup table should be added to the model before adding the arrays, unless it already exists in the dataset. How to create objects such as lookup tables will be discussed elsewhere.

**null_value**

Continuous data always uses NaN (not-a-number) as the null value, and this argument should be left as None. However, NaN cannot be used in an integer array, so RESQML allows an integer value to be specified as null for each discrete or categorical property. It is usual to use -1 as the null value unless that is a valid value for the property.

**indexable_element**

This defaults to 'cells', which most grid properties are for. For map making, the value 'columns' might well get used. There are several other possibilities. The shape of the array must be correct for the value of this argument.

**facet_type & facet**

The RESQML standard allows a property object to have any number of facets. However, the resqpy code, including this function, generally works with at most one facet per property. If no facet is applicable to the property then these arguments should be left as None. The RESQML standard lists a few common facet types, though we are free to make up new ones. Facet types currently in use include:

* 'direction': 'I', 'J', 'K', 'IJ', or 'IJK'
* 'what': 'oil', 'gas', 'water' – used by resqpy for saturation or other phase related properties
* 'netgross': 'net', or 'gross' – used for thickness properties

Other standard facet types are: 'conditions', 'statistics', or 'qualifier'. The standard facet types are defined in the RESQML schema definition file properties.xsd

**realization**

Set this to the realization number if the property is applicable to one realization within an ensemble.

**local_property_kind_uuid**

If the property kind of the array is a 'local' property kind (i.e. not specified in the RESQML standard) then the property kind must already have been added (or exist) in the model and this argument is set to its UUID.

**count_per_element**

RESQML allows more than one value to be stored together, for each indexable element. This is achieved by adding an extra dimension to the array, being the 'fastest' cycling (ie. last numpy index). For example, imagine generating an array holding a complex number for each cell. The numpy array would have shape (NK, NJ, NI, 2) and the count_per_element argument would be set to 2.

**new_epc_file**

If this argument is set to a file path, the epc_file is not modified. A new epc (and paired h5) file will be created. The grid object and the coordinate reference system it refers to are copied to the new dataset and the newly created property added.
