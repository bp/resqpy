Surface Representations
=======================

This tutorial discusses surfaces, which can be used to represent the geometry of geological horizons, fault planes, and fluid contacts. In RESQML, the two main classes of object used for surfaces (with the equivalent resqpy class in brackets) are:

* TriangulatedSetRepresentation (Surface) - a triangulated surface can be used to represent torn or untorn surfaces
* Grid2dRepresentation (Mesh) – also known as a *lattice*, has a squared grid of points and can only fully define an untorn surface

The Grid2dRepresentation class (and its resqpy equivalent, Mesh) supports various options regarding the regularity of the lattice of points.

Two more classes are also sometimes used when constructing surfaces:

* PointSetRepresentation (PointSet) - a set of points; resqpy includes a function for making a *Delaunay Triangulation* from a point set
* PolylineSetRepresentation (PolylineSet) - can be used to represent a set of fault sticks

Resqpy includes a further class (CombinedSurface) which does not have a corresponding RESQML class. It allows a set of surfaces to be treated as a single surface, which can make some application functionality easier to implement.

In this tutorial we assume the following import statements:

.. code-block:: python

    import resqpy.model as rq
    import resqpy.surface as rqs
    import resqpy.organize as rqo
    import resqpy.olio.uuid as bu

Working with an existing Surface
--------------------------------
Instantiating a resqpy Surface object for an existing RESQML TriangulatedSetRepresentation follows the familiar steps of identifying the uuid and then calling the initialisation method. In this example we start with the title of a horizon interpretation:

.. code-block:: python

    model = rq.Model('existing_geo_model.epc')
    horizon_interp_uuid = model.uuid(obj_type = 'HorizonInterpretation', title = 'Base Cretaceous')
    surface_uuid = model.uuid(obj_type = 'TriangulatedSetRepresentation', related_uuid = horizon_interp_uuid)
    base_cretaceous_surface = rqs.Surface(model, uuid = surface_uuid)

Application code will often need to access the triangulated data, for example to render the surface graphically, using the following method:

.. code-block:: python

    triangles, points = base_cretaceous_surface.triangles_and_points()

The *triangles* return value is a numpy int array of shape (NT, 3) where NT is the number of triangles and the 3 covers the three corners of a triangle. The values in *triangles* are indices into the *points* numpy float array, which has shape (NP, 3), with NP being the number of points and the 3 covering the xyz coordinates. Therefore this assertion should always pass:

.. code-block:: python

    assert np.all(triangles < len(points))

Almost all the other methods in the Surface class are for creating new surfaces in various ways.

Creating a new Surface
----------------------
The resqpy Surface class includes many methods for setting up the data for new surface. A few of these methods can be triggered by using certain arguments to the *init* method. The general way, though, is to create an empty Surface object and then call one of the *set_from_* or *set_to_* methods. Here is an example which creates a simple horizontal plane covering a rectangular area:

.. code-block:: python

    min_x, max_x = 450000.00, 520000.00
    min_y, max_y = 6400000.00, 6600000.00
    surface_depth = 2700.00
    xyz_box = np.array(((min_x, min_y, 0,0), (max_x, max_y, 0.0)))  # z values will be ignored
    # create an empty surface
    horizontal_surface = rqs.Surface(model, crs_uuid = model.crs_uuid)
    # populate the resqpy object for a horizontal plane
    horizontal_surface.set_to_horizontal_plane(depth = surface_depth, box_xyz = xyz_box)
    # write to persistent storage (not always needed, if the object is temporary)
    horizontal_surface.write_hdf5()
    horizontal_surface.create_xml()
    model.store_epc()

The *set_to_horizontal_plane* method generates a very simple surface using four points and two triangles:

.. code-block:: python

    t, p = horizontal_surface.triangles_and_points()
    assert len(t) == 2 and len(p) == 4

Here is a full list of the methods for setting up a new Surface:

* set_to_horizontal_plane - discussed above
* set_from_triangles_and_points - when the triangulation has been prepared in numpy arrays
* set_from_point_set - generates a Delaunay Triangulation for a set of points (computationally expensive)
* set_from_irregular_mesh - where the points form an irregular lattice (think of a stretched and warped piece of squared paper)
* set_from_sparse_mesh - similar to above but mesh may contain NaNs, which will result in holes in the surface
* set_from_mesh_object - starting from a resqpy Mesh object
* set_from_torn_mesh - points are in a numpy array with duplication at corners of 2D 'cells'; gaps will appear in the surface where corners of neighbouring cells are not coincident
* set_to_single_cell_faces_from_corner_points - creates a Surface representing all 6 faces of a hexahedral cell (typically from an IjkGridRepresentation geometry)
* set_to_multi_cell_faces_from_corner_points - similar to above but representing all the faces of a set of cells
* set_to_triangle - creates a Surface for a single triangle
* set_to_sail - creates a Surface with the geometry of a triangle wrapped on a sphere
* set_from_tsurf_file - import from a GOCAD tsurf file
* set_from_zmap_file - import from a zmap format ascii file
* set_from_roxar_file - import from an RMS format ascii file

If a Surface is created from a simple (untorn) mesh, with either *set_from_irregular_mesh* or *set_from_mesh_object*, then the following method can be used to locate which 2D 'cell' a particular triangle index is for. Resqpy includes functions for finding where a line intersects a triangulated surface. Those functions can return a triangle index which can be converted back to a mesh 'cell' (referred to as a column in the method name) with:

* column_from_triangle_index

Similarly, if a Surface is created using *set_to_single_cell_faces_from_corner_points* or *set_to_multi_cell_faces_from_corner_points*, the cell and face for a given triangle index can be identified with:

* cell_axis_and_polarity_from_triangle_index

The resqpy CombinedSurface class
--------------------------------
The CombinedSurface class allows a set of Surface objects to be treated as a single composite surface for some purposes. It can be useful when looking for wellbore trajectory intersections and might also be convenient in some graphical applications.

A combined surface is initialised simply from a list of resqpy Surface objects, e.g.:

.. code-block:: python

    all_horizons = rqs.CombinedSurface([top_reservoir_surface, base_triassic_horizon, base_reservoir_surface])

As this is a derived resqpy class, it is not written to persistent storage, there is no xml and it is not added to the model. There are only two useful methods. The first, *triangles_and_points* behaves just the same as the Surface method:

.. code-block:: python

    t, p = all_horizons.triangles_and_points()

And the second, *surface_index_for_triangle_index* identifies which surface, together with its local triangle index, a combined surface triangle index is equivalent to:

.. code-block:: python

    surface_index, local_triangle_index = all_horizons.surface_index_for_triangle_index(6721)

In that example, *surface_index* is an index into the list of surfaces passed to the initialisation of the combined surface object.

Introducing the Mesh class
--------------------------
The resqpy Mesh class is equivalent to the Grid2dRepresentation RESQML class. It can be used to represent a depth map for a surface such as a horizon and is characterised by usually having a regular two-dimensional lattice of points in the xy plane. RESQML allows various options for storing the data. Which option is in use is visible as the resqpy Mesh attribute *flavour* which can have the following values:

* 'explicit' - full xyz values are provided for every point, with an implied logical IJ orderliness in the xy space
* 'regular' - the xy values form a perfectly regular lattice and there are no z values
* 'reg&z' - the xy values form a perfectly regular lattice and there are explicit z values
* 'ref&z' - the xy values are stored in a separate referenced Mesh (typically of flavour 'regular'), there are explicit z values

The logical size of the lattice can be found with a pair of attributes: *ni* and *nj*. These hold the number of points in the I and J axes. Note that these hold a node or point count, not a 'cell' count.

Reading an existing Mesh
------------------------
Initialising a resqpy Mesh object for an existing RESQML Grid2dRepresentation follows the familiar steps of identifying the uuid and passing that value to the __init__ method. For example:

.. code-block:: python

    top_reservoir_mesh_uuid = model.uuid(obj_type = '', title = 'Top Reservoir')
    top_reservoir_mesh = rqs.Mesh(model, uuid = top_reservoir_mesh_uuid)

Regardless of which flavour a mesh is, a fully expanded numpy array of xyz values can be accessed with:

.. code-block:: python

    xyz_array = top_reservoir_mesh.full_array_ref()

Another generic method which will work for any flavour of Mesh object is *surface*, which generates a Surface object for the Mesh:

.. code-block:: python

    top_reservoir_surface = top_reservoir_mesh.surface(quad_triangles = True)

The *quad_triangles* boolean argument causes each 'square' (more generally, each quadrilateral) to be represented by four triangles rather than two, using an extra point at the centre of the square (the mean of the four vertices). This gives a unique representation whereas the default two triangle representation yields a different surface depending upon which diagonal is used for a non-planar quadrilateral.

For a mesh with a regular lattice of points, the origin and spacing can be found in the following way:

.. code-block:: python

    assert top_reservoir_mesh.flavour in ['regular', 'reg&z']
    origin_xyz = top_reservoir_mesh.regular_origin
    deltas_xyz = top_reservoir_mesh.regular_dxyz_dij

Those arrays contain xyz values even though the z values are typically zero and not used. The origin is a simple triple. The regular_dxyz_dij attribute is a numpy array of shape (2, 3) which holds the xyz step size for each step in I (first index zero) and J (first index one). 

Of course the geometry exists within a coordinate reference system and that can be identified with the *crs_uuid* attribute.

More on the 'ref&z' flavour of Mesh
-----------------------------------
Where two or more meshes share a common xy lattice of points, differing only in z values, it can be useful to use the 'ref&z' flavour. The xy arrangement can be represented by a Mesh of flavour 'regular', to which the other meshes refer. Alternatively, one of the meshes can act as the master and have flavour 'reg&z', with the other meshes referring to it and having flavour 'ref&z'.

The main advantage of this way of working is that it is clear that the different sets of z values 'go with' the same set of xy values. Application code can make use of this knowledge. The following snippet checks that a mesh for a base reservoir horizon references one for the top reservoir, allowing the differences in z values to be meaningfully computed.

.. code-block:: python

    assert base_reservoir_mesh.flavour == 'ref&z'
    assert bu.matching_uuids(base_reservoir_mesh.ref_uuid, top_reservoir_mesh.uuid)
    thickness = base_reservoir_mesh.full_array_ref()[2] - top_reservoir_mesh.full_array_ref()[2]

Note the use of the *ref_uuid* attribute in that snippet, to identify the mesh being referenced for the xy values.

The PointSet class
------------------
A set of points in 3D space can be held in a RESQML object of class PointSetRepresentation and the equivalent resqpy class is PointSet. That class includes a *full_array_ref* method which returns a numpy array of shape (N, 3) holding the xyz values of the points.

If a set of points is representing a surface, it is usually necessary to convert it to a Surface object using a Delaunay Triangulation, e.g.:

.. code-block:: python

    owc_point_set = rqs.PointSet(model, uuid = owc_contact_picks_point_set_uuid)
    owc_surface = rqs.Surface(model, point_set = owc_point_set, title = 'oil-water contact from picks')

Note that the Delaunay Triangulation can be a computationally expensive operation. It is probably worth storing the resulting surface as a persistent object:

.. code-block:: python

    owc_surface.write_hdf5()
    owc_surface.create_xml()
    model.store_epc()

A non-standard use of Mesh
--------------------------
The resqpy library includes a DataFrame class (and some derived classes) which, behind the scenes, map a numerical pandas dataframe onto a Mesh object of flavour 'reg&z' (or 'regular' in the case of multiple realisations, in which case the values are stored as continuous property objects). The RESQML standard did not intend Grid2dRepresentation objects to be used in this way, so such dataframes will not generally be usable by RESQML enabled software that does not use the resqpy API.
