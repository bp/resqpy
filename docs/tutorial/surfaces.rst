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
The resqpy Mesh class is equivalent to the Grid2dRepresentation RESQML class. It can be used to represent a depth map for a surface such as a horizon and is characterised by usually having a regular two-dimensional lattice of points in the xy axes. RESQML allows various options for storing the data. Which option is in use is visible as the resqpy Mesh attribute *flavour* which can have the following values:

* 'explicit' - full xyz values are provided for every point, with an implied logical IJ orderliness in the xy space
* 'regular' - the xy values form a perfectly regular lattice and there are no z values
* 'reg&z' - the xy values form a perfectly regular lattice and there are explicit z values
* 'ref&z' - the xy values are stored in a separate Mesh (typically of flavour 'regular'), there are explicit z values

The logical size of the lattice can be found with a pair of attributes: *ni* and *nj*. These hold the number of points in the I and J axes. Note that these hold a node or point count, not a 'cell' count.
