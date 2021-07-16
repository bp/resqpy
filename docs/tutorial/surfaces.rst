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
    horizontal_surface = rqs.Surface(model)
    # populate the resqpy object for a horizontal plane
    horizontal_surface.set_to_horizontal_plane(depth = surface_depth, box_xyz = xyz_box)
    # write to persistent storage (not always needed, if the object is temporary)
    horizontal_surface.write_hdf5()
    horizontal_surface.create_xml()
    model.store_epc()

The *set_to_horizontal_plane* method generates a very simple surface using 4 points and two triangles:

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
* set_from_torn_mesh - points are in a numpy array with duplication at corners of 'cells'; gaps will appear in the surface where corners of neighbouring cells are not coincident
* set_to_single_cell_faces_from_corner_points - creates a Surface representing all 6 faces of a hexahedral cell (typically from an IjkGridRepresentation geometry)
* set_to_multi_cell_faces_from_corner_points - similar to above but representing all the faces of a set of cells
* set_to_triangle - creates a Surface for a single triangle
* set_to_sail - creates a Surface with the geometry of a triangle wrapped on a sphere
* set_from_tsurf_file - import from a GOCAD tsurf file
* set_from_zmap_file - import from a zmap format ascii file
* set_from_roxar_file - import from an RMS format ascii file

If a Surface is created from a simple (untorn) mesh, with either *set_from_irregular_mesh* or *set_from_mesh_object*, then the following method can be used to locate which 'cell' a particular triangle index is for. Resqpy includes functions for finding where a line intersects a triangulated surface. Those functions can return a triangle index which can be converted back to a mesh 'cell' (referred to as a column in the method name) with:

* column_from_triangle_index

