"""High level tilted grid function."""

import logging

log = logging.getLogger(__name__)

import os

import resqpy.derived_model
import resqpy.model as rq
import resqpy.olio.vector_utilities as vec
import resqpy.olio.xml_et as rqet

import resqpy.derived_model._common as rqdm_c
import resqpy.derived_model._copy_grid as rqdm_cg


def tilted_grid(epc_file,
                source_grid = None,
                pivot_xyz = None,
                azimuth = None,
                dip = None,
                store_displacement = False,
                inherit_properties = False,
                inherit_realization = None,
                inherit_all_realizations = False,
                new_grid_title = None,
                new_epc_file = None):
    """Extends epc file with a new grid which is a version of the source grid tilted.

    arguments:
       epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
       source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
          (or one 'ROOT' grid) which is used as the source grid
       pivot_xyz (triple float): a point in 3D space on the pivot axis, which is horizontal and orthogonal to azimuth
       azimuth: the direction of tilt (orthogonal to tilt axis), as a compass bearing in degrees
       dip: the angle to tilt the grid by, in degrees; a positive value tilts points in direction azimuth downwards (needs checking!)
       store_displacement (boolean, default False): if True, 3 grid property parts are created, one each for x, y, & z
          displacement of cells' centres brought about by the tilting
       inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
          with the source grid
       inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
          inherit_properties is False
       inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
          realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
          inherit_properties is False or inherit_realization is not None
       new_grid_title (string): used as the citation title text for the new grid object
       new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
          a new epc file (& associated h5 file) is created to contain the tilted grid (& crs)

    returns:
       a new grid (grid.Grid object) which is a copy of the source grid tilted in 3D space
    """

    assert epc_file or new_epc_file, 'epc file name not specified'
    if new_epc_file and epc_file and (
        (new_epc_file == epc_file) or
        (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
        new_epc_file = None
    model, source_grid = rqdm_c._establish_model_and_source_grid(epc_file, source_grid)
    assert source_grid.grid_representation == 'IjkGrid'
    assert model is not None

    # take a copy of the grid
    grid = rqdm_cg.copy_grid(source_grid, model)

    if grid.inactive is not None:
        log.debug('copied grid inactive shape: ' + str(grid.inactive.shape))

    # tilt the grid
    grid.cache_all_geometry_arrays()  # probably already cached anyway
    vec.tilt_points(pivot_xyz, azimuth, dip, grid.points_cached)

    # build cell displacement property array(s)
    if store_displacement:
        displacement_collection = rqdm_c._displacement_properties(grid, source_grid)
    else:
        displacement_collection = None

    collection = rqdm_c._prepare_simple_inheritance(grid, source_grid, inherit_properties, inherit_realization,
                                                    inherit_all_realizations)
    if collection is None:
        collection = displacement_collection
    elif displacement_collection is not None:
        collection.inherit_imported_list_from_other_collection(displacement_collection, copy_cached_arrays = False)

    if new_grid_title is None or len(new_grid_title) == 0:
        new_grid_title = 'tilted version ({0:4.2f} degree dip) of '.format(abs(dip)) + str(
            rqet.citation_title_for_node(source_grid.root))

    # write model
    model.h5_release()
    if new_epc_file:
        rqdm_c._write_grid(new_epc_file,
                           grid,
                           property_collection = collection,
                           grid_title = new_grid_title,
                           mode = 'w')
    else:
        ext_uuid, _ = model.h5_uuid_and_path_for_node(rqet.find_nested_tags(source_grid.root, ['Geometry', 'Points']),
                                                      'Coordinates')
        rqdm_c._write_grid(epc_file,
                           grid,
                           ext_uuid = ext_uuid,
                           property_collection = collection,
                           grid_title = new_grid_title,
                           mode = 'a')

    return grid
