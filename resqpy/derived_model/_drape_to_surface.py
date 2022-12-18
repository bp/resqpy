"""High level drape to surface function."""

import logging

log = logging.getLogger(__name__)

import os
import numpy as np
import copy

import resqpy.crs as rqc
import resqpy.derived_model
import resqpy.grid_surface as rgs
import resqpy.model as rq
import resqpy.olio.intersection as meet
import resqpy.olio.xml_et as rqet

import resqpy.derived_model._common as rqdm_c
import resqpy.derived_model._copy_grid as rqdm_cg


def drape_to_surface(epc_file,
                     source_grid = None,
                     surface = None,
                     scaling_factor = None,
                     ref_k0 = 0,
                     ref_k_faces = 'top',
                     quad_triangles = True,
                     border = None,
                     store_displacement = False,
                     inherit_properties = False,
                     inherit_realization = None,
                     inherit_all_realizations = False,
                     new_grid_title = None,
                     new_epc_file = None):
    """Return a new grid with geometry draped to a surface.
    
    Extend a resqml model with a new grid where the reference layer boundary of the source grid has been re-draped
    to a surface.

    arguments:
       epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
       source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
          (or one 'ROOT' grid) which is used as the source grid
       surface (surface.Surface object, optional): the surface to drape the grid to; if None, a surface is generated from
          the reference layer boundary (which can then be scaled with the scaling_factor)
       scaling_factor (float, optional): if not None, prior to draping, the surface is stretched vertically by this factor,
          away from a horizontal plane located at the surface's shallowest depth
       ref_k0 (integer, default 0): the reference layer (zero based) to drape to the surface
       ref_k_faces (string, default 'top'): 'top' or 'base' identifying which bounding interface to use as the reference
       quad_triangles (boolean, default True): if True and surface is None, each cell face in the reference boundary layer
          is represented by 4 triangles (with a common vertex at the face centre) in the generated surface; if False,
          only 2 trianges are used for each cell face (which gives a non-unique solution)
       cell_range (integer, default 0): the number of cells away from faults which will have depths adjusted to spatially
          smooth the effect of the throw scaling (ie. reduce sudden changes in gradient due to the scaling)
       offset_decay (float, default 0.5): the factor to reduce depth shifts by with each cell step away from faults (used
          in conjunction with cell_range)
       store_displacement (boolean, default False): if True, 3 grid property parts are created, one each for x, y, & z
          displacement of cells' centres brought about by the local depth shift
       inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
          with the source grid
       inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
          inherit_properties is False
       inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
          realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
          inherit_properties is False or inherit_realization is not None
       new_grid_title (string): used as the citation title text for the new grid object
       new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
          a new epc file (& associated h5 file) is created to contain the draped grid (& crs)

    returns:
       new grid (grid.Grid object), with geometry draped to surface

    notes:
       at least one of a surface or a scaling factor must be given;
       if no surface is given, one is created from the fault-healed grid points for the reference layer interface;
       if a scaling factor other than 1.0 is given, the surface is flexed vertically, relative to its shallowest point;
       layer thicknesses measured along pillars are maintained; cell volumes may change;
       the coordinate reference systems for the surface and the grid are assumed to be the same;
       this function currently uses an exhaustive, computationally and memory intensive algorithm;
       setting quad_triangles argument to False should give a factor of 2 speed up and reduction in memory requirement;
       the epc file and associated hdf5 file are appended to (extended) with the new grid, as a side effect of this function
    """

    log.info('draping grid to surface')

    assert epc_file or new_epc_file, 'epc file name not specified'
    if new_epc_file and epc_file and (
        (new_epc_file == epc_file) or
        (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
        new_epc_file = None
    model, source_grid = rqdm_c._establish_model_and_source_grid(epc_file, source_grid)
    assert source_grid.grid_representation == 'IjkGrid'
    assert model is not None

    assert ref_k0 >= 0 and ref_k0 < source_grid.nk
    assert ref_k_faces in ['top', 'base']
    assert surface is not None or (scaling_factor is not None and scaling_factor != 1.0)

    if surface is None:
        surface = rgs.generate_untorn_surface_for_layer_interface(source_grid,
                                                                  k0 = ref_k0,
                                                                  ref_k_faces = ref_k_faces,
                                                                  quad_triangles = quad_triangles,
                                                                  border = border)

    if scaling_factor is not None and scaling_factor != 1.0:
        scaled_surf = copy.deepcopy(surface)
        scaled_surf.vertical_rescale_points(scaling_factor = scaling_factor)
        surface = scaled_surf

    # check that surface and grid use same crs; if not, convert to same
    surface_crs = rqc.Crs(surface.model, surface.crs_uuid)
    if source_grid.crs is None:
        source_grid.crs = rqc.Crs(source_grid.model, uuid = source_grid.crs_uuid)
    if surface_crs != source_grid.crs:
        surface.triangles_and_points()
        surface_crs.convert_array_to(source_grid.crs, surface.points)

    # take a copy of the grid
    log.debug('copying grid')
    grid = rqdm_cg.copy_grid(source_grid, model)
    grid.cache_all_geometry_arrays()  # probably already cached anyway

    # todo: handle pillars with no geometry defined, and cells without geometry defined
    assert grid.geometry_defined_for_all_pillars(), 'not all pillars have defined geometry'
    assert grid.geometry_defined_for_all_cells(), 'not all cells have defined geometry'

    # fetch unsplit equivalent of grid points
    log.debug('fetching unsplit equivalent grid points')
    unsplit_points = grid.unsplit_points_ref(cache_array = True)

    # assume pillars to be straight lines based on top and base points
    log.debug('setting up pillar sample points and directional vectors')
    line_p = unsplit_points[0, :, :, :].reshape((-1, 3))
    line_v = unsplit_points[-1, :, :, :].reshape((-1, 3)) - line_p
    if ref_k_faces == 'base':
        ref_k0 += 1

    # access triangulated surface as triangle node indices into array of points
    log.debug('fetching surface points and triangle corner indices')
    t, p = surface.triangles_and_points()  # will pick up cached crs converted copy if appropriate

    # compute intersections of all pillars with all triangles (sparse array returned with NaN for no intersection)
    log.debug('computing intersections of all pillars with all triangles')
    intersects = meet.line_set_triangles_intersects(line_p, line_v, p[t])

    # reduce to a single intersection point per pillar; todo: flag multiple intersections with a warning
    log.debug('selecting last intersection for each pillar (there should be only one intersection anyway)')
    picks = meet.last_intersects(intersects)

    # count the number of pillars with no intersection at surface (indicated by triple nan)
    log.debug('counting number of pillars which fail to intersect with surface')
    failures = np.count_nonzero(np.isnan(picks)) // 3
    log.info('number of pillars which do not intersect with surface: ' + str(failures))
    assert failures == 0, 'cannot proceed as some pillars do not intersect with surface'

    # compute a translation vector per pillar
    log.debug('computing translation vectors for pillars')
    translate = picks - unsplit_points[ref_k0, :, :, :].reshape((-1, 3))

    # shift all points by translation vectors
    _shift_by_translation_vectors(grid, translate)

    # check cell edge relative directions (in x,y) to ensure geometry is still coherent
    log.debug('checking grid geometry coherence')
    grid.check_top_and_base_cell_edge_directions()

    # build cell displacement property array(s)
    if store_displacement:
        log.debug('generating cell displacement property arrays')
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
        new_grid_title = 'grid flexed from ' + str(rqet.citation_title_for_node(source_grid.root))

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


def _shift_by_translation_vectors(grid, translate):
    log.debug('shifting entire grid along pillars')
    if grid.has_split_coordinate_lines:
        jip1 = (grid.nj + 1) * (grid.ni + 1)
        # adjust primary pillars
        grid.points_cached[:, :jip1, :] += translate.reshape((1, jip1, 3))  # will be broadcast over k axis
        # adjust split pillars
        for p in range(grid.split_pillars_count):
            primary = grid.split_pillar_indices_cached[p]
            grid.points_cached[:, jip1 + p, :] += translate.reshape(
                (1, jip1, 3))[0, primary, :]  # will be broadcast over k axis
    else:
        grid.points_cached[:, :, :, :] +=  \
           translate.reshape((1, grid.points_cached.shape[1], grid.points_cached.shape[2], 3))    # will be broadcast over k axis
