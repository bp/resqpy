"""Multiprocessing wrapper functions for the grid_surface module."""

import logging

log = logging.getLogger(__name__)

import numpy as np
import uuid
from typing import Tuple, Union, List, Optional, Callable
from pathlib import Path
from uuid import UUID

import resqpy.grid_surface as rqgs
import resqpy.model as rq
import resqpy.grid as grr
import resqpy.property as rqp
import resqpy.surface as rqs
import resqpy.olio.uuid as bu


def find_faces_to_represent_surface_regular_wrapper(
        index: int,
        parent_tmp_dir: str,
        use_index_as_realisation: bool,
        grid_epc: str,
        grid_uuid: Union[UUID, str],
        surface_epc: str,
        surface_uuid: Union[UUID, str],
        name: str,
        title: Optional[str] = None,
        agitate: bool = False,
        feature_type: str = 'fault',
        trimmed: bool = False,
        is_curtain = False,
        extend_fault_representation: bool = False,
        retriangulate: bool = False,
        related_uuid = None,
        progress_fn: Optional[Callable] = None,
        extra_metadata = None,
        return_properties: Optional[List[str]] = None,
        raw_bisector: bool = False) -> Tuple[int, bool, str, List[Union[UUID, str]]]:
    """Multiprocessing wrapper function of find_faces_to_represent_surface_regular_optimised.

    arguments:
        index (int): the index of the function call from the multiprocessing function
        parent_tmp_dir (str): the parent temporary directory path from the multiprocessing function
        use_index_as_realisation (bool): if True, uses the index number as the realization number on
            the property collection
        grid_epc (str): epc file path where the grid is saved
        grid_uuid (UUID or str): UUID (universally unique identifier) of the grid object
        surface_epc (str): epc file path where the surface (or point set) is saved
        surface_uuid (UUID or str): UUID (universally unique identifier) of the surface (or point set) object.
        name (str): the feature name to use in the grid connection set.
        title (str): the citation title to use for the grid connection set; defaults to name
        agitate (bool): if True, the points of the surface are perturbed by a small random
           offset, which can help if the surface has been built from a regular mesh with a periodic resonance
           with the grid
        feature_type (str, default 'fault'): one of 'fault', 'horizon', or 'geobody boundary'
        trimmed (bool, default True): if True the surface has already been trimmed
        is_curtain (bool, default False): if True, only the top layer is intersected with the surface and bisector
           is generated as a column property if requested
        extend_fault_representation (bool, default False): if True, the representation is extended with a flange
        retriangulate (bool, default False): if True, a retriangulation is performed even if not needed otherwise
        related_uuid (uuid, optional): if present, the uuid of an object to be softly related to the gcs (and to
           grid bisector and/or shadow property if requested)
        progress_fn (Callable): a callback function to be called at intervals by this function;
           the argument will progress from 0.0 to 1.0 in unspecified and uneven increments
        extra_metadata (dict, optional): extra metadata items to be added to the grid connection set
        return_properties (list of str): if present, a list of property arrays to calculate and
           return as a dictionary; recognised values in the list are 'triangle', 'depth', 'offset', 'normal vector',
           'flange bool', 'grid bisector' and 'grid shadow';
           triangle is an index into the surface triangles of the triangle detected for the gcs face; depth is
           the z value of the intersection point of the inter-cell centre vector with a triangle in the surface;
           offset is a measure of the distance between the centre of the cell face and the intersection point;
           normal vector is a unit vector normal to the surface triangle; each array has an entry for each face
           in the gcs; grid bisector is a grid cell boolean property holding True for the set of cells on one
           side of the surface, deemed to be shallower; grid shadow is a grid cell int8 property holding 1 for
           cells neither above nor below a K face, 1 for above, 2 for beneath, 3 for between;
           the returned dictionary has the passed strings as keys and numpy arrays as values
        raw_bisector (bool, default False): if True and grid bisector is requested then it is left in a raw
           form without assessing which side is shallower (True values indicate same side as origin cell)

    returns:
        Tuple containing:
            - index (int): the index passed to the function
            - success (bool): whether the function call was successful, whatever that definiton is
            - epc_file (str): the epc file path where the objects are stored
            - uuid_list (List[str]): list of UUIDs of relevant objects

    notes:
        Use this function as argument to the multiprocessing function; it will create a new model that is saved
        in a temporary epc file and returns the required values, which are used in the multiprocessing function to
        recombine all the objects into a single epc file
    """
    tmp_dir = Path(parent_tmp_dir) / f"{uuid.uuid4()}"
    tmp_dir.mkdir(parents = True, exist_ok = True)
    epc_file = str(tmp_dir / "wrapper.epc")
    model = rq.new_model(epc_file = epc_file, quiet = True)
    uuid_list = []
    g_model = rq.Model(grid_epc, quiet = True)
    g_crs_uuid = g_model.uuid(obj_type = 'LocalDepth3dCrs',
                              related_uuid = grid_uuid)  # todo: check this relationship exists
    if g_crs_uuid is not None:
        model.copy_uuid_from_other_model(g_model, g_crs_uuid)
        uuid_list.append(g_crs_uuid)
    model.copy_uuid_from_other_model(g_model, uuid = grid_uuid)
    uuid_list.append(grid_uuid)
    for prop_title in ['DX', 'DY', 'DZ']:
        prop_uuid = g_model.uuid(obj_type = 'ContinuousProperty', title = prop_title, related_uuid = grid_uuid)
        if prop_uuid is not None:
            log.debug(f'found grid cell length property {prop_title}')
            model.copy_uuid_from_other_model(g_model, uuid = prop_uuid)
            uuid_list.append(prop_uuid)
        else:
            log.warning(f'grid cell length property {prop_title} NOT found')
    grid = grr.RegularGrid(parent_model = model, uuid = grid_uuid)
    assert grid.is_aligned
    flange_radius = 5.0 * np.sum(np.array(grid.extent_kji, dtype = float) * np.array(grid.aligned_dxyz()))
    s_model = rq.Model(surface_epc, quiet = True)
    model.copy_uuid_from_other_model(s_model, uuid = str(surface_uuid))
    repr_type = model.type_of_part(model.part(uuid = surface_uuid), strip_obj = True)
    assert repr_type in ['TriangulatedSetRepresentation', 'PointSetRepresentation']
    extended = False
    retriangulated = False
    flange_bool = None
    if repr_type == 'PointSetRepresentation':
        # trim pointset to grid xyz box
        pset = rqs.PointSet(model, uuid = surface_uuid)
        surf_title = pset.title
        log.debug(f'point set {pset.title} raw point count: {len(pset.full_array_ref())}')
        pset.change_crs(grid.crs)
        if not trimmed:
            pset.trim_to_xyz_box(grid.xyz_box(local = True))
            trimmed = True
            if 'trimmed' not in surf_title:
                surf_title += ' trimmed'
        assert len(pset.full_array_ref()) >= 3,  \
            f'boundary {name} representation {pset.title} has no xyz overlap with grid'
        pset_points = pset.full_array_ref()
        log.debug(f'trimmed point set {pset.title} contains {len(pset_points)} points; about to triangulate')
        if len(pset_points) > 1000:
            log.warning(
                f'trimmed point set {pset.title} has {len(pset_points)} points, which might take a while to triangulate'
            )
        # triangulate point set to form a surface; set repr_uuid to that surface and switch repr_flavour to 'surface'
        if extend_fault_representation and not surf_title.endswith(' extended'):
            surf_title += ' extended'
        surf = rqs.Surface(model, crs_uuid = grid.crs.uuid, title = surf_title)
        flange_bool = surf.set_from_point_set(pset,
                                              convexity_parameter = 2.0,
                                              reorient = True,
                                              extend_with_flange = extend_fault_representation,
                                              flange_radial_distance = flange_radius,
                                              make_clockwise = False)
        extended = extend_fault_representation
        retriangulated = True
        surf.write_hdf5()
        surf.create_xml()
        inherit_interpretation_relationship(model, surface_uuid, surf.uuid)
        surface_uuid = surf.uuid
        if flange_bool is not None:
            flange_p = rqp.Property.from_array(parent_model = model,
                                               cached_array = flange_bool,
                                               source_info = 'flange bool array',
                                               keyword = 'flange bool',
                                               support_uuid = surface_uuid,
                                               property_kind = 'flange bool',
                                               find_local_property_kind = True,
                                               indexable_element = 'faces',
                                               discrete = True)
            uuid_list.append(flange_p.uuid)

    surface = rqs.Surface(parent_model = model, uuid = str(surface_uuid))
    surf_title = surface.title
    assert surf_title
    surface.change_crs(grid.crs)
    if not trimmed and surface.triangle_count() > 100:
        if not surf_title.endswith('trimmed'):
            surf_title += ' trimmed'
        trimmed_surf = rqs.Surface(model, crs_uuid = grid.crs.uuid, title = surf_title)
        # trimmed_surf.set_to_trimmed_surface(surf, xyz_box = xyz_box, xy_polygon = parent_seg.polygon)
        trimmed_surf.set_to_trimmed_surface(surface, xyz_box = grid.xyz_box(local = True))
        surface = trimmed_surf
        trimmed = True
    if (extend_fault_representation and not extended) or (retriangulate and not retriangulated):
        _, p = surface.triangles_and_points()
        pset = rqs.PointSet(model, points_array = p, crs_uuid = grid.crs.uuid, title = surf_title)
        if extend_fault_representation and not surf_title.endswith('extended'):
            surf_title += ' extended'
        surface = rqs.Surface(model, crs_uuid = grid.crs.uuid, title = surf_title)
        flange_bool = surface.set_from_point_set(pset,
                                                 convexity_parameter = 2.0,
                                                 reorient = True,
                                                 extend_with_flange = extend_fault_representation,
                                                 flange_radial_distance = flange_radius,
                                                 make_clockwise = False)
        del pset
        extended = extend_fault_representation
        retriangulated = True
    if not bu.matching_uuids(surface.uuid, surface_uuid):
        surface.write_hdf5()
        surface.create_xml()
        # relate modified surface to original
        model.create_reciprocal_relationship_uuids(surface.uuid, 'sourceObject', surface_uuid, 'destinationObject')
        # inherit relationship to an interpretation object, if present for original surface
        inherit_interpretation_relationship(model, surface_uuid, surface.uuid)
        surface_uuid = surface.uuid
    if flange_bool is not None:
        flange_p = rqp.Property.from_array(parent_model = model,
                                           cached_array = flange_bool,
                                           source_info = 'flange bool array',
                                           keyword = 'flange bool',
                                           support_uuid = surface_uuid,
                                           property_kind = 'flange bool',
                                           find_local_property_kind = True,
                                           indexable_element = 'faces',
                                           discrete = True)
        uuid_list.append(flange_p.uuid)
    uuid_list.append(surface_uuid)

    returns = rqgs.find_faces_to_represent_surface_regular_optimised(grid,
                                                                     surface,
                                                                     name,
                                                                     title,
                                                                     agitate,
                                                                     feature_type,
                                                                     is_curtain,
                                                                     progress_fn,
                                                                     return_properties,
                                                                     raw_bisector = raw_bisector)

    success = False

    if isinstance(returns, tuple):
        gcs = returns[0]
    else:
        gcs = returns

    if gcs.count > 0:
        success = True
        gcs.write_hdf5()
        gcs.create_xml(extra_metadata = extra_metadata)
        model.copy_uuid_from_other_model(gcs.model, uuid = gcs.uuid)
        if related_uuid is not None:
            relative_found = (model.uuid(uuid = related_uuid) is not None)
            if not relative_found:
                for m in gcs.model, g_model, s_model:
                    if m.uuid(uuid = related_uuid) is not None:
                        model.copy_uuid_from_other_model(m, related_uuid)
                        relative_found = True
                        break
            if relative_found:
                model.create_reciprocal_relationship_uuids(gcs.uuid, 'sourceObject', related_uuid, 'destinationObject')
            else:
                log.warning(f'related uuid {related_uuid} not found; relationship dropped')
        uuid_list.append(gcs.uuid)

    if success and return_properties is not None and len(return_properties):
        log.debug(f'{name} requested properties: {return_properties}')
        properties = returns[1]
        realisation = index if use_index_as_realisation else None
        property_collection = rqp.PropertyCollection(support = gcs)
        grid_pc = None
        for p_name, array in properties.items():
            log.debug(f'{name} found property {p_name}')
            if p_name == "normal vector":
                property_collection.add_cached_array_to_imported_list(
                    array,
                    f"from find_faces function for {surface.title}",
                    f'{surface.title} {p_name}',
                    discrete = False,
                    uom = "Euc",
                    property_kind = "normal vector",
                    realization = realisation,
                    indexable_element = "faces",
                    points = True,
                )
            elif p_name == "triangle":
                property_collection.add_cached_array_to_imported_list(
                    array,
                    f"from find_faces function for {surface.title}",
                    f'{surface.title} {p_name}',
                    discrete = True,
                    null_value = -1,
                    property_kind = "triangle index",
                    realization = realisation,
                    indexable_element = "faces",
                )
            elif p_name == "offset":
                property_collection.add_cached_array_to_imported_list(
                    array,
                    f"from find_faces function for {surface.title}",
                    f'{surface.title} {p_name}',
                    discrete = False,
                    uom = grid.crs.z_units,
                    property_kind = "offset",
                    realization = realisation,
                    indexable_element = "faces",
                )
            elif p_name == "depth":
                # convert values to global z inc down
                array[:] += grid.crs.z_offset
                if not grid.crs.z_inc_down:
                    array = -array
                property_collection.add_cached_array_to_imported_list(
                    array,
                    f"from find_faces function for {surface.title}",
                    f'{surface.title} {p_name}',
                    discrete = False,
                    uom = grid.crs.z_units,
                    property_kind = "depth",
                    realization = realisation,
                    indexable_element = "faces",
                )
            elif p_name == 'grid bisector':
                array, is_curtain = array
                if grid_pc is None:
                    grid_pc = rqp.PropertyCollection()
                    grid_pc.set_support(support = grid)
                grid_pc.add_cached_array_to_imported_list(
                    array,
                    f"from find_faces function for {surface.title}",
                    f'{surface.title} {p_name}',
                    discrete = True,
                    property_kind = "grid bisector",
                    facet_type = 'direction',
                    facet = 'raw' if raw_bisector else ('vertical' if is_curtain else 'sloping'),
                    realization = realisation,
                    indexable_element = "columns" if is_curtain else "cells",
                )
            elif p_name == 'grid shadow':
                if grid_pc is None:
                    grid_pc = rqp.PropertyCollection()
                    grid_pc.set_support(support = grid)
                grid_pc.add_cached_array_to_imported_list(
                    array,
                    f"from find_faces function for {surface.title}",
                    f'{surface.title} {p_name}',
                    discrete = True,
                    property_kind = "grid shadow",
                    realization = realisation,
                    indexable_element = "cells",
                )
            elif p_name == 'flange bool':
                property_collection.add_cached_array_to_imported_list(
                    array,
                    f"from find_faces function for {surface.title}",
                    f'{surface.title} {p_name}',
                    discrete = True,
                    null_value = -1,
                    property_kind = "flange bool",
                    realization = realisation,
                    indexable_element = "faces",
                )
            else:
                raise ValueError(f'unrecognised property name {p_name}')
        if property_collection.number_of_imports() > 0:
            # log.debug('writing gcs property hdf5 data')
            property_collection.write_hdf5_for_imported_list()
            uuids_properties = property_collection.create_xml_for_imported_list_and_add_parts_to_model(
                find_local_property_kinds = True)
            uuid_list.extend(uuids_properties)
        if grid_pc is not None and grid_pc.number_of_imports() > 0:
            # log.debug('writing grid property (bisector) hdf5 data')
            grid_pc.write_hdf5_for_imported_list()
            # log.debug('creating xml for grid property (bisector)')
            uuids_properties = grid_pc.create_xml_for_imported_list_and_add_parts_to_model(
                find_local_property_kinds = True)
            assert uuids_properties
            uuid_list.extend(uuids_properties)
            if related_uuid is not None:
                for p_uuid in uuids_properties:
                    # log.debug(f'creating relationship between: {p_uuid} and {related_uuid}')
                    model.create_reciprocal_relationship_uuids(p_uuid, 'sourceObject', related_uuid,
                                                               'destinationObject')
    else:
        log.debug(f'{name} no requested properties')

    # log.debug('find_faces_to_represent_surface_regular_wrapper() storing epc: {model.epc_file}')
    model.store_epc(quiet = True)

    return index, success, epc_file, uuid_list


def inherit_interpretation_relationship(model, old_repr_uuid, new_repr_uuid):
    """Inherit a relationship to an interpretation object if present for old representation."""
    for obj_type in [
            'HorizonInterpretation', 'FaultInterpretation', 'BoundaryFeatureInterpretation',
            'GeobodyBoundaryInterpretation'
    ]:
        interp_uuid = model.uuid(obj_type = obj_type, related_uuid = old_repr_uuid)
        if interp_uuid is not None:
            model.create_reciprocal_relationship_uuids(new_repr_uuid, 'sourceObject', interp_uuid, 'destinationObject')
            break
