"""Multiprocessing wrapper functions for the grid_surface module."""

import logging

log = logging.getLogger(__name__)

import numpy as np
from typing import Tuple, Union, List, Optional, Callable
import resqpy.grid_surface as rqgs
from resqpy.model import new_model
from resqpy.grid import RegularGrid
from resqpy.surface import Surface, PointSet
from resqpy.property import PropertyCollection
from pathlib import Path
from resqpy.model import Model
import resqpy.olio.uuid as bu
from uuid import UUID
import uuid


def find_faces_to_represent_surface_regular_wrapper(
    index: int,
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
    extend_fault_representation: bool = False,
    related_uuid = None,
    progress_fn: Optional[Callable] = None,
    consistent_side: bool = False,
    extra_metadata = None,
    return_properties: Optional[List[str]] = None,
) -> Tuple[int, bool, str, List[Union[UUID, str]]]:
    """Wrapper function of find_faces_to_represent_surface_regular_optimised.

    Used for multiprocessing to create a new model that is saved in a temporary epc file
    and returns the required values, which are used in the multiprocessing function to
    recombine all the objects into a single epc file.

    Args:
        index (int): the index of the function call from the multiprocessing function.
        use_index_as_realisation (bool): if True, uses the index number as the realization number on
            the property collection.
        grid_epc (str): epc file path where the grid is saved.
        grid_uuid (UUID/str): UUID (universally unique identifier) of the grid object.
        surface_epc (str): epc file path where the surface (or point set) is saved.
        surface_uuid (UUID/str): UUID (universally unique identifier) of the surface (or point set) object.
        name (str): the feature name to use in the grid connection set.
        title (str): the citation title to use for the grid connection set; defaults to name
        agitate (bool): if True, the points of the surface are perturbed by a small random
           offset, which can help if the surface has been built from a regular mesh with a periodic resonance
           with the grid
        feature_type (str, default 'fault'): one of 'fault', 'horizon', or 'geobody boundary'
        trimmed (bool, default True): if True the surface has already been trimmed
        extend_fault_representation (bool, default False): if True, the representation is extended with a flange
        related_uuid (uuid, optional): if present, the uuid of an object to be softly related to the gcs
        progress_fn (Callable): a callback function to be called at intervals by this function;
           the argument will progress from 0.0 to 1.0 in unspecified and uneven increments
        consistent_side (bool): if True, the cell pairs will be ordered so that all the first
           cells in each pair are on one side of the surface, and all the second cells on the other
        extra_metadata (dict, optional): extra metadata items to be added to the grid connection set
        return_properties (List[str]): if present, a list of property arrays to calculate and
           return as a dictionary; recognised values in the list are 'triangle', 'depth', 'offset' and 'normal vector';
           triangle is an index into the surface triangles of the triangle detected for the gcs face; depth is
           the z value of the intersection point of the inter-cell centre vector with a triangle in the surface;
           offset is a measure of the distance between the centre of the cell face and the intersection point;
           normal vector is a unit vector normal to the surface triangle; each array has an entry for each face
           in the gcs; the returned dictionary has the passed strings as keys and numpy arrays as values.

    Returns:
        Tuple containing:

            - index (int): the index passed to the function.
            - success (bool): whether the function call was successful, whatever that
                definiton is.
            - epc_file (str): the epc file path where the objects are stored.
            - uuid_list (List[str]): list of UUIDs of relevant objects.
    """
    tmp_dir = Path(f"tmp_dir/{uuid.uuid4()}")
    tmp_dir.mkdir(parents = True, exist_ok = True)
    epc_file = f"{tmp_dir}/wrapper.epc"
    model = new_model(epc_file = epc_file)
    uuid_list = []
    g_model = Model(grid_epc)
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
    grid = RegularGrid(parent_model = model, uuid = grid_uuid)
    s_model = Model(surface_epc)
    model.copy_uuid_from_other_model(s_model, uuid = str(surface_uuid))
    repr_type = model.type_of_part(model.part(uuid = surface_uuid), strip_obj = True)
    assert repr_type in ['TriangulatedSetRepresentation', 'PointSetRepresentation']
    extended = False
    if repr_type == 'PointSetRepresentation':
        # trim pointset to grid xyz box
        pset = PointSet(model, uuid = surface_uuid)
        log.debug(f'point set {pset.title} raw point count: {len(pset.full_array_ref())}')
        pset.change_crs(grid.crs)
        if not trimmed:
            pset.trim_to_xyz_box(grid.xyz_box(local = True))
            trimmed = True
        assert len(pset.full_array_ref()) >= 3,  \
            f'boundary {name} representation {pset.title} has no xyz overlap with grid'
        pset_points = pset.full_array_ref()
        log.debug(f'trimmed point set {pset.title} contains {len(pset_points)} points; about to triangulate')
        if len(pset_points) > 1000:
            log.warning(
                f'trimmed point set {pset.title} has {len(pset_points)} points, which might take a while to triangulate'
            )
        # triangulate point set to form a surface; set repr_uuid to that surface and switch repr_flavour to 'surface'
        surf = Surface(model, crs_uuid = grid.crs.uuid, title = pset.title)
        surf.set_from_point_set(pset,
                                convexity_parameter = 2.0,
                                reorient = True,
                                extend_with_flange = extend_fault_representation,
                                make_clockwise = False)
        extended = extend_fault_representation
        surf.write_hdf5()
        surf.create_xml()
        surface_uuid = surf.uuid
    surface = Surface(parent_model = model, uuid = str(surface_uuid))
    surface.change_crs(grid.crs)
    if not trimmed and surface.triangle_count() > 100:
        trimmed_surf = Surface(model, crs_uuid = grid.crs.uuid)
        # trimmed_surf.set_to_trimmed_surface(surf, xyz_box = xyz_box, xy_polygon = parent_seg.polygon)
        trimmed_surf.set_to_trimmed_surface(surface, xyz_box = grid.xyz_box(local = True))
        surface = trimmed_surf
        trimmed = True
    if extend_fault_representation and not extended:
        _, p = surface.triangles_and_points()
        pset = PointSet(model, points_array = p, crs_uuid = grid.crs.uuid, title = surface.title)
        surface = Surface(model, crs_uuid = grid.crs.uuid, title = pset.title)
        surface.set_from_point_set(pset,
                                   convexity_parameter = 2.0,
                                   reorient = True,
                                   extend_with_flange = True,
                                   make_clockwise = False)
        extended = True
    if not bu.matching_uuids(surface.uuid, surface_uuid):
        surface.write_hdf5()
        surface.create_xml()
        surface_uuid = surface.uuid
    uuid_list.append(surface_uuid)

    returns = rqgs.find_faces_to_represent_surface_regular_optimised(
        grid,
        surface,
        name,
        title,
        None,  # centres
        agitate,
        feature_type,
        progress_fn,
        consistent_side,
        return_properties,
    )

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
            model.create_reciprocal_relationship_uuids(gcs.uuid, 'sourceObject', related_uuid, 'destinationObject')
        uuid_list.append(gcs.uuid)

    if success and return_properties is not None and len(return_properties):
        log.debug(f'{name} requested properties: {return_properties}')
        properties = returns[1]
        realisation = index if use_index_as_realisation else None
        property_collection = PropertyCollection(support = gcs)
        for p_name, array in properties.items():
            log.debug(f'{name} found property {p_name}')
            if p_name == "normal vector":
                property_collection.add_cached_array_to_imported_list(
                    array,
                    "from find_faces function",
                    p_name,
                    discrete = False,
                    uom = "Euc",
                    property_kind = "continuous",
                    realization = realisation,
                    indexable_element = "faces",
                    points = True,
                )
            elif p_name == "triangle":
                property_collection.add_cached_array_to_imported_list(
                    array,
                    "from find_faces function",
                    p_name,
                    discrete = True,
                    null_value = -1,
                    property_kind = "discrete",
                    realization = realisation,
                    indexable_element = "faces",
                )
            elif p_name == "offset":
                property_collection.add_cached_array_to_imported_list(
                    array,
                    "from find_faces function",
                    p_name,
                    discrete = False,
                    uom = grid.crs.z_units,
                    property_kind = "continuous",
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
                    "from find_faces function",
                    p_name,
                    discrete = False,
                    uom = grid.crs.z_units,
                    property_kind = "depth",
                    realization = realisation,
                    indexable_element = "faces",
                )
        property_collection.write_hdf5_for_imported_list()
        uuids_properties = (property_collection.create_xml_for_imported_list_and_add_parts_to_model())
        uuid_list.extend(uuids_properties)
    else:
        log.debug(f'{name} no requested properties')

    model.store_epc()

    return index, success, epc_file, uuid_list
