"""Multiprocessing wrapper functions for the grid_surface module."""

import numpy as np
from typing import Tuple, Union, List, Optional, Callable
import resqpy.grid_surface as rqgs
from resqpy.model import new_model
from resqpy.grid import RegularGrid
from resqpy.surface import Surface
from resqpy.property import PropertyCollection
from pathlib import Path
from resqpy.model import Model
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
    centres: Optional[np.ndarray] = None,
    agitate: bool = False,
    feature_type = 'fault',
    progress_fn: Optional[Callable] = None,
    consistent_side: bool = False,
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
        surface_epc (str): epc file path where the surface is saved.
        surface_uuid (UUID/str): UUID (universally unique identifier) of the surface object.
        name (str): the feature name to use in the grid connection set.
        title (str): the citation title to use for the grid connection set; defaults to name
        centres (np.ndarray, shape (nk, nj, ni, 3)): precomputed cell centre points in
           local grid space, to avoid possible crs issues; required if grid's crs includes an origin (offset)?
        agitate (bool): if True, the points of the surface are perturbed by a small random
           offset, which can help if the surface has been built from a regular mesh with a periodic resonance
           with the grid
        feature_type (str, default 'fault'): one of 'fault', 'horizon', or 'geobody boundary'
        progress_fn (Callable): a callback function to be called at intervals by this function;
           the argument will progress from 0.0 to 1.0 in unspecified and uneven increments
        consistent_side (bool): if True, the cell pairs will be ordered so that all the first
           cells in each pair are on one side of the surface, and all the second cells on the other
        return_properties (List[str]): if present, a list of property arrays to calculate and
           return as a dictionary; recognised values in the list are 'triangle', 'offset' and 'normal vector';
           triangle is an index into the surface triangles of the triangle detected for the gcs face; offset
           is a measure of the distance between the centre of the cell face and the intersection point of the
           inter-cell centre vector with a triangle in the surface; normal vector is a unit vector normal
           to the surface triangle; each array has an entry for each face in the gcs; the returned dictionary
           has the passed strings as keys and numpy arrays as values.

    Returns:
        Tuple containing:

            - index (int): the index passed to the function.
            - success (bool): whether the function call was successful, whatever that
                definiton is.
            - epc_file (str): the epc file path where the objects are stored.
            - uuid_list (List[str]): list of UUIDs of relevant objects.
    """
    surface = Surface(parent_model = Model(surface_epc), uuid = str(surface_uuid))

    tmp_dir = Path(f"tmp_dir/{uuid.uuid4()}")
    tmp_dir.mkdir(parents = True, exist_ok = True)
    epc_file = f"{tmp_dir}/wrapper.epc"
    model = new_model(epc_file = epc_file)
    model.copy_uuid_from_other_model(Model(grid_epc), uuid = str(grid_uuid))
    model.copy_uuid_from_other_model(surface.model, uuid = str(surface_uuid))

    grid = RegularGrid(parent_model = model, uuid = str(grid_uuid))

    uuid_list = []
    uuid_list.extend([grid_uuid, surface_uuid])

    print("About to call function")

    returns = rqgs.find_faces_to_represent_surface_regular_optimised(
        grid,
        surface,
        name,
        title,
        centres,
        agitate,
        feature_type,
        progress_fn,
        consistent_side,
        return_properties,
    )

    print("Function returned")
    if return_properties is not None:
        gcs = returns[0]
        properties = returns[1]
        realisation = index if use_index_as_realisation else None
        property_collection = PropertyCollection(support = gcs)
        for name, array in properties.items():
            if name == "normal vector":
                property_collection.add_cached_array_to_imported_list(
                    array,
                    "from find_faces function",
                    name,
                    discrete = False,
                    uom = "Euc",
                    property_kind = "continuous",
                    realization = realisation,
                    indexable_element = "faces",
                    points = True,
                )
            elif name == "triangle":
                property_collection.add_cached_array_to_imported_list(
                    array,
                    "from find_faces function",
                    name,
                    discrete = True,
                    null_value = -1,
                    property_kind = "discrete",
                    realization = realisation,
                    indexable_element = "faces",
                )
            elif name == "offset":
                property_collection.add_cached_array_to_imported_list(
                    array,
                    "from find_faces function",
                    name,
                    discrete = False,
                    uom = grid.crs.z_units,
                    property_kind = "continuous",
                    realization = realisation,
                    indexable_element = "faces",
                )
        property_collection.write_hdf5_for_imported_list()
        uuids_properties = (property_collection.create_xml_for_imported_list_and_add_parts_to_model())
        uuid_list.extend(uuids_properties)
    else:
        gcs = returns

    success = False
    if gcs.count > 0:
        success = True

    gcs.write_hdf5()
    gcs.create_xml()
    model.copy_uuid_from_other_model(gcs.model, uuid = gcs.uuid)
    uuid_list.append(gcs.uuid)

    model.store_epc()

    return index, success, epc_file, uuid_list
