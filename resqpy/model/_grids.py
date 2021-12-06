"""_grids.py: functions supporting Model methods relating to grid objects."""

import resqpy.grid as grr
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet


def _root_for_ijk_grid(model, uuid = None, title = None):
    """Return xml root for IJK Grid part."""

    if title is not None:
        title = title.strip().upper()
    if uuid is None and not title:
        grid_root = model.root(obj_type = 'IjkGridRepresentation', title = 'ROOT', multiple_handling = 'oldest')
        if grid_root is None:
            grid_root = model.root(obj_type = 'IjkGridRepresentation')
    else:
        grid_root = model.root(obj_type = 'IjkGridRepresentation', uuid = uuid, title = title)

    assert grid_root is not None, 'IJK Grid part not found'

    return grid_root


def _resolve_grid_root(model, grid_root = None, uuid = None):
    """If grid root argument is None, returns the xml root for the IJK Grid part instead."""

    if grid_root is not None:
        if model.grid_root is None:
            model.grid_root = grid_root
    else:
        if model.grid_root is None:
            model.grid_root = _root_for_ijk_grid(model, uuid = uuid)
        grid_root = model.grid_root
    return grid_root


def _grid(model, title = None, uuid = None, find_properties = True):
    """Returns a shared Grid (or RegularGrid) object for this model, by default the 'main' grid."""

    if uuid is None and (title is None or title.upper() == 'ROOT'):
        if model.main_grid is not None:
            if find_properties:
                model.main_grid.extract_property_collection()
            return model.main_grid
        if title is None:
            grid_root = _resolve_grid_root(model)
        else:
            grid_root = _resolve_grid_root(model,
                                           grid_root = model.root(obj_type = 'IjkGridRepresentation', title = title))
    else:
        grid_root = model.root(obj_type = 'IjkGridRepresentation', uuid = uuid, title = title)
    assert grid_root is not None, 'IJK Grid part not found'
    if uuid is None:
        uuid = rqet.uuid_for_part_root(grid_root)
    for grid in model.grid_list:
        if grid.root is grid_root:
            if find_properties:
                grid.extract_property_collection()
            return grid
    grid = grr.any_grid(model, uuid = uuid, find_properties = find_properties)
    assert grid is not None, 'failed to instantiate grid object'
    if find_properties:
        grid.extract_property_collection()
    _add_grid(model, grid)
    return grid


def _add_grid(model, grid_object, check_for_duplicates = False):
    """Add grid object to list of shareable grids for this model."""

    if check_for_duplicates:
        for g in model.grid_list:
            if bu.matching_uuids(g.uuid, grid_object.uuid):
                return
    model.grid_list.append(grid_object)


def _grid_list_uuid_list(model):
    """Returns list of uuid's for the grid objects in the cached grid list."""

    uuid_list = []
    for grid in model.grid_list:
        uuid_list.append(grid.uuid)
    return uuid_list


def _grid_for_uuid_from_grid_list(model, uuid):
    """Returns the cached grid object matching the given uuid, if found in the grid list, otherwise None."""

    for grid in model.grid_list:
        if bu.matching_uuids(uuid, grid.uuid):
            return grid
    return None
