"""Submodule containing functions for supporting properties for a property collection."""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import resqpy.olio.uuid as bu


def _set_support_model(collection, model, support):
    if model is None and support is not None:
        model = support.model
    if model is None:
        model = collection.model
    else:
        collection.model = model
    return model


def _set_support_uuid_none(collection):
    if collection.support_uuid is not None:
        log.warning('clearing supporting representation for property collection')
    collection.support = None
    collection.support_root = None
    collection.support_uuid = None


def _set_support_uuid_notnone(collection, support, support_uuid, model, modify_parts):
    # when at global level was causing circular reference loading issues as grid imports this module
    import resqpy.fault as rqf
    import resqpy.grid as grr
    import resqpy.surface as rqs
    import resqpy.unstructured as rug
    import resqpy.well as rqw

    assert model is not None, 'model not established when setting support for property collection'
    if collection.support_uuid is not None and not bu.matching_uuids(support_uuid, collection.support_uuid):
        log.warning('changing supporting representation for property collection')
    collection.support_uuid = support_uuid
    collection.support = support
    if collection.support is None:
        _set_support_uuid_notnone_supportnone(collection, support_uuid, model)
    else:
        if type(collection.support) in [
                grr.Grid, grr.RegularGrid, rqw.WellboreFrame, rqw.BlockedWell, rqs.Mesh, rqf.GridConnectionSet,
                rug.UnstructuredGrid, rug.HexaGrid, rug.TetraGrid, rug.PrismGrid, rug.VerticalPrismGrid,
                rug.PyramidGrid, rqw.WellboreMarkerFrame, rqs.Surface
        ]:
            collection.support_root = collection.support.root
        else:
            raise TypeError('unsupported property supporting representation class: ' + str(type(collection.support)))
    if modify_parts:
        for (part, info) in collection.dict.items():
            if info[1] is not None:
                modified = list(info)
                modified[1] = support_uuid
                collection.dict[part] = tuple(modified)


def _set_support_uuid_notnone_supportnone(collection, support_uuid, model):
    import resqpy.fault as rqf
    import resqpy.grid as grr
    import resqpy.surface as rqs
    import resqpy.unstructured as rug
    import resqpy.well as rqw

    support_part = model.part_for_uuid(support_uuid)
    assert support_part is not None, 'supporting representation part missing in model'
    collection.support_root = model.root_for_part(support_part)
    support_type = model.type_of_part(support_part)
    assert support_type is not None
    if support_type == 'obj_IjkGridRepresentation':
        collection.support = grr.any_grid(model, uuid = collection.support_uuid, find_properties = False)
    elif support_type == 'obj_WellboreFrameRepresentation':
        collection.support = rqw.WellboreFrame(model, uuid = collection.support_uuid)
    elif support_type == 'obj_BlockedWellboreRepresentation':
        collection.support = rqw.BlockedWell(model, uuid = collection.support_uuid)
    elif support_type == 'obj_Grid2dRepresentation':
        collection.support = rqs.Mesh(model, uuid = collection.support_uuid)
    elif support_type == 'obj_GridConnectionSetRepresentation':
        collection.support = rqf.GridConnectionSet(model, uuid = collection.support_uuid)
    elif support_type == 'obj_TriangulatedSetRepresentation':
        collection.support = rqs.Surface(model, uuid = collection.support_uuid)
    elif support_type == 'obj_UnstructuredGridRepresentation':
        collection.support = rug.UnstructuredGrid(model,
                                                  uuid = collection.support_uuid,
                                                  geometry_required = False,
                                                  find_properties = False)
    elif support_type == 'obj_WellboreMarkerFrameRepresentation':
        collection.support = rqw.WellboreMarkerFrame(model, uuid = collection.support_uuid)
    else:
        raise TypeError('unsupported property supporting representation class: ' + str(support_type))


def _set_support_and_model_from_collection(collection, other, support_uuid, grid):
    _confirm_support_and_model_from_collection(collection, support_uuid, grid, other)

    assert collection.support_uuid is None or other.support_uuid is None or bu.matching_uuids(
        collection.support_uuid, other.support_uuid)
    if collection.support_uuid is None and collection.number_of_parts() == 0:
        collection.set_support(support_uuid = other.support_uuid, support = other.support)


def _confirm_support_and_model_from_collection(collection, support_uuid, grid, other):
    if support_uuid is None and grid is not None:
        support_uuid = grid.uuid
    if support_uuid is not None and collection.support_uuid is not None:
        assert bu.matching_uuids(support_uuid, collection.support_uuid)

    assert other is not None
    if collection.model is None:
        collection.model = other.model
    else:
        assert collection.model is other.model
