"""functions to handle different grid types"""

import warnings

import resqpy.grid as grr
import resqpy.unstructured as rug
import resqpy.olio.xml_et as rqet


def grid_flavour(grid_root):
    """Returns a string indicating type of grid geometry, currently 'IjkGrid' or 'IjkBlockGrid'."""

    if grid_root is None:
        return None
    em = rqet.load_metadata_from_xml(grid_root)
    flavour = em.get('grid_flavour')
    if flavour is None:
        node_type = rqet.node_type(grid_root, strip_obj = True)
        if node_type == 'IjkGridRepresentation':
            geom = rqet.find_tag(grid_root, 'Geometry')
            if geom is None:
                flavour = 'IjkBlockGrid'
            else:
                p_node = rqet.find_tag(geom, 'Points')
                assert p_node is not None
                p_type = rqet.node_type(p_node)
                if p_type == 'Point3dLatticeArray':
                    flavour = 'IjkBlockGrid'
                elif p_type == 'Point3dHdf5Array':
                    flavour = 'IjkGrid'
                else:
                    raise ValueError(f'grid geometry points type not supported: {p_type}')
        elif node_type == 'UnstructuredGridRepresentation':
            cell_shape = rqet.find_nested_tags_text(grid_root, ['Geometry', 'CellShape'])
            if cell_shape is None or cell_shape == 'polyhedral':
                flavour = 'UnstructuredGrid'
            elif cell_shape == 'tetrahedral':
                flavour = 'TetraGrid'
            elif cell_shape == 'hexahedral':
                flavour = 'HexaGrid'
            elif cell_shape == 'pyramidal':
                flavour = 'PyramidGrid'
            elif cell_shape == 'prism':
                flavour = 'PrismGrid'
    return flavour


def is_regular_grid(grid_root):
    """Returns True if the xml root node is for a RegularGrid."""

    return grid_flavour(grid_root) == 'IjkBlockGrid'


def any_grid(parent_model, uuid = None, find_properties = True):
    """Returns a Grid or RegularGrid or UnstructuredGrid object depending on the extra metadata in the xml.

    arguments:
        parent_model (Model): the model within which the grid exists
        uuid (UUID): the uuid of the grid object to be instantiated
        find_properties (bool, default True): passed onward to the instantiation method

    note:
        full list of resqpy grid class objects which could be returned:
        Grid, RegularGrid, UnstructuredGrid, TetraGrid, HexaGrid, PyramidGrid, PrismGrid
    """

    assert uuid is not None

    flavour = grid_flavour(parent_model.root_for_uuid(uuid))
    if flavour is None:
        return None
    if flavour == 'IjkGrid':
        return grr.Grid(parent_model, uuid = uuid, find_properties = find_properties)
    if flavour == 'IjkBlockGrid':
        return grr.RegularGrid(parent_model, extent_kji = None, uuid = uuid, find_properties = find_properties)
    if flavour == 'UnstructuredGrid':
        return rug.UnstructuredGrid(parent_model, uuid = uuid, find_properties = find_properties)
    if flavour == 'TetraGrid':
        return rug.TetraGrid(parent_model, uuid = uuid, find_properties = find_properties)
    if flavour == 'HexaGrid':
        return rug.HexaGrid(parent_model, uuid = uuid, find_properties = find_properties)
    if flavour == 'PyramidGrid':
        return rug.PyramidGrid(parent_model, uuid = uuid, find_properties = find_properties)
    if flavour == 'PrismGrid':
        return rug.PrismGrid(parent_model, uuid = uuid, find_properties = find_properties)
    return None
