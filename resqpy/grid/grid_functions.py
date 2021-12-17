import resqpy.olio.xml_et as rqet
import resqpy.property as rprop
import resqpy
import numpy as np
import resqpy.olio.point_inclusion as pip

# 'private' function


def _add_to_kelp_list(extent_kji, kelp_list, face_axis, ji):
    if isinstance(face_axis, bool):
        face_axis = 'J' if face_axis else 'I'
    # ignore external faces
    if face_axis == 'J':
        if ji[0] < 0 or ji[0] >= extent_kji[1] - 1:
            return
    elif face_axis == 'I':
        if ji[1] < 0 or ji[1] >= extent_kji[2] - 1:
            return
    else:  # ji is actually kj or ki
        assert face_axis == 'K'
        if ji[0] < 0 or ji[0] >= extent_kji[0] - 1:
            return
    pair = ji
    if pair in kelp_list:
        return  # avoid duplication
    kelp_list.append(pair)


def establish_zone_property_kind(model):
    """Returns zone local property kind object, creating the xml and adding as part if not found in model."""

    zone_pk_uuid = model.uuid(obj_type = 'LocalPropertyKind', title = 'zone')
    if zone_pk_uuid is None:
        zone_pk = rprop.PropertyKind(model, title = 'zone', parent_property_kind = 'discrete')
        zone_pk.create_xml()
    else:
        zone_pk = rprop.PropertyKind(model, uuid = zone_pk_uuid)
    return zone_pk


def extent_kji_from_root(root_node):
    """Returns kji extent as stored in xml."""

    return (rqet.find_tag_int(root_node, 'Nk'), rqet.find_tag_int(root_node, 'Nj'), rqet.find_tag_int(root_node, 'Ni'))


def grid_flavour(grid_root):
    """Returns a string indicating type of grid geometry, currently 'IjkGrid' or 'IjkBlockGrid'."""

    if grid_root is None:
        return None
    em = rqet.load_metadata_from_xml(grid_root)
    flavour = em.get('grid_flavour')
    if flavour is None:
        node_type = rqet.node_type(grid_root, strip_obj = True)
        if node_type == 'IjkGridRepresentation':
            if rqet.find_tag(grid_root, 'Geometry') is not None:
                flavour = 'IjkGrid'
            else:
                flavour = 'IjkBlockGrid'  # this might cause issues
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


def any_grid(parent_model, grid_root = None, uuid = None, find_properties = True):
    """Returns a Grid or RegularGrid or UnstructuredGrid object depending on the extra metadata in the xml."""

    import resqpy.unstructured as rug

    if uuid is None and grid_root is not None:
        uuid = rqet.uuid_for_part_root(grid_root)
    flavour = grid_flavour(parent_model.root_for_uuid(uuid))
    if flavour is None:
        return None
    if flavour == 'IjkGrid':
        return resqpy.grid.Grid(parent_model, uuid = uuid, find_properties = find_properties)
    if flavour == 'IjkBlockGrid':
        return resqpy.grid.RegularGrid(parent_model, extent_kji = None, uuid = uuid, find_properties = find_properties)
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


def find_cell_for_x_sect_xz(x_sect, x, z):
    """Returns the (k0, j0) or (k0, i0) indices of the cell containing point x,z in the cross section.

    arguments:
       x_sect (numpy float array of shape (nk, nj or ni, 2, 2, 2 or 3): the cross section x,z or x,y,z data
       x (float) x-coordinate of point of interest in the cross section space
       z (float): y-coordinate of  point of interest in the cross section space

    note:
       the x_sect data is in the form returned by x_section_corner_points() or split_gap_x_section_points();
       the 2nd of the returned pair is either a J index or I index, whichever was not the axis specified
       when generating the x_sect data; returns (None, None) if point inclusion not detected; if xyz data is
       provided, the y values are ignored; note that the point of interest x,z coordinates are in the space of
       x_sect, so if rotation has occurred, the x value is no longer an easting and is typically picked off a
       cross section plot
    """

    def test_cell(p, x_sect, k0, ji0):
        poly = np.array([
            x_sect[k0, ji0, 0, 0, 0:3:2], x_sect[k0, ji0, 0, 1, 0:3:2], x_sect[k0, ji0, 1, 1, 0:3:2], x_sect[k0, ji0, 1,
                                                                                                             0, 0:3:2]
        ])
        if np.any(np.isnan(poly)):
            return False
        return pip.pip_cn(p, poly)

    assert x_sect.ndim == 5 and x_sect.shape[2] == 2 and x_sect.shape[3] == 2 and 2 <= x_sect.shape[4] <= 3
    n_k = x_sect.shape[0]
    n_j_or_i = x_sect.shape[1]
    tolerance = 1.0e-3

    if x_sect.shape[4] == 3:
        diffs = x_sect[:, :, :, :, 0:3:2].copy()  # x,z points only
    else:
        diffs = x_sect.copy()
    diffs -= np.array((x, z))
    diffs = np.sum(diffs * diffs, axis = -1)  # square of distance of each point from given x,z
    flat_index = np.nanargmin(diffs)
    min_dist_sqr = diffs.flatten()[flat_index]
    cell_flat_k0_ji0, flat_k_ji_p = divmod(flat_index, 4)
    found_k0, found_ji0 = divmod(cell_flat_k0_ji0, n_j_or_i)
    found_kp, found_jip = divmod(flat_k_ji_p, 2)

    found = test_cell((x, z), x_sect, found_k0, found_ji0)
    if found:
        return found_k0, found_ji0
    # check cells below whilst still close to point
    while found_k0 < n_k - 1:
        found_k0 += 1
        if np.nanmin(diffs[found_k0, found_ji0]) > min_dist_sqr + tolerance:
            break
        found = test_cell((x, z), x_sect, found_k0, found_ji0)
        if found:
            return found_k0, found_ji0

    # try neighbouring column (in case of fault or point exactly on face)
    ji_neighbour = 1 if found_jip == 1 else -1
    found_ji0 += ji_neighbour
    if 0 <= found_ji0 < n_j_or_i:
        col_diffs = diffs[:, found_ji0]
        flat_index = np.nanargmin(col_diffs)
        if col_diffs.flatten()[flat_index] <= min_dist_sqr + tolerance:
            found_k0 = flat_index // 4
            found = test_cell((x, z), x_sect, found_k0, found_ji0)
            if found:
                return found_k0, found_ji0
            # check cells below whilst still close to point
            while found_k0 < n_k - 1:
                found_k0 += 1
                if np.nanmin(diffs[found_k0, found_ji0]) > min_dist_sqr + tolerance:
                    break
                found = test_cell((x, z), x_sect, found_k0, found_ji0)
                if found:
                    return found_k0, found_ji0

    return None, None
