"""well_utils.py: functions used by the classes in resqpy.well"""

# Nexus is a registered trademark of the Halliburton Company
# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.olio.grid_functions as gf
import resqpy.olio.intersection as intersect
import resqpy.olio.keyword_files as kf
import resqpy.olio.xml_et as rqet


def load_hdf5_array(object, node, array_attribute, tag = 'Values', dtype = 'float', model = None):
    """Loads the property array data as an attribute of object, from the hdf5 referenced in xml node.

    :meta private:
    """

    assert (rqet.node_type(node) in ['DoubleHdf5Array', 'IntegerHdf5Array', 'Point3dHdf5Array'])
    if model is None:
        model = object.model
    h5_key_pair = model.h5_uuid_and_path_for_node(node, tag = tag)
    if h5_key_pair is None:
        return None
    return model.h5_array_element(h5_key_pair,
                                  index = None,
                                  cache_array = True,
                                  dtype = dtype,
                                  object = object,
                                  array_attribute = array_attribute)


def extract_xyz(xyz_node):
    """Extracts an x,y,z coordinate from a solitary point xml node.

    argument:
        xyz_node: the xml node representing the solitary point (in 3D space)

    returns:
        triple float: (x, y, z) coordinates as a tuple
    """

    if xyz_node is None:
        return None
    xyz = np.zeros(3)
    for axis in range(3):
        xyz[axis] = rqet.find_tag_float(xyz_node, 'Coordinate' + str(axis + 1), must_exist = True)
    return tuple(xyz)


def well_names_in_cellio_file(cellio_file):
    """Returns a list of well names as found in the RMS blocked well export cell I/O file."""

    well_list = []
    with open(cellio_file, 'r') as fp:
        while True:
            kf.skip_blank_lines_and_comments(fp)
            line = fp.readline()  # file format version number?
            if line == '':
                break  # end of file
            fp.readline()  # 'Undefined'
            words = fp.readline().split()
            assert len(words), 'missing header info (well name) in cell I/O file'
            well_list.append(words[0])
            while not kf.blank_line(fp):
                fp.readline()  # skip to block of data for next well
    return well_list


# 'private' functions


def find_entry_and_exit(cp, entry_vector, exit_vector, well_name):
    """Returns (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz).

    :meta private:
    """

    cell_centre = np.mean(cp, axis = (0, 1, 2))
    face_triangles = gf.triangles_for_cell_faces(cp).reshape(-1, 3, 3)  # flattened first index 4 values per face
    entry_points = intersect.line_triangles_intersects(cell_centre, entry_vector, face_triangles, line_segment = True)
    entry_axis = entry_polarity = entry_xyz = exit_xyz = None
    for t in range(24):
        if not np.any(np.isnan(entry_points[t])):
            entry_xyz = entry_points[t]
            entry_axis = t // 8
            entry_polarity = (t - 8 * entry_axis) // 4
            break
    assert entry_axis is not None, 'failed to find entry face for a perforation in well ' + str(well_name)
    exit_points = intersect.line_triangles_intersects(cell_centre, exit_vector, face_triangles, line_segment = True)
    exit_axis = exit_polarity = None
    for t in range(24):
        if not np.any(np.isnan(exit_points[t])):
            exit_xyz = exit_points[t]
            exit_axis = t // 8
            exit_polarity = (t - 8 * exit_axis) // 4
            break
    assert exit_axis is not None, 'failed to find exit face for a perforation in well ' + str(well_name)

    return (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz)


def _as_optional_array(arr):
    """If not None, cast as numpy array.

    note:
        casting directly to an array can be problematic: np.array(None) creates an unsized array,
        which is potentially confusing
    """
    if arr is None:
        return None
    else:
        return np.array(arr)


def _pl(i, e = False):
    return '' if i == 1 else 'es' if e else 's'


def _derive_from_wellspec_verify_col_list(add_properties):
    """Verify additional properties to be added to the WELLSPEC file.

    argument:
       add_properties (boolean): if True, the additional properties specified will be added to the WELLSPEC file

    returns:
       list of columns to be added to the WELLSPEC file
    """

    if add_properties:
        if isinstance(add_properties, list):
            col_list = ['IW', 'JW', 'L'] + [col.upper() for col in add_properties if col not in ['IW', 'JW', 'L']]
        else:
            col_list = []
    else:
        col_list = ['IW', 'JW', 'L', 'ANGLA', 'ANGLV']
    return col_list


def _derive_from_wellspec_check_grid_name(check_grid_name, grid, col_list):
    """Verify the grid object to which the cell indices in the WELLSPEC table belong.

    arguments:
       check_grid_name (boolean): if True, the citation title of the grid will be extracted and returned
       grid (grid object): the grid object whose citation titles will be returned
       col_list (list): list of strings of column names to be added to the WELLSPEC file; if a citation title is
           extracted from the grid object, 'GRID' will be added to the col_list

    returns:
       string of grid citation title extracted from the grid object
       list of columns to be added to the WELLSPEC file
    """

    if check_grid_name:
        grid_name = rqet.citation_title_for_node(grid.root).upper()
        if not grid_name:
            name_for_check = None
        else:
            col_list.append('GRID')
            name_for_check = grid_name
    else:
        name_for_check = None
    return name_for_check, col_list
