"""well_utils.py: functions used by the classes in resqpy.well.

"""

# todo: create a trajectory from a deviation survey, assuming minimum curvature

version = '10th November 2021'

# Nexus is a registered trademark of the Halliburton Company
# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)
log.debug('well_utils.py version ' + version)

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
            exit_xyz = entry_points[t]
            exit_axis = t // 8
            exit_polarity = (t - 8 * exit_axis) // 4
            break
    assert exit_axis is not None, 'failed to find exit face for a perforation in well ' + str(well_name)

    return (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz)


def _as_optional_array(arr):
    """If not None, cast as numpy array.

    Casting directly to an array can be problematic: np.array(None) creates an unsized array, which is potentially
    confusing.
    """
    if arr is None:
        return None
    else:
        return np.array(arr)


def _pl(i, e = False):
    return '' if i == 1 else 'es' if e else 's'
