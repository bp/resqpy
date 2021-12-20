"""_gcs_functions.py: Functions for working with grid connection sets."""

version = '20th October 2021'

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import os
import numpy as np

import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo


def pinchout_connection_set(grid, skip_inactive = True, feature_name = 'pinchout'):
    """Returns a new GridConnectionSet representing non-standard K face connections across pinchouts.

    arguments:
       grid (grid.Grid): the grid for which a pinchout connection set is required
       skip_inactive (boolean, default True): if True, connections are not included where there is an inactive cell
          above or below the pinchout; if False, such connections are included
       feature_name (string, default 'pinchout'): the name to use as citation title in the feature and interpretation

    notes:
       this function does not write to hdf5, nor create xml for the new grid connection set;
       however, it does create one feature and a corresponding interpretation and creates xml for those
    """

    assert grid is not None

    po = grid.pinched_out()
    dead = grid.extract_inactive_mask() if skip_inactive else None

    cip_list = []  # cell index pair list

    for j in range(grid.nj):
        for i in range(grid.ni):
            ka = 0
            while True:
                while ka < grid.nk - 1 and po[ka, j, i]:
                    ka += 1
                while ka < grid.nk - 1 and not po[ka + 1, j, i]:
                    ka += 1
                if ka >= grid.nk - 1:
                    break
                # ka now in non-pinched out cell above pinchout
                if (skip_inactive and dead[ka, j, i]) or (grid.k_gaps and grid.k_gap_after_array[ka]):
                    ka += 1
                    continue
                kb = ka + 1
                while kb < grid.nk and po[kb, j, i]:
                    kb += 1
                if kb >= grid.nk:
                    break
                if skip_inactive and dead[kb, j, i]:
                    ka = kb + 1
                    continue
                # kb now beneath pinchout
                cip_list.append((grid.natural_cell_index((ka, j, i)), grid.natural_cell_index((kb, j, i))))
                ka = kb + 1

    log.debug(f'{len(cip_list)} pinchout connections found')

    pcs = _make_k_gcs_from_cip_list(grid, cip_list, feature_name)

    return pcs


def k_gap_connection_set(grid, skip_inactive = True, feature_name = 'k gap connection', tolerance = 0.001):
    """Returns a new GridConnectionSet representing K face connections where a K gap is zero thickness.

    arguments:
       grid (grid.Grid): the grid for which a K gap connection set is required
       skip_inactive (boolean, default True): if True, connections are not included where there is an inactive cell
          above or below the pinchout; if False, such connections are included
       feature_name (string, default 'pinchout'): the name to use as citation title in the feature and interpretation
       tolerance (float, default 0.001): the minimum vertical distance below which a K gap is deemed to be zero
          thickness; units are implicitly the z units of the coordinate reference system used by grid

    notes:
       this function does not write to hdf5, nor create xml for the new grid connection set;
       however, it does create one feature and a corresponding interpretation and creates xml for those;
       note that the entries in the connection set will be for logically K-neighbouring pairs of cells – such pairs
       are omitted from the standard transmissibilities due to the presence of the K gap layer
    """

    assert grid is not None
    if not grid.k_gaps:
        return None

    p = grid.points_ref(masked = False)
    dead = grid.extract_inactive_mask() if skip_inactive else None
    flip_z = (grid.k_direction_is_down != rqet.find_tag_bool(grid.crs_root, 'ZIncreasingDownward'))

    cip_list = []  # cell index pair list

    for k in range(grid.nk - 1):
        if not grid.k_gap_after_array[k]:
            continue
        k_gap_pillar_z = p[grid.k_raw_index_array[k + 1]][..., 2] - p[grid.k_raw_index_array[k] + 1][..., 2]
        if grid.has_split_coordinate_lines:
            pfc = grid.create_column_pillar_mapping()  # pillars for column
            k_gap_z = 0.25 * np.sum(k_gap_pillar_z[pfc], axis = (2, 3))  # resulting shape (nj, ni)
        else:
            k_gap_z = 0.25 * (k_gap_pillar_z[:-1, :-1] + k_gap_pillar_z[:-1, 1:] + k_gap_pillar_z[1:, :-1] +
                              k_gap_pillar_z[1:, 1:])  # shape (nj, ni)
        if flip_z:
            k_gap_z = -k_gap_z
        layer_mask = np.logical_and(np.logical_not(np.isnan(k_gap_z)), k_gap_z < tolerance)
        if skip_inactive:
            layer_mask = np.logical_and(layer_mask, np.logical_not(np.logical_or(dead[k], dead[k + 1])))
        # layer mask now boolean array of shape (nj, ni) set True where connection needed
        ji_list = np.stack(np.where(layer_mask)).T  # numpy array being list of [j, i] pairs
        for (j, i) in ji_list:
            cip_list.append((grid.natural_cell_index((k, j, i)), grid.natural_cell_index((k + 1, j, i))))

    log.debug(f'{len(cip_list)} k gap connections found')

    kgcs = _make_k_gcs_from_cip_list(grid, cip_list, feature_name)

    return kgcs


def add_connection_set_and_tmults(model, fault_incl, tmult_dict = None):
    """Add a grid connection set to a resqml model, based on a fault include file and a dictionary of fault:tmult pairs.

    Grid connection set added to resqml model, with extra_metadata on the fault interpretation containing the MULTFL values

    Args:
       model: resqml model object
       fault_incl: fullpath to fault include file or list of fullpaths to fault include files
       tmult_dict: dictionary of fault name/transmissibility multiplier pairs (must align with faults in include file).
          Optional, if blank values in the fault.include file will be used instead
    Returns:
       grid connection set uuid
    """

    from resqpy.fault._grid_connection_set import GridConnectionSet

    if isinstance(fault_incl, list):
        if len(fault_incl) > 1:
            # Making a concatenated version of the faultincl files
            # TODO: perhaps a better/more unique name and location could be used in future?
            temp_faults = os.path.join(os.path.dirname(model.epc_file), 'faults_combined_temp.txt')
            with open(temp_faults, 'w') as outfile:
                log.debug("combining multiple include files into one")
                for fname in fault_incl:
                    with open(fname) as infile:
                        for line in infile:
                            outfile.write(line)
        else:
            temp_faults = fault_incl[0]
    else:
        temp_faults = fault_incl

    log.info("Creating grid connection set")
    gcs = GridConnectionSet(parent_model = model,
                            ascii_load_format = 'nexus',
                            ascii_file = temp_faults,
                            create_organizing_objects_where_needed = True,
                            create_transmissibility_multiplier_property = True,
                            fault_tmult_dict = tmult_dict)
    gcs.write_hdf5(model.h5_file_name())
    gcs.create_xml(model.h5_uuid())

    model.store_epc()

    # clean up
    temp_combined_file = os.path.join(os.path.dirname(model.epc_file), 'faults_combined_temp.txt')
    if os.path.exists(temp_combined_file):
        os.remove(temp_combined_file)

    log.info("Grid connection set added")

    return gcs.uuid


# fault face table pandas dataframe functions
# these functions are for processing dataframes that have been read from (or to be written to) simulator ascii files


def zero_base_cell_indices_in_faces_df(faces, reverse = False):
    """Decrements all the cell indices in the fault face dataframe, in situ (or increments if reverse is True)."""

    if reverse:
        offset = 1
    else:
        offset = -1
    for col in ['i1', 'i2', 'j1', 'j2', 'k1', 'k2']:
        temp = faces[col] + offset
        faces[col] = temp


def standardize_face_indicator_in_faces_df(faces):
    """Sets face indicators to uppercase I, J or K, always with + or - following direction, in situ."""

    # todo: convert XYZ into IJK respectively?
    temp = faces['face'].copy().str.upper()
    temp[faces['face'].str.len() == 1] = faces['face'][faces['face'].str.len() == 1] + '+'
    for u in temp.unique():
        s = str(u)
        assert len(s) == 2, 'incorrect length to face string in fault dataframe: ' + s
        assert s[0] in 'IJK', 'unknown direction (axis) character in fault dataframe: ' + s
        assert s[1] in '+-', 'unknown face polarity character in fault dataframe: ' + s
    faces['face'] = temp


def remove_external_faces_from_faces_df(faces, extent_kji, remove_all_k_faces = False):
    """Returns a subset of the rows of faces dataframe, excluding rows on external faces."""

    # NB: assumes cell indices have been converted to zero based
    # NB: ignores grid column, ie. assumes extent_kji is applicable to all rows
    # NB: assumes single layer of cells is specified in the direction of the face
    filtered = []
    max_k0 = extent_kji[0] - 1
    max_j0 = extent_kji[1] - 1
    max_i0 = extent_kji[2] - 1
    for i in range(len(faces)):
        entry = faces.iloc[i]
        f = entry['face']
        if ((entry['i1'] <= 0 and f == 'I-') or (entry['j1'] <= 0 and f == 'J-') or (entry['k1'] <= 0 and f == 'K-') or
            (entry['i2'] >= max_i0 and f == 'I+') or (entry['j2'] >= max_j0 and f == 'J+') or
            (entry['k2'] >= max_k0 and f == 'K+')):
            continue
        if remove_all_k_faces and f[0] == 'K':
            continue
        filtered.append(i)
    return faces.loc[filtered]


def _make_k_gcs_from_cip_list(grid, cip_list, feature_name):
    # cip (cell index pair) list contains pairs of natural cell indices for which k connection is required
    # first of pair is layer above (lower k to be precise), second is below (higher k)
    # called by pinchout_connection_set() and k_gap_connection_set() functions

    from resqpy.fault._grid_connection_set import GridConnectionSet

    count = len(cip_list)

    if count == 0:
        return None

    pcs = GridConnectionSet(grid.model)
    pcs.grid_list = [grid]
    pcs.count = count
    pcs.grid_index_pairs = np.zeros((count, 2), dtype = int)
    pcs.cell_index_pairs = np.array(cip_list, dtype = int)
    pcs.face_index_pairs = np.zeros((count, 2), dtype = int)  # initialize to top faces
    pcs.face_index_pairs[:, 0] = 1  # bottom face of cells above pinchout

    pcs.feature_indices = np.zeros(count, dtype = int)  # could create seperate features by layer above or below?
    gbf = rqo.GeneticBoundaryFeature(grid.model, kind = 'horizon', feature_name = feature_name)
    gbf_root = gbf.create_xml()
    fi = rqo.HorizonInterpretation(grid.model, genetic_boundary_feature = gbf)
    fi_root = fi.create_xml(gbf_root, title_suffix = None)
    fi_uuid = rqet.uuid_for_part_root(fi_root)

    pcs.feature_list = [('obj_HorizonInterpretation', fi_uuid, str(feature_name))]

    return pcs
