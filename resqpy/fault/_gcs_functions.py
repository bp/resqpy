"""_gcs_functions.py: Functions for working with grid connection sets."""

version = '20th October 2021'

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import os
import numpy as np

import resqpy.fault as rqf
import resqpy.grid as grr
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo
import resqpy.property as rqp


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

    grid.set_crs()
    flip_z = (grid.k_direction_is_down != grid.crs.z_inc_down)

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


def cell_set_skin_connection_set(grid,
                                 cell_set_mask,
                                 feature_name,
                                 feature_type = 'geobody boundary',
                                 title = None,
                                 create_organizing_objects_where_needed = True):
    """Add a grid connection set containing external faces of selected set of cells.

    arguments:
        grid (Grid): the grid for which the connection set is required
        cell_set_mask (numpy bool array of shape grid.extent_kji): True values identify cells included in the set
        feature_name (str): the name of the skin feature
        feature_type (str, default 'geobody boundary'): 'fault', 'horizon' or 'geobody boundary'
        title (str, optional): the citation title to use for the gcs; defaults to the feature_name
        create_organizing_objects_where_needed (bool, default True): if True, feature and interpretation
            objects will be created if they do not exist

    returns:
        the newly created grid connection set

    notes:
        this function does not take into consideration split pillars, it assumes cells are neighbouring based
        on the cell indices; faces on the outer skin of the grid are not included in the connection set;
        any cell face between a cell in the cell set and one not in it will be included in the connection set,
        therefore the set may contain internal skin faces as well as the outer skin
    """

    from resqpy.fault._grid_connection_set import GridConnectionSet

    assert grid is not None
    assert cell_set_mask.shape == tuple(grid.extent_kji)
    assert feature_name, 'no feature name given for cell set skin connection set'
    assert feature_type in ['geobody boundary', 'fault', 'horizon'], f'invalid feature type: {feature_type}'
    if not title:
        title = feature_name

    k_faces = np.zeros((grid.nk - 1, grid.nj, grid.ni), dtype = bool)
    j_faces = np.zeros((grid.nk, grid.nj - 1, grid.ni), dtype = bool)
    i_faces = np.zeros((grid.nk, grid.nj, grid.ni - 1), dtype = bool)

    if grid.nk > 1:
        k_faces[np.where(cell_set_mask[1:, :, :] != cell_set_mask[:-1, :, :])] = True
    if grid.nj > 1:
        j_faces[np.where(cell_set_mask[:, 1:, :] != cell_set_mask[:, :-1, :])] = True
    if grid.ni > 1:
        i_faces[np.where(cell_set_mask[:, :, 1:] != cell_set_mask[:, :, :-1])] = True

    gcs = GridConnectionSet(grid.model,
                            grid = grid,
                            k_faces = k_faces,
                            j_faces = j_faces,
                            i_faces = i_faces,
                            feature_name = feature_name,
                            feature_type = feature_type,
                            create_organizing_objects_where_needed = create_organizing_objects_where_needed,
                            title = title)

    return gcs


def add_connection_set_and_tmults(model, fault_incl, tmult_dict = None):
    """Add a grid connection set to a resqml model, based on a fault include file and a dictionary of fault:tmult pairs.

    Grid connection set added to resqml model, with extra_metadata on the fault interpretation containing the MULTFL values

    arguments:
       model: resqml model object
       fault_incl: fullpath to fault include file or list of fullpaths to fault include files
       tmult_dict: dictionary of fault name/transmissibility multiplier pairs (must align with faults in include file).
          Optional, if blank values in the fault.include file will be used instead
    returns:
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
                            outfile.write(line.rstrip() + '\n')
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


def grid_columns_property_from_gcs_property(model,
                                            gcs_property_uuid,
                                            null_value = np.nan,
                                            title = None,
                                            multiple_handling = 'default'):
    """Derives a new grid columns property (map) from a single-grid gcs property using values for k faces.

    arguments:
        model (Model): the model in which the existing objects are to be found and the new property added
        gcs_property_uuid (UUID): the uuid of the existing grid connection set property
        null_value (float or int, default NaN): the value to use in columns where no K faces are present in the gcs
        title (str, optional): the title for the new grid property; defaults to that of the gcs property
        multiple_handling (str, default 'mean'): one of 'default', 'mean', 'min', 'max', 'min_k', 'max_k', 'exception';
            determines how a value is generated when more than one K face is present in the gcs for a column

    returns:
        uuid of the newly created Property (RESQML ContinuousProperty, DiscreteProperty, CategoricalProperty or
        PointsProperty)

    notes:
        the grid connection set which is the support for gcs_property must involve only one grid;
        the resulting columns grid property is of the same class as the original gcs property;
        the write_hdf() and create_xml() methods are called by this function, for the new property,
        which is added to the model;
        the default multiple handling mode is mean for continuous data, any for discrete (inc categorical);
        in the case of discrete (including categorical) data, a null_value of NaN will be changed to -1
    """
    assert multiple_handling in ['default', 'mean', 'min', 'max', 'any', 'exception']
    assert gcs_property_uuid is not None and bu.is_uuid(gcs_property_uuid)
    gcs_property = rqp.Property(model, uuid = gcs_property_uuid)
    assert gcs_property is not None
    support_uuid = gcs_property.collection.support_uuid
    assert support_uuid is not None
    assert model.type_of_uuid(support_uuid, strip_obj = True) == 'GridConnectionSetRepresentation'
    gcs = rqf.GridConnectionSet(model, uuid = support_uuid)
    gcs_prop_array = gcs_property.array_ref()
    if gcs_property.is_continuous():
        dtype = float
        if multiple_handling == 'default':
            multiple_handling = 'mean'
    else:
        dtype = int
        if null_value == np.nan:
            null_value = -1
        elif type(null_value) is float:
            null_value = int(null_value)
        if multiple_handling == 'default':
            multiple_handling = 'any'
        assert multiple_handling != 'mean', 'mean specified as multiple handling for non-continuous property'
    assert gcs.number_of_grids() == 1, 'only single grid gcs supported for grid columns property derivation'
    grid = gcs.grid_list[0]
    if gcs_property.is_points():
        map_shape = (grid.nj, grid.ni, 3)
    else:
        map_shape = (grid.nj, grid.ni)
    assert gcs_property.count() == 1
    map = np.full(map_shape, null_value, dtype = dtype)
    count_per_col = np.zeros((grid.nj, grid.ni), dtype = int)
    cells_and_faces = gcs.list_of_cell_face_pairs_for_feature_index(None)
    assert cells_and_faces is not None and len(cells_and_faces) == 2
    cell_pairs, face_pairs = cells_and_faces
    if cell_pairs is None or len(cell_pairs) == 0:
        log.warning('no faces found for grid connection set property {gcs_property.title}')
        return None
    assert len(cell_pairs) == len(face_pairs)
    assert len(cell_pairs) == len(gcs_prop_array)
    for index in range(len(cell_pairs)):
        if face_pairs[index, 0, 0] != 0 or face_pairs[index, 1, 0] != 0:
            continue  # not a K face
        col_j, col_i = cell_pairs[index, 0, 1:]  # assume paired cell is in same column!
        if count_per_col[col_j, col_i] == 0 or multiple_handling == 'any':
            map[col_j, col_i] = gcs_prop_array[index]
        elif multiple_handling == 'mean':
            map[col_j, col_i] = (((map[col_j, col_i] * count_per_col[col_j, col_i]) + gcs_prop_array[index]) /
                                 float(count_per_col[col_j, col_i] + 1))
        elif multiple_handling == 'min':
            if gcs_prop_array[index] < map[col_j, col_i]:
                map[col_j, col_i] = gcs_prop_array[index]
        elif multiple_handling == 'max':
            if gcs_prop_array[index] > map[col_j, col_i]:
                map[col_j, col_i] = gcs_prop_array[index]
        else:
            raise ValueError('multiple grid connection set K faces found for column')
        count_per_col[col_j, col_i] += 1
    # create an empty PropertyCollection and add the map data as a new property
    time_index = gcs_property.time_index()
    pc = rqp.PropertyCollection()
    pc.set_support(support = grid)
    pc.add_cached_array_to_imported_list(
        map,
        source_info = f'{gcs.title} property {gcs_property.title} map',
        keyword = title if title else gcs_property.title,
        discrete = not gcs_property.is_continuous(),
        uom = gcs_property.uom(),
        time_index = time_index,
        null_value = None if gcs_property.is_continuous() else null_value,
        property_kind = gcs_property.property_kind(),
        local_property_kind_uuid = gcs_property.local_property_kind_uuid(),
        facet_type = gcs_property.facet_type(),
        facet = gcs_property.facet(),
        realization = None,  # do we want to preserve the realisation number?
        indexable_element = 'columns',
        points = gcs_property.is_points())
    time_series_uuid = pc.write_hdf5_for_imported_list()
    string_lookup_uuid = gcs_property.string_lookup_uuid()
    time_series_uuid = None if time_index is None else gcs_property.time_series_uuid()
    new_uuids = pc.create_xml_for_imported_list_and_add_parts_to_model(time_series_uuid = time_series_uuid,
                                                                       string_lookup_uuid = string_lookup_uuid)
    assert len(new_uuids) == 1
    return new_uuids[0]


def combined_tr_mult_from_gcs_mults(model,
                                    gcs_tr_mult_uuid_list,
                                    merge_mode = 'minimum',
                                    sided = None,
                                    fill_value = 1.0,
                                    active_only = True,
                                    apply_baffles = False):
    """Returns a triplet of transmissibility multiplier arrays over grid faces by combining those from gcs'es.

    arguments:
        model (Model): the model containing all the relevant objects
        gcs_tr_mult_uuid_list (list of UUID): uuids of the individual grid connection set transmissibility
            multiplier properties to be combined
        merge_mode (str, default 'minimum'): one of 'minimum', 'multiply', 'maximum', 'exception'; how to
            handle multiple values applicable to the same grid face
        sided (bool, optional): whether to apply values on both sides of each gcs cell-face pair; if None, will
            default to False if merge mode is multiply, True otherwise
        fill_value (float, optional): the value to use for grid faces not present in any of the gcs'es;
            if None, NaN will be used
        active_only (bool, default True): if True and an active property exists for a grid connection set,
            then only active faces are used when combining to make the grid face arrays
        apply_baffles (bool, default False): if True, where a baffle property exists for a grid connection
            set, a transmissibility multiplier of zero will be used for faces marked as True, overriding the
            multiplier property values at such faces

    returns:
        triple numpy float arrays being transmissibility multipliers for K, J, and I grid faces; arrays have
            shapes (nk + 1, nj, ni), (nk, nj + 1, ni), and (nk, nj, ni + 1) respectively

    notes:
        each gcs, which is the supporting representation for each input tr mult property, must be for a single
        grid and that grid must be the same for all the gcs'es
    """

    assert merge_mode in ['minimum', 'multiply', 'maximum', 'exception']
    assert gcs_tr_mult_uuid_list is not None and len(gcs_tr_mult_uuid_list) > 0
    if sided is None:
        sided = (merge_mode != 'multiply')
    elif sided and merge_mode == 'multiply':
        log.error('using a gcs transmissibility multiplier merge mode of multiply is not compatible with sided True')
        # carry on anyway!

    grid = None

    for tr_mult_uuid in gcs_tr_mult_uuid_list:

        tr_mult = rqp.Property(model, uuid = tr_mult_uuid)
        assert tr_mult is not None
        gcs_uuid = tr_mult.collection.support_uuid
        assert gcs_uuid is not None
        gcs = rqf.GridConnectionSet(model, uuid = gcs_uuid)
        assert gcs is not None
        assert gcs.number_of_grids() == 1

        baffle_uuid = None
        if apply_baffles:
            baffle_part = rqp.property_part(model,
                                            obj_type = 'Discrete',
                                            property_kind = 'baffle',
                                            related_uuid = gcs.uuid)
            if baffle_part is not None:
                baffle_uuid = model.uuid_for_part(baffle_part)

        if grid is None:  # first gcs: grab grid and initialise combined tr mult arrays
            grid = gcs.grid_list[0]
            combo_trm_k = np.full((grid.nk + 1, grid.nj, grid.ni), np.NaN, dtype = float)
            combo_trm_j = np.full((grid.nk, grid.nj + 1, grid.ni), np.NaN, dtype = float)
            combo_trm_i = np.full((grid.nk, grid.nj, grid.ni + 1), np.NaN, dtype = float)
        else:  # check same grid is referenced by this gcs
            assert bu.matching_uuids(gcs.grid_list[0].uuid, grid.uuid)

        # get gcs tr mult data in form of triplet of grid faces arrays
        gcs_trm_k, gcs_trm_j, gcs_trm_i = gcs.grid_face_arrays(tr_mult_uuid,
                                                               default_value = np.NaN,
                                                               active_only = active_only,
                                                               lazy = not sided,
                                                               baffle_uuid = baffle_uuid)
        assert all([trm is not None for trm in (gcs_trm_k, gcs_trm_j, gcs_trm_i)])

        # merge in each of the three directional face arrays for this gcs with combined arrays
        for (combo_trm, gcs_trm) in [(combo_trm_k, gcs_trm_k), (combo_trm_j, gcs_trm_j), (combo_trm_i, gcs_trm_i)]:
            mask = np.logical_not(np.isnan(gcs_trm))  # true where this tr mult is present
            clash_mask = np.logical_and(mask, np.logical_not(np.isnan(combo_trm)))  # true where combined value clashes
            if np.any(clash_mask):
                if merge_mode == 'exception':
                    raise ValueError('gcs transmissibility multiplier conflict when merging')
                if merge_mode == 'minimum':
                    combo_trm[:] = np.where(clash_mask, np.minimum(combo_trm, gcs_trm), combo_trm)
                elif merge_mode == 'maximum':
                    combo_trm[:] = np.where(clash_mask, np.maximum(combo_trm, gcs_trm), combo_trm)
                elif merge_mode == 'multiply':
                    combo_trm[:] = np.where(clash_mask, combo_trm * gcs_trm, combo_trm)
                else:
                    raise Exception(f'code failure with unrecognised merge mode {merge_mode}')
                mask = np.logical_and(mask,
                                      np.logical_not(clash_mask))  # remove clash faces from mask (already handled)
            if np.any(mask):
                combo_trm[:] = np.where(mask, gcs_trm, combo_trm)  # update combined array from individual array

    # for each of the 3 combined tr mult arrays, replace unused values with the default fill value
    # also check that any set values are non-negative
    for combo_trm in (combo_trm_k, combo_trm_j, combo_trm_i):
        if fill_value is not None and not np.isnan(fill_value):
            combo_trm[:] = np.where(np.isnan(combo_trm), fill_value, combo_trm)
            assert np.all(combo_trm >= 0.0)
        else:
            assert np.all(np.logical_or(np.isnan(combo_trm), combo_trm >= 0.0))

    return (combo_trm_k, combo_trm_j, combo_trm_i)


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


def _triangulate_unsplit_grid_connection_set(gcs, feature_index = None):
    # returns triangulation as indices into grid.points_cached, for a single grid gcs only

    def flat_point_index(grid, cell_kji0, corner_index):
        ci = cell_kji0 + corner_index
        return (ci[0] * (grid.nj + 1) + ci[1]) * (grid.ni + 1) + ci[2]

    def add_to_tri(grid, tri, index, cell_kji0, axis, polarity):
        kjip = np.zeros(3, dtype = 'int')
        kjip[axis] = polarity
        a = (axis + 1) % 3
        b = (a + 1) % 3
        tri[index, :, 0] = flat_point_index(grid, cell_kji0, kjip)
        kjip[a] = 1
        kjip[b] = 1
        tri[index, :, 1] = flat_point_index(grid, cell_kji0, kjip)
        kjip[a] = 0
        tri[index, 0, 2] = flat_point_index(grid, cell_kji0, kjip)
        kjip[a] = 1
        kjip[b] = 0
        tri[index, 1, 2] = flat_point_index(grid, cell_kji0, kjip)

    if feature_index is None:
        assert gcs.number_of_features() == 1, 'no gcs feature selected for triangulation'
        feature_index = 0

    assert gcs.number_of_grids() == 1, 'more than one grid references by gcs'
    grid = gcs.grid_list[0]
    if isinstance(grid, grr.RegularGrid):
        grid.make_regular_points_cached(apply_origin_offset = False)
    else:
        assert not grid.has_split_coordinate_lines, 'grid is faulted'
        grid.cache_all_geometry_arrays()

    feature_name = gcs.feature_name_for_feature_index(feature_index)
    ci_pairs, fi_pairs = gcs.list_of_cell_face_pairs_for_feature_index(feature_index)
    cell_index_pairs = ci_pairs.copy()
    face_index_pairs = fi_pairs.copy()
    if cell_index_pairs is None or face_index_pairs is None:
        log.warning(f'no cell face pairs found in grid connection set for feature: {feature_name}')
        return None
    connection_count = cell_index_pairs.shape[0]

    tri_extent = (connection_count, 2, 3)  # 2 triangles per face, will get flattened to (-1, 3)
    tri = np.zeros(tri_extent, dtype = 'int')  # point indices
    tri_index = 0

    for connection in range(connection_count):
        cell_pair = cell_index_pairs[connection]  # shape (2, 3); values kji0
        axis, polarity = face_index_pairs[connection, 0]
        add_to_tri(grid, tri, tri_index, cell_pair[0], axis, polarity)
        tri_index += 1

    result = tri[:tri_index, :, :].reshape((-1, 3))
    if result.size == 0:
        return None

    return result
