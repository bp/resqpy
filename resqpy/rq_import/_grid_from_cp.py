"""_grid_from_cp.py: Module to generate a RESQML grid object from an input corner point array."""

import logging

log = logging.getLogger(__name__)

import numpy as np
import numpy.ma as ma

import resqpy.crs as rqc
import resqpy.grid as grr
import resqpy.olio.vector_utilities as vec


def grid_from_cp(model,
                 cp_array,
                 crs_uuid,
                 active_mask = None,
                 geometry_defined_everywhere = True,
                 treat_as_nan = None,
                 dot_tolerance = 1.0,
                 morse_tolerance = 5.0,
                 max_z_void = 0.1,
                 split_pillars = True,
                 split_tolerance = 0.01,
                 ijk_handedness = 'right',
                 known_to_be_straight = False):
    """Create a resqpy.grid.Grid object from a 7D corner point array.

    Arguments:
        model (resqpy.model.Model): model to which the grid will be added
        cp_array (numpy float array): 7 dimensional numpy array of nexus corner point data, in nexus ordering
        crs_uuid (uuid.UUID): uuid for the coordinate reference system
        active_mask (3d numpy bool array): array indicating which cells are active
        geometry_defined_everywhere (bool, default True): if False then inactive cells are marked as not having geometry
        treat_as_nan (float, default None): if a value is provided corner points with this value will be assigned nan
        dot_tolerance (float, default 1.0): minimum manhatten distance of primary diagonal of cell, below which cell is treated as inactive
        morse_tolerance (float, default 5.0): maximum ratio of i and j face vector lengths, beyond which cells are treated as inactive
        max_z_void (float, default 0.1): maximum z gap between vertically neighbouring corner points. Vertical gaps greater than this will introduce k gaps into resqml grid. Units are corp z units
        split_pillars (bool, default True): if False an unfaulted grid will be generated
        split_tolerance (float, default 0.01): maximum distance between neighbouring corner points before a pillar is considered 'split'. Applies to each of x, y, z differences
        ijk_handedness (str, default 'right'): 'right' or 'left'
        known_to_be_straight (bool, default False): if True pillars are forced to be straight

    notes:
       this function sets up all the geometry arrays in memory but does not write to hdf5 nor create xml: use Grid methods;
       geometry_defined_everywhere is deprecated, use treat_as_nan instead
    """
    grid = _GridFromCp(model, cp_array, crs_uuid, active_mask, geometry_defined_everywhere, treat_as_nan, dot_tolerance,
                       morse_tolerance, max_z_void, split_pillars, split_tolerance, ijk_handedness,
                       known_to_be_straight)

    return grid.grid


class _GridFromCp:
    """Class to build a resqpy grid from a Nexus CORP array"""

    def __init__(self,
                 model,
                 cp_array,
                 crs_uuid,
                 active_mask = None,
                 geometry_defined_everywhere = True,
                 treat_as_nan = None,
                 dot_tolerance = 1.0,
                 morse_tolerance = 5.0,
                 max_z_void = 0.1,
                 split_pillars = True,
                 split_tolerance = 0.01,
                 ijk_handedness = 'right',
                 known_to_be_straight = False):
        """Class to build a resqpy grid from a Nexus CORP array"""

        self.__model = model
        self.__cp_array = cp_array
        self.__crs_uuid = crs_uuid
        self.__active_mask = active_mask
        self.__geometry_defined_everywhere = geometry_defined_everywhere
        self.__treat_as_nan = treat_as_nan
        self.__dot_tolerance = dot_tolerance
        self.__morse_tolerance = morse_tolerance
        self.__max_z_void = max_z_void
        self.__split_pillars = split_pillars
        self.__split_tolerance = split_tolerance
        self.__ijk_handedness = ijk_handedness
        self.__known_to_be_straight = known_to_be_straight

        self.create_grid()

    def __get_treat_as_nan(self):
        if self.__treat_as_nan is None:
            if not self.__geometry_defined_everywhere:
                self.__treat_as_nan = 'morse'
        else:
            assert self.__treat_as_nan in ['none', 'dots', 'ij_dots', 'morse', 'inactive']
            if self.__treat_as_nan == 'none':
                self.__treat_as_nan = None

    def __get_extents(self):
        self.__nk, self.__nj, self.__ni = self.__cp_array.shape[:3]
        self.__nk_plus_1 = self.__nk + 1
        self.__nj_plus_1 = self.__nj + 1
        self.__ni_plus_1 = self.__ni + 1

    def __get_active_inactive_masks(self):
        if self.__active_mask is None:
            self.__active_mask = np.ones((self.__nk, self.__nj, self.__ni), dtype = 'bool')
            self.__inactive_mask = np.zeros((self.__nk, self.__nj, self.__ni), dtype = 'bool')
        else:
            assert self.__active_mask.shape == (self.__nk, self.__nj, self.__ni)
            self.__inactive_mask = np.logical_not(self.__active_mask)
        self.__all_active = np.all(self.__active_mask)

    def __get_dot_mask_dots(self):
        # for speed, only check primary diagonal of cells
        log.debug('geometry for cells with no length to primary cell diagonal being set to NaN')
        self.__dot_mask = np.all(
            np.abs(self.__cp_array[:, :, :, 1, 1, 1] - self.__cp_array[:, :, :, 0, 0, 0]) < self.__dot_tolerance,
            axis = -1)

    def __get_dot_mask_ijdots_or_morse(self):
        # check one diagonal of each I & J face
        log.debug(
            'geometry being set to NaN for inactive cells with no length to primary face diagonal for any I or J face')
        self.__dot_mask = np.zeros((self.__nk, self.__nj, self.__ni), dtype = bool)
        #              k_face_vecs = cp_array[:, :, :, :, 1, 1] - cp_array[:, :, :, :, 0, 0]
        j_face_vecs = self.__cp_array[:, :, :, 1, :, 1] - self.__cp_array[:, :, :, 0, :, 0]
        i_face_vecs = self.__cp_array[:, :, :, 1, 1, :] - self.__cp_array[:, :, :, 0, 0, :]
        self.__dot_mask[:] = np.where(np.all(np.abs(j_face_vecs[:, :, :, 0]) < self.__dot_tolerance, axis = -1), True,
                                      self.__dot_mask)
        self.__dot_mask[:] = np.where(np.all(np.abs(j_face_vecs[:, :, :, 1]) < self.__dot_tolerance, axis = -1), True,
                                      self.__dot_mask)
        self.__dot_mask[:] = np.where(np.all(np.abs(i_face_vecs[:, :, :, 0]) < self.__dot_tolerance, axis = -1), True,
                                      self.__dot_mask)
        self.__dot_mask[:] = np.where(np.all(np.abs(i_face_vecs[:, :, :, 1]) < self.__dot_tolerance, axis = -1), True,
                                      self.__dot_mask)
        log.debug(f'dot mask set for {np.count_nonzero(self.__dot_mask)} cells')

        if self.__treat_as_nan == 'morse':
            self.__get_dot_mask_morse(i_face_vecs, j_face_vecs)

    def __get_dot_mask_morse(self, i_face_vecs, j_face_vecs):
        morse_tol_sqr = self.__morse_tolerance * self.__morse_tolerance
        # compare face vecs lengths in xy against max for active cells: where much greater set to NaN
        len_j_face_vecs_sqr = np.sum(j_face_vecs[..., :2] * j_face_vecs[..., :2], axis = -1)
        len_i_face_vecs_sqr = np.sum(i_face_vecs[..., :2] * i_face_vecs[..., :2], axis = -1)
        dead_mask = self.__inactive_mask.reshape(self.__nk, self.__nj, self.__ni, 1).repeat(2, -1)
        #                  mean_len_active_j_face_vecs_sqr = np.mean(ma.masked_array(len_j_face_vecs_sqr, mask = dead_mask))
        #                  mean_len_active_i_face_vecs_sqr = np.mean(ma.masked_array(len_i_face_vecs_sqr, mask = dead_mask))
        max_len_active_j_face_vecs_sqr = np.max(ma.masked_array(len_j_face_vecs_sqr, mask = dead_mask))
        max_len_active_i_face_vecs_sqr = np.max(ma.masked_array(len_i_face_vecs_sqr, mask = dead_mask))
        self.__dot_mask = np.where(
            np.any(len_j_face_vecs_sqr > morse_tol_sqr * max_len_active_j_face_vecs_sqr, axis = -1), True,
            self.__dot_mask)
        self.__dot_mask = np.where(
            np.any(len_i_face_vecs_sqr > morse_tol_sqr * max_len_active_i_face_vecs_sqr, axis = -1), True,
            self.__dot_mask)
        log.debug(f'morse mask set for {np.count_nonzero(self.__dot_mask)} cells')

    def __get_nan_mask(self):
        if self.__all_active and self.__geometry_defined_everywhere:
            self.__cp_nan_mask = None
        else:
            self.__cp_nan_mask = np.any(np.isnan(self.__cp_array), axis = (3, 4, 5, 6))  # ie. if any nan per cell
            if not self.__geometry_defined_everywhere and not self.__all_active:
                if self.__treat_as_nan == 'inactive':
                    log.debug('all inactive cell geometry being set to NaN')
                    self.__cp_nan_mask = np.logical_or(self.__cp_nan_mask, self.__inactive_mask)
                else:
                    if self.__treat_as_nan == 'dots':
                        self.__get_dot_mask_dots()
                    elif self.__treat_as_nan in ['ij_dots', 'morse']:
                        self.__get_dot_mask_ijdots_or_morse()
                    else:
                        raise Exception('code broken')
                    self.__cp_nan_mask = np.logical_or(self.__cp_nan_mask,
                                                       np.logical_and(self.__inactive_mask, self.__dot_mask))
            self.__geometry_defined_everywhere = not np.any(self.__cp_nan_mask)
            if self.__geometry_defined_everywhere:
                self.__cp_nan_mask = None

    def __get_masked_cp_array(self):
        # set up masked version of corner point data based on cells with defined geometry
        if self.__geometry_defined_everywhere:
            full_mask = None
            self.__masked_cp_array = ma.masked_array(self.__cp_array, mask = ma.nomask)
            log.info('geometry present for all cells')
        else:
            full_mask = self.__cp_nan_mask.reshape((self.__nk, self.__nj, self.__ni, 1)).repeat(24, axis = 3).reshape(
                (self.__nk, self.__nj, self.__ni, 2, 2, 2, 3))
            self.__masked_cp_array = ma.masked_array(self.__cp_array, mask = full_mask)
            log.info('number of cells without geometry: ' + str(np.count_nonzero(self.__cp_nan_mask)))

    def __check_for_kgaps(self):
        self.__k_gaps = None
        self.__k_gap_raw_index = None
        self.__k_gap_after_layer = None

        if self.__nk > 1:
            # check for (vertical) voids, or un-pillar-like anomalies, which will require k gaps in the resqml ijk grid
            log.debug('checking for voids')
            gap = self.__masked_cp_array[1:, :, :, 0, :, :, :] - self.__masked_cp_array[:-1, :, :, 1, :, :, :]
            max_gap_by_layer_and_xyz = np.max(np.abs(gap), axis = (1, 2, 3, 4))
            max_gap = np.max(max_gap_by_layer_and_xyz)
            log.debug('maximum void distance: {0:.3f}'.format(max_gap))
            if max_gap > self.__max_z_void:
                self.__get_kgaps_details(max_gap_by_layer_and_xyz, gap)
            elif max_gap > 0.0:
                self.__close_gaps(gap)

    def __get_kgaps_details(self, max_gap_by_layer_and_xyz, gap):
        log.warning('maximum void distance exceeds limit, grid will include k gaps')
        self.__k_gaps = 0
        self.__k_gap_after_layer = np.zeros((self.__nk - 1,), dtype = bool)
        self.__k_gap_raw_index = np.empty((self.__nk,), dtype = int)
        self.__k_gap_raw_index[0] = 0
        for k in range(self.__nk - 1):
            max_layer_gap = np.max(max_gap_by_layer_and_xyz[k])
            if max_layer_gap > self.__max_z_void:
                self.__k_gap_after_layer[k] = True
                self.__k_gaps += 1
            elif max_layer_gap > 0.0:
                # close void (includes shifting x & y)
                log.debug('closing void below layer (0 based): ' + str(k))
                layer_gap = gap[k] * 0.5
                layer_gap_unmasked = np.where(gap[k].mask, 0.0, layer_gap)
                self.__masked_cp_array[k + 1, :, :, 0, :, :, :] -= layer_gap_unmasked
                self.__masked_cp_array[k, :, :, 1, :, :, :] += layer_gap_unmasked
            self.__k_gap_raw_index[k + 1] = k + self.__k_gaps

    def __close_gaps(self, gap):
        # close voids (includes shifting x & y)
        log.debug('closing voids')
        gap *= 0.5
        gap_unmasked = np.where(gap.mask, 0.0, gap)
        self.__masked_cp_array[1:, :, :, 0, :, :, :] -= gap_unmasked
        self.__masked_cp_array[:-1, :, :, 1, :, :, :] += gap_unmasked

    def __get_k_reduced_cp_array(self):
        log.debug('reducing k extent of corner point array (sharing points vertically)')
        self.__k_reduced_cp_array = ma.masked_array(np.zeros(
            (self.__nk_plus_1, self.__nj, self.__ni, 2, 2, 3)))  # (nk+1+k_gaps, nj, ni, jp, ip, xyz)
        self.__k_reduced_cp_array[0, :, :, :, :, :] = self.__masked_cp_array[0, :, :, 0, :, :, :]
        self.__k_reduced_cp_array[-1, :, :, :, :, :] = self.__masked_cp_array[-1, :, :, 1, :, :, :]
        if self.__k_gaps:
            self.__get_k_reduced_cp_array_kgaps()
        else:
            slice = self.__masked_cp_array[1:, :, :, 0, :, :, :]
            # where cell geometry undefined, if cell above is defined, take data from cell above with kp = 1 and set shared point defined
            self.__k_reduced_cp_array[1:-1, :, :, :, :, :] = np.where(slice.mask, self.__masked_cp_array[:-1, :, :,
                                                                                                         1, :, :, :],
                                                                      slice)

    def __get_k_reduced_cp_array_kgaps(self):
        raw_k = 1
        for k in range(self.__nk - 1):
            # fill reduced array slice(s) for base of layer k and top of layer k + 1
            if self.__k_gap_after_layer[k]:
                self.__k_reduced_cp_array[raw_k, :, :, :, :, :] = self.__masked_cp_array[k, :, :, 1, :, :, :]
                raw_k += 1
                self.__k_reduced_cp_array[raw_k, :, :, :, :, :] = self.__masked_cp_array[k + 1, :, :, 0, :, :, :]
                raw_k += 1
            else:  # take data from either possible cp slice, whichever is defined
                slice = self.__masked_cp_array[k + 1, :, :, 0, :, :, :]
                self.__k_reduced_cp_array[raw_k, :, :, :, :, :] = np.where(slice.mask,
                                                                           self.__masked_cp_array[k, :, :,
                                                                                                  1, :, :, :], slice)
                raw_k += 1
        assert raw_k == self.__nk + self.__k_gaps

    def __get_primary_pillar_ref(self):
        log.debug('creating primary pillar reference neighbourly indices')
        self.__primary_pillar_jip = np.zeros((self.__nj_plus_1, self.__ni_plus_1, 2),
                                             dtype = 'int')  # (nj + 1, ni + 1, jp:ip)
        self.__primary_pillar_jip[-1, :, 0] = 1
        self.__primary_pillar_jip[:, -1, 1] = 1
        for j in range(self.__nj_plus_1):
            for i in range(self.__ni_plus_1):
                if self.__active_mask_2D[j - self.__primary_pillar_jip[j, i, 0],
                                         i - self.__primary_pillar_jip[j, i, 1]]:
                    continue
                if i > 0 and self.__primary_pillar_jip[j, i, 1] == 0 and self.__active_mask_2D[
                        j - self.__primary_pillar_jip[j, i, 0], i - 1]:
                    self.__primary_pillar_jip[j, i, 1] = 1
                    continue
                if j > 0 and self.__primary_pillar_jip[j, i, 0] == 0 and self.__active_mask_2D[
                        j - 1, i - self.__primary_pillar_jip[j, i, 1]]:
                    self.__primary_pillar_jip[j, i, 0] = 1
                    continue
                if i > 0 and j > 0 and self.__primary_pillar_jip[j, i, 0] == 0 and self.__primary_pillar_jip[
                        j, i, 1] == 0 and self.__active_mask_2D[j - 1, i - 1]:
                    self.__primary_pillar_jip[j, i, :] = 1

    def __get_extra_pillar_ref(self):
        self.__extras_count = np.zeros((self.__nj_plus_1, self.__ni_plus_1),
                                       dtype = 'int')  # count (0 to 3) of extras for pillar
        self.__extras_list_index = np.zeros((self.__nj_plus_1, self.__ni_plus_1),
                                            dtype = 'int')  # index in list of 1st extra for pillar
        self.__extras_list = []  # list of (jp, ip)
        self.__extras_use = np.negative(np.ones((self.__nj, self.__ni, 2, 2),
                                                dtype = 'int'))  # (j, i, jp, ip); -1 means use primary
        if self.__split_pillars:
            self.__get_extra_pillar_ref_split()

    def __get_extra_pillar_ref_split(self):
        log.debug('building extra pillar references for split pillars')
        # loop over pillars
        for j in range(self.__nj_plus_1):
            for i in range(self.__ni_plus_1):
                self.__append_extra_pillars(i, j)

        if len(self.__extras_list) == 0:
            self.__split_pillars = False
        log.debug('number of extra pillars: ' + str(len(self.__extras_list)))

    def __append_single_pillar(self, i, j, ip, jp, col_j, p_col_i, p_col_j, primary_ip, primary_jp):
        col_i = i - ip
        if col_i < 0 or col_i >= self.__ni:
            return  # no column this side of pillar in i
        if jp == primary_jp and ip == primary_ip:
            return  # this column is the primary for this pillar
        discrepancy = np.max(
            np.abs(self.__k_reduced_cp_array[:, col_j, col_i, jp, ip, :] -
                   self.__k_reduced_cp_array[:, p_col_j, p_col_i, primary_jp, primary_ip, :]))
        if discrepancy <= self.__split_tolerance:
            return  # data for this column's corner aligns with primary
        for e in range(self.__extras_count[j, i]):
            eli = self.__extras_list_index[j, i] + e
            pillar_j_extra = j - self.__extras_list[eli][0]
            pillar_i_extra = i - self.__extras_list[eli][1]
            discrepancy = np.max(
                np.abs(self.__k_reduced_cp_array[:, col_j, col_i, jp, ip, :] -
                       self.__k_reduced_cp_array[:, pillar_j_extra, pillar_i_extra, self.__extras_list[eli][0],
                                                 self.__extras_list[eli][1], :]))
            if discrepancy <= self.__split_tolerance:  # data for this corner aligns with existing extra
                self.__extras_use[col_j, col_i, jp, ip] = e
                break
        if self.__extras_use[col_j, col_i, jp, ip] >= 0:  # reusing an existing extra for this pillar
            return
        # add this corner as an extra
        if self.__extras_count[j, i] == 0:  # create entry point for this pillar in extras
            self.__extras_list_index[j, i] = len(self.__extras_list)
        self.__extras_list.append((jp, ip))
        self.__extras_use[col_j, col_i, jp, ip] = self.__extras_count[j, i]
        self.__extras_count[j, i] += 1

    def __append_extra_pillars(self, i, j):
        primary_jp = self.__primary_pillar_jip[j, i, 0]
        primary_ip = self.__primary_pillar_jip[j, i, 1]
        p_col_j = j - primary_jp
        p_col_i = i - primary_ip
        # loop over 4 columns surrounding this pillar
        for jp in range(2):
            col_j = j - jp
            if col_j < 0 or col_j >= self.__nj:
                continue  # no column this side of pillar in j
            for ip in range(2):
                self.__append_single_pillar(i, j, ip, jp, col_j, p_col_i, p_col_j, primary_ip, primary_jp)

    def __get_points_array(self):
        log.debug('creating points array as used in resqml format')
        if self.__split_pillars:
            self.__get_points_array_split()
        else:  # unsplit pillars
            self.__points_array = np.zeros((self.__nk_plus_1, self.__nj_plus_1, self.__ni_plus_1, 3))
            for j in range(self.__nj_plus_1):
                for i in range(self.__ni_plus_1):
                    (jp, ip) = self.__primary_pillar_jip[j, i]
                    slice = self.__k_reduced_cp_array[:, j - jp, i - ip, jp, ip, :]
                    self.__points_array[:, j, i, :] = np.where(slice.mask, np.nan,
                                                               slice)  # NaN indicates undefined/invalid geometry

    def __get_points_array_split(self):
        self.__points_array = np.zeros(
            (self.__nk_plus_1, (self.__nj_plus_1 * self.__ni_plus_1) + len(self.__extras_list),
             3))  # note: nk_plus_1 might include k_gaps
        index = 0
        index = self.__get_points_array_split_primary(index)
        index = self.__get_points_array_split_extras(index)

        assert (index == (self.__nj_plus_1 * self.__ni_plus_1) + len(self.__extras_list))

    def __get_points_array_split_primary(self, index):
        # primary pillars
        for pillar_j in range(self.__nj_plus_1):
            for pillar_i in range(self.__ni_plus_1):
                (jp, ip) = self.__primary_pillar_jip[pillar_j, pillar_i]
                slice = self.__k_reduced_cp_array[:, pillar_j - jp, pillar_i - ip, jp, ip, :]
                self.__points_array[:, index, :] = np.where(slice.mask, np.nan,
                                                            slice)  # NaN indicates undefined/invalid geometry
                index += 1
        return index

    def __get_points_array_split_extras(self, index):
        # add extras for split pillars
        for pillar_j in range(self.__nj_plus_1):
            for pillar_i in range(self.__ni_plus_1):
                for e in range(self.__extras_count[pillar_j, pillar_i]):
                    eli = self.__extras_list_index[pillar_j, pillar_i] + e
                    (jp, ip) = self.__extras_list[eli]
                    pillar_j_extra = pillar_j - jp
                    pillar_i_extra = pillar_i - ip
                    slice = self.__k_reduced_cp_array[:, pillar_j_extra, pillar_i_extra, jp, ip, :]
                    self.__points_array[:, index, :] = np.where(slice.mask, np.nan,
                                                                slice)  # NaN indicates undefined/invalid geometry
                    index += 1
        return index

    def __make_basic_grid(self):
        log.debug('initialising grid object')
        self.grid = grr.Grid(self.__model)
        self.grid.grid_representation = 'IjkGrid'
        self.grid.extent_kji = np.array((self.__nk, self.__nj, self.__ni), dtype = 'int')
        self.grid.nk, self.grid.nj, self.grid.ni = self.__nk, self.__nj, self.__ni
        self.grid.k_direction_is_down = True  # assumed direction for corp; todo: determine from geometry and crs z_inc_down flag
        if self.__known_to_be_straight:
            self.grid.pillar_shape = 'straight'
        else:
            self.grid.pillar_shape = 'curved'
        self.grid.has_split_coordinate_lines = self.__split_pillars
        self.grid.k_gaps = self.__k_gaps
        self.grid.k_gap_after_array = self.__k_gap_after_layer
        self.grid.k_raw_index_array = self.__k_gap_raw_index

        self.grid.crs_uuid = self.__crs_uuid
        self.grid.crs_root = self.__model.root_for_uuid(self.__crs_uuid)
        self.__crs = rqc.Crs(self.__model, uuid = self.__crs_uuid)

        # add pillar points array to grid object
        log.debug('attaching points array to grid object')

        self.grid.points_cached = self.__points_array  # NB: reference to points_array, array not copied here

    def __get_split_pillars_lists(self, pillar_i, pillar_j, e):
        self.__split_pillar_indices_list.append((pillar_j * self.__ni_plus_1) + pillar_i)
        use_count = 0
        for jp in range(2):
            j = pillar_j - jp
            if j < 0 or j >= self.__nj:
                continue
            for ip in range(2):
                i = pillar_i - ip
                if i < 0 or i >= self.__ni:
                    continue
                if self.__extras_use[j, i, jp, ip] == e:
                    use_count += 1
                    self.__cols_for_extra_pillar_list.append((j * self.__ni) + i)
        assert (use_count > 0)
        self.__cumulative_length += use_count
        self.__cumulative_length_list.append(self.__cumulative_length)

    def __add_split_arrays_to_grid(self):
        if self.__split_pillars:
            log.debug('adding split pillar arrays to grid object')
            self.__split_pillar_indices_list = []
            self.__cumulative_length_list = []
            self.__cols_for_extra_pillar_list = []
            self.__cumulative_length = 0
            for pillar_j in range(self.__nj_plus_1):
                for pillar_i in range(self.__ni_plus_1):
                    for e in range(self.__extras_count[pillar_j, pillar_i]):
                        self.__get_split_pillars_lists(pillar_i, pillar_j, e)
            log.debug('number of extra pillars: ' + str(len(self.__split_pillar_indices_list)))
            assert (len(self.__cumulative_length_list) == len(self.__split_pillar_indices_list))
            self.grid.split_pillar_indices_cached = np.array(self.__split_pillar_indices_list, dtype = 'int')
            log.debug('number of uses of extra pillars: ' + str(len(self.__cols_for_extra_pillar_list)))
            assert (len(self.__cols_for_extra_pillar_list) == np.count_nonzero(self.__extras_use + 1))
            assert (len(self.__cols_for_extra_pillar_list) == self.__cumulative_length)
            self.grid.cols_for_split_pillars = np.array(self.__cols_for_extra_pillar_list, dtype = 'int')
            assert (len(self.__cumulative_length_list) == len(self.__extras_list))
            self.grid.cols_for_split_pillars_cl = np.array(self.__cumulative_length_list, dtype = 'int')
            self.grid.split_pillars_count = len(self.__extras_list)

    def __set_up_column_to_pillars_mapping(self):
        log.debug('setting up column to pillars mapping')
        base_pillar_count = self.__nj_plus_1 * self.__ni_plus_1
        self.grid.pillars_for_column = np.empty((self.__nj, self.__ni, 2, 2), dtype = 'int')
        for j in range(self.__nj):
            for i in range(self.__ni):
                for jp in range(2):
                    for ip in range(2):
                        if not self.__split_pillars or self.__extras_use[j, i, jp, ip] < 0:  # use primary pillar
                            pillar_index = (j + jp) * self.__ni_plus_1 + i + ip
                        else:
                            eli = self.__extras_list_index[j + jp, i + ip] + self.__extras_use[j, i, jp, ip]
                            pillar_index = base_pillar_count + eli
                        self.grid.pillars_for_column[j, i, jp, ip] = pillar_index

    def __update_grid_geometry_information(self):
        # add cell geometry defined array to model (using active cell mask unless geometry_defined_everywhere is True)
        if self.__geometry_defined_everywhere:
            self.grid.geometry_defined_for_all_cells_cached = True
            self.grid.array_cell_geometry_is_defined = None
        else:
            log.debug('using active cell mask as indicator of defined cell geometry')
            self.grid.array_cell_geometry_is_defined = self.__active_mask.copy(
            )  # a bit harsh: disallows reactivation of cells
            self.grid.geometry_defined_for_all_cells_cached = np.all(self.__active_mask)
        self.grid.geometry_defined_for_all_pillars_cached = True  # following fesapi convention of defining all pillars regardless
        # note: grid.array_pillar_geometry_is_defined not set, as line above should be sufficient

    def __update_grid_handedness(self):
        # set handedness of ijk axes
        if self.__ijk_handedness is None or self.__ijk_handedness == 'auto':
            # work out handedness from sample cell / column axes directions and handedness of crs
            sample_kji0 = tuple(np.array(self.grid.extent_kji) // 2)
            if not self.__geometry_defined_everywhere and not self.grid.array_cell_geometry_is_defined[sample_kji0]:
                where_defined = np.where(
                    np.logical_and(self.grid.array_cell_geometry_is_defined, np.logical_not(self.grid.pinched_out())))
                assert len(where_defined) == 3 and len(where_defined[0]) > 0, 'no extant cell geometries'
                sample_kji0 = (where_defined[0][0], where_defined[1][0], where_defined[2][0])
            sample_cp = self.__cp_array[sample_kji0]
            self.__cell_ijk_lefthanded = (vec.clockwise(sample_cp[0, 0, 0], sample_cp[0, 1, 0], sample_cp[0, 0, 1]) >=
                                          0.0)
            if not self.grid.k_direction_is_down:
                self.__cell_ijk_lefthanded = not self.__cell_ijk_lefthanded
            if self.__crs.is_right_handed_xyz():
                self.__cell_ijk_lefthanded = not self.__cell_ijk_lefthanded
            self.grid.grid_is_right_handed = not self.__cell_ijk_lefthanded
        else:
            assert self.__ijk_handedness in ['left', 'right']
            self.grid.grid_is_right_handed = (self.__ijk_handedness == 'right')

    def create_grid(self):
        """Make the grid"""
        # Find out which cells to treat as nans
        self.__get_treat_as_nan()

        self.__geometry_defined_everywhere = (self.__treat_as_nan is None)

        assert self.__cp_array.ndim == 7
        # Get the grid extents
        self.__get_extents()
        # Generate active and inactive masks
        self.__get_active_inactive_masks()
        # Generate the nan mask
        self.__get_nan_mask()
        # Apply nan and inactive masks
        if self.__cp_nan_mask is not None:
            self.__inactive_mask = np.logical_or(self.__inactive_mask, self.__cp_nan_mask)
            self.__active_mask = np.logical_not(self.__inactive_mask)
        # Generate the masked corner point array
        self.__get_masked_cp_array()

        # Find information on kgaps in the grid
        self.__check_for_kgaps()

        if self.__k_gaps:
            self.__nk_plus_1 += self.__k_gaps
        if self.__k_gap_raw_index is None:
            self.__k_gap_raw_index = np.arange(self.__nk, dtype = int)

        # reduce cp array extent in k
        self.__get_k_reduced_cp_array()

        # create 2D array of active columns (columns where at least one cell is active)
        log.debug('creating 2D array of active columns')
        self.__active_mask_2D = np.any(self.__active_mask, axis = 0)

        # create primary pillar reference indices as one of four column corners around pillar, active column preferred
        self.__get_primary_pillar_ref()

        # build extra pillar references for split pillars
        self.__get_extra_pillar_ref()

        # create points array as used in resqml
        self.__get_points_array()

        # create an empty grid object and fill in some basic info
        self.__make_basic_grid()

        # add split pillar arrays to grid object
        self.__add_split_arrays_to_grid()

        # following is not part of resqml standard but is used by resqml_grid module for speed optimisation
        self.__set_up_column_to_pillars_mapping()

        # add inactive cell mask to grid
        log.debug('setting inactive cell mask')
        self.grid.inactive = self.__inactive_mask.copy()

        # update grid with geometry parameters
        self.__update_grid_geometry_information()

        # tentatively add corner point array to grid object in case it is needed
        log.debug('noting corner point array in grid')
        self.grid.array_corner_points = self.__cp_array

        # update grid handedness
        self.__update_grid_handedness()
