"""fine_coarse.py: Module providing support for grid refinement and coarsening."""

# Nexus is a registered trademark of the Halliburton Company

# todo: cater for parallel to top & parallel to base style refinement proportions in k

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.olio.box_utilities as box


class FineCoarse:
    """Class for holding a mapping between fine and coarse grids."""

    def __init__(self, fine_extent_kji, coarse_extent_kji, within_fine_box = None, within_coarse_box = None):
        """Partial initialisation function, call other methods to assign ratios and proportions.

        arguments:
           fine_extent_kji (triple int): the local extent of the fine grid in k, j, i axes, ie (nk, nj, ni).
           coarse_extent_kji (triple int): the local extent of the coarse grid in k, j, i axes.
           within_fine_box (numpy int array of shape (2, 3), optional): if present, the subset of a larger
              fine grid that the mapping refers to; axes are min,max and k,j,i; values are zero based indices;
              max values are included in box (un-pythonesque); use this in the case of a local grid coarsening
           within_coarse_box (numpy int array of shape (2, 3), optional): if present, the subset of a larger
              coarse grid that the mapping refers to; axes are min,max and k,j,i; values are zero based indices;
              max values are included in box (un-pythonesque); use this in the case of a local grid refinement;
              required for write_cartref() method to work

        returns:
           newly formed FineCoarse object awaiting determination of ratios and proportions by axes.

        notes:
           at most one of within_fine_box and within_coarse_box may be passed; this information is not really
           used internally by the FineCoarse class but is noted in order to support local grid refinement and
           local grid coarsening applications;
           after intialisation, set_* methods should be called to establish the mapping
        """

        assert len(fine_extent_kji) == 3 and len(coarse_extent_kji) == 3
        assert within_fine_box is None or within_coarse_box is None

        self.fine_extent_kji = tuple(fine_extent_kji)  #: fine extent
        self.coarse_extent_kji = tuple(coarse_extent_kji)  #: coarse extent
        self._assert_extents_valid()
        if within_fine_box is not None:
            _assert_valid_box(fine_extent_kji, within_fine_box)
        if within_coarse_box is not None:
            _assert_valid_box(coarse_extent_kji, within_coarse_box)
        self.within_fine_box = within_fine_box  #: if not None, a box within an unidentified larger fine grid
        self.within_coarse_box = within_coarse_box  #: if not None, a box within an unidentified larger coarse grid

        self.constant_ratios = [None, None, None]  #: list for 3 axes kji, each None or int
        self.vector_ratios = [None, None, None]  #: list for 3 axes kji, each numpy vector of int or None
        self.equal_proportions = [True, True,
                                  True]  #: list for 3 axes kji, each boolean defaulting to equal proportions
        self.vector_proportions = [None, None, None
                                  ]  #: list for 3 axes kji, each None or list of numpy vectors of float summing to 1.0

        self.fine_to_coarse_mapping = None  # derived triplet of int vectors holding coarse cell index for each fine cell index

    def assert_valid(self):
        """Checks consistency of everything within the fine coarse mapping; raises assertion error if not valid."""

        self._assert_extents_valid()

        assert self.constant_ratios is not None and len(self.constant_ratios) == 3
        assert self.vector_ratios is not None and len(self.vector_ratios) == 3
        assert self.equal_proportions is not None and len(self.equal_proportions) == 3
        assert self.vector_proportions is not None and len(self.vector_proportions) == 3

        for axis in range(3):
            assert (self.constant_ratios[axis] is None or
                    _is_int(self.constant_ratios[axis]) and self.constant_ratios[axis] > 0)
            if self.constant_ratios[axis] is None:
                assert (self.vector_ratios[axis] is not None and isinstance(self.vector_ratios[axis], np.ndarray) and
                        self.vector_ratios[axis].ndim == 1 and str(self.vector_ratios[axis].dtype).startswith('int'))
                assert len(self.vector_ratios[axis]) == self.coarse_extent_kji[axis]
                assert np.all(self.vector_ratios[axis] > 0)
                assert np.sum(self.vector_ratios[axis]) == self.fine_extent_kji[axis]
            else:
                assert self.vector_ratios[axis] is None
                assert self.coarse_extent_kji[axis] * self.constant_ratios[axis] == self.fine_extent_kji[axis]
            assert isinstance(self.equal_proportions[axis], bool)
            if self.equal_proportions[axis]:
                assert self.vector_proportions[axis] is None
            else:
                assert (self.vector_proportions[axis] is not None and
                        isinstance(self.vector_proportions[axis], list) and  # could allow a tuple
                        len(self.vector_proportions[axis]) == self.coarse_extent_kji[axis])
                for c0 in range(self.coarse_extent_kji[axis]):
                    assert (isinstance(self.vector_proportions[axis][c0], np.ndarray) and
                            self.vector_proportions[axis][c0].ndim == 1 and
                            str(self.vector_proportions[axis][c0].dtype).startswith('float') and
                            len(self.vector_proportions[axis][c0]) == self.ratio(axis, c0))
                    assert np.min(self.vector_proportions[axis][c0]) > 0.0
                    assert abs(np.sum(self.vector_proportions[axis][c0]) - 1.0) < 1.0e-6

    def ratio(self, axis, c0):
        """Return fine:coarse ratio in given axis and coarse slice."""

        if self.constant_ratios[axis] is not None:
            return self.constant_ratios[axis]
        return self.vector_ratios[axis][c0]

    def ratios(self, c_kji0):
        """Return find:coarse ratios triplet for coarse cell."""

        return tuple(self.ratio(axis, c_kji0[axis]) for axis in range(3))

    def coarse_for_fine(self):
        """Returns triplet of numpy int vectors being the axial coarse cell indices for the axial fine cell indices."""

        if self.fine_to_coarse_mapping is None:
            self._set_fine_to_coarse_mapping()
        return self.fine_to_coarse_mapping

    def coarse_for_fine_kji0(self, fine_kji0):
        """Returns the index of the coarse cell which the given fine cell falls within."""

        if self.fine_to_coarse_mapping is None:
            self._set_fine_to_coarse_mapping()
        return (self.fine_to_coarse_mapping[0][fine_kji0[0]], self.fine_to_coarse_mapping[1][fine_kji0[1]],
                self.fine_to_coarse_mapping[2][fine_kji0[2]])

    def coarse_for_fine_axial(self, axis, f0):
        """Returns the index, for a single axis, of the coarse cell which the given fine cell falls within."""

        if self.fine_to_coarse_mapping is None:
            self._set_fine_to_coarse_mapping()
        return self.fine_to_coarse_mapping[axis][f0]

    def coarse_for_fine_axial_vector(self, axis):
        """Returns a numpy int vector, for a single axis, of the coarse cell index which each fine cell falls within."""

        if self.fine_to_coarse_mapping is None:
            self._set_fine_to_coarse_mapping()
        return self.fine_to_coarse_mapping[axis]

    def fine_base_for_coarse_axial(self, axis, c0):
        """Returns the index, for a single axis, of the 'first' fine cell within the coarse cell (lowest fine index)."""

        if self.constant_ratios[axis] is not None:
            return c0 * self.constant_ratios[axis]
        return np.sum(self.vector_ratios[axis][:c0])  # todo: check this returns zero for c0 == 0

    def fine_base_for_coarse(self, c_kji0):
        """Returns a 3-tuple being the 'first' (min) k, j, i0 in fine grid for given coarse cell."""

        base = []
        for axis in range(3):
            base.append(self.fine_base_for_coarse_axial(axis, c_kji0[axis]))
        return tuple(base)

    def fine_box_for_coarse(self, c_kji0):
        """Return the min, max for k, j, i0 in fine grid for given coarse cell.
        
        Returns:
            Numpy int array of shape (2, 3) being the min, max for k, j, i0
        """

        box = np.empty((2, 3), dtype = int)
        box[0] = self.fine_base_for_coarse(c_kji0)
        box[1] = box[0] + self.ratios(c_kji0) - 1
        return box

    def proportion(self, axis, c0):
        """Return the axial relative proportions of fine within coarse.
        
        Returns:
            numpy vector of floats, summing to one
        """

        if self.equal_proportions[axis]:
            count = self.ratio(axis, c0)
            fraction = 1.0 / float(count)
            return np.full((count,), fraction)
        return self.vector_proportions[axis][c0]

    def proportions_for_axis(self, axis):
        """Return the axial relative proportions as array of floats summing to one for each coarse slice."""

        if self.equal_proportions[axis]:
            if self.constant_ratios[axis] is None:
                assert self.vector_ratios[axis] is not None and len(
                    self.vector_ratios[axis]) == self.coarse_extent_kji[axis]
                fractions = 1.0 / self.vector_ratios[axis].astype(float)
                proportions = np.zeros((self.fine_extent_kji[axis],), dtype = float)
                fi = 0
                for ci in range(self.coarse_extent_kji[axis]):
                    proportions[fi:fi + self.vector_ratios[axis][ci]] = fractions[ci]
                    fi += self.vector_ratios[axis][ci]
                return proportions
            count = self.constant_ratios[axis]
            fraction = 1.0 / float(count)
            return np.full((self.fine_extent_kji[axis],), fraction)
        return np.concatenate(self.vector_proportions[axis])

    def interpolation(self, axis, c0):
        """Return a float array ready for interpoltion.
        
        Returns floats starting at zero and increasing monotonically to less than one.
        """
        count = self.ratio(axis, c0)
        fractions = np.zeros((count,))
        proport = self.proportion(axis, c0)
        for f0 in range(1, count):
            fractions[f0] = fractions[f0 - 1] + proport[f0 - 1]
        return fractions

    def proportions(self, c_kji0):
        """Return triplet of axial proportions for refinement of coarse cell."""

        return (self.proportion(axis, c_kji0[axis]) for axis in range(3))

    def set_constant_ratio(self, axis):
        """Set the refinement ratio for axis based on the ratio of the fine to coarse extents."""

        assert 0 <= axis < 3
        extent_ratio, remainder = divmod(self.fine_extent_kji[axis], self.coarse_extent_kji[axis])
        assert remainder == 0, 'coarse extent in ' + 'KJI'[axis] + ' is not a multiple of fine extent'
        self.constant_ratios[axis] = extent_ratio
        self.vector_ratios[axis] = None

    def set_ij_ratios_constant(self):
        """Set the refinement ratio for I & J axes based on the ratio of the fine to coarse extents."""

        for axis in (1, 2):
            self.set_constant_ratio(axis)

    def set_all_ratios_constant(self):
        """Set all refinement ratios constant based on the ratio of the fine to coarse extents."""

        for axis in range(3):
            self.set_constant_ratio(axis)

    def set_ratio_vector(self, axis, vector):
        """Set fine:coarse ratios for axis from numpy int vector of length matching coarse extent."""

        assert 0 <= axis < 3
        if isinstance(vector, list) or isinstance(vector, tuple):
            vector = np.array(vector)
        assert isinstance(vector, np.ndarray) and vector.ndim == 1
        assert str(vector.dtype).startswith('int')
        assert len(vector) == self.coarse_extent_kji[axis],  \
               f'length of vector {len(vector)} of refinement ratios ' +  \
               f'does not match coarse extent {self.coarse_extent_kji[axis]}'
        assert np.sum(vector) == self.fine_extent_kji[axis],  \
               f'sum of refinement ratios {np.sum(vector)} in vector does not match fine extent'
        minimum = np.min(vector)
        assert minimum > 0
        if minimum == np.max(vector):
            self.set_constant_ratio(axis)
            assert self.constant_ratios[axis] == minimum
        else:
            self.vector_ratios[axis] = vector.copy()
            self.constant_ratios[axis] = None

    def set_equal_proportions(self, axis):
        """Set proportions equal for axis."""

        assert 0 <= axis < 3
        self.equal_proportions[axis] = True
        self.vector_proportions[axis] = None

    def set_all_proportions_equal(self):
        """Sets proportions equal in all 3 axes."""

        for axis in range(3):
            self.set_equal_proportions(axis)

    def set_proportions_list_of_vectors(self, axis, list_of_vectors):
        """Sets the proportions for given axis, with one vector for each coarse slice in the axis."""

        assert 0 <= axis < 3
        assert len(list_of_vectors) == self.coarse_extent_kji[axis]
        if self.vector_proportions is None:
            self.vector_proportions = [None, None, None]
        fractions_list_of_vectors = []
        all_equal = True
        for c0 in range(self.coarse_extent_kji[axis]):
            vector = list_of_vectors[c0]
            count = self.ratio(axis, c0)
            if isinstance(vector, list) or isinstance(vector, tuple):
                vector = np.array(vector)
            assert isinstance(vector, np.ndarray) and vector.ndim == 1
            assert len(vector) == count, 'wrong number of proportions for axis: ' + str(axis) + ' slice(0): ' + c0
            assert not np.any(np.isnan(vector))
            assert np.all(vector > 0)
            total = float(np.sum(vector))
            fractions = np.empty((count,), dtype = float)
            fractions[:] = vector
            fractions /= total
            fractions_list_of_vectors.append(fractions)
            if np.max(fractions) - np.min(fractions) > 1.0e-6:
                all_equal = False
        if all_equal:
            self.equal_proportions[axis] = True
            self.vector_proportions[axis] = None
        else:
            self.equal_proportions[axis] = False
            self.vector_proportions[axis] = fractions_list_of_vectors

    def fine_for_coarse_natural_column_index(self, coarse_col):
        """Returns the fine equivalent natural (first) column index for coarse natural column index."""

        j, i = divmod(coarse_col, self.coarse_extent_kji[2])
        ratio_j, ratio_i = self.constant_ratios[1], self.constant_ratios[2]
        if ratio_j is None:
            if j:
                j = np.sum(self.vector_ratios[1][0:j])
        else:
            j *= ratio_j
        if ratio_i is None:
            if i:
                i = np.sum(self.vector_ratios[2][0:i])
        else:
            i *= ratio_i
        return j * self.fine_extent_kji[2] + i

    def fine_for_coarse_natural_pillar_index(self, coarse_p):
        """Returns the fine equivalent natural (first) pillar index for coarse natural pillar index."""

        p_j, p_i = divmod(coarse_p, self.coarse_extent_kji[2] + 1)
        ratio_j, ratio_i = self.constant_ratios[1], self.constant_ratios[2]
        if ratio_j is None:
            if p_j:
                p_j = np.sum(self.vector_ratios[1][0:p_j])
        else:
            p_j *= ratio_j
        if ratio_i is None:
            if p_i:
                p_i = np.sum(self.vector_ratios[2][0:p_i])
        else:
            p_i *= ratio_i
        return p_j * (self.fine_extent_kji[2] + 1) + p_i

    def write_cartref(self,
                      filename,
                      lgr_name,
                      mode = 'a',
                      root_name = None,
                      preceeding_blank_lines = 0,
                      trailing_blank_lines = 0):
        """Write Nexus ascii input format CARTREF; within_coarse_box must have been set."""

        assert self.within_coarse_box is not None, 'box within larger coarse grid required for cartref'
        assert filename and lgr_name, 'filename or lgr name missing for cartref'
        assert ' ' not in lgr_name, 'lgr name may not contain spaces'
        if len(lgr_name) > 8 or (root_name and len(root_name) > 8):
            log.warning('Nexus might only differentiate first 8 characters of grid names')
        with open(filename, mode) as fp:
            for _ in range(preceeding_blank_lines):
                fp.write('\n')
            if root_name:
                fp.write('LGR ' + str(root_name) + '\n')
            fp.write('LGR\n')
            fp.write('CARTREF ' + str(lgr_name) + '\n')
            fp.write('   ' + box.spaced_string_iijjkk1_for_box_kji0(self.within_coarse_box) + '\n')
            for axis in range(2, -1, -1):
                fp.write('  ')
                for c0 in range(self.coarse_extent_kji[axis]):
                    fp.write(' ' + str(self.ratio(axis, c0)))
                fp.write('\n')
                if not self.equal_proportions[axis]:
                    log.warning('unequal propertions in axis ' + 'KJI'[axis] + ': define Lgr corp separately')
            fp.write('ENDREF\n')
            fp.write('ENDLGR\n')
            for _ in range(trailing_blank_lines):
                fp.write('\n')

    def _assert_extents_valid(self):

        for axis in range(3):
            assert _is_int(self.fine_extent_kji[axis]) and _is_int(self.coarse_extent_kji[axis])
            assert self.fine_extent_kji[axis] > 0
            assert self.coarse_extent_kji[axis] <= self.fine_extent_kji[axis]

    def _set_fine_to_coarse_mapping(self):
        self.fine_to_coarse_mapping = [None, None, None]
        for axis in range(3):
            self.fine_to_coarse_mapping[axis] = np.empty(self.fine_extent_kji[axis], dtype = int)
            f0 = 0
            for c0 in range(self.coarse_extent_kji[axis]):
                ratio = self.ratio(axis, c0)
                self.fine_to_coarse_mapping[axis][f0:f0 + ratio] = c0
                f0 += ratio
            assert f0 == self.fine_extent_kji[axis]


def tartan_refinement(coarse_extent_kji,
                      coarse_fovea_box,
                      fovea_ratios_kji,
                      decay_rates_kji = None,
                      decay_mode = 'exponential',
                      within_coarse_box = None):
    """Returns a new FineCoarse object set to a tartan grid refinement; fine extent is determined from arguments.

    arguments:
       coarse_extent_kji (triple int): the extent of the coarse grid being refined
       coarse_fovea_box (numpy int array of shape (2, 3)): the central box within the coarse grid to receive maximum refinement
       fovea_ratios_kji (triple int): the maximum refinement ratios, to be applied in the coarse_fovea_box
       decay_rates_kji (triple float or triple int, optional): controls how quickly refinement ratio reduces in slices away from fovea;
          if None then default values will be generated; see notes for more details
       decay_mode (str, default 'exponential'): 'exponential' or 'linear'; see notes
       within_coarse_box (numpy int array of shape (2, 3), optional): if present, is preserved in FineCoarse for possible use
          in setting resqml ParentWindow or generating Nexus CARTREF

    returns:
       FineCoarse object holding the tartan refinement mapping

    notes:
       each axis is treated independently;
       the fovea (box of maximum refinement) may be a column of cells (for a vertical well) or any other logical cuboid;
       the refinement factor is reduced monotonically in slices moving away from the fovea;
       two refinement factor decay functions are available: 'exponential' and 'linear', with different meaning to decay_rates_kji;
       for exponential decay, each decay rate should be a float in the range 0.0 to 1.0, with 0.0 causing immediate change
       to no refinement (factor 1), and 1.0 causing no decay (constant refinement at fovea factor);
       for linear decay, each decay rate should typically be a non-negative integer (though float is also okay), with 0 causing
       no decay, 1 causing a reduction in refinement factor of 1 per coarse slice, 2 meaning refinement factor reduces by 2 with
       each coarse slice etc.;
       in all cases, the refinement factor is given a lower limit of 1;
       the factor is rounded to an int for each slice, when working with floats;
       if decay rates are not passed as arguments, suitable values are generated to give a gradual reduction in refinement to a
       ratio of one at the boundary of the grid
    """

    assert box.valid_box(coarse_fovea_box, coarse_extent_kji)
    assert decay_mode in ['exponential', 'linear']
    assert len(fovea_ratios_kji) == 3

    fovea_ratios_kji = np.array(fovea_ratios_kji)
    assert np.all(fovea_ratios_kji >= 1)

    # generate decay rates if not passed in
    rates = np.zeros((2, 3), dtype = float)  # -/+ side of fovea, in each axis
    if decay_rates_kji is None:
        border = np.zeros((2, 3), dtype = float)
        border[0] = coarse_fovea_box[0].astype(float) + 0.5
        border[1] = (coarse_extent_kji - coarse_fovea_box[1]).astype(float) - 0.5
        for side in (0, 1):
            if decay_mode == 'linear':
                rates[side] = (fovea_ratios_kji - 1) / border[side]
            else:  # exponential decay
                rates[side] = np.power(1.5 / fovea_ratios_kji, 1.0 / border[side])
    else:
        assert len(decay_rates_kji) == 3
        rates[:] = decay_rates_kji

    log.debug(f'decay rates: {rates}')

    ratios_list = []
    fine_extent_kji = np.zeros(3, dtype = int)

    for axis in range(3):
        ratios = np.ones(coarse_extent_kji[axis], dtype = int)
        slice_a, slice_b = coarse_fovea_box[:, axis]
        ratios[slice_a:slice_b + 1] = fovea_ratios_kji[axis]
        floating_ratio = np.ones(2, dtype = float)
        if decay_mode == 'linear':
            floating_ratio[:] = fovea_ratios_kji[axis] - rates[:, axis]
        else:
            floating_ratio[:] = fovea_ratios_kji[axis] * rates[:, axis]  # exponential decay
        rounded_ratio = np.around(floating_ratio).astype(int)
        while np.any(rounded_ratio > 1) and (slice_a > 0 or slice_b < coarse_extent_kji[axis] - 1):
            slice_a -= 1
            slice_b += 1
            if slice_a >= 0:
                ratios[slice_a] = rounded_ratio[0]
            if slice_b < coarse_extent_kji[axis]:
                ratios[slice_b] = rounded_ratio[1]
            if decay_mode == 'linear':
                floating_ratio[:] -= rates[:, axis]
            else:
                floating_ratio[:] *= rates[:, axis]  # exponential decay
            rounded_ratio = np.around(floating_ratio).astype(int)
        ratios_list.append(ratios)
        fine_extent_kji[axis] = np.sum(ratios)


#      log.debug(f'retio vector [{axis}]: {ratios}')

    fc = FineCoarse(fine_extent_kji, coarse_extent_kji, within_coarse_box = within_coarse_box)

    for axis in range(3):
        fc.set_ratio_vector(axis, ratios_list[axis])
        # todo: set proportions ?

    return fc


def axis_for_letter(letter):
    """Returns 0, 1 or 2 for 'K', 'J', or 'I'; as required for axis arguments in FineCoarse methods."""

    assert isinstance(letter, str) and len(letter) == 1
    u = letter.upper()
    return 'KJI'.index(u)


def letter_for_axis(axis):
    """Returns K, J, or I for axis; axis as required for axis arguments in FineCoarse methods."""

    assert isinstance(axis, int) and 0 <= axis < 3
    return 'KJI'[axis]


def _assert_valid_box(extent_kji, box):
    assert isinstance(box, np.ndarray) and box.shape == (2, 3) and str(box.dtype).startswith('int')
    for axis in range(3):
        assert 0 <= box[0, axis] <= box[1, axis]
        assert box[1, axis] - box[0, axis] + 1 == extent_kji[axis]


def _is_int(v):
    t = type(v)
    if t is int:
        return True
    if t in [np.int64, np.int32, np.int16, np.int8]:
        return True
    s = str(t)
    if s.startswith('int'):
        return True
    return False
