"""TriMeshStencil class for applying convolutions to TriMesh z values."""

import logging

log = logging.getLogger(__name__)

import math as maths
import numpy as np

import resqpy.surface as rqs

root_3 = maths.sqrt(3.0)
root_3_by_2 = root_3 / 2.0


class TriMeshStencil:
    """Class holding a temporary regular hexagonal symmetrical stencil for use with TriMesh objects.

    note:
       this class does not currently include store and load methods, as the stencil is regarded as a temporary
       disposable object
    """

    # at present the internal representation holds a half hexagon; if there is need for non-symmetric stencils
    # in future then this representation will change to a full hexagon

    def __init__(self, pattern, normalize = None, normalize_mode_flat = True):
        """Initialises a TriMeshStencil object from xml, or from arguments.

        arguments:
            pattern (1D numpy float array): values for one radial arm of the stencil (first value is centre)
            normalize (non zero float, optional): if present, stencil values are normalized to sum to this value;
                if None, no normalization is applied; set to one for smoothing; see also normalize_mode_flat argument
            normalize_mode_flat (bool, default True): if True (and normalize is not None), then ring values
                in the stencil preserve relative weights of unnormalized pattern values; if False, weights of
                ring values (away from centre) are decreased by a factor of the number of elements in the ring;
                in either case, the final normalization is such that the sum of all stencil elements is the
                value of the normalize argument

        returns:
            the newly created TriMeshStencil object

        note:
            the area of inflence of the stencil is hexagonal in shape with a half-axis length equal to the t_side
            length of a target tri mesh multiplied by (n - 1) where n is the length of the pattern
        """

        self.n = None  # length of pattern, including central element
        self.start_ip = None  # 1D numpy int array holding I offsets of start of stencil for each J offset (half hex)
        self.row_length = None  # 1D numpy int array holding number of elements in stencil for each J offset (half hex)
        self.half_hex = None  # 2D numpy float array of stancil values, shifted to left and right padded with NaN

        pattern = np.array(pattern, dtype = float)
        assert pattern.ndim == 1, 'tri mesh stencil pattern must be one dimensional'
        assert len(pattern) > 1, 'tri mesh stancil pattern must contain at least two elements'
        assert not np.any(np.isnan(pattern)), 'tri mesh stencil may not contain NaN'
        if len(pattern) > 50:
            log.warning(f'very large stencil pattern length: {len(pattern)}')

        self.n = len(pattern)

        if normalize is not None:
            assert not maths.isclose(normalize, 0.0), 'calling code must scale to stencil pattern to sum to zero'
            if normalize_mode_flat:
                t = pattern[0]
                for i in range(1, self.n):
                    t += 6 * i * pattern[i]
                assert not maths.isclose(t, 0.0), 'hex pattern sums to zero and cannot be normalized to another value'
                pattern *= normalize / t
            else:
                t = np.sum(pattern)
                assert not maths.isclose(t, 0.0), 'pattern sums to zero and cannot be normalized to another value'
                pattern *= normalize / t
                for i in range(1, self.n):
                    pattern[i] /= float(6 * i)

        self.pattern = pattern

        # build up half hex stencil as list (over J offsets) of row (ie. for a range of I offsets) values from pattern
        half_hex = []  # list of (ip, length, stencil values) for implicit jp in range 0..n-1
        row_length = 2 * self.n - 1
        half_hex.append((-(self.n - 1), row_length, np.concatenate((np.flip(pattern[1:]), pattern))))
        for jp in range(1, self.n):
            start_ip = -(self.n - 1) + (jp // 2)
            row_length -= 1
            row_v = np.full(row_length, pattern[jp], dtype = float)
            if jp < self.n - 1:
                sub_p = pattern[jp + 1:]
                sub_len = sub_p.size
                row_v[-sub_len:] = sub_p
                row_v[:sub_len] = np.flip(sub_p)
            half_hex.append((start_ip, row_length, row_v))

        # convert into representation more njit & numpy friendly
        self.start_ip = np.array([ip for (ip, _, _) in half_hex], dtype = int)
        self.row_length = np.array([rl for (_, rl, _) in half_hex], dtype = int)
        self.half_hex = np.full((self.n, 2 * self.n - 1), np.NaN, dtype = float)
        for jp in range(self.n):
            self.half_hex[jp, :self.row_length[jp]] = half_hex[jp][2]

    @classmethod
    def for_constant_normalized(cls, n, normalize_mode_flat = True):
        """Create a tri mesh stencil with a constant pattern, then normalized to sum to one.

        arguments:
            n (int); the length of pattern (ie. half width of hexagon); must be at least 2

        returns:
           the newly created TriMeshStencil object
        """

        assert n > 1, 'tri mesh stancil pattern must contain at least two elements'
        pattern = np.ones(n, dtype = float)
        stencil = cls(pattern, normalize = 1.0, normalize_mode_flat = normalize_mode_flat)

        return stencil

    @classmethod
    def for_constant_unnormalized(cls, n, c):
        """Create a tri mesh stencil with a constant pattern, without normalization.

        arguments:
            n (int); the length of pattern (ie. half width of hexagon); must be at least 2
            c (float): the value to use throughout the stencil

        returns:
           the newly created TriMeshStencil object
        """

        assert n > 1, 'tri mesh stancil pattern must contain at least two elements'
        pattern = np.full(n, c, dtype = float)
        stencil = cls(pattern, normalize = None)

        return stencil

    @classmethod
    def for_linear_normalized(cls, n, normalize_mode_flat = True):
        """Create a tri mesh stencil with a linearly decreasing pattern, then normalized to sum to one.

        arguments:
            n (int); the length of pattern (ie. half width of hexagon); must be at least 2

        returns:
           the newly created TriMeshStencil object
        """

        assert n > 1, 'tri mesh stancil pattern must contain at least two elements'
        pattern = np.flip((np.arange(n, dtype = int) + 1).astype(float))
        stencil = cls(pattern, normalize = 1.0, normalize_mode_flat = normalize_mode_flat)

        return stencil

    @classmethod
    def for_linear_unnormalized(cls, n, centre, outer):
        """Create a tri mesh stencil with a linearly decreasing pattern, without normalization.

        arguments:
            n (int); the length of pattern (ie. half width of hexagon); must be at least 2
            centre (float): the value of the pattern at the centre of the stencil
            outer (float): the value of the pattern in the outer ring of the stencil

        returns:
           the newly created TriMeshStencil object
        """

        assert n > 1, 'tri mesh stancil pattern must contain at least two elements'
        pattern = np.flip(np.arange(n, dtype = int).astype(float) * (centre - outer) / float(n - 1) + outer)
        stencil = cls(pattern, normalize = None)

        return stencil

    @classmethod
    def for_gaussian_normalized(cls, n, sigma = 3.0, normalize_mode_flat = True):
        """Create a tri mesh stencil with a gaussian pattern, then normalized to sum to one.

        arguments:
            n (int); the length of pattern (ie. half width of hexagon); must be at least 2
            sigma (float, default 3.0): the number of standard deviations at the outermost ring of the stencil

        returns:
           the newly created TriMeshStencil object
        """

        assert n > 1, 'tri mesh stancil pattern must contain at least two elements'
        assert sigma > 0.0, 'number of standard deviations in pattern must be positive'
        pattern = _gaussian_pattern(n, sigma)
        stencil = cls(pattern, normalize = 1.0, normalize_mode_flat = normalize_mode_flat)

        return stencil

    @classmethod
    def for_gaussian_unnormalized(cls, n, centre, sigma = 3.0):
        """Create a tri mesh stencil with a gaussian pattern, without normalization.

        arguments:
            n (int); the length of pattern (ie. half width of hexagon); must be at least 2
            centre (float): the value at the centre of the stencil (peak amplitude)
            sigma (float, default 3.0): the number of standard deviations at the outermost ring of the stencil

        returns:
           the newly created TriMeshStencil object
        """

        assert n > 1, 'tri mesh stancil pattern must contain at least two elements'
        assert sigma > 0.0, 'number of standard deviations in pattern must be positive'
        pattern = centre * _gaussian_pattern(n, sigma)
        stencil = cls(pattern, normalize = None)

        return stencil

    # todo: class methods for mexican hat

    def apply(self, tri_mesh, handle_nan = True, border_value = np.NaN, preserve_nan = False, title = None):
        """Return a new tri mesh with z values generated by applying the stencil to z values of an existing tri mesh.

        arguments:
            tri_mesh (TriMesh): an existing tri mesh to apply the stencil to
            handle_nan (bool, default True): if True, a smoothing style weighted average of non-NaN values is used;
                if False, a simple convolution is applied and will yield NaN where any input within the stencil area
                is NaN
            border_value (float, default NaN): the pre-filled value for an extended bprder around the input tri mesh;
                set to zero if partial convolution values are wanted at the edge of the tri mesh when handle_nan is False
            preserve_nan (bool, default False): if True (and handle_nan is True) then where the input tri mesh has
                a NaN z value, the output will also have NaN; if False, patches of NaNs smaller than the stencil
                will get 'smoothed over', ie. filled in, if handle_nan is True
            title (str, optional): the title to use for the new tri mesh; if None, title is inherited from input

        returns:
            a new TriMesh in the same model as the input tri mesh, with the stencil having been applied to z values

        notes:
            this method does not write hdf5 nor create xml for the new tri mesh;
            if handle_nan is False and border_value is NaN, the result will have NaN values around the edge of
            the tri mesh, to a depth equivalent to the pattern length
        """

        log.info(f'applying stencil to tri mesh: {tri_mesh.title}')

        # create a temporary tri mesh style z array as a copy of the original with a NaN border
        border = self.n
        if border % 2:
            border += 1
        e_nj = tri_mesh.nj + 2 * border
        e_ni = tri_mesh.ni + 2 * border
        z_values = np.full((e_nj, e_ni), border_value, dtype = float)
        tm_z = tri_mesh.full_array_ref()[:, :, 2]
        nan_mask = None
        if preserve_nan:
            nan_mask = np.isnan(tm_z)
            if not np.any(nan_mask):
                nan_mask = None
        z_values[border:border + tri_mesh.nj, border:border + tri_mesh.ni] = tm_z
        applied = np.full((e_nj, e_ni), np.NaN, dtype = float)

        if handle_nan:
            _apply_stencil_nanmean(self.n, self.start_ip, self.row_length, self.half_hex, z_values, applied, border,
                                   tri_mesh.nj, tri_mesh.ni)
        else:
            _apply_stencil_simple(self.n, self.start_ip, self.row_length, self.half_hex, z_values, applied, border,
                                  tri_mesh.nj, tri_mesh.ni)

        # restore NaN values where they were present in input, if requested
        tm_z_applied = applied[border:border + tri_mesh.nj, border:border + tri_mesh.ni]
        if nan_mask is not None:
            tm_z_applied[nan_mask] = np.NaN

        # create a new tri mesh object using the values from the applied array for z
        tm = rqs.TriMesh(tri_mesh.model,
                         t_side = tri_mesh.t_side,
                         nj = tri_mesh.nj,
                         ni = tri_mesh.ni,
                         origin = tri_mesh.origin,
                         z_uom = tri_mesh.z_uom,
                         title = title,
                         z_values = tm_z_applied,
                         surface_role = tri_mesh.surface_role,
                         crs_uuid = tri_mesh.crs_uuid)

        return tm

    def log(self, log_level = logging.DEBUG):
        """Outputs ascii representation of stencil to loggger.

        arguments:
            log_level (int, default DEBUG): the logging severity level to use
        """

        half_lines = []
        for jp in range(self.n):
            padding = self.n + self.start_ip[jp]
            line = ''
            if jp % 2:
                line += '    '
            line += '    x   ' * padding
            for v in self.half_hex[jp, :self.row_length[jp]]:
                line += f' {v:7.4f}'
            line += '    x   ' * padding
            half_lines.append(line)
        for i in range(1, self.n):
            log.log(log_level, half_lines[-i])
            log.log(log_level, '')
        for i in range(self.n):
            log.log(log_level, half_lines[i])
            if i < self.n - 1:
                log.log(log_level, '')


# todo: njit with parallel True
def _apply_stencil_simple(n, start_ip, row_length, half_hex, tm_z, applied, border, onj, oni):
    """Apply the stencil to the tri mesh z values as a simple convolution."""

    for j in range(border, border + onj):  # todo: change to numba prange()
        j_odd = j % 2
        for i in range(border, border + oni):
            a = 0.0
            for jp in range(n):
                js_odd = (jp % 2) * j_odd
                i_st = start_ip[jp]
                j_sm = j - jp
                j_sp = j + jp
                for ip in range(row_length[jp]):
                    i_s = i + i_st + ip + js_odd
                    v = tm_z[j_sm, i_s]
                    s = half_hex[jp, ip]
                    # if not np.isnan(v):
                    a += v * s
                    if jp:
                        v = tm_z[j_sp, i_s]
                        # if not np.isnan(v):
                        a += v * s
            applied[j, i] = a


# todo: njit with parallel True
def _apply_stencil_nanmean(n, start_ip, row_length, half_hex, tm_z, applied, border, onj, oni):
    """Apply the stencil to the tri mesh z values with a weighted nanmean (typically for smoothing)."""

    for j in range(border, border + onj):  # todo: change to numba prange()
        j_odd = j % 2
        for i in range(border, border + oni):
            a = 0.0
            ws = 0.0
            for jp in range(n):
                js_odd = (jp % 2) * j_odd
                i_st = start_ip[jp]
                j_sm = j - jp
                j_sp = j + jp
                for ip in range(row_length[jp]):
                    i_s = i + i_st + ip + js_odd
                    v = tm_z[j_sm, i_s]
                    s = half_hex[jp, ip]
                    if not np.isnan(v):
                        ws += s
                        a += v * s
                    if jp:
                        v = tm_z[j_sp, i_s]
                        if not np.isnan(v):
                            ws += s
                            a += v * s
            if ws != 0.0:
                applied[j, i] = a / ws


def _gaussian_pattern(n, sigma):
    x = np.arange(n, dtype = int)
    c = float(n - 1) / sigma
    return np.exp(-((x * x).astype(float) / (2.0 * c * c)))
