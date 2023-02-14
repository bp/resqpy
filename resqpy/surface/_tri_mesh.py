"""TriMesh class using equilateral triangles, derived from Mesh."""

import logging

log = logging.getLogger(__name__)

import math as maths
import numpy as np

import resqpy.surface as rqs
import resqpy.olio.vector_utilities as vec

root_3 = maths.sqrt(3.0)
root_3_by_2 = root_3 / 2.0


class TriMesh(rqs.Mesh):
    """Class of mesh using equilateral triangles in the xy plane."""

    def __init__(self,
                 parent_model,
                 uuid = None,
                 t_side = None,
                 nj = None,
                 ni = None,
                 origin = None,
                 z_values = None,
                 z_uom = None,
                 surface_role = 'map',
                 crs_uuid = None,
                 title = None,
                 originator = None,
                 extra_metadata = None):
        """Initialises a TriMesh object from xml, or from arguments.

        arguments:
           parent_model (model.Model object): the model to which this Mesh object will be associated
           uuid (uuid.UUID, optional): the uuid of an existing RESQML obj_Grid2dRepresentation object from which
               this resqpy Mesh object is populated
           t_side (float, optional): the length of a side of each triangle; units are xy units of crs;
               required if uuid is None
           nj (int, optional): the number of nodes (NB. not 'cells') in the j axis of the mesh;
               required if uuid is None
           ni (int, optional): the number of nodes in the i axis of the mesh; required if uuid is None
           origin (triple float, optional): the xyz origin of the regular mesh; z usually zero; defaults
               to triple zero
           z_values (numpy int array of shape (nj, ni), optional): z values; recommended if uuid is None or will
               default to zero
           z_uom (str, optional): recommended if uuid is None and z values are not in the crs z space
           surface_role (string, default 'map'): 'map' or 'pick'; ignored if uuid is not None
           crs_uuid (uuid.Uuid or string, optional): required if generating a regular mesh, the uuid of the crs
           title (str, optional): the citation title to use for a new mesh;
              ignored if uuid is not None
           originator (str, optional): the name of the person creating the mesh, defaults to login id;
              ignored if uuid is not None
           extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the mesh;
              ignored if uuid is not None

        returns:
           the newly created TriMesh object
        """

        if uuid is None:
            assert t_side > 0.0
            assert nj > 1 and ni > 1
            assert z_values is None or z_values.shape == (nj, ni)
            assert origin is None or len(origin) == 3
            assert crs_uuid is not None
            xyz = np.zeros((nj, ni, 3), dtype = float)
            xyz[:, :, 0] = np.expand_dims(np.arange(ni).astype(float), axis = 0)  # x
            xyz[1::2, :, 0] += 0.5  # offset every second row by half triangle side length
            xyz[:, :, 1] = root_3_by_2 * np.expand_dims(np.arange(nj).astype(float), axis = 1)
            xyz *= t_side
            if z_values is not None:
                xyz[:, :, 2] = z_values
            if origin is not None:
                xyz += np.expand_dims(np.expand_dims(origin, axis = 0), axis = 0)
            super().__init__(parent_model,
                             mesh_flavour = 'explicit',
                             xyz_values = xyz,
                             nj = nj,
                             ni = ni,
                             surface_role = surface_role,
                             crs_uuid = crs_uuid,
                             title = title,
                             originator = originator,
                             extra_metadata = extra_metadata)
            self.t_side = t_side
            self.origin = None
            if origin is not None:
                self.origin = np.array(origin, dtype = float)
                assert self.origin.shape == (3,)
            self.z_uom = z_uom
            self.extra_metadata['t side'] = str(t_side)
            if z_uom:
                self.extra_metadata['z uom'] = str(z_uom)
            self.represented_interpretation_root = None

        else:
            super().__init__(parent_model, uuid = uuid)
            assert self.flavour == 'explicit', 'mesh flavour must be explicit for TriMesh objects'
            assert hasattr(self, 'extra_metadata')
            t_side = self.extra_metadata.get('t side')
            assert t_side is not None, 'triangle side length missing in TriMesh extra metadata'
            self.t_side = float(t_side)
            self.z_uom = self.extra_metadata.get('z uom')
            origin = self.full_array_ref()[0, 0]
            if np.all(np.isclose(origin, 0.0)):
                self.origin = None
            else:
                self.origin = origin

    @classmethod
    def from_tri_mesh_and_z_values(cls, tri_mesh, z_values, z_uom = None, title = None, surface_role = 'map'):
        """Create a new tri mesh with the same xy pattern as an existing one, updating z."""
        assert z_values.shape == (tri_mesh.nj, tri_mesh.ni)
        if not z_uom:
            z_uom = tri_mesh.z_uom
        if not title:
            title = tri_mesh.title
        tm = cls(tri_mesh.model,
                 uuid = None,
                 t_side = tri_mesh.t_side,
                 nj = tri_mesh.nj,
                 ni = tri_mesh.ni,
                 origin = tri_mesh.origin,
                 z_values = z_values,
                 z_uom = z_uom,
                 surface_role = surface_role,
                 crs_uuid = tri_mesh.crs_uuid,
                 title = title)
        return tm

    def tji_for_xy(self, xy):
        """Return (tj, ti) indices of triangle containing point xy (x, y).

        note:
            tj triangle indices are in the range 0..nj - 2 inclusive
            ti triangle indices are in the range 0..(ni - 2) * 2 + 1 inclusive
        """
        tji, _ = self.tji_tc_for_xy(xy)
        return tji

    def tji_tc_for_xy(self, xy):
        """Return indices of triangle containing point xy (x, y), and trilinear coordinates within triangle.

        arguments:
            xy (pair of floats): coordinates x, y of a point of interest

        returns:
            ((tj, ti), (f0, f1, f2)): the indices of the triangle containing the point xy, and the trilinear
            coordinates of the point within the triangle; f2 is component from base edge towards point 2;
            f1 is component towards point 1; f0 is component towards point 0; the trilinear coordinates
            sum to one and can be used as weights to interpolate z values at point xy
        """
        x, y = xy
        if self.origin is not None:
            x -= self.origin[0]
            y -= self.origin[1]
        jp = y / (self.t_side * root_3_by_2)
        j = maths.floor(jp)
        if not 0 <= j < self.nj - 1:
            return (None, None)
        if j % 2:
            fy = float(j + 1) - jp
        else:
            fy = jp - float(j)
        ip = x / self.t_side - 0.5 * fy
        i = maths.floor(ip)
        if not 0 <= i < self.ni - 1:
            return (None, None)
        fx = ip - float(i)
        i *= 2
        if fx > 1.0 - fy:
            i += 1
            fx -= 1.0 - fy
            fy = 1.0 - fy
        return ((j, i), (1.0 - (fx + fy), fx, fy))

    def ji_and_weights_for_xy(self, xy):
        """Returns triplet of point ji indices and triplet of weights for interpolation at point xy.

        arguments:
            xy (pair of floats): coordinates x, y of a point of interest

        returns:
            numpy int array of shape (3, 2) and numpy triple float; int array holds mesh ji indices of three
            vertices of triangle containing xy; float triplet contains corresponding weights (summing to
            one) which can be used to interpolate z values at point xy
        """
        tji, tc = self.tji_tc_for_xy(xy)
        if tji is None:
            return (None, None)
        ji = self.tri_nodes_for_tji(tji)
        return (ji, tc)

    def interpolated_z(self, xy):
        """Returns z value interpolated trilinearly at point xy."""
        ji, tc = self.ji_and_weights_for_xy(xy)
        if ji is None:
            return None
        p = self.full_array_ref()
        z = np.empty(3, dtype = float)
        for vertex in range(3):
            z[vertex] = p[ji[vertex, 0], ji[vertex, 1], 2]
        return np.sum(tc * z)

    def tri_nodes_for_tji(self, tji):
        """Return mesh node indices, shape (3, 2), for triangle tji (tj, ti)."""
        j, i = tji
        tn = np.zeros((3, 2), dtype = int)
        j_odd = j % 2
        i2, i_odd = divmod(i, 2)
        assert 0 <= j < self.nj - 1 and 0 <= i < 2 * (self.ni - 1)
        if i_odd:
            base_j = j + 1 - j_odd
            tip_j = j + j_odd
        else:
            base_j = j + j_odd
            tip_j = j + 1 - j_odd
        tn[0, 0] = base_j
        tn[1, 0] = base_j
        tn[2, 0] = tip_j
        tn[0, 1] = i2
        tn[1, 1] = i2 + 1
        tn[2, 1] = i2 + i_odd
        return tn

    def all_tri_nodes(self):
        """Returns array of mesh node indices for all triangles, shape (nj - 1, 2 * (ni - 1), 3, 2)."""
        tna = np.zeros((self.nj - 1, 2 * (self.ni - 1), 3, 2), dtype = int)
        # set mesh j indices
        tna[:, :, 0, 0] = np.expand_dims(np.arange(self.nj - 1, dtype = int), axis = -1)
        tna[1::2, ::2, 0, 0] += 1
        tna[::2, 1::2, 0, 0] += 1
        tna[:, :, 1, 0] = tna[:, :, 0, 0]
        tna[:, :, 2, 0] = tna[:, :, 0, 0] + 1
        tna[1::2, ::2, 2, 0] -= 2
        tna[::2, 1::2, 2, 0] -= 2
        # set mesh i indices
        tna[:, ::2, 0, 1] = np.expand_dims(np.arange(self.ni - 1, dtype = int), axis = 0)
        tna[:, 1::2, 0, 1] = tna[:, ::2, 0, 1]
        tna[:, :, 1, 1] = tna[:, :, 0, 1] + 1
        tna[:, :, 2, 1] = tna[:, :, 0, 1]
        tna[:, 1::2, 2, 1] += 1
        return tna

    def triangles_and_points(self):
        """Returns node indices and xyz points in form suitable for a Surface (triangulated set)."""
        tna = self.all_tri_nodes()
        composite_ji = tna[:, :, :, 0] * self.ni + tna[:, :, :, 1]
        return (composite_ji.reshape((-1, 3)), self.full_array_ref().reshape((-1, 3)))

    def tji_for_triangle_index(self, ti):
        """Return triangle tji (tj, ti) for triangle index in Surface (triangulated set) protocol."""
        assert 0 <= ti < (self.nj - 1) * (self.ni - 1) * 2
        return divmod(ti, 2 * (self.ni - 1))

    def triangle_index_for_tji(self, tji):
        """Return triangle index in Surface (triangulated set) protocol for triangle tji (tj, ti)."""
        tj, ti = tji
        assert 0 <= tj < self.nj - 1 and 0 <= ti < 2 * (self.ni - 1)
        return 2 * (self.ni - 1) * tj + ti

    def tri_nodes_in_triangles(self, triangles):
        """Return indices of nodes of this tri mesh which are within other triangles, in xy space.

        arguments:
            triangles (numpy float array of shape (N, 3, 2)): other triangles' vertices in 2D (last axis is x,y)

        returns:
            numpy int array (list-like) of shape (M, 3) where last axis is (triangle index, nj, ni)

        note:
            other triangles must be in the same crs as this tri mesh; they do not need to be equilateral
        """

        assert triangles.ndim == 3 and triangles.shape[1] == 3 and triangles.shape[2] == 2

        tp = triangles.copy()
        if self.origin is not None:
            tp[:] -= np.expand_dims(np.expand_dims(self.origin[:2], axis = 0), axis = 0)

        # test odd node j rows of tri mesh points; note that vec function assumes a half 'cell' offset!
        tn_b = vec.points_in_triangles_aligned_optimised(self.ni, self.nj // 2, self.t_side,
                                                         2.0 * root_3_by_2 * self.t_side, tp)
        tn_b[:, 1] *= 2  # node j
        tn_b[:, 1] += 1

        # shift other triangles' points so as to compensate for half cell offset, for even j node rows
        offset = np.array((0.5, root_3_by_2), dtype = float) * self.t_side
        tp[:] += np.expand_dims(np.expand_dims(offset, axis = 0), axis = 0)

        # test even node j rows of tri mesh points
        tn_a = vec.points_in_triangles_aligned_optimised(self.ni, (self.nj + 1) // 2, self.t_side, root_3 * self.t_side,
                                                         tp)
        tn_a[:, 1] *= 2  # node j

        return np.concatenate((tn_a, tn_b), axis = 0)

    def edge_zero_crossings(self, z_values = None):
        """Returns numpy list of points from edges where z values cross zero (or given value).

        arguments:
            z_values (numpy float array of shape (nj, ni), optional): if present, these values
                are used to determine the crossing points, though returned xyz values are from
                the tri mesh's array

        returns:
            numpy float array of shape (N, 3) being points on tri mesh triangle edges where
            z values cross zero
        """

        xyz = self.full_array_ref()
        if z_values is None:
            z_values = xyz[..., 2]
        else:
            assert z_values.shape == (self.nj, self.ni)
        abs_ze = np.expand_dims(np.abs(z_values), axis = -1)
        p = np.empty((0, 3), dtype = float)

        # i edges
        crossing = np.where(z_values[:, :-1] * z_values[:, 1:] < 0.0)  # zero crossing i edge indices
        if len(crossing[0]):
            s = abs_ze[:, :-1, :] + abs_ze[:, 1:, :]
            # TODO: ignore divide by zero
            w = (xyz[:, :-1, :] * abs_ze[:, 1:, :] + xyz[:, 1:, :] * abs_ze[:, :-1, :]) / s
            # TODO: unignore
            p = np.concatenate((p, w[crossing]))

        # one set of j edges
        crossing = np.where(z_values[:-1, :] * z_values[1:, :] < 0.0)  # zero crossing j edge indices
        if len(crossing[0]):
            s = abs_ze[:-1, :, :] + abs_ze[1:, :, :]
            w = (xyz[:-1, :, :] * abs_ze[1:, :, :] + xyz[1:, :, :] * abs_ze[:-1, :, :]) / s
            p = np.concatenate((p, w[crossing]))

        # other set of j edges (even j base)
        crossing = np.where(z_values[:-1:2, 1:] * z_values[1::2, :-1] < 0.0)  # zero crossing j edge indices
        if len(crossing[0]):
            s = abs_ze[:-1:2, 1:, :] + abs_ze[1::2, :-1, :]
            w = (xyz[:-1:2, 1:, :] * abs_ze[1::2, :-1, :] + xyz[1::2, :-1, :] * abs_ze[:-1:2, 1:, :]) / s
            p = np.concatenate((p, w[crossing]))

        # other set of j edges (odd j base)
        crossing = np.where(z_values[1:-1:2, :-1] * z_values[2::2, 1:] < 0.0)  # zero crossing j edge indices
        if len(crossing[0]):
            s = abs_ze[1:-1:2, :-1, :] + abs_ze[2::2, 1:, :]
            w = (xyz[1:-1:2, :-1, :] * abs_ze[2::2, 1:, :] + xyz[2::2, 1:, :] * abs_ze[1:-1:2, :-1, :]) / s
            p = np.concatenate((p, w[crossing]))

        return p
