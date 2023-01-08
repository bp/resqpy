"""TriMesh class using equilateral triangles, derived from Mesh."""

import logging

log = logging.getLogger(__name__)

import math as maths
import numpy as np

import resqpy.surface as rqs

root_3_by_2 = maths.sqrt(3.0) / 2.0


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
                xyz += np.expand_dims(np.expand_dims(origin, axis = -1), axis = -1)
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

    # todo: sample_z method based on trilinear interpolation within triangle
