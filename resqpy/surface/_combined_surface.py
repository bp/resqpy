"""Combined surface class."""

import logging

log = logging.getLogger(__name__)

import numpy as np


class CombinedSurface:
    """Class allowing a collection of Surface objects to be treated as a single surface.
    
    Not a RESQML class in its own right.
    """

    def __init__(self, surface_list, crs_uuid = None):
        """Initialise a CombinedSurface object from a list of Surface (and/or CombinedSurface) objects.

        arguments:
           surface_list (list of Surface and/or CombinedSurface objects): the new object is the combination of these surfaces
           crs_uuid (uuid.UUID, optional): if present, all contributing surfaces must refer to this crs

        note:
           all contributing surfaces should be established before initialising this object;
           all contributing surfaces must refer to the same crs; this class of object is not part of the RESQML
           standard and cannot be saved in a RESQML dataset - it is a high level derived object class
        """

        assert len(surface_list) > 0
        self.surface_list = surface_list
        self.crs_uuid = crs_uuid
        if self.crs_uuid is None:
            self.crs_uuid = surface_list[0].crs_uuid
        self.patch_count_list = []
        self.triangle_count_list = []
        self.points_count_list = []
        self.is_combined_list = []
        self.triangles = None
        self.points = None
        for surface in surface_list:
            is_combined = isinstance(surface, CombinedSurface)
            self.is_combined_list.append(is_combined)
            if is_combined:
                self.patch_count_list.append(sum(surface.patch_count_list))
            else:
                self.patch_count_list.append(len(surface.patch_list))
            t, p = surface.triangles_and_points()
            self.triangle_count_list.append(len(t))
            self.points_count_list.append(len(p))

    def surface_index_for_triangle_index(self, tri_index):
        """Return the index of the surface containing the triangle and local triangle index.
        
        Arguments:
            tri_index: triangle index in the combined surface
        """
        for s_i in range(len(self.surface_list)):
            if tri_index < self.triangle_count_list[s_i]:
                return s_i, tri_index
            tri_index -= self.triangle_count_list[s_i]
        return None

    def triangles_and_points(self):
        """Returns the composite triangles and points for the combined surface."""

        if self.triangles is None:
            points_offset = 0
            for surface in self.surface_list:
                (t, p) = surface.triangles_and_points()
                if points_offset == 0:
                    self.triangles = t.copy()
                    self.points = p.copy()
                else:
                    self.triangles = np.concatenate((self.triangles, t.copy() + points_offset))
                    self.points = np.concatenate((self.points, p.copy()))
                points_offset += p.shape[0]

        return self.triangles, self.points
