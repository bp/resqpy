"""PointSet class based on RESQML standard."""

# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company
# GOCAD is also a trademark of Emerson

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.crs as rcrs
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
import resqpy.surface
import resqpy.surface._base_surface as rqsb
from resqpy.olio.xml_namespaces import curly_namespace as ns


class PointSet(rqsb.BaseSurface):
    """Class for RESQML Point Set Representation within resqpy model object."""  # TODO: Work in Progress

    resqml_type = 'PointSetRepresentation'

    def __init__(self,
                 parent_model,
                 uuid = None,
                 load_hdf5 = False,
                 points_array = None,
                 crs_uuid = None,
                 polyset = None,
                 polyline = None,
                 random_point_count = None,
                 charisma_file = None,
                 irap_file = None,
                 title = None,
                 originator = None,
                 extra_metadata = None):
        """Creates an empty Point Set object and optionally populates from xml or other source.

        arguments:
           parent_model (model.Model object): the model to which the new point set belongs
           uuid (uuid.UUID, optional): if present, the object is populated from the RESQML PointSetRepresentation
              with this uuid
           load_hdf5 (boolean, default False): if True and uuid is present, the actual points are
              pre-loaded into a numpy array; otherwise the points will be loaded on demand
           points_array (numpy float array of shape (..., 2 or 3), optional): if present, the xy(&z) data which
              will constitute the point set; missing z will be set to zero; ignored if uuid is not None
           crs_uuid (uuid.UUID, optional): if present, identifies the coordinate reference system for the points;
              ignored if uuid is not None; if None, 'imported' points will be associated with the
              default crs of the parent model
           polyset (optional): if present, creates a pointset from points in a polylineset
           polyline (optional): if present and random_point_count is None or zero, creates a pointset from
              points in a polyline; if present and random_point_count is set, creates random points within
              the (closed, convex) polyline
           random_point_count (int, optional): if present and polyline is present then the number of random
              points to generate within the (closed) polyline in the xy plane, with z set to 0.0
           charisma_file (optional): if present, creates a pointset from a charisma 3d interpretation file
           irap_file (optional): if present, creates a pointset from an IRAP classic points format file
           title (str, optional): the citation title to use for a new point set;
              ignored if uuid is not None
           originator (str, optional): the name of the person creating the point set, defaults to login id;
              ignored if uuid is not None
           extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the point set;
              ignored if uuid is not None

        returns:
           newly created PointSet object

        :meta common:
        """

        self.crs_uuid = crs_uuid
        self.patch_count = None
        self.patch_ref_list = []  # ordered list of (patch hdf5 ext uuid, path in hdf5, point count)
        self.patch_array_list = []  # ordered list of numpy float arrays (or None before loading), each of shape (N, 3)
        self.full_array = None  # composite points (all patches)
        self.represented_interpretation_root = None
        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata)

        if self.root is not None:
            if load_hdf5:
                self.load_all_patches()

        elif points_array is not None:
            assert self.crs_uuid is not None, 'missing crs uuid when establishing point set from array'
            self.add_patch(points_array)

        elif polyline is not None:  # Points from or within polyline
            self.from_polyline(polyline, random_point_count)

        elif polyset is not None:  # Points from polylineSet
            self.from_polyset(polyset)

        elif charisma_file is not None:  # Points from Charisma 3D interpretation lines
            self.from_charisma(charisma_file)

        elif irap_file is not None:  # Points from IRAP simple points
            self.from_irap(irap_file)

        if not self.title:
            self.title = 'point set'

    def _load_from_xml(self):
        assert self.root is not None
        self.patch_count = rqet.count_tag(self.root, 'NodePatch')
        assert self.patch_count, 'no patches found in xml for point set'
        self.patch_array_list = [None for _ in range(self.patch_count)]
        patch_index = 0
        for child in rqet.list_of_tag(self.root, 'NodePatch'):
            point_count = rqet.find_tag_int(child, 'Count')
            geom_node = rqet.find_tag(child, 'Geometry')
            assert geom_node is not None, 'geometry missing in xml for point set patch'
            crs_uuid = rqet.find_nested_tags_text(geom_node, ['LocalCrs', 'UUID'])
            assert crs_uuid, 'crs uuid missing in geometry xml for point set patch'
            self.check_crs_match(crs_uuid)
            ext_uuid = rqet.find_nested_tags_text(geom_node, ['Points', 'Coordinates', 'HdfProxy', 'UUID'])
            assert ext_uuid, 'missing hdf5 uuid in geometry xml for point set patch'
            hdf5_path = rqet.find_nested_tags_text(geom_node, ['Points', 'Coordinates', 'PathInHdfFile'])
            assert hdf5_path, 'missing internal hdf5 path in geometry xml for point set patch'
            self.patch_ref_list.append((ext_uuid, hdf5_path, point_count))
            patch_index += 1

        ref_node = rqet.find_tag(self.root, 'RepresentedInterpretation')
        if ref_node is not None:
            interp_root = self.model.referenced_node(ref_node)
            self.set_represented_interpretation_root(interp_root)
        # note: load of patches handled elsewhere

    def from_charisma(self, charisma_file):
        """Instantiate a pointset using points from an input charisma file

        arguments:
            charisma_file: a charisma 3d interpretation file
        """
        with open(charisma_file, 'r') as surf:
            cpoints = np.loadtxt(surf, usecols = (6, 7, 8), encoding = 'uft-8')
        self.add_patch(cpoints)
        assert self.crs_uuid is not None, 'crs uuid missing when establishing point set from charisma file'
        if not self.title:
            self.title = charisma_file

    def from_irap(self, irap_file):
        """Instantiate a pointset using points from an input irap file.

        arguments:
            irap_file: a IRAP classic points format file
        """
        with open(irap_file, 'r') as points:
            cpoints = np.loadtxt(points, encoding = 'uft-8')
            # for i, line in enumerate(points.readlines()):
            #     if i == 0:
            #         cpoints = np.array([[float(x) for x in line.split(" ")]])
            #     else:
            #         curr = np.array([[float(x) for x in line.split(" ")]])
            #         cpoints = np.concatenate((cpoints, curr))
        self.add_patch(cpoints)
        assert self.crs_uuid is not None, 'crs uuid missing when establishing point set from irap file'
        if not self.title:
            self.title = irap_file

    def from_polyline(self, polyline, random_point_count = None):
        """Instantiates a pointset using points from an input polyline (PolylineRepresentation) object

        arguments:
            polyline (resqpy.lines.Polyline object): if random_point_count is None or zero, creates a pointset from
              points in a polyline; if present and random_point_count is set, creates random points within
              the (closed, convex) polyline
           random_point_count (int, optional): if present then the number of random
              points to generate within the (closed) polyline in the xy plane, with z set to 0.0

        """
        if random_point_count:
            assert polyline.is_convex()
            points = np.zeros((random_point_count, 3))
            points[:, :2] = np.random.random((random_point_count, 2))
            for p_i in range(random_point_count):
                points[p_i, :2] = polyline.denormalised_xy(points[p_i, 0], points[p_i, 1])
            self.add_patch(points)
        else:
            self.add_patch(polyline.coordinates)
            if polyline.rep_int_root is not None:
                self.set_represented_interpretation_root(polyline.rep_int_root)
        self.check_crs_match(polyline.crs_uuid)
        if not self.title:
            self.title = polyline.title

    def from_polyset(self, polyset):
        """Instantiates a pointset using points from an input polylineset (PolylineSetRepresentation) object

        arguments:
            polyset (resqpy.lines.PolylineSet object): a polylineset object to generate the pointset from
        """
        master_crs = rcrs.Crs(self.model, uuid = polyset.polys[0].crs_uuid)
        if polyset.polys[0].isclosed and vec.isclose(polyset.polys[0].coordinates[0], polyset.polys[0].coordinates[-1]):
            poly_coords = polyset.polys[0].coordinates[:-1].copy()
        else:
            poly_coords = polyset.polys[0].coordinates.copy()
        for poly in polyset.polys[1:]:
            curr_crs = rcrs.Crs(self.model, uuid = poly.crs_uuid)
            assert master_crs is not None
            assert poly_coords is not None
            if not curr_crs.is_equivalent(master_crs):
                shifted = curr_crs.convert_array_to(master_crs, poly.coordinates)
                poly_coords = concat_polyset_points(poly.isclosed, shifted, poly_coords)
            else:
                poly_coords = concat_polyset_points(poly.isclosed, poly.coordinates, poly_coords)
        self.add_patch(poly_coords)
        if polyset.rep_int_root is not None:
            self.set_represented_interpretation_root(polyset.rep_int_root)
        self.check_crs_match(master_crs.uuid)
        if not self.title:
            self.title = polyset.title

    def check_crs_match(self, crs_uuid):
        """Checks if the model crs_uuid matches a given crs_uuid. If the current model crs_uuid is None, the new crs_uuid is used as the model crs_uuid. If a mismatch is found an assertion error is raised."""
        if self.crs_uuid is None:
            self.crs_uuid = crs_uuid
        else:
            assert bu.matching_uuids(crs_uuid, self.crs_uuid), 'mixed coordinate reference systems in point set'

    def set_represented_interpretation_root(self, interp_root):
        """Makes a note of the xml root of the represented interpretation."""

        self.represented_interpretation_root = interp_root

    def single_patch_array_ref(self, patch_index):
        """Load numpy array for one patch of the point set from hdf5, cache and return it."""

        assert 0 <= patch_index < self.patch_count, 'point set patch index out of range'
        if self.patch_array_list[patch_index] is not None:
            return self.patch_array_list[patch_index]
        h5_key_pair = (self.patch_ref_list[patch_index][0], self.patch_ref_list[patch_index][1]
                      )  # ext uuid, path in hdf5
        try:
            self.model.h5_array_element(h5_key_pair,
                                        cache_array = True,
                                        object = self,
                                        array_attribute = 'temp_points',
                                        dtype = 'float')
        except Exception:
            log.exception('hdf5 points failure for point set patch ' + str(patch_index))
        assert self.temp_points.ndim == 2 and self.temp_points.shape[
            1] == 3, 'unsupported dimensionality to points array'
        self.patch_array_list[patch_index] = self.temp_points.copy()
        delattr(self, 'temp_points')
        return self.patch_array_list[patch_index]

    def load_all_patches(self):
        """Load hdf5 data for all patches and cache as separate numpy arrays; not usually called directly."""

        for patch_index in range(self.patch_count):
            self.single_patch_array_ref(patch_index)

    def change_crs(self, required_crs):
        """Changes the crs of the point set, also sets a new uuid if crs changed.

        notes:
           this method is usually used to change the coordinate system for a temporary resqpy object;
           to add as a new part, call write_hdf5() and create_xml() methods
        """

        old_crs = rcrs.Crs(self.model, uuid = self.crs_uuid)
        self.crs_uuid = required_crs.uuid
        if required_crs == old_crs or not self.patch_ref_list:
            log.debug(f'no crs change needed for {self.title}')
            return
        log.debug(f'crs change needed for {self.title} from {old_crs.title} to {required_crs.title}')
        self.load_all_patches()
        self.patch_ref_list = []
        for patch_points in self.patch_array_list:
            required_crs.convert_array_from(old_crs, patch_points)
            self.patch_ref_list.append((None, None, len(patch_points)))
        self.full_array = None  # clear cached full array for point set
        self.uuid = bu.new_uuid()  # hope this doesn't cause problems

    def trim_to_xyz_box(self, xyz_box):
        """Converts point set to a single patch, holding only those points within the xyz box.

        arguments:
           xyz_box (numpy float array of shape (2, 3)): the minimum and maximum range to keep in x,y,z

        notes:
           usually used to reduce the point set volume for a temprary object; a new uuid is assigned;
           to add as a new part, call write_hdf5() and create_xml() methods
        """
        points = self.full_array_ref()
        keep_mask = np.where(
            np.logical_and(np.all(points >= np.expand_dims(xyz_box[0], axis = 0), axis = -1),
                           np.all(points <= np.expand_dims(xyz_box[1], axis = 0), axis = -1)))
        self.patch_count = 0
        self.patch_ref_list = []
        self.patch_array_list = []
        self.full_array = None
        self.add_patch(points[keep_mask, :].copy())
        self.uuid = bu.new_uuid()  # hope this doesn't cause problems

    def trim_to_xy_polygon(self, xy_polygon):
        """Converts point set to a single patch, holding only those points within the polygon when projected in xy.

        arguments:
           xy_polygon (closed convex resqpy.lines.Polyline): the polygon outlining the area in xy within which
              points are kept

        notes:
           usually used to reduce the point set volume for a temprary object; a new uuid is assigned;
           to add as a new part, call write_hdf5() and create_xml() methods
        """
        points = self.full_array_ref()
        keep_mask = xy_polygon.points_are_inside_xy(points)
        self.patch_count = 0
        self.patch_ref_list = []
        self.patch_array_list = []
        self.full_array = None
        self.add_patch(points[keep_mask, :].copy())
        self.uuid = bu.new_uuid()  # hope this doesn't cause problems

    def minimum_xy_area_rectangle(self, delta_theta = 5.0):
        """Returns the xy projection rectangle of minimum area that contains the points.

        arguments:
           delta_theta (float, default 5.0): the incremental angle in degrees to test against

        returns:
           (d1, d2, r) where d1 and d2 are lengths of sides of an xy plane rectangle that just contains the
           points, and d1 <= d2, and r is a bearing in degrees of a d2 (longer) side between 0.0 and 180.0
        """

        def try_angle(pset, theta):
            p = pset.full_array_ref()
            m = vec.rotation_matrix_3d_axial(2, theta)
            p = vec.rotate_array(m, p)[:, :2]
            dxy = np.nanmax(p, axis = 0) - np.nanmin(p, axis = 0)
            return dxy

        assert self.patch_count > 0
        theta = 0.0
        min_dxy = try_angle(self, theta)
        min_area = min_dxy[0] * min_dxy[1]
        min_theta = theta
        while theta < 180.0 - delta_theta:
            theta += delta_theta
            dxy = try_angle(self, theta)
            area = dxy[0] * dxy[1]
            if area < min_area:
                min_dxy = dxy.copy()
                min_area = area
                min_theta = theta
        if min_dxy[0] <= min_dxy[1]:
            return min_dxy[0], min_dxy[1], min_theta
        return min_dxy[1], min_dxy[0], min_theta + 90.0 if min_theta < 90.0 else min_theta - 90.0

    def full_array_ref(self):
        """Return a single numpy float array of shape (N, 3) containing all points from all patches.

        :meta common:
        """

        if self.full_array is not None:
            return self.full_array
        self.load_all_patches()
        if self.patch_count == 1:  # optimisation, as usually the case
            self.full_array = self.patch_array_list[0]
            return self.full_array
        point_count = 0
        for patch_index in range(self.patch_count):
            point_count += self.patch_ref_list[patch_index][2]
        self.full_array = np.empty((point_count, 3))
        full_index = 0
        for patch_index in range(self.patch_count):
            self.full_array[full_index:full_index +
                            self.patch_ref_list[patch_index][2]] = self.patch_array_list[patch_index]
            full_index += self.patch_ref_list[patch_index][2]
        assert full_index == point_count, 'point count mismatch when constructing full array for point set'
        return self.full_array

    def add_patch(self, points_array):
        """Extend the current point set with a new patch of points."""

        assert points_array.ndim >= 2 and points_array.shape[-1] in [2, 3]
        if points_array.shape[-1] == 2:
            shape = list(points_array.shape)
            shape[-1] = 3
            p = np.zeros(shape)
            p[..., :2] = points_array
            points_array = p
        self.patch_array_list.append(points_array.reshape(-1, 3).copy())
        self.patch_ref_list.append((None, None, points_array.shape[0]))
        self.full_array = None
        if self.patch_count is None:
            self.patch_count = 0
        self.patch_count += 1

    def write_hdf5(self, file_name = None, mode = 'a'):
        """Create or append to an hdf5 file, writing datasets for the point set patches after caching arrays.

        :meta common:
        """

        if not file_name:
            file_name = self.model.h5_file_name()
        if self.uuid is None:
            self.uuid = bu.new_uuid()
        # NB: patch arrays must all have been set up prior to calling this function
        h5_reg = rwh5.H5Register(self.model)
        for patch_index in range(self.patch_count):
            h5_reg.register_dataset(self.uuid, 'points_{}'.format(patch_index), self.patch_array_list[patch_index])
        h5_reg.write(file_name, mode = mode)

    def create_xml(self,
                   ext_uuid = None,
                   add_as_part = True,
                   add_relationships = True,
                   title = None,
                   originator = None):
        """Creates a point set representation xml node from this point set object and optionally adds as part of model.

        arguments:
            ext_uuid (uuid.UUID): the uuid of the hdf5 external part holding the points array(s)
            add_as_part (boolean, default True): if True, the newly created xml node is added as a part
                in the model
            add_relationships (boolean, default True): if True, a relationship xml part is created relating the
                new point set part to the crs part (and optional interpretation part)
            title (string): used as the citation Title text; should be meaningful to a human
            originator (string, optional): the name of the human being who created the point set representation part;
                default is to use the login name

        returns:
            the newly created point set representation xml node

        :meta common:
        """

        if ext_uuid is None:
            ext_uuid = self.model.h5_uuid()

        ps_node = super().create_xml(add_as_part = False, title = title, originator = originator)

        if self.crs_uuid is None:
            self.crs_uuid = self.model.crs_uuid  # maverick use of model's default crs

        if self.represented_interpretation_root is not None:
            interp_root = self.represented_interpretation_root
            interp_uuid = bu.uuid_from_string(interp_root.attrib['uuid'])
            interp_part = self.model.part_for_uuid(interp_uuid)
            self.model.create_ref_node('RepresentedInterpretation',
                                       rqet.find_nested_tags_text(interp_root, ['Citation', 'Title']),
                                       interp_uuid,
                                       content_type = self.model.type_of_part(interp_part),
                                       root = ps_node)

        for patch_index in range(self.patch_count):
            p_node = rqet.SubElement(ps_node, ns['resqml2'] + 'NodePatch')
            p_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'NodePatch')
            p_node.text = '\n'

            pi_node = rqet.SubElement(p_node, ns['resqml2'] + 'PatchIndex')
            pi_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
            pi_node.text = str(patch_index)

            ct_node = rqet.SubElement(p_node, ns['resqml2'] + 'Count')
            ct_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
            ct_node.text = str(self.patch_ref_list[patch_index][2])

            geom = rqet.SubElement(p_node, ns['resqml2'] + 'Geometry')
            geom.set(ns['xsi'] + 'type', ns['resqml2'] + 'PointGeometry')
            geom.text = '\n'

            self.model.create_crs_reference(crs_uuid = self.crs_uuid, root = geom)

            points_node = rqet.SubElement(geom, ns['resqml2'] + 'Points')
            points_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dHdf5Array')
            points_node.text = '\n'

            coords = rqet.SubElement(points_node, ns['resqml2'] + 'Coordinates')
            coords.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            coords.text = '\n'

            self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'points_{}'.format(patch_index), root = coords)

        if add_as_part:
            self.model.add_part('obj_PointSetRepresentation', self.uuid, ps_node)
            if add_relationships:
                # todo: add multiple crs'es (one per patch)?
                crs_root = self.model.root_for_uuid(self.crs_uuid)
                self.model.create_reciprocal_relationship(ps_node, 'destinationObject', crs_root, 'sourceObject')
                if self.represented_interpretation_root is not None:
                    self.model.create_reciprocal_relationship(ps_node, 'destinationObject',
                                                              self.represented_interpretation_root, 'sourceObject')
                ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
                ext_node = self.model.root_for_part(ext_part)
                self.model.create_reciprocal_relationship(ps_node, 'mlToExternalPartProxy', ext_node,
                                                          'externalPartProxyToMl')

        return ps_node

    def convert_to_charisma(self, file_name):
        """Output to Charisma 3D interpretation format from a pointset.

        If file_name exists, it will be overwritten.

        args:
            file_name: output file name to save to
        """
        #      hznname = self.title.replace(" ","_")
        lines = []
        self.load_all_patches()
        for patch in self.patch_array_list:
            for points in patch:
                lines.append(f"INLINE :\t1 XLINE :\t1\t{points[0]}\t{points[1]}\t{points[2]}\n")
        with open(file_name, 'w') as f:
            for item in lines:
                f.write(item)

    def convert_to_irap(self, file_name):
        """Output to IRAP simple points format from a pointset.

        If file_name exists, it will be overwritten.

        args:
            file_name: output file name to save to
        """
        #      hznname = self.title.replace(" ","_")
        lines = []
        self.load_all_patches()
        for patch in self.patch_array_list:
            for points in patch:
                lines.append(f"{points[0]} {points[1]} {points[2]}\n")
        with open(file_name, 'w') as f:
            for item in lines:
                f.write(item)


def concat_polyset_points(closed, coordinates, poly_coords):
    """Concatenates two numpy arrays of coordinates for polylines, omitting the final (duplicated) point in a polyline if the polyline is closed.

    arguments:
        closed (bool): True if polyline is closed
        coordinates (np.array): Coordinates to be concatenated
        poly_coords (np.array): Array to concatenate the given coordinates to

    returns:
        poly_coords (np.array) with additional coordinates concatenated
    """
    if closed and vec.isclose(coordinates[0], coordinates[-1]):
        poly_coords = np.concatenate((poly_coords, coordinates[:-1]))
    else:
        poly_coords = np.concatenate((poly_coords, coordinates))
    return poly_coords
