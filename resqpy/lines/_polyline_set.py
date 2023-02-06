"""_polyline_set.py: Resqml polyline set module."""

import logging

log = logging.getLogger(__name__)

import os
import numpy as np

import resqpy.crs as rcrs
import resqpy.lines
import resqpy.olio.simple_lines as rsl
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vu
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
import resqpy.lines._common as rql_c
from resqpy.olio.xml_namespaces import curly_namespace as ns


class PolylineSet(rql_c._BasePolyline):
    """Class for RESQML polyline set representation."""

    resqml_type = 'PolylineSetRepresentation'

    def __init__(self,
                 parent_model,
                 uuid = None,
                 polylines = None,
                 irap_file = None,
                 charisma_file = None,
                 crs_uuid = None,
                 title = None,
                 originator = None,
                 extra_metadata = None):
        """Initialises a new PolylineSet object.

        arguments:
            parent_model (model.Model object): the model which the new PolylineSetRepresentation belongs to
            uuid (uuid.UUID, optional): the uuid of an existing RESQML PolylineSetRepresentation object from
                which to initialise this resqpy PolylineSet
            polylines (optional): list of polyline objects from which to build the polylineset
            irap_file (str, optional): the name of a file in irap format from which to import the polyline set
            charisma_file (str, optional): the name of a file in charisma format from which to import the polyline set
            crs_uuid (UUID, optional): required if loading from an IRAP or Charisma file
            title (str, optional): the citation title to use for a new polyline set;
                ignored if uuid is not None
            originator (str, optional): the name of the person creating the polyline set, defaults to login id;
                ignored if uuid is not None
            extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the polyline set;
                ignored if uuid is not None

        returns:
            the newly instantiated PolylineSet object

        note:
            this initialiser does not perform any conversion between CRSes; if loading from a list of Polylines,
            they must all share a common CRS (which must also match crs_uuid if supplied)

        :meta common:
        """

        self.model = parent_model
        self.coordinates = None
        self.count_perpol = None
        self.polys = []
        self.rep_int_root = None
        self.save_polys = False
        self.boolnotconstant = None
        self.boolvalue = None
        self.crs_uuid = crs_uuid

        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata)

        if self.root is not None:
            return

        if polylines is not None:  # Create from list of polylines
            crs_list = [] if crs_uuid is None else [bu.uuid_from_string(crs_uuid)]
            for poly in polylines:
                crs_list.append(poly.crs_uuid)
            crs_set = set(crs_list)
            # following assumes consistent use of UUID or str
            assert len(crs_set) == 1, 'More than one CRS found in input polylines for polyline set'
            for crs_uuid in crs_set:
                self.crs_uuid = crs_uuid
                if self.crs_uuid is not None:
                    break
            self.polys = polylines
            # Setting the title of the first polyline given as the PolylineSet title
            if len(polylines) > 1:
                self.title = f"{polylines[0].title} + {len(polylines)-1} polylines"
            else:
                self.title = polylines[0].title

        elif irap_file is not None:  # Create from an input IRAP file
            if crs_uuid is None:
                log.warning(f'applying generic model CRS when loading polylines from IRAP file: {irap_file}')
            self._set_from_irap(irap_file)

        elif charisma_file is not None:
            if crs_uuid is None:
                log.warning(f'applying generic model CRS when loading polylines from Charisma file: {charisma_file}')
            self._set_from_charisma(charisma_file)

    def _load_from_xml(self):

        assert self.root is not None  # polyline set xml node specified
        root = self.root

        self.rep_int_root = self.model.referenced_node(rqet.find_tag(root, 'RepresentedInterpretation'))

        for patch_node in rqet.list_of_tag(root, 'LinePatch'):  # Loop over all LinePatches - likely just the one
            assert patch_node is not None  # Required field

            geometry_node = rqet.find_tag(patch_node, 'Geometry')
            assert geometry_node is not None  # Required field

            crs_root = self.model.referenced_node(rqet.find_tag(geometry_node, 'LocalCrs'))
            assert crs_root is not None  # Required field
            uuid = rqet.uuid_for_part_root(crs_root)
            assert uuid is not None  # Required field
            if self.crs_uuid is not None:
                assert bu.matching_uuids(self.crs_uuid, uuid), 'mixed CRS uuids in use in PolylineSet'
            self.crs_uuid = uuid

            closed_node = rqet.find_tag(patch_node, 'ClosedPolylines')
            assert closed_node is not None  # Required field
            # The ClosedPolylines could be a BooleanConstantArray, or a BooleanArrayFromIndexArray
            closed_array = self.get_bool_array(closed_node)

            count_node = rqet.find_tag(patch_node, 'NodeCountPerPolyline')
            rql_c.load_hdf5_array(self, count_node, 'count_perpol', tag = 'Values')

            points_node = rqet.find_tag(geometry_node, 'Points')
            assert points_node is not None  # Required field
            rql_c.load_hdf5_array(self, points_node, 'coordinates', tag = 'Coordinates')

            # Check that the number of bools aligns with the number of count_perpoly
            # Check that the total of the count_perpoly aligns with the number of coordinates
            assert len(self.count_perpol) == len(closed_array)
            assert np.sum(self.count_perpol) == len(self.coordinates)

            subpolys = self.convert_to_polylines(closed_array, self.count_perpol, self.coordinates, self.crs_uuid,
                                                 self.rep_int_root)
            # Check we have the right number of polygons
            assert len(subpolys) == len(self.count_perpol)

            # Remove duplicate coordinates and count arrays (exist in polylines now)
            # delattr(self,'coordinates')
            # delattr(self,'count_perpol')

            self.polys.extend(subpolys)

    def _set_from_irap(self, irap_file):
        inpoints = rsl.read_lines(irap_file)
        self.count_perpol = []
        closed_array = []
        self.title = os.path.basename(irap_file).split(".")[0]
        for i, poly in enumerate(inpoints):
            if len(poly) > 1:  # Polylines must have at least 2 points
                self.count_perpol.append(len(poly))
                if vu.isclose(poly[0], poly[-1]):
                    closed_array.append(True)
                else:
                    closed_array.append(False)
                if i == 0:
                    self.coordinates = poly
                else:
                    self.coordinates = np.concatenate((self.coordinates, poly))
        self.count_perpol = np.array(self.count_perpol)
        if self.crs_uuid is None:  # If no crs_uuid is provided, assume the main model crs is valid
            self.crs_uuid = self.model.crs_uuid
        self.polys = self.convert_to_polylines(closed_array, self.count_perpol, self.coordinates, self.crs_uuid,
                                               self.rep_int_root)

    def _set_from_charisma(self, charisma_file):
        with open(charisma_file) as f:
            inpoints = f.readlines()
        self.count_perpol = []
        closed_array = []
        self.title = os.path.basename(charisma_file).split(".")[0]
        for i, line in enumerate(inpoints):
            line = line.split()
            coord_entry = np.array([[float(line[3]), float(line[4]), float(line[5])]])
            if i == 0:
                self.coordinates = coord_entry
                stick = line[7]
                count = 1
            else:
                self.coordinates = np.concatenate((self.coordinates, coord_entry))
                count += 1
                if stick != line[7] or i == len(inpoints) - 1:
                    if count <= 2 and stick != line[7]:  # Line has fewer than 2 points
                        log.warning(f"Polylines must contain at least 2 points - ignoring point {self.coordinates[-2]}")
                        self.coordinates = np.delete(self.coordinates, -2, 0)  # Remove the second to last entry
                    else:
                        self.count_perpol.append(count - 1)
                        closed_array.append(False)
                    count = 1
                    stick = line[7]
        self.count_perpol = np.array(self.count_perpol)
        if self.crs_uuid is None:  # If no crs_uuid is provided, assume the main model crs is valid
            self.crs_uuid = self.model.crs_uuid
        self.polys = self.convert_to_polylines(closed_array, self.count_perpol, self.coordinates, self.crs_uuid,
                                               self.rep_int_root)

    def poly_index_containing_point_in_xy(self, p, mode = 'crossing'):
        """Returns the index of the first (closed) polyline containing point p in the xy plane, or None.

        :meta common:
        """

        assert mode in ['crossing', 'winding'], 'unrecognised mode when looking for polygon containing point'

        for i, poly in enumerate(self.polys):
            if not poly.isclosed:
                continue
            if poly.point_is_inside_xy(p, mode = mode):
                return i
        return None

    def create_xml(self,
                   ext_uuid = None,
                   add_as_part = True,
                   add_relationships = True,
                   title = None,
                   originator = None,
                   save_polylines = False):
        """Create xml from polylineset.

        args:
            save_polylines: If true, polylines are also saved individually

        :meta common:
        """

        if ext_uuid is None:
            ext_uuid = self.model.h5_uuid()

        self.save_polys = save_polylines

        if title:
            self.title = title
        if not self.title:
            self.title = 'polyline set'

        if self.save_polys:
            for poly in self.polys:
                poly.create_xml(ext_uuid, add_relationships = add_relationships, originator = originator)

        polyset = super().create_xml(add_as_part = False, originator = originator)

        if self.rep_int_root is not None:
            rep_int = self.rep_int_root
            if "FaultInterpretation" in str(rqet.content_type(rep_int)):
                content_type = 'obj_FaultInterpretation'
            else:
                content_type = 'obj_HorizonInterpretation'
            self.model.create_ref_node('RepresentedInterpretation',
                                       rqet.citation_title_for_node(rep_int),
                                       rep_int.attrib['uuid'],
                                       content_type = content_type,
                                       root = polyset)

        # We convert all Polylines to the CRS of the first Polyline in the set, so set this as crs_uuid

        patch = rqet.SubElement(polyset, ns['resqml2'] + 'LinePatch')
        patch.set(ns['xsi'] + 'type', ns['resqml2'] + 'PolylineSetPatch')
        patch.text = '\n'

        pindex = rqet.SubElement(patch, ns['resqml2'] + 'PatchIndex')
        pindex.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
        pindex.text = '0'

        if self.boolnotconstant:
            # We have mixed data - use a BooleanArrayFromIndexArray
            closed = rqet.SubElement(patch, ns['resqml2'] + 'ClosedPolylines')
            closed.set(ns['xsi'] + 'type', ns['xsd'] + 'BooleanArrayFromIndexArray')
            closed.text = '\n'

            bool_val = rqet.SubElement(closed, ns['resqml2'] + 'Value')
            bool_val.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
            bool_val.text = str(self.boolvalue).lower()

            ind_val = rqet.SubElement(closed, ns['resqml2'] + 'Indices')
            ind_val.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            ind_val.text = '\n'

            count = rqet.SubElement(closed, ns['resqml2'] + 'Count')
            count.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
            count.text = str(len(self.count_perpol))

            self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'indices_patch0', root = ind_val)
        else:
            # All bools are the same - use a BooleanConstantArray
            closed = rqet.SubElement(patch, ns['resqml2'] + 'ClosedPolylines')
            closed.set(ns['xsi'] + 'type', ns['resqml2'] + 'BooleanConstantArray')
            closed.text = '\n'
            bool_val = rqet.SubElement(closed, ns['resqml2'] + 'Value')
            bool_val.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
            bool_val.text = str(self.boolvalue).lower()
            count = rqet.SubElement(closed, ns['resqml2'] + 'Count')
            count.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
            count.text = str(len(self.count_perpol))

        count_pp = rqet.SubElement(patch, ns['resqml2'] + 'NodeCountPerPolyline')
        count_pp.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
        count_pp.text = '\n'

        null = rqet.SubElement(count_pp, ns['resqml2'] + 'NullValue')
        null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
        null.text = '0'

        count_val = rqet.SubElement(count_pp, ns['resqml2'] + 'Values')
        count_val.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        count_val.text = '\n'

        self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'NodeCountPerPolyline_patch0', root = count_val)

        geom = rqet.SubElement(patch, ns['resqml2'] + 'Geometry')
        geom.set(ns['xsi'] + 'type', ns['resqml2'] + 'PointGeometry')
        geom.text = '\n'

        self.model.create_crs_reference(crs_uuid = self.crs_uuid, root = geom)

        points = rqet.SubElement(geom, ns['resqml2'] + 'Points')
        points.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dHdf5Array')
        points.text = '\n'

        coords = rqet.SubElement(points, ns['resqml2'] + 'Coordinates')
        coords.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        coords.text = '\n'

        self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'points_patch0', root = coords)

        if add_as_part:
            self.model.add_part('obj_PolylineSetRepresentation', self.uuid, polyset)
            if add_relationships:
                crs_root = self.model.root_for_uuid(self.crs_uuid)
                self.model.create_reciprocal_relationship(polyset, 'destinationObject', crs_root, 'sourceObject')
                if self.rep_int_root is not None:  # Optional
                    self.model.create_reciprocal_relationship(polyset, 'destinationObject', self.rep_int_root,
                                                              'sourceObject')
                if self.save_polys:
                    for poly in self.polys:
                        self.model.create_reciprocal_relationship(polyset, 'destinationObject', poly.root_node,
                                                                  'sourceObject')

                ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
                ext_node = self.model.root_for_part(ext_part)
                self.model.create_reciprocal_relationship(polyset, 'mlToExternalPartProxy', ext_node,
                                                          'externalPartProxyToMl')

        return polyset

    def write_hdf5(self, file_name = None, mode = 'a', save_polylines = False):
        """Create or append the coordinates, counts and indices hdf5 arrays to hdf5 file.

        :meta common:
        """

        if self.uuid is None:
            self.uuid = bu.new_uuid()
        self.combine_polylines(self.polys)
        self.bool_array_format(self.closed_array)
        self.save_polys = save_polylines
        if self.save_polys:
            for poly in self.polys:
                poly.write_hdf5(file_name)

        h5_reg = rwh5.H5Register(self.model)
        h5_reg.register_dataset(self.uuid, 'points_patch0', self.coordinates)
        h5_reg.register_dataset(self.uuid, 'NodeCountPerPolyline_patch0', self.count_perpol.astype(np.int32))
        if self.boolnotconstant:
            h5_reg.register_dataset(self.uuid, 'indices_patch0', self.indices)
        h5_reg.write(file_name, mode = mode)

    def get_bool_array(self, closed_node):
        # TODO: Check if also defined boolean arrays
        """Returns a boolean array using details in the node location.

        If type of boolean array is BooleanConstantArray, uses the array value and count to generate the array. If type of boolean array is BooleanArrayFromIndexArray, find the "other" value bool and indices of the "other" values, and insert these into an array opposite to the main bool.

        args:
            closed_node: the node under which the boolean array information sits
        """
        if rqet.node_type(closed_node) == 'BooleanConstantArray':
            count = rqet.find_tag_int(closed_node, 'Count')
            value = rqet.bool_from_text(rqet.node_text(rqet.find_tag(closed_node, 'Value')))
            return np.full((count), value)
        elif rqet.node_type(closed_node) == 'BooleanArrayFromIndexArray':
            count = rqet.find_tag_int(closed_node, 'Count')
            indices_arr = rql_c.load_hdf5_array(self, closed_node, 'indices_arr', tag = 'Indices')
            istrue = rqet.bool_from_text(rqet.node_text(rqet.find_tag(closed_node, 'IndexIsTrue')))
            out = np.full((count), not istrue)
            out[indices_arr] = istrue
            return out

    def convert_to_polylines(self,
                             closed_array = None,
                             count_perpol = None,
                             coordinates = None,
                             crs_uuid = None,
                             rep_int_root = None):
        """Returns a list of Polylines objects from a PolylineSet.

        note:
            all arguments are optional and by default the data will be taken from self

        args:
            closed_array: array containing a bool for each polygon in if it is open (False) or closed (True)
            count_perpol: array containing a list of polygon "lengths" for each polygon
            coordinates: array containing coordinates for all the polygons
            crs_uuid: crs_uuid for polylineset
            rep_int_root: represented interpretation root (optional)

        returns:
            list of polyline objects

        :meta common:
        """

        from resqpy.lines._polyline import Polyline

        if count_perpol is None:
            count_perpol = self.count_perpol
        if closed_array is None:
            closed_array = np.zeros(len(count_perpol), dtype = bool)
            closed_node = rqet.find_nested_tags(self.root, ['LinePatch', 'ClosedPolylines'])
            if closed_node is not None:
                closed_array[:] = self.get_bool_array(closed_node)
        if coordinates is None:
            coordinates = self.coordinates
        if crs_uuid is None:
            crs_uuid = self.crs_uuid
        if rep_int_root is None:
            rep_int_root = self.rep_int_root
        polys = []
        count = 0
        for i in range(len(count_perpol)):
            if i != len(count_perpol) - 1:
                subset = coordinates[count:int(count_perpol[i]) + count].copy()
            else:
                subset = coordinates[count:int(count_perpol[i]) + count + 1].copy()
            if vu.isclose(subset[0], subset[-1]):
                isclosed = True
            else:
                isclosed = closed_array[i]
            count += int(count_perpol[i])
            subtitle = f"{self.title} {i+1}"
            polys.append(
                Polyline(self.model,
                         is_closed = isclosed,
                         set_coord = subset,
                         set_crs = crs_uuid,
                         title = subtitle,
                         rep_int_root = rep_int_root))

        return polys

    def combine_polylines(self, polylines):
        """Combines the isclosed boolean array, coordinates and count data for a list of polyline objects.

        args:
            polylines: list of polyline objects
        """

        self.count_perpol = []
        self.closed_array = []

        for poly in polylines:
            if poly == polylines[0]:
                master_crs = rcrs.Crs(self.model, uuid = poly.crs_uuid)
                self.crs_uuid = poly.crs_uuid
                self.coordinates = poly.coordinates.copy()
            else:
                curr_crs = rcrs.Crs(self.model, uuid = poly.crs_uuid)
                if not curr_crs.is_equivalent(master_crs):
                    shifted = curr_crs.convert_array_to(master_crs, poly.coordinates)
                    self.coordinates = np.concatenate((self.coordinates, shifted))
                else:
                    self.coordinates = np.concatenate((self.coordinates, poly.coordinates))

            self.closed_array.append(poly.isclosed)
            self.count_perpol.append(int(len(poly.coordinates)))

        self.count_perpol = np.array(self.count_perpol)

        assert len(self.closed_array) == len(self.count_perpol)
        assert np.sum(self.count_perpol) == len(self.coordinates)

        self.polys = polylines

    def bool_array_format(self, closed_array):
        """Determines an appropriate output boolean array format from an input array of bools.

        self.boolnotconstant - set to True if all are not open or all closed
        self.boolvalue - value of isclosed for all polylines, or for the majority of polylines if mixed
        self.indices - array of indices where the values are not self.boolvalue, if the polylines are mixed
        """

        self.indices = []
        self.boolnotconstant = False
        if all(closed_array):
            self.boolvalue = True
        elif not all(closed_array) and not any(closed_array):
            self.boolvalue = False
        else:
            if np.count_nonzero(closed_array) > (len(closed_array) / 2):
                self.boolvalue = True
                for i, val in enumerate(closed_array):
                    if not val:
                        self.indices.append(i)
            else:
                self.boolvalue = False
                for i, val in enumerate(closed_array):
                    if val:
                        self.indices.append(i)
        if len(self.indices) > 0:
            self.boolnotconstant = True

    def set_interpretation_root(self, rep_int_root, recursive = True):
        """Updates the rep_int_root for the polylineset.

        args:
            rep_int_root: new rep_int_root
            recursive: boolean, if true will update individual polys with same root
        """

        self.rep_int_root = rep_int_root

        if recursive:
            for poly in self.polys:
                poly.rep_int_root = rep_int_root

    def convert_to_irap(self, file_name):
        """Output an irap file from a polyline set.

        If file_name exists, it will be overwritten.

        args:
            file_name: output file name for polyline set representation
        """

        end_of_line = np.array([[999.0, 999.0, 999.0]])
        for poly in self.polys:
            if poly == self.polys[0]:
                out_coords = poly.coordinates
            else:
                out_coords = np.concatenate((out_coords, poly.coordinates))
            out_coords = np.concatenate((out_coords, end_of_line))
        np.savetxt(file_name, out_coords, delimiter = ' ')

    def convert_to_charisma(self, file_name):
        """Output to Charisma fault sticks from a polyline set.

        If file_name exists, it will be overwritten.

        args:
            file_name: output file name for polyline set representation
        """

        faultname = self.title.replace(" ", "_") if self.title else 'fault'
        lines = []
        for i, poly in enumerate(self.polys):
            for point in poly.coordinates:
                lines.append(f"INLINE-\t0\t0\t{point[0]}\t{point[1]}\t{point[2]}\t{faultname}\t{i+1}\n")
        with open(file_name, 'w') as f:
            for item in lines:
                f.write(item)
