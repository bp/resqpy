"""RESQML coordinate reference systems."""

import logging

log = logging.getLogger(__name__)

import math as maths
import uuid
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import resqpy.model as rq
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec
import resqpy.olio.xml_et as rqet
import resqpy.weights_and_measures as wam
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns

PointType = Union[Tuple[float, float, float], List[float], np.ndarray]


class Crs(BaseResqpy):
    """Coordinate reference system object."""

    @property
    def resqml_type(self):
        """Returns the RESQML class for this crs."""

        return 'LocalTime3dCrs' if hasattr(self, 'time_units') and self.time_units else 'LocalDepth3dCrs'

    valid_axis_orders = ("easting northing", "northing easting", "westing southing", "southing westing",
                         "northing westing", "westing northing")

    def __init__(self,
                 parent_model: 'rq.Model',
                 uuid: Optional[uuid.UUID] = None,
                 x_offset: float = 0.0,
                 y_offset: float = 0.0,
                 z_offset: float = 0.0,
                 rotation: float = 0.0,
                 rotation_units: str = 'dega',
                 xy_units: str = 'm',
                 z_units: str = 'm',
                 z_inc_down: bool = True,
                 axis_order: str = 'easting northing',
                 time_units: Optional[str] = None,
                 epsg_code: Optional[str] = None,
                 title: Optional[str] = None,
                 originator: Optional[str] = None,
                 extra_metadata: Optional[Dict[str, str]] = None):
        """Create a new coordinate reference system object.

        arguments:
            parent_model (model.Model): the model to which the new Crs object will belong
            crs_uuid (uuid.UUID): the uuid of an existing RESQML crs object in model, from which to instantiate the
                resqpy Crs object; if present, all the remaining arguments are ignored
            x_offset, y_offset, z_offset (floats, default zero): the local origin within an implicit parent crs
            rotation (float, default zero): a projected view rotation (in the xy plane), in radians, relative to an
                implicit parent crs; positive value indicates local y axis is clockwise from global y axis.
            rotation_units (str, default 'dega'): the units applicable to rotation, usually 'dega' or 'rad'
            xy_units (str, default 'm'): the units applicable to x & y values; must be one of the RESQML length uom strings
            z_units (str, default 'm'): the units applicable to z depth values; must be one of the RESQML length uom strings
            z_inc_down (boolean, default True): if True, increasing z values indicate greater depth (ie. in direction of
                gravity); if False, increasing z values indicate greater elevation
            axis_order (str, default 'easting northing'): the compass directions of the positive x and y axes respectively
            time_units (str, optional): if present, the time units of the z values (for seismic datasets), in which case the
                z_units argument is irrelevant
            epsg_code (str, optional): if present, the EPSG code of the implicit parent crs
            title (str, optional): the citation title to use for a new crs;
                ignored if uuid is not None
            originator (str, optional): the name of the person creating the crs, defaults to login id;
                ignored if uuid is not None
            extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the crs;
                ignored if uuid is not None

        returns:
            a new resqpy Crs object

        notes:
            although resqpy does not have full support for identifying a parent crs, the modifiers such as a local origin
            may be used on the assumption that all crs objects refer to the same implicit parent crs;
            there are two equivalent RESQML classes and which one is generated depends on whether the time_units argument
            is used, when instantiating from values;
            it is strongly encourage to call the create_xml() method immediately after instantiation (unless the resqpy
            crs is a temporary object) as the uuid may be modified at that point to re-use any existing equivalent RESQML
            crs object in the model;
            a positive rotation implies that local coordinates are clockwise from global coordinates

        :meta common:
        """

        self.xy_units = xy_units
        self.z_units = z_units
        self.time_units = time_units  # if None, z values are depth; if not None, z values are time (from seismic)
        self.z_inc_down = z_inc_down
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset
        self.rotation = rotation
        self.rotation_units = rotation_units
        self.axis_order = axis_order
        self.epsg_code = epsg_code
        # following are derived attributes, set below
        self.rotated = None
        self.rotation_matrix = None
        self.reverse_rotation_matrix = None

        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata)

        assert self.xy_units in wam.valid_uoms(quantity = 'length'), f'invalid CRS xy units: {self.xy_units}'
        assert self.z_units == self.xy_units or self.z_units in wam.valid_uoms(
            quantity = 'length'), f'invalid CRS z units: {self.z_units}'
        assert self.rotation_units in wam.valid_uoms(quantity='plane angle'), \
            f'invalid CRS rotation units: {self.rotation_units}'
        if self.time_units:
            assert self.time_units in wam.valid_uoms(quantity = 'time'), f'invalid CRS time units: {self.time_units}'
        assert self.axis_order in self.valid_axis_orders, 'invalid CRS axis order: ' + str(axis_order)

        self.set_rotation_matrices()

    def _load_from_xml(self):
        root_node = self.root
        assert root_node is not None
        flavour = rqet.node_type(root_node)
        assert flavour in ['obj_LocalDepth3dCrs', 'obj_LocalTime3dCrs'], f'bad crs node type: {flavour}'
        if flavour == 'obj_LocalTime3dCrs':
            self.time_units = rqet.find_tag_text(root_node, 'TimeUom')
            assert self.time_units
        else:
            self.time_units = None
        self.xy_units = rqet.find_tag_text(root_node, 'ProjectedUom')
        self.axis_order = rqet.find_tag_text(root_node, 'ProjectedAxisOrder')
        self.z_units = rqet.find_tag_text(root_node, 'VerticalUom')
        self.z_inc_down = rqet.find_tag_bool(root_node, 'ZIncreasingDownward')
        self.x_offset = rqet.find_tag_float(root_node, 'XOffset')
        self.y_offset = rqet.find_tag_float(root_node, 'YOffset')
        self.z_offset = rqet.find_tag_float(root_node, 'ZOffset')
        self.rotation = rqet.find_tag_float(root_node, 'ArealRotation')
        rotation_node = rqet.find_tag(root_node, 'ArealRotation')
        self.rotation_units = rotation_node.attrib.get('uom')
        parent_xy_crs = rqet.find_tag(root_node, 'ProjectedCrs')
        if parent_xy_crs is not None and rqet.node_type(parent_xy_crs) == 'ProjectedCrsEpsgCode':
            self.epsg_code = rqet.find_tag_text(parent_xy_crs, 'EpsgCode')  # should be an integer?
        else:
            self.epsg_code = None

    def set_rotation_matrices(self):
        """Sets the rotation matrices, and the rotated and null_transform flags, call after changing rotation."""

        rotation_deg = wam.convert(self.rotation, self.rotation_units, 'dega', quantity = 'plane angle')
        # log.debug(f'setting rotation matrices for dega: {rotation_deg}')
        self.rotated = (not maths.isclose(rotation_deg, 0.0, abs_tol = 1e-8) and
                        not maths.isclose(rotation_deg, 360.0, abs_tol = 1e-8))
        if self.rotated:
            self.rotation_matrix = vec.rotation_matrix_3d_axial(2, rotation_deg)
            self.reverse_rotation_matrix = vec.rotation_matrix_3d_axial(2, -rotation_deg)
            if self.is_right_handed_xy():
                self.rotation_matrix, self.reverse_rotation_matrix = self.reverse_rotation_matrix, self.rotation_matrix
        else:
            self.rotation_matrix = None
            self.reverse_rotation_matrix = None
        self.null_transform = (maths.isclose(self.x_offset, 0.0, abs_tol = 1e-8) and
                               maths.isclose(self.y_offset, 0.0, abs_tol = 1e-8) and
                               maths.isclose(self.z_offset, 0.0, abs_tol = 1e-8) and not self.rotated)

    def is_right_handed_xy(self) -> bool:
        """Returns True if the xy axes are right handed when viewed from above; False if left handed."""

        return bool(self.axis_order in ["northing easting", "southing westing", "westing northing"])

    def is_right_handed_xyz(self) -> bool:
        """Returns True if the xyz axes are right handed; False if left handed."""

        return self.is_right_handed_xy() is bool(self.z_inc_down)

    def global_to_local(self, xyz: PointType, global_z_inc_down: bool = True) -> Tuple[float, float, float]:
        """Convert a single xyz point from the parent coordinate reference system to this one."""

        x, y, z = xyz
        if self.x_offset != 0.0:
            x -= self.x_offset
        if self.y_offset != 0.0:
            y -= self.y_offset
        if global_z_inc_down != self.z_inc_down:
            z = -z
        if self.z_offset != 0.0:
            z -= self.z_offset
        if self.rotated:
            (x, y, z) = vec.rotate_vector(self.rotation_matrix, np.array((x, y, z)))
        return (x, y, z)

    def global_to_local_array(self, xyz: np.ndarray, global_z_inc_down: bool = True):
        """Convert in situ a numpy array of xyz points from the parent coordinate reference system to this one."""

        if self.x_offset != 0.0:
            xyz[..., 0] -= self.x_offset
        if self.y_offset != 0.0:
            xyz[..., 1] -= self.y_offset
        if global_z_inc_down != self.z_inc_down:
            z = np.negative(xyz[..., 2])
            xyz[..., 2] = z
        if self.z_offset != 0.0:
            xyz[..., 2] -= self.z_offset
        if self.rotated:
            a = vec.rotate_array(self.rotation_matrix, xyz)
            xyz[:] = a

    def local_to_global(self, xyz: PointType, global_z_inc_down: bool = True) -> Tuple[float, float, float]:
        """Convert a single xyz point from this coordinate reference system to the parent one."""

        if self.rotated:
            (x, y, z) = vec.rotate_vector(self.reverse_rotation_matrix, np.array(xyz))
        else:
            (x, y, z) = xyz
        if self.x_offset != 0.0:
            x += self.x_offset
        if self.y_offset != 0.0:
            y += self.y_offset
        if self.z_offset != 0.0:
            z += self.z_offset
        if global_z_inc_down != self.z_inc_down:
            z = -z
        return (x, y, z)

    def local_to_global_array(self, xyz: np.ndarray, global_z_inc_down: bool = True):
        """Convert in situ a numpy array of xyz points from this coordinate reference system to the parent one."""

        if self.rotated:
            a = vec.rotate_array(self.reverse_rotation_matrix, xyz)
            xyz[:] = a
        if self.x_offset != 0.0:
            xyz[..., 0] += self.x_offset
        if self.y_offset != 0.0:
            xyz[..., 1] += self.y_offset
        if self.z_offset != 0.0:
            xyz[..., 2] += self.z_offset
        if global_z_inc_down != self.z_inc_down:
            z = np.negative(xyz[..., 2])
            xyz[..., 2] = z

    def has_same_epsg_code(self, other_crs: 'Crs') -> bool:
        """Returns True if either of the crs'es has a null EPSG code, or if they are the same."""
        return not self.epsg_code or not other_crs.epsg_code or self.epsg_code == other_crs.epsg_code

    def is_equivalent(self, other_crs: 'Crs') -> bool:
        """Returns True if this crs is effectively the same as the other crs."""

        # log.debug('testing crs equivalence')
        if other_crs is None:
            return False
        if self is other_crs:
            return True
        if bu.matching_uuids(self.uuid, other_crs.uuid):
            return True
        if self.resqml_type != other_crs.resqml_type:
            return False
        if self.xy_units != other_crs.xy_units or self.z_units != other_crs.z_units:
            return False
        if self.z_inc_down != other_crs.z_inc_down:
            return False
        if (self.time_units is not None or
                other_crs.time_units is not None) and self.time_units != other_crs.time_units:
            return False
        if self.axis_order != other_crs.axis_order:
            return False
        if not self.has_same_epsg_code(other_crs):
            return False
        if not _matching_extra_metadata(self, other_crs):
            return False
        if self.null_transform and other_crs.null_transform:
            return True
        if (maths.isclose(self.x_offset, other_crs.x_offset, abs_tol = 1e-4) and
                maths.isclose(self.y_offset, other_crs.y_offset, abs_tol = 1e-4) and
                maths.isclose(self.z_offset, other_crs.z_offset, abs_tol = 1e-4) and
                maths.isclose(self.rotation, other_crs.rotation, abs_tol = 1e-4)):
            return True
        # todo: handle and check rotation units; modularly equivalent rotations
        return False

    def convert_to(self, other_crs: 'Crs', xyz: PointType) -> Tuple[float, float, float]:
        """Converts a single xyz point from this coordinate reference system to the other.

        :meta common:
        """

        if self is other_crs:
            return _as_xyz_tuple(xyz)
        assert self.resqml_type == other_crs.resqml_type
        # assert self.has_same_epsg_code(other_crs)
        if not self.has_same_epsg_code(other_crs):
            log.warning("converting between crs'es with different epsg codes")
        xyz = self.local_to_global(xyz)
        # yapf: disable
        if self.resqml_type == 'LocalDepth3dCrs':
            xyz = (wam.convert_lengths(xyz[0], self.xy_units, other_crs.xy_units),
                   wam.convert_lengths(xyz[1], self.xy_units, other_crs.xy_units),
                   wam.convert_lengths(xyz[2], self.z_units, other_crs.z_units))
        else:
            xyz = (wam.convert_lengths(xyz[0], self.xy_units, other_crs.xy_units),
                   wam.convert_lengths(xyz[1], self.xy_units, other_crs.xy_units),
                   wam.convert_times(xyz[2], self.time_units, other_crs.time_units))
        # yapf: enable
        xyz = other_crs.global_to_local(xyz)
        return _as_xyz_tuple(xyz)

    def convert_array_to(self, other_crs: 'Crs', xyz: np.ndarray):
        """Converts in situ a numpy array of xyz points from this coordinate reference system to the other.

        :meta common:
        """

        if self.is_equivalent(other_crs):
            return xyz
        assert self.resqml_type == other_crs.resqml_type
        # assert self.has_same_epsg_code(other_crs)
        if not self.has_same_epsg_code(other_crs):
            log.warning("converting between crs'es with different epsg codes")
        self.local_to_global_array(xyz)
        if self.resqml_type == 'LocalDepth3dCrs':
            if self.xy_units == self.z_units and other_crs.xy_units == other_crs.z_units:
                wam.convert_lengths(xyz, self.xy_units, other_crs.xy_units)
            else:
                wam.convert_lengths(xyz[..., :2], self.xy_units, other_crs.xy_units)
                wam.convert_lengths(xyz[..., 2], self.z_units, other_crs.z_units)
        else:
            wam.convert_lengths(xyz[..., :2], self.xy_units, other_crs.xy_units)
            wam.convert_times(xyz[..., 2], self.time_units, other_crs.time_units)
        other_crs.global_to_local_array(xyz)
        return xyz

    def convert_from(self, other_crs: 'Crs', xyz: PointType) -> Tuple[float, float, float]:
        """Converts a single xyz point from the other coordinate reference system to this one.

        :meta common:
        """

        if self is other_crs:
            return _as_xyz_tuple(xyz)
        assert self.resqml_type == other_crs.resqml_type
        # assert self.has_same_epsg_code(other_crs)
        if not self.has_same_epsg_code(other_crs):
            log.warning("converting between crs'es with different epsg codes")
        xyz = other_crs.local_to_global(xyz)
        # yapf: disable
        if self.resqml_type == 'LocalDepth3dCrs':
            xyz = (wam.convert_lengths(xyz[0], other_crs.xy_units, self.xy_units),
                   wam.convert_lengths(xyz[1], other_crs.xy_units, self.xy_units),
                   wam.convert_lengths(xyz[2], other_crs.z_units, self.z_units))
        else:
            xyz = (wam.convert_lengths(xyz[0], other_crs.xy_units, self.xy_units),
                   wam.convert_lengths(xyz[1], other_crs.xy_units, self.xy_units),
                   wam.convert_times(xyz[2], other_crs.time_units, self.time_units))
        # yapf: enable
        xyz = self.global_to_local(xyz)
        return _as_xyz_tuple(xyz)

    def convert_array_from(self, other_crs: 'Crs', xyz: np.ndarray):
        """Converts in situ a numpy array of xyz points from the other coordinate reference system to this one.

        :meta common:
        """

        if self.is_equivalent(other_crs):
            return xyz
        assert self.resqml_type == other_crs.resqml_type
        # assert self.has_same_epsg_code(other_crs)
        if not self.has_same_epsg_code(other_crs):
            log.warning("converting between crs'es with different epsg codes")
        other_crs.local_to_global_array(xyz)
        if self.resqml_type == 'LocalDepth3dCrs':
            if self.xy_units == self.z_units and other_crs.xy_units == other_crs.z_units:
                wam.convert_lengths(xyz, other_crs.xy_units, self.xy_units)
            else:
                wam.convert_lengths(xyz[..., :2], other_crs.xy_units, self.xy_units)
                wam.convert_lengths(xyz[..., 2], other_crs.z_units, self.z_units)
        else:
            wam.convert_lengths(xyz[..., :2], other_crs.xy_units, self.xy_units)
            wam.convert_times(xyz[..., 2], other_crs.time_units, self.time_units)
        self.global_to_local_array(xyz)
        return xyz

    def create_xml(self,
                   title: Optional[str] = None,
                   originator: Optional[str] = None,
                   extra_metadata: Optional[Dict[str, str]] = None,
                   add_as_part: bool = True,
                   reuse: bool = True):
        """Creates a Coordinate Reference System xml node and optionally adds as a part in the parent model.

        arguments:
            title (string, optional): used as the Title text in the citation node
            originator (string, optional): the name of the human being who created the crs object;
                default is to use the login name
            extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the crs
            add_as_part (boolean, default True): if True the newly created crs node is added to the model
                as a part
            reuse (boolean, default True): if True and an equivalent crs already exists in the model then
                the uuid for this Crs is modified to match that of the existing object and the existing
                xml node is returned without anything new being added

        returns:
            newly created (or reused) coordinate reference system xml node

        notes:
            if the reuse argument is True, it is strongly recommended to call this method immediately after
            a new Crs has been instantiated from explicit values, as the uuid may be modified here;
            if reuse is True, the title is not regarded as important and a match with an existing object
            may occur even if the titles differ

        :meta common:
        """

        if reuse and self.try_reuse():
            return self.root  # check for reusable (equivalent) object

        crs = super().create_xml(add_as_part = False,
                                 title = title,
                                 originator = originator,
                                 extra_metadata = extra_metadata)

        xoffset = rqet.SubElement(crs, ns['resqml2'] + 'XOffset')
        xoffset.set(ns['xsi'] + 'type', ns['xsd'] + 'double')
        xoffset.text = '{0:5.3f}'.format(self.x_offset)

        yoffset = rqet.SubElement(crs, ns['resqml2'] + 'YOffset')
        yoffset.set(ns['xsi'] + 'type', ns['xsd'] + 'double')
        yoffset.text = '{0:5.3f}'.format(self.y_offset)

        zoffset = rqet.SubElement(crs, ns['resqml2'] + 'ZOffset')
        zoffset.set(ns['xsi'] + 'type', ns['xsd'] + 'double')
        zoffset.text = '{0:6.4f}'.format(self.z_offset)

        rotation = rqet.SubElement(crs, ns['resqml2'] + 'ArealRotation')
        rotation.set('uom', self.rotation_units)
        rotation.set(ns['xsi'] + 'type', ns['eml'] + 'PlaneAngleMeasure')
        rotation.text = '{0:8.6f}'.format(self.rotation)

        if self.axis_order is None:
            axes = 'easting northing'
        else:
            axes = self.axis_order.lower()
        axis_order = rqet.SubElement(crs, ns['resqml2'] + 'ProjectedAxisOrder')
        axis_order.set(ns['xsi'] + 'type', ns['eml'] + 'AxisOrder2d')
        axis_order.text = axes

        xy_uom = rqet.SubElement(crs, ns['resqml2'] + 'ProjectedUom')
        xy_uom.set(ns['xsi'] + 'type', ns['eml'] + 'LengthUom')
        xy_uom.text = wam.rq_length_unit(self.xy_units)

        z_uom = rqet.SubElement(crs, ns['resqml2'] + 'VerticalUom')
        z_uom.set(ns['xsi'] + 'type', ns['eml'] + 'LengthUom')
        z_uom.text = wam.rq_length_unit(self.z_units)

        if self.time_units is not None:
            t_uom = rqet.SubElement(crs, ns['resqml2'] + 'TimeUom')
            t_uom.set(ns['xsi'] + 'type', ns['eml'] + 'TimeUom')  # todo: check this
            t_uom.text = self.time_units

        z_sense = rqet.SubElement(crs, ns['resqml2'] + 'ZIncreasingDownward')
        z_sense.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
        z_sense.text = str(self.z_inc_down).lower()

        xy_crs = rqet.SubElement(crs, ns['resqml2'] + 'ProjectedCrs')
        xy_crs.text = rqet.null_xml_text
        if self.epsg_code is None:
            xy_crs.set(ns['xsi'] + 'type', ns['eml'] + 'ProjectedUnknownCrs')
            self.model.create_unknown(root = xy_crs)
        else:
            xy_crs.set(ns['xsi'] + 'type', ns['eml'] + 'ProjectedCrsEpsgCode')
            epsg_node = rqet.SubElement(xy_crs, ns['resqml2'] + 'EpsgCode')
            epsg_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
            epsg_node.text = str(self.epsg_code)

        z_crs = rqet.SubElement(crs, ns['resqml2'] + 'VerticalCrs')
        if self.epsg_code is None:
            z_crs.set(ns['xsi'] + 'type', ns['eml'] + 'VerticalUnknownCrs')
            z_crs.text = rqet.null_xml_text
            self.model.create_unknown(root = z_crs)
        else:  # not sure if this is appropriate for the vertical crs
            z_crs.set(ns['xsi'] + 'type', ns['eml'] + 'VerticalCrsEpsgCode')
            epsg_node = rqet.SubElement(z_crs, ns['resqml2'] + 'EpsgCode')
            epsg_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
            epsg_node.text = str(self.epsg_code)

        if add_as_part:
            self.model.add_part('obj_' + self.resqml_type, bu.uuid_from_string(crs.attrib['uuid']), crs)
        if self.model.crs_uuid is None:
            self.model.crs_uuid = self.uuid  # mark's as 'main' (ie. first) crs for model

        return crs


def _as_xyz_tuple(xyz):
    """Coerce into 3-tuple of floats."""

    return tuple((float(xyz[0]), float(xyz[1]), float(xyz[2])))


def _matching_extra_metadata(a, b):
    if not hasattr(a, 'extra_metadata') and not hasattr(b, 'extra_metadata'):
        return True
    if not hasattr(a, 'extra_metadata') or not hasattr(b, 'extra_metadata'):
        return False
    if len(a.extra_metadata) != len(b.extra_metadata):
        return False
    if len(a.extra_metadata) == 0:
        return True
    for i in a.extra_metadata.items():
        if i not in b.extra_metadata.items():
            return False
    return True
