"""Class for RESQML Fault Interpretation organizational objects."""

import math as maths

from ._utils import (equivalent_extra_metadata, alias_for_attribute, extract_has_occurred_during,
                     equivalent_chrono_pairs, create_xml_has_occurred_during)

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns
from .tectonic_boundary_feature import TectonicBoundaryFeature


class FaultInterpretation(BaseResqpy):
    """Class for RESQML Fault Interpretation organizational objects.

    RESQML documentation:

       A type of boundary feature, this class contains the data describing an opinion
       about the characterization of the fault, which includes the attributes listed below.
    """

    resqml_type = "FaultInterpretation"
    valid_domains = ('depth', 'time', 'mixed')

    # note: many of the attributes could be deduced from geometry

    def __init__(self,
                 parent_model,
                 uuid = None,
                 title = None,
                 tectonic_boundary_feature = None,
                 domain = 'depth',
                 is_normal = None,
                 is_listric = None,
                 maximum_throw = None,
                 mean_azimuth = None,
                 mean_dip = None,
                 extra_metadata = None):
        """Initialises a Fault interpretation organisational object."""

        # note: will create a paired TectonicBoundaryFeature object when loading from xml
        # if not extracting from xml,:
        # tectonic_boundary_feature is required and must be a TectonicBoundaryFeature object
        # domain is required and must be one of 'depth', 'time' or 'mixed'
        # is_listric is required if the fault is not normal (and must be None if normal)
        # max throw, azimuth & dip are all optional
        # the throw interpretation list is not supported for direct initialisation

        self.tectonic_boundary_feature = tectonic_boundary_feature  # InterpretedFeature RESQML field, when not loading from xml
        self.feature_root = None if self.tectonic_boundary_feature is None else self.tectonic_boundary_feature.root
        if (not title) and self.tectonic_boundary_feature is not None:
            title = self.tectonic_boundary_feature.feature_name
        self.main_has_occurred_during = (None, None)
        self.is_normal = is_normal  # extra field, not explicitly in RESQML
        self.domain = domain
        # RESQML xml business rule: IsListric must be present if the fault is normal; must not be present if the fault is not normal
        self.is_listric = is_listric
        self.maximum_throw = maximum_throw
        self.mean_azimuth = mean_azimuth
        self.mean_dip = mean_dip
        self.throw_interpretation_list = None  # list of (list of throw kind, (base chrono uuid, top chrono uuid)))

        super().__init__(model = parent_model, uuid = uuid, title = title, extra_metadata = extra_metadata)

    @property
    def feature_uuid(self):
        """Returns the UUID of the interpreted feature"""
        # TODO: rewrite using uuid as primary key
        return rqet.uuid_for_part_root(self.feature_root)

    def _load_from_xml(self):
        root_node = self.root
        self.domain = rqet.find_tag_text(root_node, 'Domain')
        interp_feature_ref_node = rqet.find_tag(root_node, 'InterpretedFeature')
        assert interp_feature_ref_node is not None
        self.feature_root = self.model.referenced_node(interp_feature_ref_node)
        if self.feature_root is not None:
            self.tectonic_boundary_feature = TectonicBoundaryFeature(self.model,
                                                                     uuid = self.feature_root.attrib['uuid'],
                                                                     feature_name = self.model.title_for_root(
                                                                         self.feature_root))
            self.main_has_occurred_during = extract_has_occurred_during(root_node)
            self.is_listric = rqet.find_tag_bool(root_node, 'IsListric')
            self.is_normal = (self.is_listric is None)
            self.maximum_throw = rqet.find_tag_float(root_node, 'MaximumThrow')
            # todo: check that type="eml:LengthMeasure" is simple float
            self.mean_azimuth = rqet.find_tag_float(root_node, 'MeanAzimuth')
            self.mean_dip = rqet.find_tag_float(root_node, 'MeanDip')
            throw_interpretation_nodes = rqet.list_of_tag(root_node, 'ThrowInterpretation')
            if throw_interpretation_nodes is not None and len(throw_interpretation_nodes):
                self.throw_interpretation_list = []
                for ti_node in throw_interpretation_nodes:
                    hod_pair = extract_has_occurred_during(ti_node)
                    throw_kind_list = rqet.list_of_tag(ti_node, 'Throw')
                    for tk_node in throw_kind_list:
                        self.throw_interpretation_list.append((tk_node.text, hod_pair))

    def is_equivalent(self, other, check_extra_metadata = True):
        """Returns True if this interpretation is essentially the same as the other; otherwise False."""

        if other is None or not isinstance(other, FaultInterpretation):
            return False
        if self is other or bu.matching_uuids(self.uuid, other.uuid):
            return True

        attr_list = ['tectonic_boundary_feature', 'root', 'maximum_throw', 'mean_azimuth', 'mean_dip']
        one_none_attr_list = [(getattr(self, v) is None) != (getattr(other, v) is None) for v in attr_list]
        if any(one_none_attr_list):
            # If only one of self or other has a None attribute
            return False

        # List of attributes that are not None in either self or other
        non_none_attr_list = [a for a in attr_list if getattr(self, a) is not None]

        # Additional tests for attributes that are not None
        check_dict = {
            'tectonic_boundary_feature':
                lambda: self.tectonic_boundary_feature.is_equivalent(other.tectonic_boundary_feature),
            'root':
                lambda: rqet.citation_title_for_node(self.root) != rqet.citation_title_for_node(other.root),
            'maximum_throw':
                lambda: maths.isclose(self.maximum_throw, other.maximum_throw, rel_tol = 1e-3),
            'mean_azimuth':
                lambda: maths.isclose(self.mean_azimuth, other.mean_azimuth, abs_tol = 0.5),
            'mean_dip':
                lambda: maths.isclose(self.mean_dip, other.mean_dip, abs_tol = 0.5)
        }

        check_outcomes = [check_dict[v]() for v in non_none_attr_list]
        if not all(check_outcomes):  # If any of the Additional tests fail then self and other are not equivalent
            return False

        if (not equivalent_chrono_pairs(self.main_has_occurred_during, other.main_has_occurred_during) or
                self.is_normal != other.is_normal or self.domain != other.domain or
                self.is_listric != other.is_listric):
            return False

        if check_extra_metadata and not equivalent_extra_metadata(self, other):
            return False
        if not self.throw_interpretation_list and not other.throw_interpretation_list:
            return True
        if not self.throw_interpretation_list or not other.throw_interpretation_list:
            return False
        if len(self.throw_interpretation_list) != len(other.throw_interpretation_list):
            return False
        for this_ti, other_ti in zip(self.throw_interpretation_list, other.throw_interpretation_list):
            if this_ti[0] != other_ti[0]:
                return False  # throw kind
            if not equivalent_chrono_pairs(this_ti[1], other_ti[1]):
                return False
        return True

    def create_xml(self,
                   tectonic_boundary_feature_root = None,
                   add_as_part = True,
                   add_relationships = True,
                   originator = None,
                   title_suffix = None,
                   reuse = True):
        """Creates a fault interpretation organisational xml node from a fault interpretation object."""

        # note: related tectonic boundary feature node should be created first and referenced here

        assert self.is_normal == (self.is_listric is None)
        if not self.title:
            if tectonic_boundary_feature_root is not None:
                title = rqet.find_nested_tags_text(tectonic_boundary_feature_root, ['Citation', 'Title'])
            else:
                title = 'fault interpretation'
            if title_suffix:
                title += ' ' + title_suffix

        if reuse and self.try_reuse():
            return self.root

        fi = super().create_xml(add_as_part = False, originator = originator)

        if self.tectonic_boundary_feature is not None:
            tbf_root = self.tectonic_boundary_feature.root
            if tbf_root is not None:
                if tectonic_boundary_feature_root is None:
                    tectonic_boundary_feature_root = tbf_root
                else:
                    assert tbf_root is tectonic_boundary_feature_root, 'tectonic boundary feature mismatch'
        assert tectonic_boundary_feature_root is not None

        assert self.domain in self.valid_domains, 'illegal domain value for fault interpretation'
        dom_node = rqet.SubElement(fi, ns['resqml2'] + 'Domain')
        dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
        dom_node.text = self.domain

        self.model.create_ref_node('InterpretedFeature',
                                   self.model.title_for_root(tectonic_boundary_feature_root),
                                   tectonic_boundary_feature_root.attrib['uuid'],
                                   content_type = 'obj_TectonicBoundaryFeature',
                                   root = fi)

        create_xml_has_occurred_during(self.model, fi, self.main_has_occurred_during)

        if self.is_listric is not None:
            listric = rqet.SubElement(fi, ns['resqml2'] + 'IsListric')
            listric.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
            listric.text = str(self.is_listric).lower()

        # todo: check eml:LengthMeasure and PlaneAngleMeasure structures: uom?
        if self.maximum_throw is not None:
            max_throw = rqet.SubElement(fi, ns['resqml2'] + 'MaximumThrow')
            max_throw.set(ns['xsi'] + 'type', ns['eml'] + 'LengthMeasure')
            max_throw.text = str(self.maximum_throw)

        if self.mean_azimuth is not None:
            azimuth = rqet.SubElement(fi, ns['resqml2'] + 'MeanAzimuth')
            azimuth.set(ns['xsi'] + 'type', ns['eml'] + 'PlaneAngleMeasure')
            azimuth.text = str(self.mean_azimuth)

        if self.mean_dip is not None:
            dip = rqet.SubElement(fi, ns['resqml2'] + 'MeanDip')
            dip.set(ns['xsi'] + 'type', ns['eml'] + 'PlaneAngleMeasure')
            dip.text = str(self.mean_azimuth)

        if self.throw_interpretation_list is not None and len(self.throw_interpretation_list):
            previous_has_occurred_during = ('never', 'never')
            ti_node = None
            for (throw_kind, (base_chrono_uuid, top_chrono_uuid)) in self.throw_interpretation_list:
                if (str(base_chrono_uuid), str(top_chrono_uuid)) != previous_has_occurred_during:
                    previous_has_occurred_during = (str(base_chrono_uuid), str(top_chrono_uuid))
                    ti_node = rqet.SubElement(fi, 'ThrowInterpretation')
                    ti_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'FaultThrow')
                    ti_node.text = rqet.null_xml_text
                    create_xml_has_occurred_during(self.model, ti_node, (base_chrono_uuid, top_chrono_uuid))
                tk_node = rqet.SubElement(ti_node, 'Throw')
                tk_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ThrowKind')
                tk_node.text = throw_kind

        if add_as_part:
            self.model.add_part('obj_FaultInterpretation', self.uuid, fi)
            if add_relationships:
                self.model.create_reciprocal_relationship(fi, 'destinationObject', tectonic_boundary_feature_root,
                                                          'sourceObject')

        return fi
