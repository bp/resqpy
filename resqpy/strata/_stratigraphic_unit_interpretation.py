"""_stratigraphic_unit_interpretation.py: RESQML StratigraphicUnitInterpretation class."""

# NB: in this module, the term 'unit' refers to a geological stratigraphic unit, i.e. a layer of rock, not a unit of measure

import logging

log = logging.getLogger(__name__)

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.strata
import resqpy.strata._strata_common as rqstc
import resqpy.strata._geologic_unit_interpretation as rqsgui
import resqpy.strata._stratigraphic_unit_feature as rqsui
import resqpy.weights_and_measures as wam
from resqpy.olio.xml_namespaces import curly_namespace as ns


class StratigraphicUnitInterpretation(rqsgui.GeologicUnitInterpretation):
    """Class for RESQML Stratigraphic Unit Interpretation objects.

    RESQML documentation:

       Interpretation of a stratigraphic unit which includes the knowledge of the top, the bottom,
       the deposition mode.
    """

    resqml_type = 'StratigraphicUnitInterpretation'

    def __init__(self,
                 parent_model,
                 uuid = None,
                 title = None,
                 domain = 'depth',
                 stratigraphic_unit_feature = None,
                 composition = None,
                 material_implacement = None,
                 deposition_mode = None,
                 min_thickness = None,
                 max_thickness = None,
                 thickness_uom = None,
                 extra_metadata = None):
        """Initialises a stratigraphic unit interpretation object.

        arguments:
           parent_model (model.Model): the model with which the new interpretation will be associated
           uuid (uuid.UUID, optional): the uuid of an existing RESQML stratigraphic unit interpretation from which
              this object will be initialised
           title (str, optional): the citation title (feature name) of the new interpretation;
              ignored if uuid is not None
           domain (str, default 'time'): 'time', 'depth' or 'mixed', being the domain of the interpretation;
              ignored if uuid is not None
           stratigraphic_unit_feature (StratigraphicUnitFeature, optional): the feature which this object is
              an interpretation of; ignored if uuid is not None
           composition (str, optional): the interpreted composition of the stratigraphic unit; if present, must be
              in valid_compositions; ignored if uuid is not None
           material_implacement (str, optional): the interpeted material implacement of the stratigraphic unit;
              if present, must be in valid_implacements, ie. 'autochtonous' or 'allochtonous';
              ignored if uuid is not None
           deposition_mode (str, optional): indicates whether deposition within the unit is interpreted as parallel
              to top, base or another boundary, or is proportional to thickness; if present, must be in
              valid_deposition_modes; ignored if uuid is not None
           min_thickness (float, optional): the minimum thickness of the unit; ignored if uuid is not None
           max_thickness (float, optional): the maximum thickness of the unit; ignored if uuid is not None
           thickness_uom (str, optional): the length unit of measure of the minimum and maximum thickness; required
              if either thickness argument is provided and uuid is None; if present, must be a valid length uom
           extra_metadata (dict, optional): extra metadata items for the new interpretation

        returns:
           a new stratigraphic unit interpretation resqpy object

        notes:
           if given, the thickness_uom must be a valid RESQML length unit of measure; the set of valid uoms is
           returned by: weights_and_measures.valid_uoms(quantity = 'length');
           the RESQML 2.0.1 schema definition includes a spurious trailing space in the names of two compositions;
           resqpy removes such spaces in the composition attribute as presented to calling code (but includes them
           in xml)
        """

        self.deposition_mode = deposition_mode
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.thickness_uom = thickness_uom
        super().__init__(parent_model,
                         uuid = uuid,
                         title = title,
                         domain = domain,
                         geologic_unit_feature = stratigraphic_unit_feature,
                         composition = composition,
                         material_implacement = material_implacement,
                         extra_metadata = extra_metadata)
        if self.deposition_mode is not None:
            assert self.deposition_mode in rqstc.valid_deposition_modes
        if self.min_thickness is not None or self.max_thickness is not None:
            assert self.thickness_uom in wam.valid_uoms(quantity = 'length')

    @property
    def stratigraphic_unit_feature(self):
        """Returns the interpreted geologic unit feature."""

        return self.geologic_unit_feature

    def _load_from_xml(self):
        """Loads class specific attributes from xml for an existing RESQML object; called from BaseResqpy."""
        super()._load_from_xml()
        root_node = self.root
        assert root_node is not None
        feature_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(root_node, ['InterpretedFeature', 'UUID']))
        if feature_uuid is not None:
            self.geologic_unit_feature = rqsui.StratigraphicUnitFeature(self.model,
                                                                        uuid = feature_uuid,
                                                                        title = self.model.title(uuid = feature_uuid))
        # load deposition mode and min & max thicknesses (& uom), if present
        self.deposition_mode = rqet.find_tag_text(root_node, 'DepositionMode')
        for min_max in ['Min', 'Max']:
            thick_node = rqet.find_tag(root_node, min_max + 'Thickness')
            if thick_node is not None:
                thick = float(thick_node.text)
                if min_max == 'Min':
                    self.min_thickness = thick
                else:
                    self.max_thickness = thick
                thick_uom = thick_node.attrib['uom']  # todo: check this is correct uom representation
                if self.thickness_uom is None:
                    self.thickness_uom = thick_uom
                else:
                    assert thick_uom == self.thickness_uom, 'inconsistent length units of measure for stratigraphic thicknesses'

    def is_equivalent(self, other, check_extra_metadata = True):
        """Returns True if this interpretation is essentially the same as the other; otherwise False.

        arguments:
           other (StratigraphicUnitInterpretation): the other interpretation to compare this one against
           check_extra_metadata (bool, default True): if True, then extra metadata items must match for the two
              interpretations to be deemed equivalent; if False, extra metadata is ignored in the comparison

        returns:
           bool: True if this interpretation is essentially the same as the other; False otherwise
        """
        if not super().is_equivalent(other):
            return False
        if self.deposition_mode is not None and other.deposition_mode is not None:
            return self.deposition_mode == other.deposition_mode
        # note: thickness range information might be lost as not deemed of significance in comparison
        return True

    def create_xml(self, add_as_part = True, add_relationships = True, originator = None, reuse = True):
        """Creates a stratigraphic unit interpretation xml tree.

        arguments:
           add_as_part (bool, default True): if True, the interpretation is added to the parent model as a high level part
           add_relationships (bool, default True): if True and add_as_part is True, a relationship is created with
              the referenced stratigraphic unit feature
           originator (str, optional): if present, is used as the originator field of the citation block
           reuse (bool, default True): if True, the parent model is inspected for any equivalent interpretation and, if found,
              the uuid of this interpretation is set to that of the equivalent part

        returns:
           lxml.etree._Element: the root node of the newly created xml tree for the interpretation
        """

        if reuse and self.try_reuse():
            return self.root

        sui = super().create_xml(add_as_part = add_as_part,
                                 add_relationships = add_relationships,
                                 originator = originator,
                                 reuse = False)
        assert sui is not None

        if self.deposition_mode is not None:
            assert self.deposition_mode in rqstc.valid_deposition_modes,  \
                f'invalid deposition mode {self.deposition_mode} for stratigraphic unit interpretation'
            dm_node = rqet.SubElement(sui, ns['resqml2'] + 'DepositionMode')
            dm_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'DepositionMode')
            dm_node.text = self.deposition_mode

        if self.min_thickness is not None or self.max_thickness is not None:
            assert self.thickness_uom in wam.valid_uoms(quantity = 'length')

        if self.min_thickness is not None:
            min_thick_node = rqet.SubElement(sui, ns['resqml2'] + 'MinThickness')
            min_thick_node.set(ns['xsi'] + 'type', ns['eml'] + 'LengthMeasure')
            min_thick_node.set('uom', self.thickness_uom)  # todo: check this
            min_thick_node.text = str(self.min_thickness)

        if self.max_thickness is not None:
            max_thick_node = rqet.SubElement(sui, ns['resqml2'] + 'MaxThickness')
            max_thick_node.set(ns['xsi'] + 'type', ns['eml'] + 'LengthMeasure')
            max_thick_node.set('uom', self.thickness_uom)
            max_thick_node.text = str(self.max_thickness)

        return sui
