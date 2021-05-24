import pytest

import resqpy.property as rqp
import resqpy.olio.weights_and_measures as bwam


# ---- Test uom from string ---


@pytest.mark.parametrize("case_sensitive, input_uom, expected_uom", [
   (False, 'gapi', 'gAPI'),
   (True, 'gapi', 'Euc'),
   (False, 'm', 'm'),
   (False, 'M', 'm'),
   (True, 'foobar', 'Euc'),
   (True, '', 'Euc'),
   (False, 'gal[UK]/mi', 'gal[UK]/mi'),
   (False, 'GAL[UK]/MI', 'gal[UK]/mi'),
   (False, None, 'Euc'),
])
def test_uom_from_string(case_sensitive, input_uom, expected_uom):
   validated_uom = rqp.validate_uom_from_string(input_uom, case_sensitive=case_sensitive)
   assert expected_uom == validated_uom


# ---- Test property kind parsing ---


def test_property_kinds():
   prop_kinds = bwam.properties_data()['property_kinds']
   assert "angle per time" in prop_kinds.keys()
   assert "foobar" not in prop_kinds.keys()
   assert prop_kinds['area'] is None
   assert prop_kinds['amplitude'] == "Amplitude of the acoustic signal recorded. " \
                                     "It is not a physical property, only a value."


# ---- Test infer_property_kind ---

@pytest.mark.parametrize("input_name, input_unit, kind, facet_type, facet", [
   ('gamma ray API unit', 'gAPI', 'gamma ray API unit', None, None),  # Just a placeholder for now
   ('', '', 'Unknown', None, None),  # What should default return?
])
def test_infer_property_kind(input_name, input_unit, kind, facet_type, facet):
   kind_, facet_type_, facet_ = rqp.infer_property_kind(input_name, input_unit)
   assert kind == kind_
   assert facet_type == facet_type_
   assert facet == facet_
