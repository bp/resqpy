import pytest

import resqpy.model as rq
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


# ---- Test bespoke property kind reuse ---

def test_bespoke_property_kind():
   model = rq.Model(create_basics = True)
   em = {'something': 'important', 'and_another_thing': 42}
   pk1 = rqp.PropertyKind(model, title = 'my kind of property', extra_metadata = em)
   pk1.create_xml()
   assert len(model.uuids(obj_type = 'PropertyKind')) == 1
   pk2 = rqp.PropertyKind(model, title = 'my kind of property', extra_metadata = em)
   pk2.create_xml()
   assert len(model.uuids(obj_type = 'PropertyKind')) == 1
   pk3 = rqp.PropertyKind(model, title = 'your kind of property', extra_metadata = em)
   pk3.create_xml()
   assert len(model.uuids(obj_type = 'PropertyKind')) == 2
   pk4 = rqp.PropertyKind(model, title = 'my kind of property')
   pk4.create_xml()
   assert len(model.uuids(obj_type = 'PropertyKind')) == 3
   pk5 = rqp.PropertyKind(model, title = 'my kind of property', parent_property_kind = 'discrete', extra_metadata = em)
   pk5.create_xml()
   assert len(model.uuids(obj_type = 'PropertyKind')) == 4
   pk6 = rqp.PropertyKind(model, title = 'my kind of property', parent_property_kind = 'continuous')
   pk6.create_xml()
   assert len(model.uuids(obj_type = 'PropertyKind')) == 4
   assert pk6 == pk4


# ---- Test string lookup ---

def test_string_lookup():
   model = rq.Model(create_basics = True)
   d = {1: 'peaty',
        2: 'muddy',
        3: 'sandy',
        4: 'granite'}
   sl = rqp.StringLookup(model, int_to_str_dict = d, title = 'stargazing')
   sl.create_xml()
   assert sl.length() == 5 if sl.stored_as_list else 4
   assert sl.get_string(3) == 'sandy'
   assert sl.get_index_for_string('muddy') == 2
   d2 = {0: 'nothing',
         27: 'something',
         173: 'amazing',
         1072: 'brilliant'}
   sl2 = rqp.StringLookup(model, int_to_str_dict = d2, title = 'head in the clouds')
   assert not sl.stored_as_list
   assert sl.length() == 4
   assert sl.get_string(1072) == 'brilliant'
   assert sl.get_string(555) is None
   assert sl.get_index_for_string('amazing') == 173
   sl2.create_xml()
   assert set(model.titles(obj_type = 'StringTableLookup')) == set(['stargazing', 'head in the clouds'])
   assert sl != sl2

