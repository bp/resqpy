import pytest

import os
import numpy as np

import resqpy.model as rq
import resqpy.grid as grr
import resqpy.property as rqp
import resqpy.olio.weights_and_measures as bwam


# ---- Test PropertyCollection ---

# TODO


# ---- Test Property ---

def test_property(tmp_path):
   epc = os.path.join(tmp_path, 'test_prop.epc')
   model = rq.new_model(epc)
   grid = grr.RegularGrid(model, extent_kji = (2, 3, 4))
   grid.write_hdf5()
   grid.create_xml()
   a1 = np.random.random(grid.extent_kji)
   p1 = rqp.Property(model)
   p1.from_array(a1, source_info = 'random', keyword = 'NETGRS', support_uuid = grid.uuid,
                 property_kind = 'net to gross ratio', indexable_element = 'cells', uom = 'm3/m3')
   pk = rqp.PropertyKind(model, title = 'facies', parent_property_kind = 'discrete')
   pk.create_xml()
   facies_dict = {0: 'background'}
   for i in range(1, 10): facies_dict[i] = 'facies ' + str(i)
   sl = rqp.StringLookup(model, int_to_str_dict = facies_dict, title = 'facies table')
   sl.create_xml()
   shape2 = tuple(list(grid.extent_kji).append(2))
   a2 = (np.random.random(shape2) * 10.0).astype(int)
   p2 = rqp.Property(model)
   p2.from_array(a2, source_info = 'random', keyword = 'FACIES', support_uuid = grid.uuid,
                 local_property_kind_uuid = pk.uuid, indexable_element = 'cells', discrete = True,
                 null_value = 0, count = 2, string_lookup_uuid = sl.uuid)
   model.store_epc()
   model = rq.Model(epc)
   ntg_uuid = model.uuid(obj_type = p1.resqml_type, title = 'NETGRS')
   assert ntg_uuid is not None
   p1p = rqp.Property(model, uuid = ntg_uuid)
   assert np.all(p1p.array_ref() == p1.array_ref())
   facies_uuid = model.uuid(obj_type = p2.resqml_type, title = 'FACIES')
   assert facies_uuid is not None
   p2p = rqp.Property(model, uuid = facies_uuid)
   assert np.all(p2p.array_ref() == p2.array_ref())
   grid = model.grid()
   assert grid.property_collection.number_of_parts() == 2
   

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
