import os
import math as maths

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.model as rq
import resqpy.grid as grr
import resqpy.property as rqp
import resqpy.derived_model as rqdm
import resqpy.olio.weights_and_measures as bwam
from resqpy.olio.exceptions import IncompatibleUnitsError

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
   p1 = rqp.Property.from_array(model, a1, source_info = 'random', keyword = 'NETGRS', support_uuid = grid.uuid,
                                property_kind = 'net to gross ratio', indexable_element = 'cells', uom = 'm3/m3')
   pk = rqp.PropertyKind(model, title = 'facies', parent_property_kind = 'discrete')
   pk.create_xml()
   facies_dict = {0: 'background'}
   for i in range(1, 10): facies_dict[i] = 'facies ' + str(i)
   sl = rqp.StringLookup(model, int_to_str_dict = facies_dict, title = 'facies table')
   sl.create_xml()
   shape2 = tuple(list(grid.extent_kji) + [2])
   a2 = (np.random.random(shape2) * 10.0).astype(int)
   p2 = rqp.Property.from_array(model, a2, source_info = 'random', keyword = 'FACIES', support_uuid = grid.uuid,
                                local_property_kind_uuid = pk.uuid, indexable_element = 'cells', discrete = True,
                                null_value = 0, count = 2, string_lookup_uuid = sl.uuid)
   model.store_epc()
   model = rq.Model(epc)
   ntg_uuid = model.uuid(obj_type = p1.resqml_type, title = 'NETGRS')
   assert ntg_uuid is not None
   p1p = rqp.Property(model, uuid = ntg_uuid)
   assert np.all(p1p.array_ref() == p1.array_ref())
   assert p1p.uom() == 'm3/m3'
   facies_uuid = model.uuid(obj_type = p2.resqml_type, title = 'FACIES')
   assert facies_uuid is not None
   p2p = rqp.Property(model, uuid = facies_uuid)
   assert np.all(p2p.array_ref() == p2.array_ref())
   assert p2p.null_value() is not None and p2p.null_value() == 0
   grid = model.grid()
   assert grid.property_collection.number_of_parts() == 5  # two created here, plus 3 regular grid cell lengths properties


def test_create_Property_from_singleton_collection(tmp_model):

   # Arrange
   grid = grr.RegularGrid(tmp_model, extent_kji = (2, 3, 4))
   grid.write_hdf5()
   grid.create_xml(add_cell_length_properties = True)
   collection = rqp.selective_version_of_collection(grid.property_collection, property_kind = 'cell length',
                                                    facet_type = 'direction', facet='J')
   assert collection.number_of_parts() == 1
   # Check property can be created
   prop = rqp.Property.from_singleton_collection(collection)
   assert np.all(np.isclose(prop.array_ref(), 1.0))


# ---- Test uoms ---


def test_valid_uoms():

   # Good units
   assert "0.001 gal[US]/gal[US]" in bwam.valid_uoms()
   assert "Btu[IT]/min" in bwam.valid_uoms()
   assert "%" in bwam.valid_uoms()

   # Bad units
   assert "foo barr" not in bwam.valid_uoms()
   assert "" not in bwam.valid_uoms()
   assert None not in bwam.valid_uoms()


def test_uom_aliases():
   for uom, aliases in bwam.ALIASES.items():
      assert uom in bwam.valid_uoms(), f"Bad uom {uom}"
      for alias in aliases:
         if alias != uom:
            assert alias not in bwam.valid_uoms(), f"Bad alias {alias}"

   for alias, uom in bwam.ALIAS_MAP.items():
      assert uom in bwam.valid_uoms(), f"Bad uom {uom}"
      if alias != uom:
         assert alias not in bwam.valid_uoms(), f"Bad alias {alias}"


@pytest.mark.parametrize("input_uom, expected_uom", [
   ('gapi', 'gAPI'),
   ('m', 'm'),
   ('M', 'm'),
   ('gal[UK]/mi', 'gal[UK]/mi'),
   ("0.001 gal[US]/gal[US]", "0.001 gal[US]/gal[US]"),
])
def test_uom_from_string(input_uom, expected_uom):
   validated_uom = bwam.rq_uom(input_uom)
   assert expected_uom == validated_uom


# ---- Test property conversions -------

@pytest.mark.parametrize("unit_from, unit_to, value, expected", [
   # Straightforward conversions
   ("m", "m", 1, 1),
   ("m", "km", 1, 0.001),
   ("ft", "m", 1, 0.3048),
   ("ft", "ft[US]", 1, 0.999998),

   # Aliases of common units
   ("metres", "m", 1, 1),
   ("meters", "m", 1, 1),
   ("pu", "%", 1, 1),
   ("p.u.", "%", 1, 1),

   # Different base units!
   ("%", "v/v", 1, 0.01),
   ("pu", "v/v", 1, 0.01),
   ("m3/m3", "%", 1, 100),
   ("D", "ft2", 10, 1.062315e-10),
])
@pytest.mark.filterwarnings("ignore:Assuming base units")
def test_unit_conversion(unit_from, unit_to, value, expected):
   result = bwam.convert(value, unit_from, unit_to)
   assert maths.isclose(result, expected, rel_tol=1e-4)


def test_convert_array():
   # Duck typing should work
   value = np.array([1,2,3])
   expected = np.array([1000, 2000, 3000])
   result = bwam.convert(value, unit_from="km", unit_to="m")
   assert_array_almost_equal(result, expected)


def test_conversion_factors_are_numeric():
   for uom in bwam.valid_uoms():
      base_unit, dimension, factors = bwam.get_conversion_factors(uom)
      assert base_unit in bwam.valid_uoms()
      assert len(dimension) > 0
      assert len(factors) == 4, f"Issue with {uom}"
      assert all(isinstance(f, (int, float)) for f in factors), f"Issue with {uom}"


# Test incompatible units raise an Error

@pytest.mark.parametrize("unit_from, unit_to", [
   ("m", "gAPI"),
   ("%", "bbl"),
   ("%", "ft"),
   ("m", "m3"),
])
def test_incompatible_units(unit_from, unit_to):
   with pytest.raises(IncompatibleUnitsError):
      bwam.convert(1, unit_from, unit_to)



# ---- Test property kind parsing ---


def test_property_kinds():
   prop_kinds = bwam.valid_property_kinds()
   assert "angle per time" in prop_kinds
   assert "foobar" not in prop_kinds


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
   assert not sl2.stored_as_list
   assert sl2.length() == 4
   assert sl2.get_string(1072) == 'brilliant'
   assert sl2.get_string(555) is None
   assert sl2.get_index_for_string('amazing') == 173
   sl2.create_xml()
   assert set(model.titles(obj_type = 'StringTableLookup')) == set(['stargazing', 'head in the clouds'])
   assert sl != sl2



def test_property_extra_metadata(tmp_path):
   # set up test model with a grid
   epc = os.path.join(tmp_path, 'em_test.epc')
   model = rq.new_model(epc)
   grid = grr.RegularGrid(model, extent_kji = (2,3,4))
   grid.write_hdf5()
   grid.create_xml()
   model.store_epc()
   # add a grid property with some extra metadata
   a = np.arange(24).astype(float).reshape((2, 3, 4))
   em = {'important': 'something', 'also': 'nothing'}
   uuid = rqdm.add_one_grid_property_array(epc, a, 'length', title = 'nonsense', uom = 'm', extra_metadata = em)
   # re-open the model and check that extra metadata has been preserved
   model = rq.Model(epc)
   p = rqp.Property(model, uuid = uuid)
   for item in em.items():
       assert item in p.extra_metadata.items()
