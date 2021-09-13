import os

import pytest
import numpy as np

import resqpy.model as rq
import resqpy.grid as grr
import resqpy.property as rqp
import resqpy.time_series as rqts
import resqpy.derived_model as rqdm
import resqpy.weights_and_measures as bwam
import resqpy.olio.vector_utilities as vec

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
   p1 = rqp.Property.from_array(model,
                                a1,
                                source_info = 'random',
                                keyword = 'NETGRS',
                                support_uuid = grid.uuid,
                                property_kind = 'net to gross ratio',
                                indexable_element = 'cells',
                                uom = 'm3/m3')
   a3 = np.random.random((grid.nk + 1, grid.nj + 1, grid.ni + 1))
   p3 = rqp.Property.from_array(model,
                                a3,
                                source_info = 'random',
                                keyword = 'jiggle shared nodes',
                                support_uuid = grid.uuid,
                                property_kind = 'length',
                                indexable_element = 'nodes',
                                uom = 'm3')
   a4 = np.random.random((grid.nk, grid.nj, grid.ni, 2, 2, 2))
   p4 = rqp.Property.from_array(model,
                                a4,
                                source_info = 'random',
                                keyword = 'jiggle nodes per cell',
                                support_uuid = grid.uuid,
                                property_kind = 'length',
                                indexable_element = 'nodes per cell',
                                uom = 'm3')
   pk = rqp.PropertyKind(model, title = 'facies', parent_property_kind = 'discrete')
   pk.create_xml()
   facies_dict = {0: 'background'}
   for i in range(1, 10):
      facies_dict[i] = 'facies ' + str(i)
   sl = rqp.StringLookup(model, int_to_str_dict = facies_dict, title = 'facies table')
   sl.create_xml()
   shape2 = tuple(list(grid.extent_kji) + [2])
   a2 = (np.random.random(shape2) * 10.0).astype(int)
   p2 = rqp.Property.from_array(model,
                                a2,
                                source_info = 'random',
                                keyword = 'FACIES',
                                support_uuid = grid.uuid,
                                local_property_kind_uuid = pk.uuid,
                                indexable_element = 'cells',
                                discrete = True,
                                null_value = 0,
                                count = 2,
                                string_lookup_uuid = sl.uuid)
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
   jiggle_parts = model.parts(title = 'jiggle', title_mode = 'starts')
   assert len(jiggle_parts) == 2
   jiggle_shared_uuid = model.uuid(parts_list = jiggle_parts, title = 'shared', title_mode = 'contains')
   assert jiggle_shared_uuid is not None
   p3p = rqp.Property(model, uuid = jiggle_shared_uuid)
   assert p3p is not None
   assert p3p.array_ref().shape == (grid.nk + 1, grid.nj + 1, grid.ni + 1)
   jiggle_per_cell_uuid = model.uuid(parts_list = jiggle_parts, title = 'per cell', title_mode = 'ends')
   assert jiggle_per_cell_uuid is not None
   # four properties created here, plus 3 regular grid cell lengths properties
   assert grid.property_collection.number_of_parts() == 7
   collection = rqp.selective_version_of_collection(grid.property_collection,
                                                    property_kind = 'length',
                                                    uuid = jiggle_per_cell_uuid)
   assert collection is not None
   assert collection.number_of_parts() == 1
   p4p = rqp.Property.from_singleton_collection(collection)
   assert p4p is not None
   assert p4p.array_ref().shape == (grid.nk, grid.nj, grid.ni, 2, 2, 2)


def test_create_Property_from_singleton_collection(tmp_model):

   # Arrange
   grid = grr.RegularGrid(tmp_model, extent_kji = (2, 3, 4))
   grid.write_hdf5()
   grid.create_xml(add_cell_length_properties = True)
   collection = rqp.selective_version_of_collection(grid.property_collection,
                                                    property_kind = 'cell length',
                                                    facet_type = 'direction',
                                                    facet = 'J')
   assert collection.number_of_parts() == 1
   # Check property can be created
   prop = rqp.Property.from_singleton_collection(collection)
   assert np.all(np.isclose(prop.array_ref(), 1.0))


# ---- Test property kind parsing ---


def test_property_kinds():
   prop_kinds = bwam.valid_property_kinds()
   assert "angle per time" in prop_kinds
   assert "foobar" not in prop_kinds


# ---- Test infer_property_kind ---


@pytest.mark.parametrize(
   "input_name, input_unit, kind, facet_type, facet",
   [
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
   d = {1: 'peaty', 2: 'muddy', 3: 'sandy', 4: 'granite'}
   sl = rqp.StringLookup(model, int_to_str_dict = d, title = 'stargazing')
   sl.create_xml()
   assert sl.length() == 5 if sl.stored_as_list else 4
   assert sl.get_string(3) == 'sandy'
   assert sl.get_index_for_string('muddy') == 2
   d2 = {0: 'nothing', 27: 'something', 173: 'amazing', 1072: 'brilliant'}
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
   grid = grr.RegularGrid(model, extent_kji = (2, 3, 4))
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


def test_points_properties(tmp_path):

   epc = os.path.join(tmp_path, 'points_test.epc')
   model = rq.new_model(epc)

   extent_kji = (2, 3, 4)
   ensemble_size = 5
   time_series_size = 6

   # create a geological time series
   years = [int(t) for t in np.linspace(-252170000, -66000000, num = time_series_size, dtype = int)]
   ts = rqts.GeologicTimeSeries.from_year_list(model, year_list = years, title = 'Mesozoic time series')
   ts.create_xml()
   ts_uuid = ts.uuid
   assert ts.timeframe == 'geologic'

   # create a simple grid without an explicit geometry and ensure it has a property collection initialised
   grid = grr.RegularGrid(model, extent_kji = extent_kji, origin = (0.0, 0.0, 1000.0))
   grid.create_xml()
   if grid.property_collection is None:
      grid.property_collection = rqp.PropertyCollection(support = grid)
   pc = grid.property_collection

   # shape of points property arrays
   points_shape = tuple(list(extent_kji) + [3])

   # create a static points property with multiple realisations
   for r in range(ensemble_size):
      stress = vec.unit_vectors(np.random.random(points_shape) + 0.1)
      pc.add_cached_array_to_imported_list(stress,
                                           'random stress vectors',
                                           'stress direction',
                                           uom = 'm',
                                           property_kind = 'length',
                                           realization = r,
                                           indexable_element = 'cells',
                                           points = True)
   pc.write_hdf5_for_imported_list()
   pc.create_xml_for_imported_list_and_add_parts_to_model()

   # create a dynamic points property (indexable cells) related to the geological time series
   for r in range(ensemble_size):
      centres = grid.centre_point().copy()
      for time_index in range(time_series_size):
         pc.add_cached_array_to_imported_list(centres,
                                              'dynamic cell centres',
                                              'centres',
                                              uom = 'm',
                                              property_kind = 'length',
                                              realization = r,
                                              time_index = time_index,
                                              indexable_element = 'cells',
                                              points = True)
         centres[..., 2] += 100.0 * (r + 1)
   pc.write_hdf5_for_imported_list()
   pc.create_xml_for_imported_list_and_add_parts_to_model(time_series_uuid = ts_uuid)

   # create a dynamic points property (indexable nodes) related to the geological time series
   for r in range(ensemble_size):
      nodes = grid.points_ref().copy()
      for time_index in range(time_series_size):
         pc.add_cached_array_to_imported_list(nodes,
                                              'dynamic nodes',
                                              'points',
                                              uom = 'm',
                                              property_kind = 'length',
                                              realization = r,
                                              time_index = time_index,
                                              indexable_element = 'nodes',
                                              points = True)
         nodes[..., 2] += 100.0 * (r + 1)
   pc.write_hdf5_for_imported_list()
   pc.create_xml_for_imported_list_and_add_parts_to_model(time_series_uuid = ts_uuid)

   # store the xml
   model.store_epc()

   # re-open the model and access the grid properties
   model = rq.Model(epc)
   grid = model.grid()
   pc = grid.property_collection

   # load a static points property for a single realisation
   sample_stress = pc.single_array_ref(realization = ensemble_size // 2,
                                       citation_title = 'stress direction',
                                       points = True)
   assert sample_stress is not None and sample_stress.ndim == 4
   assert np.all(sample_stress.shape[:3] == extent_kji)
   assert sample_stress.shape[3] == 3
   assert np.count_nonzero(np.isnan(sample_stress)) == 0

   # select the dynamic points properties related to the geological time series and indexable by cells
   cc = rqp.selective_version_of_collection(grid.property_collection,
                                            indexable = 'cells',
                                            points = True,
                                            time_series_uuid = ts_uuid)
   assert cc.number_of_parts() == ensemble_size * time_series_size

   # check that 5 dimensional numpy arrays can be set up, each covering time indices for a single realisation
   for r in range(ensemble_size):
      rcc = rqp.selective_version_of_collection(cc, realization = r)
      assert rcc.number_of_parts() == time_series_size
      dynamic_nodes = rcc.time_series_array_ref()
      assert dynamic_nodes.ndim == 5
      assert dynamic_nodes.shape[0] == time_series_size
      assert np.all(dynamic_nodes.shape[1:4] == np.array(extent_kji))
      assert dynamic_nodes.shape[-1] == 3
      mean_depth = np.nanmean(dynamic_nodes[..., 2])
      assert mean_depth > 1000.0

   # select the dynamic points properties related to the geological time series and indexable by nodes
   nc = rqp.selective_version_of_collection(grid.property_collection,
                                            indexable = 'nodes',
                                            points = True,
                                            time_series_uuid = ts_uuid)
   assert nc.number_of_parts() == ensemble_size * time_series_size

   # check that 5 dimensional numpy arrays can be set up, each covering realisations for a single time index
   for ti in range(time_series_size):
      tnc = rqp.selective_version_of_collection(nc, time_index = ti)
      assert tnc.number_of_parts() == ensemble_size
      dynamic_nodes = tnc.realizations_array_ref()
      assert dynamic_nodes.ndim == 5
      assert dynamic_nodes.shape[0] == ensemble_size
      assert np.all(dynamic_nodes.shape[1:4] == np.array(extent_kji) + 1)
      assert dynamic_nodes.shape[-1] == 3
      mean_depth = np.nanmean(dynamic_nodes[..., 2])
      assert mean_depth > 1000.0
