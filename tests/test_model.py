from numpy.lib.arraysetops import isin
import pytest
import os
import numpy as np

import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.well as rqw
import resqpy.property as rqp
import resqpy.olio.xml_et as rqet
import resqpy.olio.uuid as bu
import resqpy.olio.write_hdf5 as rwh5


def test_model(tmp_path):

   epc = os.path.join(tmp_path, 'model.epc')
   model = rq.new_model(epc)
   assert model is not None
   crs = rqc.Crs(model)
   crs_root = crs.create_xml()
   model.store_epc()
   assert os.path.exists(epc)
   md_datum_1 = rqw.MdDatum(model, location = (0.0, 0.0, -50.0), crs_uuid = crs.uuid)
   md_datum_1.create_xml(title = 'Datum 1')
   md_datum_2 = rqw.MdDatum(model, location = (3.0, 0.0, -50.0), crs_uuid = crs.uuid)
   md_datum_2.create_xml(title = 'Datum 2')
   assert len(model.uuids(obj_type = 'MdDatum')) == 2
   model.store_epc()

   model = rq.Model(epc)
   assert model is not None
   assert len(model.uuids(obj_type = 'MdDatum')) == 2
   datum_part_1 = model.part(obj_type = 'MdDatum', title = '1', title_mode = 'ends')
   datum_part_2 = model.part(obj_type = 'MdDatum', title = '2', title_mode = 'ends')
   assert datum_part_1 is not None and datum_part_2 is not None and datum_part_1 != datum_part_2
   datum_uuid_1 = rqet.uuid_in_part_name(datum_part_1)
   datum_uuid_2 = rqet.uuid_in_part_name(datum_part_2)
   assert not bu.matching_uuids(datum_uuid_1, datum_uuid_2)
   p1 = model.uuid_part_dict[bu.uuid_as_int(datum_uuid_1)]
   p2 = model.uuid_part_dict[bu.uuid_as_int(datum_uuid_2)]
   assert p1 == datum_part_1 and p2 == datum_part_2


def test_model_iterators(example_model_with_well):

   model, well_interp, datum, traj = example_model_with_well

   w = next(model.iter_wellbore_interpretations())
   d = next(model.iter_md_datums())
   t = next(model.iter_trajectories())

   assert bu.matching_uuids(w.uuid, well_interp.uuid)
   assert bu.matching_uuids(d.uuid, datum.uuid)
   assert bu.matching_uuids(t.uuid, traj.uuid)
   assert w == well_interp
   assert d == datum
   assert t == traj


def test_model_iterate_objects(example_model_with_well):

   from resqpy.organize import WellboreFeature, WellboreInterpretation

   model, well_interp, _, _ = example_model_with_well

   # Try iterating over wellbore features

   wells = model.iter_objs(WellboreFeature)
   wells = list(wells)
   w = wells[0]
   assert len(wells) == 1
   assert isinstance(wells[0], WellboreFeature)
   assert wells[0].title == "well A"

   # Try iterating over wellbore interpretations

   interps = model.iter_objs(WellboreInterpretation)
   interps = list(interps)
   assert len(interps) == 1
   assert isinstance(interps[0], WellboreInterpretation)
   assert interps[0] == well_interp


def test_model_iter_crs(example_model_and_crs):

   model, crs_1 = example_model_and_crs

   crs_list = list(model.iter_crs())
   assert len(crs_list) == 1

   crs_2 = crs_list[0]
   assert crs_2 == crs_1


def test_model_iter_crs_empty(tmp_model):

   # Should raise an exception if no CRS exists
   with pytest.raises(StopIteration):
      next(tmp_model.iter_crs())


def test_model_as_graph(example_model_with_well):

   model, well_interp, datum, traj = example_model_with_well
   crs = next(model.iter_crs())

   nodes, edges = model.as_graph()

   # Check nodes
   for obj in [well_interp, datum, traj, crs]:
      assert str(obj.uuid) in nodes.keys()
      if hasattr(obj, 'title'):  # TODO: remove this when all objects have this attribute
         assert obj.title == nodes[str(obj.uuid)]['title']

   # Check edges
   assert frozenset([str(traj.uuid), str(datum.uuid)]) in edges
   assert frozenset([str(traj.uuid), str(crs.uuid)]) in edges
   assert frozenset([str(traj.uuid), str(well_interp.uuid)]) in edges
   assert frozenset([str(datum.uuid), str(traj.uuid)]) in edges

   # Test uuid subset
   nodes, edges = model.as_graph(uuids_subset = [datum.uuid])
   assert len(nodes.keys()) == 1
   assert len(edges) == 0


def test_model_copy_all_parts(example_model_with_properties):

   epc = example_model_with_properties.epc_file
   dir = example_model_with_properties.epc_directory
   copied_epc = os.path.join(dir, 'copied.epc')

   # test copying without consolidation
   original = rq.Model(epc)
   assert original is not None
   copied = rq.new_model(copied_epc)
   copied.copy_all_parts_from_other_model(original, consolidate = False)

   assert set(original.uuids()) == set(copied.uuids())
   assert set(original.parts()) == set(copied.parts())

   # test without consolidation of two crs objects
   copied = rq.new_model(copied_epc)
   new_crs = rqc.Crs(copied)
   new_crs.create_xml()

   copied.copy_all_parts_from_other_model(original, consolidate = False)

   assert len(copied.parts()) == len(original.parts()) + 1
   assert set(original.parts()).issubset(set(copied.parts()))
   assert len(copied.parts(obj_type = 'LocalDepth3dCrs')) == 2

   # test with consolidation of two crs objects
   copied = rq.new_model(copied_epc)
   new_crs = rqc.Crs(copied)
   new_crs.create_xml()

   copied.copy_all_parts_from_other_model(original, consolidate = True)

   assert len(copied.parts()) == len(original.parts())
   assert len(copied.parts(obj_type = 'LocalDepth3dCrs')) == 1

   crs_uuid = copied.uuid(obj_type = 'LocalDepth3dCrs')
   assert (bu.matching_uuids(crs_uuid, new_crs.uuid) or
           bu.matching_uuids(crs_uuid, original.uuid(obj_type = 'LocalDepth3dCrs')))

   # test write and re-load of copied model
   copied.store_epc()
   re_opened = rq.Model(copied_epc)
   assert re_opened is not None

   assert len(copied.parts()) == len(original.parts())

   crs_uuid = re_opened.uuid(obj_type = 'LocalDepth3dCrs')
   assert (bu.matching_uuids(crs_uuid, new_crs.uuid) or
           bu.matching_uuids(crs_uuid, original.uuid(obj_type = 'LocalDepth3dCrs')))


def test_model_copy_all_parts_non_resqpy_hdf5_paths(example_model_with_properties):
   # this test uses low level methods to override the hdf5 internal paths for some new objects

   epc = example_model_with_properties.epc_file
   dir = example_model_with_properties.epc_directory
   copied_epc = os.path.join(dir, 'copied.epc')
   extent_kji = (3, 5, 5)  # needs to match extent of grid in example model

   # add some properties with bespoke internal hdf5 paths
   original = rq.Model(epc)
   assert original is not None
   grid_uuid = original.uuid(obj_type = 'IjkGridRepresentation')
   assert grid_uuid is not None
   data = np.linspace(0.0, 1000.0, num = 75).reshape(extent_kji)

   hdf5_paths = [
      '/RESQML/uuid_waffle/values', '/RESQML/waffle_uuid/unusual_name', 'RESQML/class_uuid_waffle/values0',
      'RESQML/something_interesting/array', '/RESQML/abrupt', 'no_resqml/uuid/values'
   ]

   prop_list = []
   path_list = []  # elements will have 'uuid' replaced with uuid string
   h5_reg = rwh5.H5Register(original)
   for i, hdf5_path in enumerate(hdf5_paths):
      prop = rqp.Property(original, support_uuid = grid_uuid)
      if 'uuid' in hdf5_path:
         path = hdf5_path.replace('uuid', str(prop.uuid))
      else:
         path = hdf5_path
      prop.prepare_import(cached_array = data,
                          source_info = 'test data',
                          keyword = f'TEST{i}',
                          property_kind = 'continuous',
                          uom = 'm')
      for entry in prop.collection.imported_list:  # only one entry
         # override internal hdf5 path
         h5_reg.register_dataset(entry[0], 'values_patch0', prop.collection.__dict__[entry[3]], hdf5_path = path)
      prop_list.append(prop)
      path_list.append(path)
   h5_reg.write(mode = 'a')

   for prop, hdf5_path in zip(prop_list, path_list):
      root_node = prop.create_xml(find_local_property_kind = False)
      assert root_node is not None
      # override the hdf5 internal path in the xml tree
      path_node = rqet.find_nested_tags(root_node, ['PatchOfValues', 'Values', 'Values', 'PathInHdfFile'])
      assert path_node is not None
      path_node.text = hdf5_path

   # rewrite model
   original.store_epc()

   # re-open model and test copying
   original = rq.Model(epc)
   assert original is not None
   copied = rq.new_model(copied_epc)
   copied.copy_all_parts_from_other_model(original, consolidate = False)

   assert set(original.uuids()) == set(copied.uuids())
   assert set(original.parts()) == set(copied.parts())
