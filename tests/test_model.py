from numpy.lib.arraysetops import isin
import pytest
import os

import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.well as rqw
import resqpy.olio.xml_et as rqet
import resqpy.olio.uuid as bu


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
