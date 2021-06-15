import uuid
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
   md_datum_1 = rqw.MdDatum(model, location = (0.0, 0.0, -50.0), crs_root = crs_root)
   md_datum_1.create_xml(title = 'Datum 1')
   md_datum_2 = rqw.MdDatum(model, location = (3.0, 0.0, -50.0), crs_root = crs_root)
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

def test_iter_wells(example_model_with_well):
   model: rq.Model
   model, well_interp, datum, traj = example_model_with_well

   w = next(model.iter_wellbore_interpretations())
   d = next(model.iter_md_datums())
   t = next(model.iter_trajectories())

   assert w.uuid == well_interp.uuid
   assert w == well_interp
   assert d.uuid == datum.uuid
   assert d == datum
   assert t.uuid == traj.uuid
   assert t == traj