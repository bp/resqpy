import pytest
import os

import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.olio.xml_et as rqet
import resqpy.olio.uuid as bu

def test_model(tmp_path):

   epc = os.path.join(tmp_path, 'model.epc')
   model = rq.new_model(epc)
   assert model is not None
   crs = rqc.Crs()
   crs_root = crs.create_xml()
   model.store_epc()
   assert os.path.exists(epc)
   md_datum_1 = rqw.MdDatum(model, location = (0.0, 0.0, -50.0), crs_root = crs_root)
   md_datum_1.create_xml(title = 'Datum 1')
   md_datum_2 = rqw.MdDatum(model, location = (3.0, 0.0, -50.0), crs_root = crs_root)
   md_datum_2.create_xml(title = 'Datum 2')
   assert len(model.uuids(obj_type = 'MdDatum)) == 2
   model.store_epc()

   model = rq.Model(epc)
   assert model is not None
   assert len(model.uuids(obj_type = 'MdDatum)) == 2
   datum_part_1 = model.part(obj_type = 'MdDatum', title = '1', title_mode = 'endswith')
   datum_part_2 = model.part(obj_type = 'MdDatum', title = '2', title_mode = 'endswith')
   assert datum_part_1 is not None and datum_part_2 is not None and datum_part_1 != datum_part_2
   datum_uuid_1 = rqet.uuid_in_part_name(datum_part_1)
   datum_uuid_2 = rqet.uuid_in_part_name(datum_part_2)
   assert not bu.matching_uuids(datum_uuid_1, datum_uuid_2)
   p1 = model.uuid_part_dict[bu.uuid_as_int(datum_uuid_1)]
   p2 = model.uuid_part_dict[bu.uuid_as_int(datum_uuid_2)]
   assert p1 == datum_part_1 and p2 == datum_part_2
