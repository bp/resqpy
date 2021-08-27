import pytest
import os

import resqpy.model as rq
import resqpy.fault as rqf
import resqpy.olio.xml_et as rqet


@pytest.mark.parametrize('inc_list,tmult_dict,expected_mult', [(['fault_1.inc'], {}, {
   'fault_1': 1
}), (['fault_1.inc'], {
   'fault_1': 2
}, {
   'fault_1': 2
}), (['fault_1.inc', 'fault_2.inc'], {
   'fault_1': 2
}, {
   'fault_1': 2,
   'fault_2': 2
})])
def test_add_connection_set_and_tmults(example_model_with_properties, test_data_path, inc_list, tmult_dict,
                                       expected_mult):
   model = example_model_with_properties

   inc_list = [os.path.join(test_data_path, inc) for inc in inc_list]

   gcs_uuid = rqf.add_connection_set_and_tmults(model, inc_list, tmult_dict)

   assert gcs_uuid is not None, 'Grid connection set not generated'

   reload_model = rq.Model(epc_file = model.epc_file)

   faults = reload_model.parts_list_of_type(('obj_FaultInterpretation'))
   assert len(faults) == len(
      expected_mult.keys()), f'Expected a {len(expected_mult.keys())} faults, found {len(faults)}'
   for fault in faults:
      metadata = rqet.load_metadata_from_xml(reload_model.root_for_part(fault))
      title = reload_model.citation_title_for_part(fault)
      assert metadata["Transmissibility multiplier"] == str(
         float(expected_mult[title])
      ), f'Expected mult for fault {title} to be {float(expected_mult[title])}, found {metadata["Transmissibility multiplier"]}'


def test_add_connection_set_and_tmults_fails(example_model_with_properties, test_data_path, include = 'fault_3.inc'):
   model = example_model_with_properties

   inc_list = [os.path.join(test_data_path, include)]

   with pytest.raises(NotImplementedError):
      rqf.add_connection_set_and_tmults(model, inc_list)
