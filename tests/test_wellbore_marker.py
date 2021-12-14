import numpy as np
import pandas as pd

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo
import resqpy.well


def test_load_from_xml(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    epc_path = model.epc_file
    elevation = 100
    # Create a measured depth datum
    datum = resqpy.well.MdDatum(parent_model = model,
                                crs_uuid = crs.uuid,
                                location = (0, 0, -elevation),
                                md_reference = 'kelly bushing')
    datum.create_xml()
    mds = np.array([300, 310, 330, 340])
    zs = mds - elevation
    well_name = 'JubJub'
    source_dataframe = pd.DataFrame({
        'MD': mds,
        'X': [1, 2, 3, 4],
        'Y': [1, 2, 3, 4],
        'Z': zs,
        'WELL': ['JubJub', 'JubJub', 'JubJub', 'JubJub']
    })

    # Create a trajectory from dataframe
    trajectory = resqpy.well.Trajectory(parent_model = model,
                                        data_frame = source_dataframe,
                                        well_name = well_name,
                                        md_datum = datum,
                                        length_uom = 'm')
    trajectory.write_hdf5()
    trajectory.create_xml()
    trajectory_uuid = trajectory.uuid

    # Create a wellbore marker frame
    wellbore_marker_frame = resqpy.well.WellboreMarkerFrame(parent_model = model,
                                                            trajectory_uuid = trajectory_uuid,
                                                            title = 'WMF1',
                                                            originator = 'Person1',
                                                            extra_metadata = {'target_reservoir': 'r1'})
    wellbore_marker_frame.create_xml()

    # Create several boundary features and interpretations
    horizon_feature_1 = rqo.GeneticBoundaryFeature(parent_model = model,
                                                   kind = 'horizon',
                                                   feature_name = 'horizon_feature_1')
    horizon_feature_1.create_xml()
    horizon_interp_1 = rqo.HorizonInterpretation(parent_model = model,
                                                 title = 'horizon_interp_1',
                                                 genetic_boundary_feature = horizon_feature_1,
                                                 sequence_stratigraphy_surface = 'flooding',
                                                 boundary_relation_list = ['conformable'])
    horizon_interp_1.create_xml()
    horizon_interp_uuid = horizon_interp_1.uuid

    fluid_contact_feature1 = rqo.FluidBoundaryFeature(parent_model = model,
                                                      kind = "gas oil contact",
                                                      feature_name = 'goc_1')
    fluid_contact_feature1.create_xml()

    # --------- Act ----------
    # Create a wellbore marker object
    wellbore_marker_1 = resqpy.well.WellboreMarker(parent_model = model,
                                                   parent_frame = wellbore_marker_frame,
                                                   marker_index = 0,
                                                   marker_type = 'horizon',
                                                   interpretation_uuid = horizon_interp_uuid,
                                                   title = 'Horizon1_marker',
                                                   extra_metadata = {'FormationName': 'Banoffee'})
    wellbore_marker_1_uuid = wellbore_marker_1.uuid
    wellbore_marker_2 = resqpy.well.WellboreMarker(parent_model = model,
                                                   parent_frame = wellbore_marker_frame,
                                                   marker_index = 1,
                                                   marker_type = 'gas oil contact',
                                                   title = 'GOC_marker')
    # Create xml for new wellbore markers
    wbm_node_1 = wellbore_marker_1.create_xml(parent_node = wellbore_marker_frame.root)
    wbm_node_2 = wellbore_marker_2.create_xml(parent_node = wellbore_marker_frame.root)

    # Load a new wellbore marker using a marker node
    wellbore_marker_3 = resqpy.well.WellboreMarker(parent_model = model,
                                                   parent_frame = wellbore_marker_frame,
                                                   marker_index = 0,
                                                   marker_node = wbm_node_1)

    # --------- Assert ----------
    assert rqet.find_tag_text(root = wbm_node_1, tag_name = 'GeologicBoundaryKind') == 'horizon'
    assert rqet.find_tag_text(root = wbm_node_2, tag_name = 'FluidContact') == 'gas oil contact'
    assert wellbore_marker_3 is not None
    assert wellbore_marker_3.title == 'Horizon1_marker'
    assert wellbore_marker_3.marker_type == 'horizon'
    assert bu.matching_uuids(wellbore_marker_3.interpretation_uuid, horizon_interp_uuid)
    assert wellbore_marker_3.extra_metadata == {'FormationName': 'Banoffee'}
    assert bu.matching_uuids(wellbore_marker_3.uuid, wellbore_marker_1_uuid)
