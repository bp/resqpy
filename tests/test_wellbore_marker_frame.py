import numpy as np
import pandas as pd

import resqpy.well
import resqpy.organize as rqo
from resqpy.model import Model


def test_WellboreMarkerFrame(example_model_and_crs):
    # Test that all attributes are correctly saved and loaded from disk

    # --------- Arrange ----------
    # Create a WellboreMarkerFrame object in memory
    # Load example model from a fixture
    model, crs = example_model_and_crs
    epc_path = model.epc_file
    # Create a trajectory
    well_name = 'Banoffee'
    elevation = 100
    datum = resqpy.well.MdDatum(parent_model = model,
                                crs_uuid = crs.uuid,
                                location = (0, 0, -elevation),
                                md_reference = 'kelly bushing')
    mds = np.array([300, 310, 330])
    zs = mds - elevation
    source_dataframe = pd.DataFrame({
        'MD': mds,
        'X': [1, 2, 3],
        'Y': [1, 2, 3],
        'Z': zs,
    })
    trajectory = resqpy.well.Trajectory(parent_model = model,
                                        data_frame = source_dataframe,
                                        well_name = well_name,
                                        md_datum = datum,
                                        length_uom = 'm')
    trajectory.write_hdf5()
    trajectory.create_xml()
    # Create features and interpretations
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

    horizon_feature_2 = rqo.GeneticBoundaryFeature(parent_model = model,
                                                   kind = 'horizon',
                                                   feature_name = 'horizon_feature_2')
    horizon_feature_2.create_xml()
    horizon_interp_2 = rqo.HorizonInterpretation(parent_model = model,
                                                 title = 'horizon_interp_2',
                                                 genetic_boundary_feature = horizon_feature_1,
                                                 sequence_stratigraphy_surface = 'transgressive',
                                                 boundary_relation_list = ['conformable'])
    horizon_interp_2.create_xml()

    fault_feature_1 = rqo.TectonicBoundaryFeature(parent_model = model,
                                                  kind = 'fault',
                                                  feature_name = 'fault_feature_1')
    fault_feature_1.create_xml()
    fault_interp_1 = rqo.FaultInterpretation(parent_model = model,
                                             title = 'fault_interp_1',
                                             tectonic_boundary_feature = fault_feature_1,
                                             is_normal = True,
                                             maximum_throw = 15)
    fault_interp_1.create_xml()

    wellbore_marker_frame_dataframe = pd.DataFrame({
        'MD': [300, 310, 330],
        'X': [1, 2, 3],
        'Y': [1, 2, 3],
        'Z': [200, 210, 230],
        'Type': ['horizon', 'horizon', 'fault'],
        'Feature_Name': ['horizon_feature_1', 'horizon_feature_2', 'fault_feature_1'],
        'Interp_Name': ['horizon_interp_1', 'horizon_interp_2', 'fault_interp_1'],
        'Well': well_name
    })

    # Create a wellbore marker frame object
    wellbore_marker_frame = resqpy.well.WellboreMarkerFrame(parent_model = model, trajectory = trajectory)
    uuid = wellbore_marker_frame.uuid
    wellbore_marker_frame.load_from_data_frame(dataframe = wellbore_marker_frame_dataframe)
    # print(wellbore_marker_frame.trajectory.dataframe())
    # print(wellbore_marker_frame.dataframe())
    # print(uuid)
    # ----------- Act ---------
    # Save to disk
    wellbore_marker_frame.write_hdf5()
    wellbore_marker_frame.create_xml()
    model.store_epc()
    model.h5_release()

    # # Clear memory
    # del model, datum, trajectory, wellbore_marker_frame

    # # Reload from disk
    # model2 = Model(epc_file = epc_path)
    # wellbore_marker_frame2 = resqpy.well.WellboreMarkerFrame(parent_model = model2,
    #                                                          uuid = uuid
    #                                                          )
