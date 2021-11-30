import numpy as np
import pandas as pd

import resqpy.well
import resqpy.organize as rqo
from resqpy.model import Model


def test_from_dataframe(example_model_and_crs):
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
        'Type': ['horizon', 'horizon', 'fault'],
        'Surface': ['marker_horizon_1', 'marker_horizon_2', 'marker_fault_1'],
        'Interp_Surface': ['horizon_interp_1', 'horizon_interp_2', 'fault_interp_1'],
        'Well': well_name
    })

    # Create a wellbore marker frame from a dataframe
    wellbore_marker_frame = resqpy.well.WellboreMarkerFrame.from_dataframe(parent_model = model,
                                                                           dataframe = wellbore_marker_frame_dataframe,
                                                                           trajectory = trajectory)
    # Save to disk
    wellbore_marker_frame.write_hdf5()
    wellbore_marker_frame.create_xml()
    uuid_wmf = wellbore_marker_frame.uuid  # called after create_xml method as it can alter the uuid
    model.store_epc()
    model.h5_release()

    # Clear memory
    del model, wellbore_marker_frame

    # Reload from disk
    model2 = Model(epc_file = epc_path)
    wellbore_marker_frame2 = resqpy.well.WellboreMarkerFrame(parent_model = model2, uuid = uuid_wmf)
    # --------- Assert ----------
    # test that the attributes were reloaded correctly
    assert wellbore_marker_frame2.trajectory == trajectory
    assert wellbore_marker_frame2.node_count == len(wellbore_marker_frame_dataframe)
    assert wellbore_marker_frame2.get_marker_count() == 3
    np.testing.assert_equal(wellbore_marker_frame2.node_mds, wellbore_marker_frame_dataframe['MD'].values)


def test_dataframe(example_model_and_crs):
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

    wellbore_marker_frame_dataframe = pd.DataFrame({
        'MD': [305],
        'Type': ['horizon'],
        'Surface': ['marker_horizon_1'],
        'Interp_Surface': ['horizon_interp_1'],
        'Well': well_name
    })

    # Create a wellbore marker frame from a dataframe
    wellbore_marker_frame = resqpy.well.WellboreMarkerFrame.from_dataframe(parent_model = model,
                                                                           dataframe = wellbore_marker_frame_dataframe,
                                                                           trajectory = trajectory)
    # Create the expected dataframe that will be created from the wellbore marker frame object
    expected_dataframe = pd.DataFrame({
        'X': np.array([1.5]),
        'Y': np.array([1.5]),
        'Z': np.array([205.0]),
        'MD': np.array([305]),
        'Type': 'Horizon',
        'Surface': '"' + 'horizon_interp_1' + '"',
        'Well': '"' + 'Banoffee' + '"'
    })
    # --------- Assert ----------
    # test that the attributes were reloaded correctly
    assert all(wellbore_marker_frame.dataframe() == expected_dataframe)
    # TODO: get pandas.testing.assert_frame_equal to work


def test_get_interpretation_object(example_model_and_crs):

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
    horizon_interp_2_uuid = horizon_interp_2.uuid

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
        'Type': ['horizon', 'horizon', 'fault'],
        'Surface': ['marker_horizon_1', 'marker_horizon_2', 'marker_fault_1'],
        'Interp_Surface': ['horizon_interp_1', 'horizon_interp_2', 'fault_interp_1'],
        'Well': well_name
    })

    # Create a wellbore marker frame from a dataframe
    wellbore_marker_frame = resqpy.well.WellboreMarkerFrame.from_dataframe(parent_model = model,
                                                                           dataframe = wellbore_marker_frame_dataframe,
                                                                           trajectory = trajectory)
    # --------- Act ----------
    # Get an interpretation based on an uuid
    found_interpretation = wellbore_marker_frame.get_interpretation_obj(interpretation_uuid = horizon_interp_2_uuid)

    # --------- Assert ----------
    found_interpretation == horizon_interp_2


def test_find_marker_from_interp(example_model_and_crs):

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
    horizon_interp_2_uuid = horizon_interp_2.uuid

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
        'Type': ['horizon', 'horizon', 'fault'],
        'Surface': ['marker_horizon_1', 'marker_horizon_2', 'marker_fault_1'],
        'Interp_Surface': ['horizon_interp_1', 'horizon_interp_2', 'fault_interp_1'],
        'Well': well_name
    })

    # Create a wellbore marker frame from a dataframe
    wellbore_marker_frame = resqpy.well.WellboreMarkerFrame.from_dataframe(parent_model = model,
                                                                           dataframe = wellbore_marker_frame_dataframe,
                                                                           trajectory = trajectory)
    # --------- Act ----------
    # Get the marker object from an interpretation object and interpretation object uuid
    found_marker1 = wellbore_marker_frame.find_marker_from_interp(interpetation_obj = fault_interp_1)
    found_marker2 = wellbore_marker_frame.find_marker_from_interp(uuid = horizon_interp_2_uuid)

    # --------- Assert ----------
    assert found_marker1 == wellbore_marker_frame.wellbore_marker_list[2]
    assert found_marker2 == wellbore_marker_frame.wellbore_marker_list[1]


def test_find_marker_from_index(example_model_and_crs):

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
        'Type': ['horizon', 'horizon', 'fault'],
        'Surface': ['marker_horizon_1', 'marker_horizon_2', 'marker_fault_1'],
        'Interp_Surface': ['horizon_interp_1', 'horizon_interp_2', 'fault_interp_1'],
        'Well': well_name
    })

    # Create a wellbore marker frame from a dataframe
    wellbore_marker_frame = resqpy.well.WellboreMarkerFrame.from_dataframe(parent_model = model,
                                                                           dataframe = wellbore_marker_frame_dataframe,
                                                                           trajectory = trajectory)
    print(wellbore_marker_frame.wellbore_marker_list)
    # --------- Act ----------
    # Get the marker object from an interpretation object and interpretation object uuid
    found_marker1 = wellbore_marker_frame.find_marker_from_index(1)

    # --------- Assert ----------
    assert found_marker1[1] == 'horizon'
