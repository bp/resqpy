import numpy as np
import pandas as pd

import resqpy.well
from resqpy.model import Model


def test_WellboreFrame(example_model_and_crs):
    # Test that all attributes are correctly saved and loaded from disk

    # --------- Arrange ----------
    # Create a WellboreFrame object in memory
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

    # Create a wellbore frame object
    wellbore_frame_mds = np.array([305, 315])
    wellbore_frame = resqpy.well.WellboreFrame(parent_model = model,
                                               trajectory = trajectory,
                                               mds = wellbore_frame_mds,
                                               title = 'WellboreFrame_1',
                                               originator = 'Person_1')

    # ----------- Act ---------
    # Save to disk
    wellbore_frame.write_hdf5()
    wellbore_frame.create_xml()
    wellbore_frame_uuid = wellbore_frame.uuid
    model.store_epc()
    model.h5_release()

    # Clear memory
    del model, datum, trajectory, wellbore_frame

    # Reload from disk
    model2 = Model(epc_file = epc_path)
    wellbore_frame_2 = resqpy.well.WellboreFrame(parent_model = model2, uuid = wellbore_frame_uuid)

    # ----------- Assert ---------
    assert wellbore_frame_2.node_count == 2
    np.testing.assert_equal(wellbore_frame_2.node_mds, wellbore_frame_mds)


def test_create_feature_and_intrepretation(example_model_and_crs):
    # Test that WellboreFeature and WellboreInterpretation objects can be added to WellboreFrame object

    # --------- Arrange ----------
    # Create a WellboreFrame object without a well feature
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

    # Create a wellbore frame object
    wellbore_frame_mds = np.array([305, 315])
    wellbore_frame = resqpy.well.WellboreFrame(parent_model = model,
                                               trajectory = trajectory,
                                               mds = wellbore_frame_mds,
                                               title = 'WellboreFrame_1',
                                               originator = 'Person_1')

    assert wellbore_frame.wellbore_feature is None
    assert wellbore_frame.wellbore_interpretation is None

    # ----------- Act ---------
    wellbore_frame.create_feature_and_interpretation()

    # ----------- Assert ---------
    assert wellbore_frame.wellbore_feature is not None
    assert wellbore_frame.wellbore_feature.feature_name == wellbore_frame.title
