import os

import numpy as np
import pandas as pd

from resqpy.grid import RegularGrid
import resqpy.well

import pytest

def test_WellboreMarkerFrame(example_model_and_crs):
    # Test that all attributes are correctly saved and loaded from disk

    # --------- Arrange ----------
    # Create a WellboreMarkerFrame object in memory
    data = dict(
        title = 'Majestic Umlaut ö',
        originator = 'Thor, god of sparkles'
    )

    # Load example model from a fixture
    model, crs = example_model_and_crs

    # Create a trajectory
    elevation = 100
    datum = resqpy.well.MdDatum(parent_model=model,
                                crs_uuid=crs.uuid,
                                location=(0, 0, -elevation),
                                md_reference='kelly bushing')
    mds = np.array([300, 310, 330, 340])
    zs = mds - elevation
    source_dataframe = pd.DataFrame({
        'MD': mds,
        'X': [1, 2, 3, 4],
        'Y': [1, 2, 3, 4],
        'Z': zs,
    })
    trajectory = resqpy.well.Trajectory(parent_model=model,
                                        data_frame=source_dataframe,
                                        md_datum=datum,
                                        length_uom='m')
    wellbore_marker_frame = resqpy.well.WellboreMarkerFrame(parent_model = model,
                                                            trajectory = trajectory)
    assert wellbore_marker_frame is not None
