import pytest

import resqpy.lines
import resqpy.organize
import numpy as np


def test_lines(example_model_and_crs):

    # Set up a Polyline
    title = 'Nazca'
    model, crs = example_model_and_crs
    line = resqpy.lines.Polyline(
        parent_model=model, title=title,
        set_crs=crs.uuid, set_crsroot=crs.crs_root,
        set_bool=True, set_coord=np.array([[0,0,0],[1,1,1]])
    )
    line.create_xml()

    # Add a interpretation
    assert line.rep_int_root is None
    line.create_interpretation_and_feature(kind='fault')
    assert line.rep_int_root is not None

    # Check fault can be loaded in again
    model.store_epc()
    fault_interp = resqpy.organize.FaultInterpretation(
        model, uuid=line.rep_int_uuid
    )
    fault_feature = resqpy.organize.TectonicBoundaryFeature(
        model, uuid=fault_interp.feature_uuid
    )

    # Check title matches expected title
    assert fault_feature.feature_name == title
