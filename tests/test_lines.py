import resqpy.lines
import resqpy.organize
import numpy as np


def test_lines(example_model):

    # Set up a Polyline
    title = 'Nazca'
    model, crs = example_model
    line = resqpy.lines.Polyline(
        parent_model=model, set_title=title, 
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
        model, root_node=line.rep_int_root
    )
    fault_feature = resqpy.organize.TectonicBoundaryFeature(
        model, root_node=fault_interp.feature_root
    )

    # Check title matches expected title
    assert fault_feature.feature_name == title
