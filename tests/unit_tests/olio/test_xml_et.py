import pytest
import resqpy.olio.xml_et as rqet


def test_list_obj_references(example_model_with_well):

    model, well_interp, datum, traj = example_model_with_well

    i_refs = rqet.list_obj_references(well_interp.root)
    d_refs = rqet.list_obj_references(datum.root)
    t_refs = rqet.list_obj_references(traj.root)
    t_refs_with_hdf5 = rqet.list_obj_references(traj.root, skip_hdf5 = False)

    assert len(i_refs) == 1  # feature reference
    assert len(d_refs) == 1  # crs reference
    assert len(t_refs) == 3  # interp, crs & datum references
    assert len(t_refs_with_hdf5) == 5  # as above, plus mds and control points hdf5 references
