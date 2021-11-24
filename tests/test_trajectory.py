import numpy as np

import resqpy.well


def test_Trajectory_add_well_feature_and_interp(example_model_and_crs):

    # Prepare an example Trajectory without a well feature
    wellname = "Hullabaloo"
    model, crs = example_model_and_crs
    datum = resqpy.well.MdDatum(parent_model = model,
                                crs_uuid = crs.uuid,
                                location = (0, 0, -100),
                                md_reference = 'kelly bushing')
    datum.create_xml()
    traj = resqpy.well.Trajectory(parent_model = model, md_datum = datum, well_name = wellname)

    # Add the well interp
    assert traj.wellbore_feature is None
    assert traj.wellbore_interpretation is None
    traj.create_feature_and_interpretation()

    # Check well is present
    assert traj.wellbore_feature is not None
    assert traj.wellbore_feature.feature_name == wellname

def test_compute_from_deviation_survey(example_model_with_well):

    # --------- Arrange ----------
    # Create a Deviation Survey object in memory

    # Load example model from a fixture
    model, well_interp, datum, _ = example_model_with_well
    epc_path = model.epc_file

    # Create a deviation survey
    data = dict(
        title='Majestic Umlaut ö',
        originator='Thor, god of sparkles',
        md_uom='ft',
        angle_uom='rad',
        is_final=True,
    )
    array_data = dict(
        measured_depths=np.array([1, 2, 3], dtype=float) + 1000.0,
        azimuths=np.array([4, 5, 6], dtype=float),
        inclinations=np.array([7, 8, 9], dtype=float),
        first_station=np.array([0, -1, 999], dtype=float),
    )

    survey = resqpy.well.DeviationSurvey(
        parent_model=model,
        represented_interp=well_interp,
        md_datum=datum,
        **data,
        **array_data,
    )

    # --------- Act ----------
    # Create a trajectory from that deviation survey
    trajectory_from_deviation_survey = resqpy.well.Trajectory(parent_model = model, deviation_survey = survey)

    # --------- Assert ----------
    assert trajectory_from_deviation_survey is not None
    assert trajectory_from_deviation_survey.knot_count == survey.station_count == trajectory_from_deviation_survey.control_points.shape[0]
    assert trajectory_from_deviation_survey.md_datum == datum


