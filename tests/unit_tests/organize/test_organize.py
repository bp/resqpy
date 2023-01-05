import pytest

import resqpy.organize as rqo
from resqpy.model import Model
from resqpy.organize._utils import equivalent_extra_metadata, equivalent_chrono_pairs


# Test saving and loading from disk
@pytest.mark.parametrize(
    "cls, data",
    [
        (
            rqo.OrganizationFeature,
            dict(feature_name = 'hello', organization_kind = 'stratigraphic'),
        ),
        (
            rqo.GeobodyFeature,
            dict(feature_name = 'hi'),
        ),
        (
            rqo.BoundaryFeature,
            dict(feature_name = 'foobar'),
        ),
        (
            rqo.FrontierFeature,
            dict(feature_name = 'foobar'),
        ),
        (
            rqo.GeologicUnitFeature,
            dict(feature_name = 'foobar'),
        ),
        (
            rqo.FluidBoundaryFeature,
            dict(feature_name = 'foobar', kind = 'gas oil contact'),
        ),
        (
            rqo.TectonicBoundaryFeature,
            dict(feature_name = 'foobar', kind = 'fracture'),
        ),
        (
            rqo.GeneticBoundaryFeature,
            dict(feature_name = 'foobar',
                 kind = 'geobody boundary'),  # TODO: Allow age to be passed into init so coverage can be increased
        ),
        (
            rqo.WellboreFeature,
            dict(feature_name = 'foobar'),
        ),
    ])
def test_organize_classes(tmp_model, cls, data):
    # Load example model from a fixture
    model = tmp_model
    epc = model.epc_file

    # Create the feature
    obj = cls(parent_model = model, **data)
    uuid = obj.uuid

    # Save to disk
    obj.create_xml()
    model.store_epc()
    model.h5_release()

    # Reload from disk
    del model, obj
    model2 = Model(epc_file = epc)
    obj2 = cls(parent_model = model2, uuid = uuid)

    # Check all attributes were loaded correctly
    for key, expected_value in data.items():
        assert getattr(obj2, key) == expected_value, f"Error for {key}"


def test_RockFluidUnitFeature(tmp_model):
    # Create the features
    top = rqo.BoundaryFeature(tmp_model, feature_name = 'the top')
    base = rqo.BoundaryFeature(tmp_model, feature_name = 'the base')
    rfuf_1 = rqo.RockFluidUnitFeature(parent_model = tmp_model,
                                      feature_name = 'foobar',
                                      phase = 'seal',
                                      top_boundary_feature = top,
                                      base_boundary_feature = base)
    uuid = rfuf_1.uuid

    # Save to disk
    top.create_xml()
    base.create_xml()
    rfuf_1.create_xml()
    tmp_model.store_epc()

    # Reload from disk
    rfuf_2 = rqo.RockFluidUnitFeature(parent_model = tmp_model, uuid = uuid)

    # Check properties the same
    assert rfuf_2.feature_name == 'foobar'
    assert rfuf_2.phase == 'seal'
    assert rfuf_2.top_boundary_feature.feature_name == 'the top'
    assert rfuf_2.base_boundary_feature.feature_name == 'the base'


def test_FaultInterp(tmp_model):
    title = "fault interpretation"
    tect_boundary = rqo.TectonicBoundaryFeature(tmp_model, kind = 'fault')
    fault_interp = rqo.FaultInterpretation(
        tmp_model,
        tectonic_boundary_feature = tect_boundary,
        title = title,
        domain = "depth",
        is_normal = True,
        maximum_throw = 3,
        mean_dip = 1,
        mean_azimuth = 2,
    )

    tect_boundary.create_xml()
    fault_interp.create_xml()

    fault_interp_2 = rqo.FaultInterpretation(tmp_model, uuid = fault_interp.uuid)
    assert fault_interp_2.title == title
    assert fault_interp_2.maximum_throw == 3


def test_EarthModelInterp(tmp_model):
    title = 'gaia'
    org_feat = rqo.OrganizationFeature(tmp_model, feature_name = 'marie kondo', organization_kind = "earth model")
    em1 = rqo.EarthModelInterpretation(tmp_model, title = title, organization_feature = org_feat)

    org_feat.create_xml()
    em1.create_xml()

    em2 = rqo.EarthModelInterpretation(tmp_model, uuid = em1.uuid)
    assert em2.title == title


def test_GenericInterp(tmp_model):
    title = 'anything goes'
    org_feat = rqo.OrganizationFeature(tmp_model, feature_name = 'mystery', organization_kind = "earth model")
    gi = rqo.GenericInterpretation(tmp_model, title = title, feature_uuid = org_feat.uuid, domain = "time")

    org_feat.create_xml()
    gi.create_xml()

    gi_reincarnated = rqo.GenericInterpretation(tmp_model, uuid = gi.uuid)
    assert gi_reincarnated.title == title


def test_HorizonInterp(tmp_model):
    gen = rqo.GeneticBoundaryFeature(tmp_model, kind = 'horizon')
    hor = rqo.HorizonInterpretation(tmp_model,
                                    genetic_boundary_feature = gen,
                                    sequence_stratigraphy_surface = 'maximum flooding')

    gen.create_xml()
    hor.create_xml()

    hor2 = rqo.HorizonInterpretation(tmp_model, uuid = hor.uuid)
    assert hor2.sequence_stratigraphy_surface == hor.sequence_stratigraphy_surface


def test_GeobodyBoundaryInterp(tmp_model):
    gen = rqo.GeneticBoundaryFeature(tmp_model, kind = 'geobody boundary')
    gb = rqo.GeobodyBoundaryInterpretation(tmp_model, genetic_boundary_feature = gen)

    gen.create_xml()
    gb.create_xml()
    gb2 = rqo.GeobodyBoundaryInterpretation(tmp_model, uuid = gb.uuid)
    assert gb == gb2


def test_GeobodyInterp(tmp_model):
    gi = rqo.GeobodyFeature(tmp_model)
    gb = rqo.GeobodyInterpretation(tmp_model,
                                   geobody_feature = gi,
                                   domain = 'depth',
                                   composition = 'intrusive clay',
                                   material_implacement = 'autochtonous',
                                   geobody_shape = 'silt')

    gi.create_xml()
    gb.create_xml()
    gb2 = rqo.GeobodyInterpretation(tmp_model, uuid = gb.uuid)
    assert gb == gb2


def test_BoundaryFeatureInterp(tmp_model):
    gen = rqo.BoundaryFeature(tmp_model)
    bfi = rqo.BoundaryFeatureInterpretation(tmp_model, boundary_feature = gen)

    gen.create_xml()
    bfi.create_xml()
    bfi2 = rqo.BoundaryFeatureInterpretation(tmp_model, uuid = bfi.uuid)
    assert bfi == bfi2


def test_wellbore_interp_title(tmp_model):
    # Create a feature and interp objects
    feature_name = 'well A'
    well_feature = rqo.WellboreFeature(tmp_model, feature_name = feature_name)
    well_feature.create_xml()
    well_interp_1 = rqo.WellboreInterpretation(tmp_model, wellbore_feature = well_feature, is_drilled = True)
    well_interp_1.create_xml()

    # Create a duplicate object, loading from XML
    well_interp_2 = rqo.WellboreInterpretation(tmp_model, uuid = well_interp_1.uuid)

    # Check feature name is present
    assert well_interp_1.title == feature_name
    assert well_interp_2.title == feature_name


@pytest.mark.parametrize("pair_one, pair_two, model", [(
    None,
    'string',
    None,
), (
    'string_2',
    None,
    None,
), (
    (None, None),
    'string_3',
    None,
), (
    'string_4',
    (None, None),
    None,
), (
    'string_5',
    'string_6',
    'Model',
)])
def test_equivalent_chrono_pairs(pair_one, pair_two, model):
    # Arrange

    # Act
    result = equivalent_chrono_pairs(pair_one, pair_two)

    # Assert
    assert result is False


def test_equivalent_extra_metadata_false(tmp_model, mocker):
    # Arrange
    bf_1 = tmp_model
    bf_2 = rqo.BoundaryFeature("test")

    mocker.patch('resqpy.olio.xml_et.load_metadata_from_xml', return_value = [1, 2, 3, 4, 5])

    # Act
    result = equivalent_extra_metadata(bf_1, bf_2)

    # Assert
    assert result is False


def test_equivalent_extra_metadata_true(tmp_model, mocker):
    # Arrange
    bf_1 = tmp_model
    bf_2 = tmp_model

    mocker.patch('resqpy.olio.xml_et.load_metadata_from_xml', return_value = [1, 2, 3, 4, 5])

    # Act
    result = equivalent_extra_metadata(bf_1, bf_2)

    # Assert
    assert result is True


def test_BoundaryFeatureInterpretation_is_equivalent_true(tmp_model):
    # Arrange
    gen = rqo.BoundaryFeature(tmp_model)
    bfi_1 = rqo.BoundaryFeatureInterpretation(tmp_model, boundary_feature = gen)
    bfi_2 = bfi_1

    # Act
    result = bfi_1.is_equivalent(bfi_2)

    # Assert
    assert result is True


@pytest.mark.parametrize(
    "bfi_2", [(None,), (12),
              (rqo.BoundaryFeatureInterpretation(
                  parent_model = "test", boundary_feature = rqo.BoundaryFeature("test_2"), title = 'title'))])
def test_BoundaryFeatureInterpretation_is_equivalent_false(tmp_model, bfi_2):
    # Arrange
    gen = rqo.BoundaryFeature(tmp_model)
    bfi_1 = rqo.BoundaryFeatureInterpretation(tmp_model, boundary_feature = gen)

    # Act
    result = bfi_1.is_equivalent(bfi_2)

    # Assert
    assert result is False


def test_EarthModelInterpretation_is_equivalent_true(tmp_model):
    # Arrange
    org = rqo.OrganizationFeature(tmp_model)
    emi_1 = rqo.EarthModelInterpretation(tmp_model, organization_feature = org)
    emi_2 = emi_1

    # Act
    result = emi_1.is_equivalent(emi_2)

    # Assert
    assert result is True


@pytest.mark.parametrize("emi_2", [(None,), (12),
                                   (rqo.EarthModelInterpretation(parent_model = "test", title = 'title'))])
def test_EarthModelInterpretation_is_equivalent_false(tmp_model, emi_2):
    # Arrange
    org = rqo.OrganizationFeature(tmp_model)
    emi_1 = rqo.EarthModelInterpretation(tmp_model, organization_feature = org)

    # Act
    result = emi_1.is_equivalent(emi_2)

    # Assert
    assert result is False


def test_FaultInterpretation_is_equivalent_true(tmp_model):
    # Arrange
    fi_1 = rqo.FaultInterpretation(tmp_model)
    fi_2 = fi_1

    fi_1.throw_interpretation_list = [1, 2, 3]
    fi_2.throw_interpretation_list = [1, 2, 3]

    # Act
    result = fi_1.is_equivalent(fi_2)

    # Assert
    assert result is True


@pytest.mark.parametrize("fi_2", [
    (None),
    (12),
    (rqo.FaultInterpretation('model', mean_azimuth = 1, mean_dip = 0)),
    (rqo.FaultInterpretation('model', mean_azimuth = 0, mean_dip = 1)),
    (rqo.FaultInterpretation('model', mean_azimuth = 0, mean_dip = 1)),
    (rqo.FaultInterpretation('model', mean_azimuth = 0, mean_dip = 1)),
    (rqo.FaultInterpretation('model', mean_azimuth = 0, mean_dip = 0, extra_metadata = {'metadata': 'metadata'})),
    (rqo.FaultInterpretation('model', mean_azimuth = 0, mean_dip = 0, extra_metadata = {'metadata': 'metadata'})),
    (rqo.FaultInterpretation('model', mean_azimuth = 0, mean_dip = 0, extra_metadata = {'metadata': 'metadata'})),
    (rqo.FaultInterpretation('model', mean_azimuth = 0, mean_dip = 0, extra_metadata = {'metadata': 'metadata'})),
])
def test_FaultInterpretation_is_equivalent_false(tmp_model, fi_2):
    # Arrange
    fi_1 = rqo.FaultInterpretation(tmp_model, mean_azimuth = 0, mean_dip = 0)

    # Act
    result = fi_1.is_equivalent(fi_2, check_extra_metadata = True)

    # Assert
    assert result is False


@pytest.mark.parametrize("fi_1_throw_list, fi_2_throw_list, expected_output", [
    (
        ['string', 'left', 'right'],
        ['string', 'left', 'right'],
        True,
    ),
    (
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        False,
    ),
    (
        [1, 2, 3, 4],
        None,
        False,
    ),
    (
        ['string', 'left', 'right'],
        ['ptring', 'left', 'right'],
        False,
    ),
    (
        ['string', 'left', 'right'],
        ['string', 'reft', 'right'],
        False,
    ),
])
def test_FaultInterpretation_is_equivalent_throw_interpretation_list_false(tmp_model, fi_1_throw_list, fi_2_throw_list,
                                                                           expected_output):
    # Arrange
    fi_1 = rqo.FaultInterpretation(tmp_model, mean_azimuth = 0, mean_dip = 0)
    fi_2 = rqo.FaultInterpretation(tmp_model, mean_azimuth = 0, mean_dip = 0)
    fi_2.throw_interpretation_list = fi_2_throw_list
    fi_1.throw_interpretation_list = fi_1_throw_list

    # Act
    result = fi_1.is_equivalent(fi_2, check_extra_metadata = True)

    # Assert
    assert result is expected_output


def test_FluidBoundaryFeature_is_equivalent_true(tmp_model):
    # Arrange
    fb_1 = rqo.FluidBoundaryFeature(tmp_model)
    fb_2 = fb_1

    # Act
    result = fb_1.is_equivalent(fb_2)

    # Assert
    assert result is True


@pytest.mark.parametrize("fb_2", [
    (None),
    (12),
    (rqo.FluidBoundaryFeature('model', feature_name = 'name')),
    (rqo.FluidBoundaryFeature('model', extra_metadata = {'metadata': 'metadata'})),
])
def test_FluidBoundaryFeature_is_equivalent_false(tmp_model, fb_2):
    # Arrange
    fb_1 = rqo.FluidBoundaryFeature(tmp_model)

    # Act
    result = fb_1.is_equivalent(fb_2, check_extra_metadata = True)

    # Assert
    assert result is False


def test_FrontierFeature_is_equivalent_true(tmp_model):
    # Arrange
    ff_1 = rqo.FrontierFeature(tmp_model)
    ff_2 = ff_1

    # Act
    result = ff_1.is_equivalent(ff_2)

    # Assert
    assert result is True


@pytest.mark.parametrize("ff_2", [
    (None),
    (12),
    (rqo.FrontierFeature('model', feature_name = 'name')),
    (rqo.FrontierFeature('model', extra_metadata = {'metadata': 'metadata'})),
])
def test_FrontierFeature_is_equivalent_false(tmp_model, ff_2):
    # Arrange
    ff_1 = rqo.FrontierFeature(tmp_model)

    # Act
    result = ff_1.is_equivalent(ff_2, check_extra_metadata = True)

    # Assert
    assert result is False


def test_GenericInterpretation_is_equivalent_true(tmp_model):
    # Arrange
    gi_1 = rqo.GenericInterpretation(tmp_model)
    gi_2 = gi_1

    # Act
    result = gi_1.is_equivalent(gi_1)

    # Assert
    assert result is True


@pytest.mark.parametrize("gi_2", [
    (None),
    (12),
    (rqo.GenericInterpretation('model')),
    (rqo.GenericInterpretation('model', domain = 'depth')),
    (rqo.GenericInterpretation('model', title = 'title')),
    (rqo.GenericInterpretation('model', extra_metadata = {'metadata': 'metadata'})),
])
def test_GenericInterpretation_is_equivalent_false(tmp_model, gi_2):
    # Arrange
    gi_1 = rqo.GenericInterpretation(tmp_model)

    # Act
    result = gi_1.is_equivalent(gi_2, check_extra_metadata = True)

    # Assert
    assert result is False


def test_GeneticBoundaryFeature_is_equivalent_true(tmp_model):
    # Arrange
    gbf_1 = rqo.GeneticBoundaryFeature(tmp_model)
    gbf_2 = gbf_1

    # Act
    result = gbf_1.is_equivalent(gbf_2)

    # Assert
    assert result is True


@pytest.mark.parametrize("gbf_2", [
    (None),
    (12),
    (rqo.GeneticBoundaryFeature('model', feature_name = 'name')),
    (rqo.GeneticBoundaryFeature('model', extra_metadata = {'metadata': 'metadata'})),
])
def test_GeneticBoundaryFeature_is_equivalent_false(tmp_model, gbf_2):
    # Arrange
    gbf_1 = rqo.GeneticBoundaryFeature(tmp_model)

    # Act
    result = gbf_1.is_equivalent(gbf_2, check_extra_metadata = True)

    # Assert
    assert result is False


def test_GeobodyBoundaryInterpretation_is_equivalent_true(tmp_model):
    # Arrange
    gbi_1 = rqo.GeobodyBoundaryInterpretation(tmp_model)
    gbi_2 = gbi_1

    # Act
    result = gbi_1.is_equivalent(gbi_2)

    # Assert
    assert result is True


@pytest.mark.parametrize("gbi_2", [
    (None),
    (12),
])
def test_GeobodyBoundaryInterpretation_is_equivalent_false(tmp_model, gbi_2):
    # Arrange
    gbf = rqo.GeneticBoundaryFeature(tmp_model)
    gbi_1 = rqo.GeobodyBoundaryInterpretation(tmp_model, genetic_boundary_feature = gbf)

    # Act
    result = gbi_1.is_equivalent(gbi_2, check_extra_metadata = True)

    # Assert
    assert result is False


def test_GeobodyBoundaryInterpretation_is_equivalent_false_2(tmp_model):
    # Arrange
    gbf = rqo.GeneticBoundaryFeature(tmp_model)
    gbi_1 = rqo.GeobodyBoundaryInterpretation(tmp_model, genetic_boundary_feature = gbf)
    gbi_2 = rqo.GeobodyBoundaryInterpretation(tmp_model)

    # Act
    result = gbi_1.is_equivalent(gbi_2, check_extra_metadata = True)
    result_2 = gbi_2.is_equivalent(gbi_1, check_extra_metadata = True)

    # Assert
    assert result is False
    assert result_2 is False


def test_GeobodyBoundaryInterpretation_is_equivalent_false_3(tmp_model):
    # Arrange
    gbf = rqo.GeneticBoundaryFeature(tmp_model)
    gbi_1 = rqo.GeobodyBoundaryInterpretation(tmp_model, genetic_boundary_feature = gbf, domain = 'depth')
    gbi_2 = rqo.GeobodyBoundaryInterpretation(tmp_model,
                                              genetic_boundary_feature = gbf,
                                              domain = 'depth',
                                              extra_metadata = {'metadata': 'metadata'})

    # Act
    result = gbi_1.is_equivalent(gbi_2, check_extra_metadata = True)
    result_2 = gbi_2.is_equivalent(gbi_1, check_extra_metadata = True)

    # Assert
    assert result is False
    assert result_2 is False


def test_GeobodyBoundaryInterpretation_is_equivalent_false_4(tmp_model):
    # Arrange
    gbf = rqo.GeneticBoundaryFeature(tmp_model)
    gbi_1 = rqo.GeobodyBoundaryInterpretation(tmp_model,
                                              genetic_boundary_feature = gbf,
                                              domain = 'depth',
                                              extra_metadata = {'metadata': 'metadata'})
    gbi_2 = rqo.GeobodyBoundaryInterpretation(tmp_model,
                                              genetic_boundary_feature = gbf,
                                              domain = 'test',
                                              extra_metadata = {'metadata': 'metadata'})

    # Act
    result = gbi_1.is_equivalent(gbi_2, check_extra_metadata = True)
    result_2 = gbi_2.is_equivalent(gbi_1, check_extra_metadata = True)

    # Assert
    assert result is False
    assert result_2 is False


@pytest.mark.parametrize("relation_list_1, relation_list_2, expected_result", [(
    None,
    None,
    True,
), (
    None,
    [1, 2, 3, 4],
    False,
), ([1, 2, 3], [1, 2, 3], True), ([1, 2, 3, 4], [1, 2, 3], False)])
def test_GeobodyBoundaryInterpretation_is_equivalent_5(tmp_model, relation_list_1, relation_list_2, expected_result):
    # Arrange
    gbf = rqo.GeneticBoundaryFeature(tmp_model)
    gbi_1 = rqo.GeobodyBoundaryInterpretation(tmp_model,
                                              genetic_boundary_feature = gbf,
                                              domain = 'depth',
                                              extra_metadata = {'metadata': 'metadata'},
                                              boundary_relation_list = relation_list_1)
    gbi_2 = rqo.GeobodyBoundaryInterpretation(tmp_model,
                                              genetic_boundary_feature = gbf,
                                              domain = 'depth',
                                              extra_metadata = {'metadata': 'metadata'},
                                              boundary_relation_list = relation_list_2)

    # Act
    result = gbi_1.is_equivalent(gbi_2, check_extra_metadata = True)
    result_2 = gbi_2.is_equivalent(gbi_1, check_extra_metadata = True)

    # Assert
    assert result is expected_result
    assert result_2 is expected_result


def test_GeobodyFeature_is_equivalent_true(tmp_model):
    # Arrange
    gf_1 = rqo.GeobodyFeature(tmp_model, feature_name = 'name')
    gf_2 = gf_1

    # Act
    result = gf_1.is_equivalent(gf_2)

    # Assert
    assert result is True


@pytest.mark.parametrize("gf_2", [
    (None),
    (12),
    (rqo.GeobodyFeature('model', feature_name = 'name')),
])
def test_GeobodyFeature_is_equivalent_false(tmp_model, gf_2):
    # Arrange
    gf_1 = rqo.GeobodyFeature(tmp_model, feature_name = 'test')

    # Act
    result = gf_1.is_equivalent(gf_2, check_extra_metadata = True)

    # Assert
    assert result is False


def test_GeobodyInterpretation_is_equivalent_true(tmp_model):
    # Arrange
    gi_1 = rqo.GeobodyInterpretation(tmp_model)
    gi_2 = gi_1

    # Act
    result = gi_1.is_equivalent(gi_2)

    # Assert
    assert result is True


@pytest.mark.parametrize("gi_2", [(None), (12)])
def test_GeobodyInterpretation_is_equivalent_false(tmp_model, gi_2):
    # Arrange
    gi_1 = rqo.GeobodyInterpretation(tmp_model)

    # Act
    result = gi_1.is_equivalent(gi_2, check_extra_metadata = True)

    # Assert
    assert result is False


@pytest.mark.parametrize("extra_metadata_1, extra_metadata_2, domain_1, domain_2, expected_result", [({
    'metadata': 'metadata'
}, {
    'metadata': 'test'
}, None, None, False), ({
    'metadata': 'metadata'
}, {
    'metadata': 'metadata'
}, 'depth', 'depth', True), ({
    'metadata': 'metadata'
}, {
    'metadata': 'metadata'
}, 'depth', 'false', False)])
def test_GeobodyInterpretation_is_equivalent_false_2(tmp_model, extra_metadata_1, extra_metadata_2, domain_1, domain_2,
                                                     expected_result):
    # Arrange
    gf = rqo.GeobodyFeature(tmp_model)
    gi_1 = rqo.GeobodyInterpretation(
        tmp_model,
        geobody_feature = gf,
        extra_metadata = extra_metadata_1,
        domain = domain_1,
    )
    gi_2 = rqo.GeobodyInterpretation(tmp_model,
                                     geobody_feature = gf,
                                     extra_metadata = extra_metadata_2,
                                     domain = domain_2)

    # Act
    result = gi_1.is_equivalent(gi_2, check_extra_metadata = True)

    # Assert
    assert result is expected_result


def test_GeobodyInterpretation_is_equivalent_false_3(tmp_model):
    # Arrange
    gf = rqo.GeobodyFeature(tmp_model)
    gi_1 = rqo.GeobodyInterpretation(tmp_model, geobody_feature = gf)
    gi_2 = rqo.GeobodyInterpretation(tmp_model, geobody_feature = None)

    # Act
    result = gi_1.is_equivalent(gi_2, check_extra_metadata = True)

    # Assert
    assert result is False


def test_GeobodyInterpretation_is_equivalent_false_4(tmp_model):
    # Arrange
    gf_1 = rqo.GeobodyFeature(tmp_model)
    gf_2 = rqo.GeobodyFeature(tmp_model, feature_name = 'title')
    gi_1 = rqo.GeobodyInterpretation(tmp_model, geobody_feature = gf_1)
    gi_2 = rqo.GeobodyInterpretation(tmp_model, geobody_feature = gf_2)

    # Act
    result = gi_1.is_equivalent(gi_2, check_extra_metadata = True)

    # Assert
    assert result is False


def test_GeobodyInterpretation_is_equivalent_false_5(tmp_model):
    # Arrange
    gf = rqo.GeobodyFeature(tmp_model)
    gi_1 = rqo.GeobodyInterpretation(tmp_model, geobody_feature = None)
    gi_2 = rqo.GeobodyInterpretation(tmp_model, geobody_feature = gf)

    # Act
    result = gi_1.is_equivalent(gi_2, check_extra_metadata = True)

    # Assert
    assert result is False


def test_GeologicalUnitFeature_is_equivalent_true(tmp_model):
    # Arrange
    guf_1 = rqo.GeologicUnitFeature(tmp_model, feature_name = 'name')
    guf_2 = guf_1

    # Act
    result = guf_1.is_equivalent(guf_2)

    # Assert
    assert result is True


@pytest.mark.parametrize("guf_2", [
    (None),
    (12),
    (rqo.GeologicUnitFeature(
        'model',
        feature_name = 'name',
        extra_metadata = {'metadata': 'metadata'},
    )),
])
def test_GeologicUnitFeature_is_equivalent_false(tmp_model, guf_2):
    # Arrange
    guf_1 = rqo.GeologicUnitFeature(
        tmp_model,
        feature_name = 'test',
        extra_metadata = {'metadata': 'metadata'},
    )

    # Act
    result = guf_1.is_equivalent(guf_2, check_extra_metadata = True)

    # Assert
    assert result is False


def test_HorizonInterpretation_is_equivalent_true(tmp_model):
    # Arrange
    hi_1 = rqo.HorizonInterpretation(tmp_model)
    hi_2 = hi_1

    # Act
    result = hi_1.is_equivalent(hi_2)

    # Assert
    assert result is True


@pytest.mark.parametrize("hi_2", [
    (None),
    (12),
])
def test_HorizonInterpretation_is_equivalent_false(tmp_model, hi_2):
    # Arrange
    hi_1 = rqo.HorizonInterpretation(
        tmp_model,
        extra_metadata = {'metadata': 'test'},
    )

    # Act
    result = hi_1.is_equivalent(hi_2, check_extra_metadata = True)

    # Assert
    assert result is False


def test_HorizonInterpretation_is_equivalent_false_2(tmp_model):
    # Arrange
    hi_1 = rqo.HorizonInterpretation(
        tmp_model,
        extra_metadata = {'metadata': 'metadata'},
    )
    hi_2 = rqo.HorizonInterpretation(
        tmp_model,
        extra_metadata = {'metadata': 'test'},
    )

    # Act
    result = hi_1.is_equivalent(hi_2, check_extra_metadata = True)

    # Assert
    assert result is False


def test_RockFluidUnitFeature_is_equivalent_true(tmp_model):
    # Arrange
    top = rqo.BoundaryFeature(tmp_model, feature_name = 'the top')
    base = rqo.BoundaryFeature(tmp_model, feature_name = 'the base')
    rfuf_1 = rqo.RockFluidUnitFeature(parent_model = tmp_model,
                                      feature_name = 'foobar',
                                      phase = 'seal',
                                      top_boundary_feature = top,
                                      base_boundary_feature = base)
    rfuf_2 = rqo.RockFluidUnitFeature(parent_model = tmp_model,
                                      feature_name = 'foobar',
                                      phase = 'seal',
                                      top_boundary_feature = top,
                                      base_boundary_feature = base)
    # Act
    result = rfuf_1.is_equivalent(rfuf_2)

    # Assert
    assert result is True


@pytest.mark.parametrize("rfuf_2", [(None,), (12), (rqo.RockFluidUnitFeature(parent_model = "test"))])
def test_RockFluidUnitFeature_is_equivalent_false(tmp_model, rfuf_2):
    # Arrange
    top = rqo.BoundaryFeature(tmp_model, feature_name = 'the top')
    base = rqo.BoundaryFeature(tmp_model, feature_name = 'the base')
    rfuf_1 = rqo.RockFluidUnitFeature(parent_model = tmp_model,
                                      feature_name = 'foobar',
                                      phase = 'seal',
                                      top_boundary_feature = top,
                                      base_boundary_feature = base)
    # Act
    result = rfuf_1.is_equivalent(rfuf_2)

    # Assert
    assert result is False
